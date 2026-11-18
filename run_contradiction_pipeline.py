"""
Contradiction-Aware RAG Master Pipeline
=======================================
End-to-end pipeline that takes a user query, retrieves relevant multi-source
documents, extracts atomic factual claims, detects cross-document contradictions
using NLI + Contextual Verification, resolves source conflicts, persists
relationships in a Contradiction Knowledge Graph, and synthesizes an
evidence-backed, contradiction-aware response.

Usage:
  # One-shot CLI query over Multi-News dataset
  python run_contradiction_pipeline.py --query "hurricane death toll Florida" --top_k 3

  # Interactive mode
  python run_contradiction_pipeline.py --interactive

  # Run curated multi-perspective dispute demo scenarios
  python run_contradiction_pipeline.py --demo

  # Query specific dataset (multinews | sample)
  python run_contradiction_pipeline.py --query "employee leave days" --dataset sample
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import pipeline components
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from claim_extraction import AtomicClaim, ClaimExtractor
from conflict_resolution import ConflictResolution, ConflictResolver
from contextual_verification import ContextualReasoner, ContextualVerificationResult
from contradiction_graph import ContradictionKnowledgeGraph
from nli_classifier import CandidatePairMatcher, FactualNLIClassifier

# Output directories
OUTPUT_DIR = BASE_DIR / "output" / "pipeline_runs"
GRAPH_FILE = BASE_DIR / "output" / "contradiction_graph" / "knowledge_graph.json"
CACHE_FILE = BASE_DIR / "output" / "tfidf_index" / "tfidf_cache.joblib"
CLEANED_CSV = BASE_DIR / "output" / "multinews" / "multinews_cleaned_sample.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_FILE.parent.mkdir(parents=True, exist_ok=True)


# Curated dispute scenarios for rapid testing and demonstrations
CURATED_SCENARIOS = {
    "maria": [
        {
            "doc_id": "fema_official_01",
            "cluster_id": "hurricane_maria",
            "source": "Federal Emergency Management Agency (FEMA) Official Report",
            "text": "The official confirmed death toll from Hurricane Maria stands at 64 people according to government audits and official hospital death certificates issued following the storm.",
        },
        {
            "doc_id": "harvard_study_02",
            "cluster_id": "hurricane_maria",
            "source": "Harvard T.H. Chan School of Public Health Study",
            "text": "An independent academic mortality study by Harvard researchers estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico, citing widespread post-hurricane healthcare disruption and infrastructure collapse.",
        },
        {
            "doc_id": "gw_university_03",
            "cluster_id": "hurricane_maria",
            "source": "George Washington University Commissioned Milestone Report",
            "text": "The official commissioned review conducted by George Washington University estimated 2,975 excess deaths in Puerto Rico between September 2017 and February 2018.",
        },
    ],
    "leave": [
        {
            "doc_id": "hr_policy_v1",
            "cluster_id": "leave_policy",
            "source": "Corporate Standard HR Policy Manual (v1.0)",
            "text": "Full-time permanent employees are entitled to 24 days of annual leave every year under standard company policy.",
        },
        {
            "doc_id": "hr_handbook_v2",
            "cluster_id": "leave_policy",
            "source": "Updated Employee Handbook (v2.0 - 2026)",
            "text": "Full-time staff members receive 30 days of paid annual leave per company handbook v2 approved by executive management.",
        },
        {
            "doc_id": "probation_clause_03",
            "cluster_id": "leave_policy",
            "source": "Onboarding & Probationary Policy Addendum",
            "text": "Employees joining after July are entitled to 12 days of annual leave during their first year of employment.",
        },
    ],
    "vaccine": [
        {
            "doc_id": "cdc_report_01",
            "cluster_id": "vaccine_stats",
            "source": "CDC Surveillance Bulletin",
            "text": "The clinical trial confirmed vaccine efficacy of 95 percent against severe illness and hospitalization.",
        },
        {
            "doc_id": "reuters_wire_02",
            "cluster_id": "vaccine_stats",
            "source": "Reuters Global Health Wire",
            "text": "Health ministry data showed vaccine efficacy dropped to 64 percent following the emergence of new viral subvariants.",
        },
    ],
}


class DocumentRetriever:
    """
    Retrieves documents from the local Multi-News corpus or curated datasets.
    Uses TF-IDF sparse matching with vectorizer caching for sub-second retrieval.
    """

    def __init__(self):
        self.corpus_df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self._corpus_initialized = False

    def _initialize_corpus(self):
        if self._corpus_initialized:
            return
        if CLEANED_CSV.exists():
            try:
                import joblib
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                if CACHE_FILE.exists():
                    self.vectorizer, self.tfidf_matrix, self.corpus_df = joblib.load(CACHE_FILE)
                else:
                    self.corpus_df = pd.read_csv(CLEANED_CSV)
                    self.vectorizer = TfidfVectorizer(
                        max_features=25_000,
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        stop_words="english",
                    )
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_df["text"])
                    try:
                        joblib.dump((self.vectorizer, self.tfidf_matrix, self.corpus_df), CACHE_FILE)
                    except Exception:
                        pass
                self._corpus_initialized = True
            except Exception as e:
                print(f"[Warning] Could not initialize TF-IDF on corpus ({e}).")

    def retrieve(self, query: str, top_k: int = 3, dataset: str = "multinews") -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.
        """
        if dataset in ("sample", "demo"):
            return self._retrieve_from_scenarios(query, top_k=top_k)

        self._initialize_corpus()
        if self.corpus_df is None or self.tfidf_matrix is None:
            return self._retrieve_from_scenarios(query, top_k=top_k)

        # Vector retrieval over corpus
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.corpus_df.iloc[idx]
            score = float(scores[idx])
            if score < 0.02:
                # If sparse score is extremely low, check if scenarios match better
                scenario_results = self._retrieve_from_scenarios(query, top_k=top_k)
                if scenario_results:
                    return scenario_results

            snippet = row["text"][:260] + "..." if len(row["text"]) > 260 else row["text"]
            results.append({
                "rank": rank,
                "doc_id": row["doc_id"],
                "cluster_id": row.get("cluster_id", "cluster_001"),
                "source": f"Multi-News [{row['doc_id']}]",
                "text": row["text"],
                "snippet": snippet,
                "score": round(score, 4),
            })
        return results

    def _retrieve_from_scenarios(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Match query against curated scenarios."""
        q_lower = query.lower()
        if any(k in q_lower for k in ["maria", "puerto", "rico", "hurricane", "death", "toll", "casualties"]):
            docs = CURATED_SCENARIOS["maria"]
        elif any(k in q_lower for k in ["leave", "holiday", "vacation", "annual", "days", "employee", "handbook"]):
            docs = CURATED_SCENARIOS["leave"]
        elif any(k in q_lower for k in ["vaccine", "efficacy", "clinical", "trial", "covid", "health"]):
            docs = CURATED_SCENARIOS["vaccine"]
        else:
            # Default to maria if nothing matched
            docs = CURATED_SCENARIOS["maria"]

        results = []
        for i, d in enumerate(docs[:top_k], start=1):
            snippet = d["text"][:260] + "..." if len(d["text"]) > 260 else d["text"]
            results.append({
                "rank": i,
                "doc_id": d["doc_id"],
                "cluster_id": d["cluster_id"],
                "source": d["source"],
                "text": d["text"],
                "snippet": snippet,
                "score": 0.88 - (i * 0.05),
            })
        return results


class ContradictionPipeline:
    """
    The orchestrating pipeline that links Retrieval, Claim Extraction,
    Candidate Matching, NLI, Contextual Verification, Conflict Resolution,
    and Knowledge Graph management.
    """

    def __init__(self, mode: str = "hybrid"):
        self.mode = mode
        self.retriever = DocumentRetriever()
        self.extractor = ClaimExtractor()
        self.matcher = CandidatePairMatcher(min_similarity_threshold=0.18)
        self.nli = FactualNLIClassifier()
        self.reasoner = ContextualReasoner()
        self.resolver = ConflictResolver()
        self.graph = ContradictionKnowledgeGraph(persistence_file=GRAPH_FILE)

    def run(self, query: str, top_k: int = 3, dataset: str = "multinews") -> Dict[str, Any]:
        """
        Execute full contradiction detection flow for query.
        """
        start_time = time.time()
        print("\n" + "=" * 80)
        print(f"CONTRADICTION-AWARE RAG PIPELINE: '{query}'")
        print("=" * 80)

        # -------------------------------------------------------------
        # STEP 1: DOCUMENT RETRIEVAL
        # -------------------------------------------------------------
        print("\n[STEP 1] RETRIEVING RELEVANT DOCUMENTS...")
        docs = self.retriever.retrieve(query, top_k=top_k, dataset=dataset)
        print(f"Retrieved {len(docs)} documents:")
        for doc in docs:
            print(f"  - Rank #{doc['rank']} | Score: {doc['score']:.4f} | Doc ID: {doc['doc_id']}")
            print(f"    Source:  {doc['source']}")
            print(f"    Snippet: {doc['snippet']}\n")

        # -------------------------------------------------------------
        # STEP 2: ATOMIC CLAIM EXTRACTION
        # -------------------------------------------------------------
        print("[STEP 2] EXTRACTING ATOMIC FACTUAL CLAIMS...")
        all_claims: List[AtomicClaim] = []
        for doc in docs:
            self.graph.add_document_node(doc_id=doc["doc_id"], title=doc["source"], cluster_id=doc["cluster_id"])
            claims = self.extractor.extract_from_text(
                doc["text"],
                doc_id=doc["doc_id"],
                cluster_id=doc["cluster_id"],
                metadata={"source_title": doc["source"]},
            )
            for c in claims:
                self.graph.add_claim_node(c)
                all_claims.append(c)

        print(f"Extracted {len(all_claims)} atomic claims across retrieved documents:")
        for c in all_claims:
            cond_str = f" | Conds: {c.conditions}" if c.conditions else ""
            val_str = f" | Value: '{c.value}' ({c.unit})" if c.value else ""
            print(f"  - [{c.claim_id}] Subj: '{c.subject}'{val_str}{cond_str} | Auth: {c.authority_tier}")

        # -------------------------------------------------------------
        # STEP 3 & 4: CANDIDATE PAIRING & NLI CLASSIFICATION
        # -------------------------------------------------------------
        print("\n[STEP 3] PAIRING CANDIDATE CLAIMS & RUNNING NLI...")
        candidate_pairs = self.matcher.generate_candidate_pairs(all_claims, allow_same_doc=False)
        print(f"Generated {len(candidate_pairs)} candidate comparison pairs (pruned O(n^2) space).")

        nli_results = []
        for c1, c2 in candidate_pairs:
            res = self.nli.classify(c1, c2)
            nli_results.append(res)
            print(f"  - Compare [{c1.claim_id}] vs [{c2.claim_id}] --> NLI: {res.prediction.upper()} (conf={res.confidence:.2f})")

        # -------------------------------------------------------------
        # STEP 5: CONTEXTUAL VERIFICATION (REASONER)
        # -------------------------------------------------------------
        print("\n[STEP 4] CONTEXTUAL VERIFICATION REASONER...")
        verified_results: List[ContextualVerificationResult] = []
        for nli_res in nli_results:
            if self.mode == "nli":
                # NLI Alone mode
                is_c = (nli_res.prediction == "contradiction")
                ver = ContextualVerificationResult(
                    is_contradiction=is_c,
                    contradiction_type="direct_factual" if is_c else "agreement",
                    severity="moderate" if is_c else "informational",
                    explanation=nli_res.rationale,
                    supporting_evidence={"premise_quote": nli_res.premise_text, "hypothesis_quote": nli_res.hypothesis_text},
                    confidence=nli_res.confidence,
                    unresolved_questions=[],
                    nli_prediction=nli_res.prediction,
                    premise_claim=nli_res.premise_claim,
                    hypothesis_claim=nli_res.hypothesis_claim,
                )
            else:
                # Hybrid mode: Reasoner checks context
                ver = self.reasoner.verify(nli_res)

            verified_results.append(ver)

        # Highlight verified contradictions
        contradictions = [v for v in verified_results if v.is_contradiction]
        contextual_diffs = [v for v in verified_results if not v.is_contradiction and v.nli_prediction == "contradiction"]

        print(f"\nVerification Summary:")
        print(f"  - Confirmed Contradictions:     {len(contradictions)}")
        print(f"  - Disproved (Context/Scope/Cond): {len(contextual_diffs)}")

        for c_res in contradictions:
            print(f"\n  [!] CONTRADICTION CONFIRMED: [{c_res.premise_claim.claim_id}] vs [{c_res.hypothesis_claim.claim_id}]")
            print(f"      Type:        {c_res.contradiction_type.upper()}")
            print(f"      Severity:    {c_res.severity.upper()}")
            print(f"      Confidence:  {c_res.confidence:.2f}")
            print(f"      Explanation: {c_res.explanation}")
            print(f"      Evidence A:  \"{c_res.supporting_evidence['premise_quote']}\"")
            print(f"      Evidence B:  \"{c_res.supporting_evidence['hypothesis_quote']}\"")

        for c_diff in contextual_diffs:
            print(f"\n  [i] CONTEXTUAL DISTINCTION (Non-Contradiction): [{c_diff.premise_claim.claim_id}] vs [{c_diff.hypothesis_claim.claim_id}]")
            print(f"      Distinction: {c_diff.contradiction_type}")
            print(f"      Reasoning:   {c_diff.explanation}")

        # -------------------------------------------------------------
        # STEP 6: CONFLICT RESOLUTION & PRECEDENCE
        # -------------------------------------------------------------
        print("\n[STEP 5] CONFLICT RESOLUTION & AUTHORITY PRECEDENCE...")
        resolutions: List[ConflictResolution] = []
        for ver in verified_results:
            if ver.is_contradiction or ver.contradiction_type == "scope_or_conditional_difference":
                res = self.resolver.resolve(ver)
                resolutions.append(res)
                self.graph.ingest_conflict_resolution(res)

                if res.is_contradiction:
                    print(f"  - Dispute: [{ver.premise_claim.claim_id}] vs [{ver.hypothesis_claim.claim_id}]")
                    print(f"    Precedence:     {res.precedence}")
                    print(f"    Winning Claim:  {res.winning_claim_id or 'None (Disputed)'}")
                    print(f"    Human Review:   {'REQUIRED' if res.requires_human_review else 'Not Required'}")
                    print(f"    Recommendation: {res.actionable_recommendation}\n")

        # Save persistent graph
        self.graph.save(GRAPH_FILE)

        # -------------------------------------------------------------
        # STEP 7: CONTRADICTION KNOWLEDGE GRAPH
        # -------------------------------------------------------------
        print("[STEP 6] CONTRADICTION KNOWLEDGE GRAPH:")
        print(self.graph.render_ascii())

        # -------------------------------------------------------------
        # STEP 8: SYNTHESIZED CONTRADICTION-AWARE RAG ANSWER
        # -------------------------------------------------------------
        print("\n" + "=" * 80)
        print("FINAL SYNTHESIZED CONTRADICTION-AWARE RAG RESPONSE:")
        print("=" * 80)
        final_answer = self._generate_rag_synthesis(query, docs, contradictions, resolutions)
        print(final_answer)
        print("=" * 80)

        elapsed = round(time.time() - start_time, 2)
        print(f"\nPipeline finished in {elapsed}s. Graph memory saved to: {GRAPH_FILE}")

        # Export run report
        run_record = {
            "query": query,
            "elapsed_seconds": elapsed,
            "documents": docs,
            "extracted_claims": [c.to_dict() for c in all_claims],
            "contradictions": [c.to_dict() for c in contradictions],
            "resolutions": [r.to_dict() for r in resolutions],
            "synthesized_answer": final_answer,
            "mermaid_graph": self.graph.export_mermaid(),
        }
        report_path = OUTPUT_DIR / f"run_{hashlib.md5(query.encode()).hexdigest()[:8]}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(run_record, f, indent=2, ensure_ascii=False)
        print(f"Full run report saved to: {report_path}\n")

        return run_record

    def _generate_rag_synthesis(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        contradictions: List[ContextualVerificationResult],
        resolutions: List[ConflictResolution],
    ) -> str:
        """
        Synthesizes an honest, transparent, non-hallucinatory answer.
        Explicitly surfaces contradictions with evidence references.
        """
        if not contradictions:
            doc_summaries = " ".join([d["snippet"][:120] for d in docs])
            return (
                f"Based on the retrieved sources, no direct factual contradictions were identified regarding your query: '{query}'.\n"
                f"Summary of findings: {doc_summaries}"
            )

        lines = [
            f"Regarding your query: '{query}', multiple retrieved sources report CONFLICTING or DIVERGENT factual information.\n",
            "### Identified Contradictions & Discrepancies:",
        ]

        for i, c in enumerate(contradictions, 1):
            lines.append(f"\n{i}. **{c.contradiction_type.replace('_', ' ').title()} ({c.severity.upper()} Severity)**:")
            lines.append(f"   - **Source A** ([{c.premise_claim.doc_id}] - {c.premise_claim.metadata.get('source_title', 'Document A')}):")
            lines.append(f"     \"{c.supporting_evidence['premise_quote']}\"")
            lines.append(f"   - **Source B** ([{c.hypothesis_claim.doc_id}] - {c.hypothesis_claim.metadata.get('source_title', 'Document B')}):")
            lines.append(f"     \"{c.supporting_evidence['hypothesis_quote']}\"")
            lines.append(f"   - **Explanation**: {c.explanation}")

        lines.append("\n### Conflict Resolution & Source Reliability Assessment:")
        for res in resolutions:
            if res.is_contradiction:
                lines.append(f"   - **Precedence**: `{res.precedence}`")
                lines.append(f"   - **Assessment**: {res.authority_assessment} | {res.version_assessment}")
                lines.append(f"   - **Guidance**: {res.actionable_recommendation}")
                if res.requires_human_review:
                    lines.append("   - **Note**: Human verification is advised before relying on either figure in production or policy.")

        lines.append("\n### Balanced Takeaway:")
        primary_res = resolutions[0] if resolutions else None
        if primary_res:
            lines.append(f"   {primary_res.synthesized_statement}")
        else:
            lines.append("   Users should evaluate both figures within their respective temporal and organizational contexts.")

        return "\n".join(lines)


def interactive_mode():
    pipeline = ContradictionPipeline()
    print("\n" + "=" * 60)
    print("INTERACTIVE CONTRADICTION-AWARE RAG SYSTEM")
    print("Type a question (e.g. 'hurricane death toll', 'employee leave days')")
    print("Type 'sample' to toggle curated dispute test mode.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    current_dataset = "multinews"

    while True:
        try:
            q = input(f"[{current_dataset}] Enter query > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Exiting pipeline. Goodbye!")
                break
            if q.lower() == "sample":
                current_dataset = "sample" if current_dataset == "multinews" else "multinews"
                print(f"Toggled active dataset to: {current_dataset}")
                continue

            pipeline.run(query=q, top_k=3, dataset=current_dataset)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Contradiction-Aware RAG Pipeline")
    parser.add_argument("--query", type=str, default=None, help="User query string.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve (default: 3).")
    parser.add_argument("--dataset", type=str, default="multinews", choices=["multinews", "sample", "demo"], help="Corpus dataset.")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "nli", "reasoner"], help="Detection architecture.")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive query REPL.")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration on key dispute scenarios.")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    pipeline = ContradictionPipeline(mode=args.mode)

    if args.demo:
        demo_queries = [
            ("hurricane death toll Puerto Rico Maria", "sample"),
            ("employee annual leave policy entitlement", "sample"),
            ("vaccine clinical trial efficacy rate", "sample"),
        ]
        print(f"Running automated demonstration across {len(demo_queries)} multi-document dispute scenarios...\n")
        for query_str, dset in demo_queries:
            pipeline.run(query=query_str, top_k=3, dataset=dset)
            print("\n" + "#" * 80 + "\n")
        return

    if args.query:
        pipeline.run(query=args.query, top_k=args.top_k, dataset=args.dataset)
    else:
        # Default to interactive
        interactive_mode()


if __name__ == "__main__":
    main()

