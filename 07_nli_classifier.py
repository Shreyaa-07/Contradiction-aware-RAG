"""
Stage 2: Candidate Pair Matching & NLI Classification for Contradiction-Aware RAG
================================================================================
Command-line execution and demonstration wrapper for the nli_classifier module.
Demonstrates claim affinity candidate pairing and pairwise NLI inference.

Usage:
    python 07_nli_classifier.py
"""

from claim_extraction import ClaimExtractor
from nli_classifier import CandidatePairMatcher, FactualNLIClassifier, NLIResult


def run_demo() -> None:
    extractor = ClaimExtractor()
    classifier = FactualNLIClassifier()
    matcher = CandidatePairMatcher()

    docs = [
        "The official death toll from Hurricane Maria stands at 64 people according to government reports.",
        "A Harvard academic study estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico.",
        "Employees are entitled to 24 days of annual leave every year under standard policy.",
        "Employees receive 30 days of leave per company handbook v2.",
        "Employees receive 24 days of annual leave.",
    ]

    claims = []
    for i, doc in enumerate(docs, 1):
        claims.extend(extractor.extract_from_text(doc, doc_id=f"doc_{i:02d}"))

    print(f"Extracted {len(claims)} atomic claims.")
    candidate_pairs = matcher.generate_candidate_pairs(claims)
    print(f"Generated {len(candidate_pairs)} candidate comparison pairs.\n")

    for c1, c2 in candidate_pairs:
        result = classifier.classify(c1, c2)
        print(f"Pair: [{c1.claim_id}] vs [{c2.claim_id}]")
        print(f"  P: {c1.text}")
        print(f"  H: {c2.text}")
        print(f"  --> Prediction: {result.prediction.upper()} (conf: {result.confidence:.3f})")
        print(f"  --> Rationale:  {result.rationale}\n")


if __name__ == "__main__":
    run_demo()
