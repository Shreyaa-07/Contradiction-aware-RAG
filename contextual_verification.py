"""
Stage 3: Contextual Verification Reasoner for Contradiction-Aware RAG
=====================================================================
Performs in-depth contextual verification on candidate pairs flagged by NLI.

Crucial Distinctions:
1. Actual Contradiction vs. Conditional Exception:
   - "24 days" vs "12 days for first-year employees joining after July"
   -> Not a contradiction; conditional exception.
2. Actual Contradiction vs. Temporal Revision / Timeline Update:
   - "Death toll at 16 (preliminary)" vs "Death toll at 64 (confirmed)"
   -> Temporal update / revision.
3. Actual Contradiction vs. Geographic / Entity Scope Difference:
   - "30 casualties in Puerto Rico" vs "4 casualties in Florida"
   -> Different regions, not a contradiction.
4. Direct Factual Contradiction:
   - "Entitlement is 24 days" vs "Entitlement is 30 days" for the exact same scope.

Strict Output JSON Contract:
- is_contradiction: bool
- contradiction_type: "direct_factual" | "numerical_discrepancy" | "temporal_revision" | "scope_or_conditional_difference" | "unresolved"
- severity: "critical" | "moderate" | "informational"
- explanation: str
- supporting_evidence: {"premise_quote": ..., "hypothesis_quote": ...}
- confidence: float
- unresolved_questions: list
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from claim_extraction import AtomicClaim
from nli_classifier import NLIResult


@dataclass
class ContextualVerificationResult:
    is_contradiction: bool
    contradiction_type: str  # direct_factual | numerical_discrepancy | temporal_revision | scope_or_conditional_difference | unresolved
    severity: str  # critical | moderate | informational
    explanation: str
    supporting_evidence: Dict[str, str]
    confidence: float
    unresolved_questions: List[str]
    nli_prediction: str
    premise_claim: AtomicClaim
    hypothesis_claim: AtomicClaim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_contradiction": self.is_contradiction,
            "contradiction_type": self.contradiction_type,
            "severity": self.severity,
            "explanation": self.explanation,
            "supporting_evidence": self.supporting_evidence,
            "confidence": round(self.confidence, 4),
            "unresolved_questions": self.unresolved_questions,
            "nli_prediction": self.nli_prediction,
            "premise_id": self.premise_claim.claim_id,
            "hypothesis_id": self.hypothesis_claim.claim_id,
        }


class ContextualReasoner:
    """
    Contextual verification engine.
    Supports either an external LLM (Qwen3, Llama 3, OpenAI, Gemini)
    or a built-in deterministic contextual verification reasoner.
    """

    SYSTEM_PROMPT = """You are a contradiction verification engine. Given two factual claims and their source context:
1. Determine whether the claims actually contradict each other.
2. Check whether they apply to the same subject.
3. Check temporal validity and conditions.
4. Distinguish contradiction from missing information.
5. Do not assume that one document is correct.
6. Do not invent facts.
Return JSON with:
- is_contradiction (boolean)
- contradiction_type (direct_factual | numerical_discrepancy | temporal_revision | scope_or_conditional_difference | unresolved)
- severity (critical | moderate | informational)
- explanation (string)
- supporting_evidence (object with premise_quote and hypothesis_quote)
- confidence (float between 0.0 and 1.0)
- unresolved_questions (list of strings)"""

    def __init__(self, llm_callable: Optional[Callable[[str, str], str]] = None):
        self.llm_callable = llm_callable

    def verify(self, nli_result: NLIResult) -> ContextualVerificationResult:
        """
        Verify an NLI-flagged claim pair.
        """
        if self.llm_callable is not None:
            return self._verify_with_llm(nli_result)
        else:
            return self._verify_heuristic(nli_result)

    def _verify_heuristic(self, nli_result: NLIResult) -> ContextualVerificationResult:
        c1 = nli_result.premise_claim
        c2 = nli_result.hypothesis_claim

        # 1. Conditionality check: If either claim has restrictive conditions
        conds_1 = c1.conditions
        conds_2 = c2.conditions

        if conds_1 or conds_2:
            # Check if one is a conditional exception to the general rule
            cond_text = "; ".join(conds_1 + conds_2)
            explanation = (
                f"Conditional exception detected: Claim '{c2.claim_id if conds_2 else c1.claim_id}' "
                f"applies under specific conditions ({cond_text}), whereas the other represents "
                f"a baseline policy or general scope. These statements describe distinct cases rather than a direct contradiction."
            )
            return ContextualVerificationResult(
                is_contradiction=False,
                contradiction_type="scope_or_conditional_difference",
                severity="informational",
                explanation=explanation,
                supporting_evidence={"premise_quote": c1.text, "hypothesis_quote": c2.text},
                confidence=0.91,
                unresolved_questions=["Does the conditional clause completely supersede or operate as an exception to the base policy?"],
                nli_prediction=nli_result.prediction,
                premise_claim=c1,
                hypothesis_claim=c2,
            )

        # 2. Scope / Location check: Different geographic locations or entities
        loc1 = self._extract_geographic_scope(c1.text)
        loc2 = self._extract_geographic_scope(c2.text)
        if loc1 and loc2 and loc1.lower() != loc2.lower():
            explanation = (
                f"Geographic scope difference: Claim A refers to {loc1}, while Claim B refers to {loc2}. "
                f"They report on different regional impacts rather than contradicting each other."
            )
            return ContextualVerificationResult(
                is_contradiction=False,
                contradiction_type="scope_or_conditional_difference",
                severity="informational",
                explanation=explanation,
                supporting_evidence={"premise_quote": c1.text, "hypothesis_quote": c2.text},
                confidence=0.93,
                unresolved_questions=[f"Are regional totals for {loc1} and {loc2} part of an aggregate nationwide figure?"],
                nli_prediction=nli_result.prediction,
                premise_claim=c1,
                hypothesis_claim=c2,
            )

        # 3. Temporal evolution / Revision check
        # E.g., one preliminary vs one official final; or different timestamps
        is_temporal_rev = False
        rev_reason = ""
        if c1.version_status != c2.version_status and "preliminary" in (c1.version_status + c2.version_status):
            is_temporal_rev = True
            rev_reason = f"One source provides a preliminary estimate ({c1.version_status if 'preliminary' in c1.version_status else c2.version_status}) while the other provides an official or updated count."

        if c1.temporal and c2.temporal and c1.temporal.lower() != c2.temporal.lower():
            is_temporal_rev = True
            rev_reason += f" Reports reflect different temporal anchors ({c1.temporal} vs {c2.temporal})."

        if is_temporal_rev and nli_result.prediction == "contradiction":
            severity = "critical" if any(k in (c1.subject + c2.subject).lower() for k in ["death", "toll", "casualties", "killed"]) else "moderate"
            return ContextualVerificationResult(
                is_contradiction=True,
                contradiction_type="temporal_revision",
                severity=severity,
                explanation=f"Temporal revision detected: {rev_reason.strip()} The claims reflect evolving data rather than static disagreement.",
                supporting_evidence={"premise_quote": c1.text, "hypothesis_quote": c2.text},
                confidence=0.92,
                unresolved_questions=["Has the later figure been universally adopted as the finalized official record?"],
                nli_prediction=nli_result.prediction,
                premise_claim=c1,
                hypothesis_claim=c2,
            )

        # 4. Direct Numerical or Factual Contradiction
        if nli_result.prediction == "contradiction":
            val_a = c1.value
            val_b = c2.value
            c_type = "numerical_discrepancy" if (val_a and val_b) else "direct_factual"

            # Determine severity
            subj_lower = (c1.subject + " " + c2.subject).lower()
            if any(k in subj_lower for k in ["death", "toll", "casualties", "killed", "fatalities", "safety", "cancer"]):
                severity = "critical"
            elif any(k in subj_lower for k in ["revenue", "budget", "cost", "fine", "settlement", "annual leave"]):
                severity = "moderate"
            else:
                severity = "informational"

            val_note = f" (Value: '{val_a}' vs '{val_b}')" if (val_a and val_b) else ""
            explanation = (
                f"Direct {c_type.replace('_', ' ')}: Both claims assert mutually incompatible facts "
                f"regarding the same subject '{c1.subject}'{val_note} without conditional qualifiers or differing scopes."
            )

            return ContextualVerificationResult(
                is_contradiction=True,
                contradiction_type=c_type,
                severity=severity,
                explanation=explanation,
                supporting_evidence={"premise_quote": c1.text, "hypothesis_quote": c2.text},
                confidence=nli_result.confidence,
                unresolved_questions=[
                    f"Which source conducted the primary direct measurement?",
                    f"Is there an official regulatory or judicial determination resolving this discrepancy?",
                ],
                nli_prediction=nli_result.prediction,
                premise_claim=c1,
                hypothesis_claim=c2,
            )

        # 5. Non-contradictory (Entailment or Neutral)
        return ContextualVerificationResult(
            is_contradiction=False,
            contradiction_type="scope_or_conditional_difference" if nli_result.prediction == "neutral" else "agreement",
            severity="informational",
            explanation=f"NLI classified relation as '{nli_result.prediction}'. Contextual analysis confirms statements are consistent or unrelated.",
            supporting_evidence={"premise_quote": c1.text, "hypothesis_quote": c2.text},
            confidence=nli_result.confidence,
            unresolved_questions=[],
            nli_prediction=nli_result.prediction,
            premise_claim=c1,
            hypothesis_claim=c2,
        )

    def _verify_with_llm(self, nli_result: NLIResult) -> ContextualVerificationResult:
        """Call external LLM using structured prompt."""
        c1 = nli_result.premise_claim
        c2 = nli_result.hypothesis_claim

        user_content = json.dumps({
            "claim_a": c1.to_dict(),
            "claim_b": c2.to_dict(),
            "nli_prediction": nli_result.prediction,
            "nli_confidence": nli_result.confidence,
            "nli_rationale": nli_result.rationale,
        }, indent=2)

        try:
            raw_response = self.llm_callable(self.SYSTEM_PROMPT, user_content)
            # Parse JSON
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return ContextualVerificationResult(
                    is_contradiction=bool(data.get("is_contradiction", False)),
                    contradiction_type=data.get("contradiction_type", "unresolved"),
                    severity=data.get("severity", "moderate"),
                    explanation=data.get("explanation", ""),
                    supporting_evidence=data.get("supporting_evidence", {"premise_quote": c1.text, "hypothesis_quote": c2.text}),
                    confidence=float(data.get("confidence", 0.85)),
                    unresolved_questions=data.get("unresolved_questions", []),
                    nli_prediction=nli_result.prediction,
                    premise_claim=c1,
                    hypothesis_claim=c2,
                )
        except Exception as e:
            # Fall back to heuristic if LLM call fails
            return self._verify_heuristic(nli_result)

        return self._verify_heuristic(nli_result)

    def _extract_geographic_scope(self, text: str) -> Optional[str]:
        places = ["Puerto Rico", "Florida", "Illinois", "Chicago", "Midwest", "California", "Texas", "New York", "London", "Europe"]
        for p in places:
            if re.search(rf"\b{p}\b", text, re.IGNORECASE):
                return p
        return None


if __name__ == "__main__":
    from claim_extraction import ClaimExtractor
    from nli_classifier import CandidatePairMatcher, FactualNLIClassifier

    extractor = ClaimExtractor()
    classifier = FactualNLIClassifier()
    matcher = CandidatePairMatcher()
    reasoner = ContextualReasoner()

    test_pairs = [
        # Pair 1: Direct numerical contradiction
        (
            "The official death toll from Hurricane Maria stands at 64 people according to government reports.",
            "A Harvard academic study estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico.",
        ),
        # Pair 2: Conditional exception (should NOT be marked contradiction)
        (
            "Employees are entitled to 24 days of annual leave every year under standard policy.",
            "Employees joining after July are entitled to 12 days of annual leave during their first year.",
        ),
        # Pair 3: Scope difference (different locations)
        (
            "Officials confirmed 30 casualties in Puerto Rico following the storm.",
            "Police reported 4 casualties in Florida from flooding.",
        ),
    ]

    print("Running Contextual Verification Reasoner on benchmark pairs:\n")
    for idx, (t1, t2) in enumerate(test_pairs, 1):
        c1 = extractor.extract_from_text(t1, doc_id=f"doc_A{idx}")[0]
        c2 = extractor.extract_from_text(t2, doc_id=f"doc_B{idx}")[0]
        nli_res = classifier.classify(c1, c2)
        verified = reasoner.verify(nli_res)

        print(f"================ Pair #{idx} ================")
        print(f"Claim 1: {t1}")
        print(f"Claim 2: {t2}")
        print(f"NLI Output:           {nli_res.prediction.upper()} (conf={nli_res.confidence:.2f})")
        print(f"Is Contradiction:     {verified.is_contradiction}")
        print(f"Contradiction Type:   {verified.contradiction_type}")
        print(f"Severity:             {verified.severity.upper()}")
        print(f"Reasoning / Evidence: {verified.explanation}")
        if verified.unresolved_questions:
            print(f"Unresolved Questions: {verified.unresolved_questions[0]}")
        print()

