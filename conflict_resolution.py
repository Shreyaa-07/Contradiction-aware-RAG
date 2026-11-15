"""
Stage 4: Conflict Resolution Layer for Contradiction-Aware RAG
==============================================================
Evaluates verified contradictions to determine whether one source takes
precedence or whether human review / active dispute disclosure is required.

Evaluation Factors:
1. Authority: Official government/regulatory > Academic research > News outlet > Informal notes
2. Recency: Evolving timeline / newer confirmed reporting vs older preliminary reports
3. Version Status: Approved / ratified version vs preliminary estimate or draft
4. Scope: General baseline vs conditional exception
5. Evidence Quality: Direct measurement vs secondary inference

Precedence Decisions:
- CLAIM_A_SUPERSEDES
- CLAIM_B_SUPERSEDES
- MUTUALLY_EXCLUSIVE_DISPUTE (Active dispute: requires human review)
- COMPATIBLE_CONTEXTUAL_VARIATION (Both valid within respective scopes)
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from claim_extraction import AtomicClaim
from contextual_verification import ContextualVerificationResult


@dataclass
class ConflictResolution:
    is_contradiction: bool
    precedence: str  # CLAIM_A_SUPERSEDES | CLAIM_B_SUPERSEDES | MUTUALLY_EXCLUSIVE_DISPUTE | COMPATIBLE_CONTEXTUAL_VARIATION
    winning_claim_id: Optional[str]
    confidence: float
    authority_assessment: str
    recency_assessment: str
    version_assessment: str
    requires_human_review: bool
    actionable_recommendation: str
    synthesized_statement: str
    verification_details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_contradiction": self.is_contradiction,
            "precedence": self.precedence,
            "winning_claim_id": self.winning_claim_id,
            "confidence": round(self.confidence, 4),
            "authority_assessment": self.authority_assessment,
            "recency_assessment": self.recency_assessment,
            "version_assessment": self.version_assessment,
            "requires_human_review": self.requires_human_review,
            "actionable_recommendation": self.actionable_recommendation,
            "synthesized_statement": self.synthesized_statement,
            "verification_details": self.verification_details,
        }


class ConflictResolver:
    """
    Arbitrates conflicts across verified claim pairs.
    """

    AUTHORITY_WEIGHTS = {
        "official_government": 4,
        "academic_research": 3,
        "news_outlet": 2,
        "informal_social": 1,
    }

    VERSION_WEIGHTS = {
        "official_final": 4,
        "finalized": 3,
        "preliminary_estimate": 2,
        "unspecified": 2,
        "draft": 1,
    }

    def resolve(self, verified_result: ContextualVerificationResult) -> ConflictResolution:
        c1 = verified_result.premise_claim
        c2 = verified_result.hypothesis_claim

        # Case 1: If contextual verification confirmed it is NOT a contradiction
        if not verified_result.is_contradiction:
            return ConflictResolution(
                is_contradiction=False,
                precedence="COMPATIBLE_CONTEXTUAL_VARIATION",
                winning_claim_id=None,
                confidence=verified_result.confidence,
                authority_assessment="Both sources valid within their respective condition or geographic scope.",
                recency_assessment="No temporal conflict.",
                version_assessment="Non-conflicting versions.",
                requires_human_review=False,
                actionable_recommendation="Surface both claims with their specific qualifiers (e.g. conditional rules or geographic regions).",
                synthesized_statement=f"Both statements are valid in context: '{c1.text}' applies generally, while '{c2.text}' applies conditionally or to a distinct scope.",
                verification_details=verified_result.to_dict(),
            )

        # Case 2: Temporal Revision
        if verified_result.contradiction_type == "temporal_revision":
            # Check version and authority
            # If one is official_final and other is preliminary
            auth1 = self.AUTHORITY_WEIGHTS.get(c1.authority_tier, 2)
            auth2 = self.AUTHORITY_WEIGHTS.get(c2.authority_tier, 2)
            ver1 = self.VERSION_WEIGHTS.get(c1.version_status, 2)
            ver2 = self.VERSION_WEIGHTS.get(c2.version_status, 2)

            if ver2 > ver1:
                winning = c2.claim_id
                precedence = "CLAIM_B_SUPERSEDES"
                rec = f"Claim B ({c2.claim_id}) represents an updated/finalized account superseding earlier preliminary reporting."
            elif ver1 > ver2:
                winning = c1.claim_id
                precedence = "CLAIM_A_SUPERSEDES"
                rec = f"Claim A ({c1.claim_id}) represents an updated/finalized account superseding earlier preliminary reporting."
            else:
                # Both have similar version weights; check authority or timestamps
                if auth2 > auth1:
                    winning = c2.claim_id
                    precedence = "CLAIM_B_SUPERSEDES"
                    rec = f"Claim B ({c2.claim_id}) is backed by higher source authority ({c2.authority_tier})."
                elif auth1 > auth2:
                    winning = c1.claim_id
                    precedence = "CLAIM_A_SUPERSEDES"
                    rec = f"Claim A ({c1.claim_id}) is backed by higher source authority ({c1.authority_tier})."
                else:
                    winning = None
                    precedence = "MUTUALLY_EXCLUSIVE_DISPUTE"
                    rec = "Evolving reporting without consensus; cite both preliminary and updated figures."

            requires_review = (winning is None) or (verified_result.severity == "critical")
            synth = (
                f"Reports reflect differing stages of verification: '{c1.text}' was reported with status '{c1.version_status}', "
                f"whereas '{c2.text}' represents '{c2.version_status}'. "
                f"{'The updated figure supersedes earlier numbers.' if winning else 'Both figures should be presented to the user with caveats.'}"
            )

            return ConflictResolution(
                is_contradiction=True,
                precedence=precedence,
                winning_claim_id=winning,
                confidence=verified_result.confidence,
                authority_assessment=f"Authority comparison: [{c1.claim_id}]={c1.authority_tier} vs [{c2.claim_id}]={c2.authority_tier}",
                recency_assessment=f"Temporal timeline: [{c1.claim_id}]={c1.temporal or 'unspecified'} vs [{c2.claim_id}]={c2.temporal or 'unspecified'}",
                version_assessment=f"Version status: [{c1.claim_id}]={c1.version_status} vs [{c2.claim_id}]={c2.version_status}",
                requires_human_review=requires_review,
                actionable_recommendation=rec,
                synthesized_statement=synth,
                verification_details=verified_result.to_dict(),
            )

        # Case 3: Direct Factual or Numerical Contradiction
        auth1 = self.AUTHORITY_WEIGHTS.get(c1.authority_tier, 2)
        auth2 = self.AUTHORITY_WEIGHTS.get(c2.authority_tier, 2)
        ver1 = self.VERSION_WEIGHTS.get(c1.version_status, 2)
        ver2 = self.VERSION_WEIGHTS.get(c2.version_status, 2)

        score_a = auth1 * 1.5 + ver1
        score_b = auth2 * 1.5 + ver2

        if score_b > score_a + 1.5:
            winning = c2.claim_id
            precedence = "CLAIM_B_SUPERSEDES"
            rec = f"Claim B ({c2.claim_id}) takes precedence due to superior source authority ({c2.authority_tier}) and formal status ({c2.version_status})."
            review = verified_result.severity == "critical"
        elif score_a > score_b + 1.5:
            winning = c1.claim_id
            precedence = "CLAIM_A_SUPERSEDES"
            rec = f"Claim A ({c1.claim_id}) takes precedence due to superior source authority ({c1.authority_tier}) and formal status ({c1.version_status})."
            review = verified_result.severity == "critical"
        else:
            # Significant unresolvable conflict between peers (e.g. government vs academic study, or contradictory official reports)
            winning = None
            precedence = "MUTUALLY_EXCLUSIVE_DISPUTE"
            rec = "Direct factual dispute between reputable or peer sources. Human review is recommended; system should present both claims side-by-side."
            review = True

        synth = (
            f"Factual dispute detected regarding {c1.subject}: "
            f"Source A ({c1.doc_id}) reports: '{c1.text}', "
            f"while Source B ({c2.doc_id}) reports: '{c2.text}'. "
            f"{rec}"
        )

        return ConflictResolution(
            is_contradiction=True,
            precedence=precedence,
            winning_claim_id=winning,
            confidence=verified_result.confidence,
            authority_assessment=f"Authority: [{c1.claim_id}]={c1.authority_tier} (weight {auth1}) vs [{c2.claim_id}]={c2.authority_tier} (weight {auth2})",
            recency_assessment=f"Recency: [{c1.claim_id}]={c1.temporal or 'standard'} vs [{c2.claim_id}]={c2.temporal or 'standard'}",
            version_assessment=f"Version: [{c1.claim_id}]={c1.version_status} vs [{c2.claim_id}]={c2.version_status}",
            requires_human_review=review,
            actionable_recommendation=rec,
            synthesized_statement=synth,
            verification_details=verified_result.to_dict(),
        )


if __name__ == "__main__":
    from claim_extraction import ClaimExtractor
    from contextual_verification import ContextualReasoner
    from nli_classifier import FactualNLIClassifier

    extractor = ClaimExtractor()
    classifier = FactualNLIClassifier()
    reasoner = ContextualReasoner()
    resolver = ConflictResolver()

    # Benchmark: Hurricane Maria
    t1 = "The official death toll from Hurricane Maria stands at 64 people according to government reports."
    t2 = "A Harvard academic study estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico."
    c1 = extractor.extract_from_text(t1, doc_id="fema_report_01")[0]
    c2 = extractor.extract_from_text(t2, doc_id="harvard_study_01")[0]

    nli_res = classifier.classify(c1, c2)
    verified = reasoner.verify(nli_res)
    resolution = resolver.resolve(verified)

    print("================ Conflict Resolution Result ================")
    print(f"Precedence Decision:     {resolution.precedence}")
    print(f"Winning Claim:           {resolution.winning_claim_id}")
    print(f"Requires Human Review:   {resolution.requires_human_review}")
    print(f"Authority Assessment:    {resolution.authority_assessment}")
    print(f"Version Assessment:      {resolution.version_assessment}")
    print(f"Recommendation:          {resolution.actionable_recommendation}")
    print(f"Synthesized RAG Answer:  {resolution.synthesized_statement}")

