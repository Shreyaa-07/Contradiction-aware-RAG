"""
Stage 1: Atomic Claim Extraction for Contradiction-Aware RAG
============================================================
Extracts structured atomic factual statements from retrieved documents.
Instead of comparing raw multi-page articles (which leads to hallucination
and context dilution), this module breaks texts down into granular claims:

    Claim Schema:
    - claim_id: Unique identifier (e.g., C101)
    - doc_id: Source document ID (e.g., mn_01394_02)
    - cluster_id: News/event cluster ID
    - text: The exact sentence or claim assertion
    - subject: Who or what the statement is about
    - predicate: Property, relationship, or action asserted
    - value: Numeric, date, or factual value
    - unit: Unit of measurement (e.g. deaths, days, percent, dollars)
    - conditions: Applicable conditional constraints
    - temporal: Temporal anchor/timestamp (e.g., 2024, Friday, 3 months after)
    - authority_tier: Estimated source reliability tier (official, news, informal)
    - version_status: Status (e.g., official, preliminary, estimate, draft)
"""

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AtomicClaim:
    claim_id: str
    doc_id: str
    cluster_id: str
    text: str
    subject: str
    predicate: str
    value: Optional[str] = None
    unit: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    temporal: Optional[str] = None
    authority_tier: str = "news_outlet"  # official | research | news_outlet | informal
    version_status: str = "unspecified"  # official | finalized | preliminary | estimate | draft
    confidence: float = 0.90
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicClaim":
        return cls(**data)


class ClaimExtractor:
    """
    Extracts atomic factual claims from raw text passages.
    Combines high-precision linguistic pattern matching (for numerical,
    temporal, status, and entity assertions) with an extensible interface
    for LLM-based claim extraction.
    """

    # Numerical & statistical patterns (death tolls, monetary amounts, days, percentages)
    NUMERICAL_PATTERNS = [
        # Casualty / death toll: "death toll from Hurricane Maria stands at 64 people", "death toll reached 64"
        re.compile(
            r"(?P<subj>death toll|casualties|deaths?|fatalit(?:y|ies))(?:\s+from\s+[A-Za-z0-9\s]+?)?\s+(?:of|is|was|were|stands? at|reached|mounted to|stood at|rose to|at)?\s*(?P<val>\d+(?:,\d+)*(?:\.\d+)?|\b(?:tens|hundreds|thousands|millions)\b)\s*(?P<unit>people|deaths|lives|victims)?",
            re.IGNORECASE,
        ),
        # "estimates that 4,645 people died", "killed 45 people"
        re.compile(
            r"(?:estimates? that|reported that|states? that|killed|claimed|left)?\s*(?P<val>\d+(?:,\d+)*)\s*(?P<unit>people|deaths|victims)?\s*(?:died|were killed|perished|lost their lives)\s*(?:as a result of|in|across|throughout)?\s*(?P<subj>[A-Za-z0-9\s,]+)?",
            re.IGNORECASE,
        ),
        # "confirmed 30 casualties in Puerto Rico"
        re.compile(
            r"(?:confirmed|reported|counted|suffered|recorded)?\s*(?P<val>\d+(?:,\d+)*)\s*(?P<unit>casualties|deaths|fatalities|injuries)\s*(?:in|across|throughout)?\s*(?P<subj>[A-Za-z0-9\s,]+)?",
            re.IGNORECASE,
        ),
        # Financial / budget / revenue: "revenue dropped to 8.2 billion dollars", "$12 million"
        re.compile(
            r"(?P<subj>[a-zA-Z\s]{2,30})?\s*(?:cost|budget|revenue|damage|loss|fine|settlement|worth|priced at|spending)\s+(?:of|is|was|were|estimated at|dropped to|rose to|reached|stood at)?\s*(?P<curr>\$|€|£)?\s*(?P<val>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<mag>million|billion|trillion|thousand)?\s*(?P<unit>dollars|euros|pounds)?",
            re.IGNORECASE,
        ),
        # Policy / leave / duration: "entitled to 24 days of leave", "leave is 30 days"
        re.compile(
            r"(?:(?P<subj>\b[a-zA-Z\s]{3,30}\b)\s+)?(?:are\s+|is\s+)?(?:entitled to|receive|receives|granted|allowance is|entitlement is|offered)\s*(?P<val>\d+)\s*(?P<unit>days|weeks|months|hours)\s*(?:of)?\s*(?P<pred>[a-zA-Z\s]+)?",
            re.IGNORECASE,
        ),
        # Percentage / efficacy (style 1: "efficacy of 95 percent")
        re.compile(
            r"(?P<subj>[a-zA-Z\s]{3,30})?\s*(?:efficacy|rate|growth|share|increase|decrease|tax|discount)\s*(?:is|was|reached|of)?\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|percentage points)",
            re.IGNORECASE,
        ),
        # Percentage / efficacy (style 2: "demonstrated 95 percent efficacy")
        re.compile(
            r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>%|percent|percentage points)\s*(?P<subj>efficacy|growth|rate|share|tax|discount)",
            re.IGNORECASE,
        ),
        # Generic scalar fact: "open 24 hours", "at least 15 miles"
        re.compile(
            r"(?P<subj>[a-zA-Z\s]{2,30})?\s*(?P<pred>open|remains open|lasts|extends|measures|operates)?\s*(?P<val>\d+(?:,\d+)*(?:\.\d+)?)\s*(?P<unit>hours|days|weeks|months|miles|kilometers|dollars|percent)",
            re.IGNORECASE,
        ),
    ]

    # Conditional markers
    CONDITIONAL_PATTERNS = [
        re.compile(r"\b(?:if|when|provided that|assuming|in the event of|for employees who|joining after|during (?:the )?first year|unless|subject to)\s+([^,.;]+)", re.IGNORECASE),
        re.compile(r"\b(?:only applicable to|restricted to|specifically for)\s+([^,.;]+)", re.IGNORECASE),
    ]

    # Temporal indicators
    TEMPORAL_PATTERNS = [
        re.compile(r"\b(?:as of|in|on|by|since|during)\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?|\d{4}|(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b", re.IGNORECASE),
        re.compile(r"\b(\d+\s+(?:hours|days|weeks|months|years)\s+after\s+[^,.;]+)\b", re.IGNORECASE),
        re.compile(r"\b(initially|preliminary|earlier|subsequently|later|updated|revised|current|latest)\b", re.IGNORECASE),
    ]

    # Authority and version markers
    OFFICIAL_MARKERS = ["official", "government", "police", "fema", "governor", "cdc", "who", "department", "authorities", "court", "ministry"]
    ESTIMATE_MARKERS = ["estimate", "estimated", "preliminary", "initial", "unconfirmed", "approximate", "projected", "draft"]
    FINAL_MARKERS = ["confirmed", "final", "official", "ratified", "audit", "verified", "concluded"]

    def __init__(self, use_llm: bool = False, llm_callable: Optional[Any] = None):
        self.use_llm = use_llm
        self.llm_callable = llm_callable

    def extract_from_text(
        self,
        text: str,
        doc_id: str = "doc_001",
        cluster_id: str = "cluster_001",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[AtomicClaim]:
        """
        Extract atomic claims from document text.
        Splits into candidate factual sentences and parses structured elements.
        """
        metadata = metadata or {}
        claims: List[AtomicClaim] = []
        sentences = self._split_sentences(text)

        claim_idx = 1
        for sent in sentences:
            sent_clean = sent.strip()
            if len(sent_clean.split()) < 5:
                continue

            extracted_claims = self._extract_claims_from_sentence(
                sent_clean,
                doc_id=doc_id,
                cluster_id=cluster_id,
                claim_idx_start=claim_idx,
                metadata=metadata,
            )
            claims.extend(extracted_claims)
            claim_idx += len(extracted_claims)

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Sentence splitter with abbreviation awareness."""
        text = text.replace("NEWLINE_CHAR", "\n")
        abbrs = ["U.S.", "Dr.", "Mr.", "Mrs.", "Gov.", "Sen.", "Rep.", "vs.", "Inc.", "Ltd.", "Jan.", "Feb.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
        for abbr in abbrs:
            text = text.replace(abbr, abbr.replace(".", "<DOT>"))

        raw_sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", text)
        clean_sents = []
        for s in raw_sents:
            s = s.replace("<DOT>", ".").strip()
            if s:
                clean_sents.append(s)
        return clean_sents

    def _extract_claims_from_sentence(
        self,
        sentence: str,
        doc_id: str,
        cluster_id: str,
        claim_idx_start: int,
        metadata: Dict[str, Any],
    ) -> List[AtomicClaim]:
        claims: List[AtomicClaim] = []

        # 1. Detect conditions
        conditions = []
        for cond_pat in self.CONDITIONAL_PATTERNS:
            for match in cond_pat.finditer(sentence):
                cond_text = match.group(0).strip()
                if cond_text and cond_text not in conditions:
                    conditions.append(cond_text)

        # 2. Detect temporal anchor
        temporal = None
        for temp_pat in self.TEMPORAL_PATTERNS:
            temp_match = temp_pat.search(sentence)
            if temp_match:
                temporal = temp_match.group(0).strip()
                break

        # 3. Detect authority tier & version status
        authority = "news_outlet"
        sent_lower = sentence.lower()
        if any(marker in sent_lower for marker in self.OFFICIAL_MARKERS):
            authority = "official_government"
        elif "study" in sent_lower or "research" in sent_lower or "university" in sent_lower or "harvard" in sent_lower:
            authority = "academic_research"
        elif "blog" in sent_lower or "twitter" in sent_lower or "unverified" in sent_lower:
            authority = "informal_social"

        version_status = "unspecified"
        if any(marker in sent_lower for marker in self.FINAL_MARKERS):
            version_status = "official_final"
        elif any(marker in sent_lower for marker in self.ESTIMATE_MARKERS):
            version_status = "preliminary_estimate"

        # 4. Extract structured numerical/factual claims
        matched_any = False
        for pat in self.NUMERICAL_PATTERNS:
            for match in pat.finditer(sentence):
                matched_any = True
                mdict = match.groupdict()
                raw_val = mdict.get("val", "")
                if not raw_val:
                    continue

                mag = mdict.get("mag")
                if mag:
                    val_str = f"{raw_val} {mag}".strip()
                else:
                    val_str = raw_val.strip()

                subj = (mdict.get("subj") or "").strip().lower()
                subj = re.sub(r"^(the|a|an)\s+", "", subj)
                if not subj or subj == "subject":
                    subj = self._infer_subject_from_sentence(sentence)

                unit = mdict.get("unit") or mdict.get("curr") or "count"
                pred = mdict.get("pred") or "has_value"

                cid = f"{doc_id}_C{claim_idx_start + len(claims)}"
                claim = AtomicClaim(
                    claim_id=cid,
                    doc_id=doc_id,
                    cluster_id=cluster_id,
                    text=sentence,
                    subject=subj,
                    predicate=pred.strip().replace(" ", "_"),
                    value=val_str,
                    unit=unit,
                    conditions=conditions,
                    temporal=temporal,
                    authority_tier=authority,
                    version_status=version_status,
                    confidence=0.92,
                    metadata={"match_type": "numerical_assertion", **metadata},
                )
                claims.append(claim)

        # 5. Fallback: If no claims extracted yet, extract base assertion
        if len(claims) == 0:
            subj = self._infer_subject_from_sentence(sentence)
            pred, val = self._extract_predicate_and_value(sentence)
            num_match = re.search(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s*(billion|million|thousand)?\s*(%|percent|dollars|days|hours|deaths|casualties)?\b", sentence, re.IGNORECASE)
            num_val = None
            num_unit = None
            if num_match:
                raw_n, mag, unit = num_match.groups()
                num_val = f"{raw_n} {mag}".strip() if mag else raw_n
                num_unit = unit
            cid = f"{doc_id}_C{claim_idx_start}"
            claim = AtomicClaim(
                claim_id=cid,
                doc_id=doc_id,
                cluster_id=cluster_id,
                text=sentence,
                subject=subj,
                predicate=pred,
                value=num_val or val,
                unit=num_unit,
                conditions=conditions,
                temporal=temporal,
                authority_tier=authority,
                version_status=version_status,
                confidence=0.85,
                metadata={"match_type": "factual_statement", **metadata},
            )
            claims.append(claim)

        return claims

    def _infer_subject_from_sentence(self, sent: str) -> str:
        """Heuristic subject extraction from sentence."""
        words = sent.split()
        if not words:
            return "general_entity"
        for term in ["death toll", "annual leave", "hurricane maria", "hurricane michael", "vaccine efficacy", "casualty count", "revenue", "illinois national guard"]:
            if term in sent.lower():
                return term
        first_chunk = " ".join(words[:4]).lower()
        first_chunk = re.sub(r"[,\.:;\"']", "", first_chunk)
        return first_chunk

    def _extract_predicate_and_value(self, sent: str):
        verbs = ["is", "was", "were", "are", "reported", "stated", "confirmed", "denied", "approved", "rejected", "banned"]
        sent_lower = sent.lower()
        for v in verbs:
            pattern = rf"\b{v}\b"
            match = re.search(pattern, sent_lower)
            if match:
                start, end = match.span()
                predicate = v
                value_phrase = sent[end:].strip()[:80]
                return predicate, value_phrase
        return "asserts", sent[:60]

    def _is_informative_assertion(self, sent: str) -> bool:
        words = sent.split()
        if len(words) < 7:
            return False
        sent_lower = sent.lower()
        uninformative = ["click here", "read more", "all rights reserved", "photo by", "subscribe", "advertisement"]
        if any(u in sent_lower for u in uninformative):
            return False
        return bool(re.search(r"\b(is|was|were|reported|claimed|caused|ordered|decimated|reached|approved|banned|entitled)\b", sent_lower))
