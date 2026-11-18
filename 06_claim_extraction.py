"""
Stage 1: Atomic Claim Extraction for Contradiction-Aware RAG
============================================================
Command-line execution and demonstration wrapper for the claim_extraction module.
Extracts structured atomic factual statements from text passages with metadata,
conditions, temporal anchors, and authority tiers.

Usage:
    python 06_claim_extraction.py
"""

from claim_extraction import AtomicClaim, ClaimExtractor


def run_demo() -> None:
    extractor = ClaimExtractor()
    test_docs = [
        "The official death toll from Hurricane Maria stands at 64 people according to government reports.",
        "A Harvard academic study estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico.",
        "Employees are entitled to 24 days of annual leave every year under standard policy.",
        "Employees joining after July are entitled to 12 days of annual leave during their first year.",
        "Employees receive 30 days of leave per company handbook v2.",
    ]

    print(f"Running atomic claim extraction on {len(test_docs)} test sentences...\n")
    for i, doc in enumerate(test_docs, 1):
        extracted = extractor.extract_from_text(doc, doc_id=f"doc_{i:02d}", cluster_id="test_cluster")
        print(f"--- Doc {i}: {doc} ---")
        for c in extracted:
            print(f"  [Claim ID: {c.claim_id}]")
            print(f"    Subject:   {c.subject}")
            print(f"    Predicate: {c.predicate}")
            print(f"    Value:     {c.value} ({c.unit})")
            print(f"    Conds:     {c.conditions}")
            print(f"    Temporal:  {c.temporal}")
            print(f"    Authority: {c.authority_tier} | Status: {c.version_status}\n")


if __name__ == "__main__":
    run_demo()
