"""
Contradiction Knowledge Graph for Contradiction-Aware RAG
=========================================================
Represents claims and source documents as nodes, and their semantic/factual
relationships as directed edges:
- CONTRADICTS
- SUPERSEDES
- CONDITIONAL_EXCEPTION
- SUPPORTS

Persistent memory: Saves to output/contradiction_graph/knowledge_graph.json
so previously discovered contradictions can be reused across queries without
re-computing.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from claim_extraction import AtomicClaim
from conflict_resolution import ConflictResolution


@dataclass
class GraphNode:
    id: str
    node_type: str  # "claim" | "document"
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str  # CONTRADICTS | SUPERSEDES | CONDITIONAL_EXCEPTION | SUPPORTS
    confidence: float
    severity: str = "moderate"
    properties: Dict[str, Any] = field(default_factory=dict)


class ContradictionKnowledgeGraph:
    """
    In-memory and file-backed Knowledge Graph for contradiction tracking.
    """

    def __init__(self, persistence_file: Optional[Path] = None):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.persistence_file = persistence_file
        if persistence_file and persistence_file.exists():
            self.load(persistence_file)

    def add_claim_node(self, claim: AtomicClaim) -> None:
        if claim.claim_id not in self.nodes:
            self.nodes[claim.claim_id] = GraphNode(
                id=claim.claim_id,
                node_type="claim",
                label=f"{claim.subject} = {claim.value or 'assertion'}",
                properties=claim.to_dict(),
            )

    def add_document_node(self, doc_id: str, title: str = "", cluster_id: str = "") -> None:
        if doc_id not in self.nodes:
            self.nodes[doc_id] = GraphNode(
                id=doc_id,
                node_type="document",
                label=f"Doc: {doc_id}",
                properties={"doc_id": doc_id, "title": title, "cluster_id": cluster_id},
            )

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        confidence: float,
        severity: str = "moderate",
        explanation: str = "",
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> None:
        props = {"explanation": explanation}
        if extra_props:
            props.update(extra_props)

        # Avoid exact duplicate edges
        for edge in self.edges:
            if edge.source_id == source_id and edge.target_id == target_id and edge.relation == relation:
                return

        self.edges.append(
            GraphEdge(
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                confidence=confidence,
                severity=severity,
                properties=props,
            )
        )

    def ingest_conflict_resolution(self, resolution: ConflictResolution) -> None:
        """Add nodes and relationships from a ConflictResolution object."""
        details = resolution.verification_details
        premise_id = details.get("premise_id")
        hypothesis_id = details.get("hypothesis_id")
        confidence = resolution.confidence
        severity = details.get("severity", "moderate")
        explanation = details.get("explanation", "")

        if not premise_id or not hypothesis_id:
            return

        if resolution.is_contradiction:
            # Add bidirectional or directed contradiction edge
            self.add_relationship(
                source_id=premise_id,
                target_id=hypothesis_id,
                relation="CONTRADICTS",
                confidence=confidence,
                severity=severity,
                explanation=explanation,
            )

            # Add precedence / supersedes edge if determined
            if resolution.precedence == "CLAIM_A_SUPERSEDES":
                self.add_relationship(
                    source_id=premise_id,
                    target_id=hypothesis_id,
                    relation="SUPERSEDES",
                    confidence=confidence,
                    severity=severity,
                    explanation=resolution.actionable_recommendation,
                )
            elif resolution.precedence == "CLAIM_B_SUPERSEDES":
                self.add_relationship(
                    source_id=hypothesis_id,
                    target_id=premise_id,
                    relation="SUPERSEDES",
                    confidence=confidence,
                    severity=severity,
                    explanation=resolution.actionable_recommendation,
                )
        else:
            if details.get("contradiction_type") == "scope_or_conditional_difference":
                self.add_relationship(
                    source_id=hypothesis_id,
                    target_id=premise_id,
                    relation="CONDITIONAL_EXCEPTION",
                    confidence=confidence,
                    severity="informational",
                    explanation=explanation,
                )
            elif details.get("contradiction_type") == "agreement":
                self.add_relationship(
                    source_id=premise_id,
                    target_id=hypothesis_id,
                    relation="SUPPORTS",
                    confidence=confidence,
                    severity="informational",
                    explanation=explanation,
                )

    def find_contradictions_for_subject(self, keyword: str) -> List[Dict[str, Any]]:
        """Query persistent graph for known contradictions mentioning keyword."""
        results = []
        kw = keyword.lower()
        for edge in self.edges:
            if edge.relation == "CONTRADICTS":
                n1 = self.nodes.get(edge.source_id)
                n2 = self.nodes.get(edge.target_id)
                if n1 and n2:
                    text_blob = f"{n1.label} {n2.label} {n1.properties.get('text', '')} {n2.properties.get('text', '')}".lower()
                    if kw in text_blob:
                        results.append({
                            "source_node": asdict(n1),
                            "target_node": asdict(n2),
                            "edge": asdict(edge),
                        })
        return results

    def save(self, filepath: Optional[Path] = None) -> None:
        target = filepath or self.persistence_file
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "nodes": {nid: asdict(node) for nid, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: Path) -> None:
        if not filepath.exists():
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.nodes = {nid: GraphNode(**node_data) for nid, node_data in data.get("nodes", {}).items()}
            self.edges = [GraphEdge(**edge_data) for edge_data in data.get("edges", [])]
        except Exception:
            pass

    def render_ascii(self) -> str:
        """Render readable ASCII diagram of graph edges."""
        if not self.edges:
            return "Contradiction Graph is empty (no relationships logged)."

        lines = ["=== Contradiction Knowledge Graph ==="]
        for edge in self.edges:
            n1 = self.nodes.get(edge.source_id)
            n2 = self.nodes.get(edge.target_id)
            lbl1 = f"[{edge.source_id}] ({n1.label if n1 else 'Claim'})"
            lbl2 = f"[{edge.target_id}] ({n2.label if n2 else 'Claim'})"
            arrow = f"--[{edge.relation} (sev={edge.severity.upper()}, conf={edge.confidence:.2f})]-->"
            lines.append(f"  {lbl1}\n      {arrow}\n  {lbl2}\n")
        return "\n".join(lines)

    def export_mermaid(self) -> str:
        """Export graph as Mermaid diagram string."""
        lines = ["graph TD"]
        for edge in self.edges:
            s_id = edge.source_id.replace("-", "_").replace(".", "_")
            t_id = edge.target_id.replace("-", "_").replace(".", "_")
            n1 = self.nodes.get(edge.source_id)
            n2 = self.nodes.get(edge.target_id)
            lbl1 = (n1.label if n1 else edge.source_id).replace('"', "'")
            lbl2 = (n2.label if n2 else edge.target_id).replace('"', "'")
            lines.append(f'    {s_id}["{lbl1}"] -- {edge.relation} --> {t_id}["{lbl2}"]')
        return "\n".join(lines)


if __name__ == "__main__":
    from claim_extraction import ClaimExtractor
    from contextual_verification import ContextualReasoner
    from conflict_resolution import ConflictResolver
    from nli_classifier import FactualNLIClassifier

    extractor = ClaimExtractor()
    classifier = FactualNLIClassifier()
    reasoner = ContextualReasoner()
    resolver = ConflictResolver()

    graph = ContradictionKnowledgeGraph()

    claims_data = [
        ("The official death toll from Hurricane Maria stands at 64 people according to government reports.", "fema_01"),
        ("A Harvard academic study estimates that 4,645 people died as a result of Hurricane Maria in Puerto Rico.", "harvard_01"),
        ("Employees receive 24 days of annual leave under policy v1.", "policy_v1"),
        ("Employees receive 30 days of leave per company handbook v2.", "policy_v2"),
        ("Employees joining after July are entitled to 12 days of annual leave during their first year.", "policy_v2_clause"),
    ]

    claims = []
    for text, doc_id in claims_data:
        extracted = extractor.extract_from_text(text, doc_id=doc_id)[0]
        claims.append(extracted)
        graph.add_claim_node(extracted)

    # Compare Maria
    res_maria = resolver.resolve(reasoner.verify(classifier.classify(claims[0], claims[1])))
    graph.ingest_conflict_resolution(res_maria)

    # Compare Leave v1 vs v2
    res_leave = resolver.resolve(reasoner.verify(classifier.classify(claims[2], claims[3])))
    graph.ingest_conflict_resolution(res_leave)

    # Compare Leave v1 vs clause
    res_clause = resolver.resolve(reasoner.verify(classifier.classify(claims[2], claims[4])))
    graph.ingest_conflict_resolution(res_clause)

    print(graph.render_ascii())
    print("\n--- Mermaid Export ---")
    print(graph.export_mermaid())

