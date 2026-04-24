"""
Knowledge Dependency Graph

Maps relationships between CVEs, ATT&CK techniques, CWEs,
and affected products to build a dependency graph of security knowledge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeNode:
    """A node in the knowledge dependency graph."""
    id: str
    type: str  # 'cve', 'technique', 'cwe', 'product', 'exploit'
    data: dict = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeNode":
        return cls(
            id=data["id"],
            type=data.get("type", "unknown"),
            data=data.get("data", {}),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class DependencyEdge:
    """An edge in the knowledge dependency graph."""
    source: str
    target: str
    relation: str  # 'requires', 'enables', 'mitigates', 'related', 'variant'
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            relation=data.get("relation", "related"),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


class KnowledgeDependencyGraph:
    """
    Knowledge dependency graph for security intelligence.

    Links CVEs to ATT&CK techniques, CWEs, products, and exploits,
    enabling intelligent attack path discovery.
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[DependencyEdge] = []
        self._outgoing: Dict[str, List[int]] = {}
        self._incoming: Dict[str, List[int]] = {}
        self._type_index: Dict[str, Set[str]] = {}

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a knowledge node."""
        self.nodes[node.id] = node
        self._type_index.setdefault(node.type, set()).add(node.id)
        if node.id not in self._outgoing:
            self._outgoing[node.id] = []
        if node.id not in self._incoming:
            self._incoming[node.id] = []

    def add_edge(self, edge: DependencyEdge) -> None:
        """Add a dependency edge."""
        idx = len(self.edges)
        self.edges.append(edge)
        self._outgoing.setdefault(edge.source, []).append(idx)
        self._incoming.setdefault(edge.target, []).append(idx)

    def add_cve(self, cve) -> None:
        """Add a CVE entry as a knowledge node."""
        from .cve_db import CVEEntry

        if isinstance(cve, CVEEntry):
            node = KnowledgeNode(
                id=cve.cve_id,
                type="cve",
                data=cve.to_dict(),
                confidence=min(1.0, cve.cvss_score / 10.0),
            )
        elif isinstance(cve, dict):
            node = KnowledgeNode(
                id=cve.get("cve_id", ""),
                type="cve",
                data=cve,
                confidence=min(1.0, cve.get("cvss_score", 0) / 10.0),
            )
        else:
            return

        self.add_node(node)

        # Link to CWEs
        cwe_ids = node.data.get("cwe_ids", [])
        for cwe_id in cwe_ids:
            cwe_node_id = f"CWE_{cwe_id}"
            if cwe_node_id not in self.nodes:
                self.add_node(KnowledgeNode(
                    id=cwe_node_id,
                    type="cwe",
                    data={"cwe_id": cwe_id},
                ))
            self.add_edge(DependencyEdge(
                source=node.id,
                target=cwe_node_id,
                relation="has_weakness",
            ))

    def add_technique(self, technique) -> None:
        """Add an ATT&CK technique as a knowledge node."""
        from .attack_db import ATTACKTechnique

        if isinstance(technique, ATTACKTechnique):
            node = KnowledgeNode(
                id=technique.technique_id,
                type="technique",
                data=technique.to_dict(),
            )
        elif isinstance(technique, dict):
            node = KnowledgeNode(
                id=technique.get("technique_id", ""),
                type="technique",
                data=technique,
            )
        else:
            return

        self.add_node(node)

    def link_cve_technique(self, cve_id: str, technique_id: str, weight: float = 1.0) -> None:
        """Create an edge linking a CVE to an ATT&CK technique."""
        if cve_id not in self.nodes:
            self.add_node(KnowledgeNode(id=cve_id, type="cve"))
        if technique_id not in self.nodes:
            self.add_node(KnowledgeNode(id=technique_id, type="technique"))

        self.add_edge(DependencyEdge(
            source=cve_id,
            target=technique_id,
            relation="enables",
            weight=weight,
        ))

    def link_cve_exploit(self, cve_id: str, exploit_id: str, reliability: str = "good") -> None:
        """Link a CVE to an available exploit."""
        if cve_id not in self.nodes:
            self.add_node(KnowledgeNode(id=cve_id, type="cve"))
        if exploit_id not in self.nodes:
            self.add_node(KnowledgeNode(id=exploit_id, type="exploit",
                                        data={"reliability": reliability}))

        weight = {"great": 1.0, "good": 0.8, "average": 0.5}.get(reliability, 0.6)
        self.add_edge(DependencyEdge(
            source=exploit_id,
            target=cve_id,
            relation="exploits",
            weight=weight,
        ))

    def find_prerequisites(self, technique_id: str) -> List[str]:
        """Find prerequisites for a technique."""
        prereqs = []
        for idx in self._incoming.get(technique_id, []):
            edge = self.edges[idx]
            if edge.relation in ("requires", "enables"):
                prereqs.append(edge.source)
        return prereqs

    def find_consequences(self, cve_id: str) -> List[str]:
        """Find what a CVE enables (techniques, lateral movement)."""
        consequences = []
        for idx in self._outgoing.get(cve_id, []):
            edge = self.edges[idx]
            consequences.append({
                "id": edge.target,
                "relation": edge.relation,
                "weight": edge.weight,
            })
        return consequences

    def get_attack_chain(self, start_cve: str, target_goal: str) -> List[str]:
        """
        Find a chain from a CVE to a target goal using dependency traversal.

        Uses BFS to find the shortest dependency chain.
        """
        from collections import deque

        if start_cve not in self.nodes:
            return []

        visited: Set[str] = {start_cve}
        queue: deque = deque([(start_cve, [start_cve])])

        while queue:
            current, path = queue.popleft()

            # Check if we've reached the goal
            current_node = self.nodes.get(current)
            if current_node:
                if current == target_goal:
                    return path
                if target_goal.lower() in current_node.data.get("tactics", []):
                    return path
                if target_goal.lower() in current_node.data.get("name", "").lower():
                    return path

            # Traverse edges
            for idx in self._outgoing.get(current, []):
                edge = self.edges[idx]
                if edge.target not in visited and edge.weight > 0.3:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge.target]))

        return []

    def get_nodes_by_type(self, node_type: str) -> List[KnowledgeNode]:
        """Get all nodes of a specific type."""
        ids = self._type_index.get(node_type, set())
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                ntype: len(ids) for ntype, ids in self._type_index.items()
            },
            "edges_by_relation": self._count_edge_relations(),
        }

    def _count_edge_relations(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def to_json(self) -> dict:
        """Serialize to JSON."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_json(cls, data: dict) -> "KnowledgeDependencyGraph":
        """Deserialize from JSON."""
        graph = cls()
        for nid, ndata in data.get("nodes", {}).items():
            graph.add_node(KnowledgeNode.from_dict(ndata))
        for edata in data.get("edges", []):
            graph.add_edge(DependencyEdge.from_dict(edata))
        return graph

    def auto_link(self) -> None:
        """
        Automatically link CVEs to ATT&CK techniques based on
        keyword matching and known mappings.
        """
        cve_nodes = self.get_nodes_by_type("cve")
        tech_nodes = self.get_nodes_by_type("technique")

        # CVE keyword -> Technique mapping
        keyword_mappings = {
            "sql injection": ["T1190"],
            "remote code execution": ["T1190", "T1059"],
            "privilege escalation": ["T1068", "T1548"],
            "buffer overflow": ["T1190", "T1068"],
            "cross-site scripting": ["T1190"],
            "authentication bypass": ["T1190", "T1078"],
            "directory traversal": ["T1190"],
            "denial of service": ["T1498", "T1499"],
            "credentials": ["T1552", "T1003"],
            "brute force": ["T1110"],
            "lateral movement": ["T1021"],
            "persistence": ["T1053", "T1547"],
            "exfiltration": ["T1048"],
            "password": ["T1110", "T1003"],
        }

        for cve_node in cve_nodes:
            desc = cve_node.data.get("description", "").lower()
            for keyword, tech_ids in keyword_mappings.items():
                if keyword in desc:
                    for tech_id in tech_ids:
                        if tech_id in self.nodes:
                            self.add_edge(DependencyEdge(
                                source=cve_node.id,
                                target=tech_id,
                                relation="enables",
                                weight=0.7,
                            ))
