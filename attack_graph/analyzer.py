"""
Attack Graph Analyzer

Provides analysis capabilities for attack graphs:
- Find attack paths
- Calculate risk scores
- Identify critical nodes
- Simulate attacks
- Generate mitigations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import heapq

from .graph import AttackGraph, AttackNode, AttackEdge, NodeType, EdgeType


@dataclass
class AttackPath:
    """Represents a potential attack path."""
    nodes: List[str]
    edges: List[AttackEdge]
    probability: float
    impact: float
    risk_score: float

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "probability": self.probability,
            "impact": self.impact,
            "risk_score": self.risk_score,
        }


@dataclass
class AttackResult:
    """Result of a simulated attack."""
    success: bool
    path: List[str]
    compromised_nodes: List[str]
    failed_at: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "path": self.path,
            "compromised_nodes": self.compromised_nodes,
            "failed_at": self.failed_at,
            "error": self.error,
        }


@dataclass
class MitigationAction:
    """Recommended mitigation action."""
    target_node: str
    action_type: str
    description: str
    priority: int  # 1 = highest
    estimated_effort: str  # "low", "medium", "high"

    def to_dict(self) -> dict:
        return {
            "target_node": self.target_node,
            "action_type": self.action_type,
            "description": self.description,
            "priority": self.priority,
            "estimated_effort": self.estimated_effort,
        }


class AttackGraphAnalyzer:
    """Analyzes attack graphs for security insights."""

    def __init__(self):
        self._risk_weights = {
            NodeType.VULNERABILITY: 0.4,
            NodeType.HOST: 0.3,
            NodeType.SERVICE: 0.2,
            NodeType.CREDENTIAL: 0.3,
        }

    def find_attack_paths(
        self,
        graph: AttackGraph,
        target: str,
        max_paths: int = 10,
        min_probability: float = 0.1,
    ) -> List[AttackPath]:
        """
        Find all attack paths to a target node.

        Returns paths sorted by risk score (highest first).
        """
        if target not in graph.nodes:
            return []

        network_nodes = [
            n.id for n in graph.nodes.values()
            if n.type == NodeType.NETWORK
        ]

        if not network_nodes:
            return []

        all_paths: List[AttackPath] = []

        for network in network_nodes:
            raw_paths = graph.find_paths(network, target, max_paths=max_paths)
            for raw_path in raw_paths:
                attack_path = self._build_attack_path(graph, raw_path)
                if attack_path.probability >= min_probability:
                    all_paths.append(attack_path)

        return sorted(all_paths, key=lambda p: p.risk_score, reverse=True)[:max_paths]

    def _build_attack_path(self, graph: AttackGraph, node_path: List[str]) -> AttackPath:
        """Build an AttackPath from node IDs."""
        edges: List[AttackEdge] = []
        prob = 1.0
        impact = 0.0

        for i in range(len(node_path) - 1):
            src, tgt = node_path[i], node_path[i + 1]
            edge = graph.get_edge_by_nodes(src, tgt)
            if edge:
                edges.append(edge)
                prob *= edge.probability
            tgt_node = graph.nodes.get(tgt)
            if tgt_node:
                impact += self._calculate_node_impact(tgt_node)

        risk_score = prob * impact
        return AttackPath(
            nodes=node_path,
            edges=edges,
            probability=prob,
            impact=impact,
            risk_score=risk_score,
        )

    def _calculate_node_impact(self, node: AttackNode) -> float:
        """Calculate impact score for a node."""
        base_impact = 1.0

        if node.type == NodeType.VULNERABILITY:
            severity = node.properties.get("severity", 0)
            base_impact = min(10.0, severity)
        elif node.type == NodeType.HOST:
            base_impact = 5.0
        elif node.type == NodeType.CREDENTIAL:
            base_impact = 7.0
        elif node.type == NodeType.SERVICE:
            base_impact = 3.0

        return base_impact * node.confidence

    def calculate_risk_score(self, graph: AttackGraph) -> float:
        """
        Calculate overall risk score for the graph.

        Higher score = more dangerous attack surface.
        """
        if not graph.nodes:
            return 0.0

        total_risk = 0.0
        for node in graph.nodes.values():
            if node.type == NodeType.VULNERABILITY:
                severity = node.properties.get("severity", 0)
                weight = self._risk_weights.get(node.type, 0.2)
                total_risk += severity * weight * node.confidence

        # Consider graph connectivity
        connectivity = graph.edge_count() / max(1, graph.node_count())
        total_risk *= (1 + connectivity * 0.5)

        return min(100.0, total_risk)

    def find_critical_nodes(
        self,
        graph: AttackGraph,
        top_n: int = 10,
    ) -> List[AttackNode]:
        """
        Find the most critical nodes using betweenness centrality approximation.

        Critical nodes are those that appear on many attack paths.
        """
        node_scores: Dict[str, float] = {nid: 0.0 for nid in graph.nodes}

        network_nodes = [
            n.id for n in graph.nodes.values()
            if n.type == NodeType.NETWORK
        ]

        # Count how many times each node appears on paths
        for target in graph.nodes:
            if graph.nodes[target].type in (NodeType.HOST, NodeType.VULNERABILITY):
                for network in network_nodes:
                    paths = graph.find_paths(network, target, max_paths=20)
                    for path in paths:
                        for node_id in path:
                            node_scores[node_id] += 1.0

        # Sort by score
        sorted_nodes = sorted(
            node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [graph.nodes[nid] for nid, _ in sorted_nodes if nid in graph.nodes]

    def simulate_attack(
        self,
        graph: AttackGraph,
        path: List[str],
        exploitation_success_prob: float = 0.7,
    ) -> AttackResult:
        """
        Simulate an attack along a given path.

        Returns simulation result with success/failure status.
        """
        import random

        compromised: List[str] = []

        for i, node_id in enumerate(path):
            if node_id not in graph.nodes:
                return AttackResult(
                    success=False,
                    path=path,
                    compromised_nodes=compromised,
                    failed_at=node_id,
                    error=f"Node {node_id} not found",
                )

            node = graph.nodes[node_id]

            # Check if exploitation succeeds
            if node.type == NodeType.VULNERABILITY:
                severity = node.properties.get("severity", 0)
                success_prob = min(0.95, exploitation_success_prob * (severity / 10.0))
                if random.random() > success_prob:
                    return AttackResult(
                        success=False,
                        path=path,
                        compromised_nodes=compromised,
                        failed_at=node_id,
                        error="Exploitation failed",
                    )

            compromised.append(node_id)

        # All nodes compromised
        return AttackResult(
            success=True,
            path=path,
            compromised_nodes=compromised,
        )

    def generate_mitigation(self, graph: AttackGraph) -> List[MitigationAction]:
        """
        Generate prioritized mitigation recommendations.

        Prioritizes actions that block the highest-risk paths.
        """
        mitigations: List[MitigationAction] = []
        critical_nodes = self.find_critical_nodes(graph, top_n=10)
        vuln_nodes = [
            n for n in graph.nodes.values()
            if n.type == NodeType.VULNERABILITY
        ]

        # Sort vulnerabilities by severity
        vuln_nodes.sort(
            key=lambda n: n.properties.get("severity", 0),
            reverse=True,
        )

        priority = 1
        added_mitigations: Set[str] = set()

        # Mitigate high-severity vulnerabilities first
        for vuln in vuln_nodes[:5]:
            cve_id = vuln.properties.get("cve_id", vuln.id)
            key = f"patch_{cve_id}"
            if key not in added_mitigations:
                mitigations.append(MitigationAction(
                    target_node=vuln.id,
                    action_type="patch",
                    description=f"Patch vulnerability {cve_id}: {vuln.name}",
                    priority=priority,
                    estimated_effort="medium" if vuln.properties.get("severity", 0) > 7 else "low",
                ))
                added_mitigations.add(key)
                priority += 1

        # Isolate critical hosts
        for node in critical_nodes:
            if node.type == NodeType.HOST:
                key = f"isolate_{node.id}"
                if key not in added_mitigations:
                    mitigations.append(MitigationAction(
                        target_node=node.id,
                        action_type="isolate",
                        description=f"Isolate or restrict access to {node.name}",
                        priority=priority,
                        estimated_effort="medium",
                    ))
                    added_mitigations.add(key)
                    priority += 1

        # Add network segmentation
        mitigations.append(MitigationAction(
            target_node="network",
            action_type="segment",
            description="Implement network segmentation to limit lateral movement",
            priority=priority,
            estimated_effort="high",
        ))

        return mitigations

    def find_shortest_exploit_path(
        self,
        graph: AttackGraph,
        target_host: str,
    ) -> Optional[AttackPath]:
        """Find the shortest path to exploit a target host."""
        network_nodes = [
            n.id for n in graph.nodes.values()
            if n.type == NodeType.NETWORK
        ]

        shortest: Optional[AttackPath] = None
        for network in network_nodes:
            path_nodes = graph.find_shortest_path(network, target_host)
            if path_nodes:
                attack_path = self._build_attack_path(graph, path_nodes)
                if shortest is None or attack_path.risk_score > shortest.risk_score:
                    shortest = attack_path

        return shortest

    def assess_lateral_movement_risk(
        self,
        graph: AttackGraph,
        compromised_host: str,
    ) -> List[Tuple[str, float]]:
        """
        Assess risk of lateral movement from a compromised host.

        Returns list of (target_host_id, risk_score) pairs.
        """
        risks: List[Tuple[str, float]] = []

        outgoing = graph.get_outgoing_edges(compromised_host)
        for edge in outgoing:
            if edge.type == EdgeType.LATERAL:
                target = edge.target
                if target in graph.nodes:
                    target_node = graph.nodes[target]
                    # Risk based on accessible vulnerabilities
                    vulns = [
                        n for n in graph.nodes.values()
                        if n.type == NodeType.VULNERABILITY
                        and n.properties.get("host") == target_node.properties.get("ip")
                    ]
                    vuln_severity = sum(
                        v.properties.get("severity", 0)
                        for v in vulns
                    )
                    risk = edge.probability * (1 + vuln_severity * 0.1)
                    risks.append((target, risk))

        return sorted(risks, key=lambda x: x[1], reverse=True)
