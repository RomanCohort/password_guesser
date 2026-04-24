"""
Attack Graph Core Data Structures

Defines nodes, edges, and the attack graph itself.
Nodes represent hosts, services, vulnerabilities, credentials, and privileges.
Edges represent attack actions like exploits, scans, lateral movement, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from collections import deque
import json


class NodeType(Enum):
    HOST = "host"
    NETWORK = "network"
    SERVICE = "service"
    VULNERABILITY = "vuln"
    CREDENTIAL = "cred"
    PRIVILEGE = "priv"


class EdgeType(Enum):
    EXPLOIT = "exploit"
    SCAN = "scan"
    LATERAL = "lateral"
    ESCALATE = "escalate"
    EXFILTRATE = "exfil"
    DEPENDS = "depends"
    ENABLES = "enables"


@dataclass
class AttackNode:
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    compromised: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": self.properties,
            "compromised": self.compromised,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackNode":
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            name=data["name"],
            properties=data.get("properties", {}),
            compromised=data.get("compromised", False),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class AttackEdge:
    source: str
    target: str
    type: EdgeType
    cve_id: Optional[str] = None
    technique_id: Optional[str] = None
    probability: float = 1.0
    preconditions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "cve_id": self.cve_id,
            "technique_id": self.technique_id,
            "probability": self.probability,
            "preconditions": self.preconditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttackEdge":
        return cls(
            source=data["source"],
            target=data["target"],
            type=EdgeType(data["type"]),
            cve_id=data.get("cve_id"),
            technique_id=data.get("technique_id"),
            probability=data.get("probability", 1.0),
            preconditions=data.get("preconditions", []),
            metadata=data.get("metadata", {}),
        )


class AttackGraph:
    """
    Directed graph representing attack paths.

    Nodes are states/assets (hosts, services, vulnerabilities).
    Edges are attack actions (exploits, lateral movement, etc.).
    """

    def __init__(self):
        self.nodes: Dict[str, AttackNode] = {}
        self.edges: List[AttackEdge] = []
        # Adjacency list: node_id -> list of (edge_index, target_node_id)
        self._outgoing: Dict[str, List[int]] = {}
        self._incoming: Dict[str, List[int]] = {}

    def add_node(self, node: AttackNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self._outgoing:
            self._outgoing[node.id] = []
        if node.id not in self._incoming:
            self._incoming[node.id] = []

    def add_edge(self, edge: AttackEdge) -> None:
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' not in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' not in graph")

        idx = len(self.edges)
        self.edges.append(edge)
        self._outgoing[edge.source].append(idx)
        self._incoming[edge.target].append(idx)

    def remove_edge(self, edge_index: int) -> None:
        if 0 <= edge_index < len(self.edges):
            edge = self.edges[edge_index]
            self._outgoing[edge.source].remove(edge_index)
            self._incoming[edge.target].remove(edge_index)
            self.edges[edge_index] = None  # type: ignore

    def get_outgoing_edges(self, node_id: str) -> List[AttackEdge]:
        return [self.edges[i] for i in self._outgoing.get(node_id, [])
                if i < len(self.edges) and self.edges[i] is not None]

    def get_incoming_edges(self, node_id: str) -> List[AttackEdge]:
        return [self.edges[i] for i in self._incoming.get(node_id, [])
                if i < len(self.edges) and self.edges[i] is not None]

    def get_neighbors(self, node_id: str) -> List[str]:
        return [e.target for e in self.get_outgoing_edges(node_id)]

    def find_paths(self, source: str, target: str, max_paths: int = 10) -> List[List[str]]:
        """Find all simple paths from source to target using BFS."""
        if source not in self.nodes or target not in self.nodes:
            return []

        all_paths: List[List[str]] = []
        queue: deque = deque([[source]])
        visited_set: Set[frozenset] = set()

        while queue and len(all_paths) < max_paths:
            path = queue.popleft()
            current = path[-1]

            if current == target:
                all_paths.append(path)
                continue

            path_set = frozenset(path)
            if path_set in visited_set:
                continue
            visited_set.add(path_set)

            for neighbor in self.get_neighbors(current):
                if neighbor not in path:  # Simple path - no cycles
                    queue.append(path + [neighbor])

        return all_paths

    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path using BFS."""
        if source not in self.nodes or target not in self.nodes:
            return None

        visited: Set[str] = {source}
        queue: deque = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()

            if current == target:
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def get_exploitable_nodes(self) -> List[AttackNode]:
        """Get nodes that have incoming exploit edges."""
        exploitable_ids: Set[str] = set()
        for edge in self.edges:
            if edge is not None and edge.type == EdgeType.EXPLOIT:
                exploitable_ids.add(edge.target)
        return [self.nodes[nid] for nid in exploitable_ids if nid in self.nodes]

    def get_compromised_nodes(self) -> List[AttackNode]:
        return [n for n in self.nodes.values() if n.compromised]

    def update_compromised(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id].compromised = True

    def get_edge_by_nodes(self, source: str, target: str) -> Optional[AttackEdge]:
        for edge in self.get_outgoing_edges(source):
            if edge.target == target:
                return edge
        return None

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len([e for e in self.edges if e is not None])

    def to_json(self) -> dict:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges if e is not None],
        }

    @classmethod
    def from_json(cls, data: dict) -> "AttackGraph":
        graph = cls()
        for nid, ndata in data.get("nodes", {}).items():
            graph.add_node(AttackNode.from_dict(ndata))
        for edata in data.get("edges", []):
            graph.add_edge(AttackEdge.from_dict(edata))
        return graph

    def to_json_str(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json_str(cls, json_str: str) -> "AttackGraph":
        return cls.from_json(json.loads(json_str))
