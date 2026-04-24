"""
Attack Graph Module

Provides graph-based modeling of attack paths for penetration testing.
"""

from .graph import AttackGraph, AttackNode, AttackEdge, NodeType, EdgeType
from .builder import AttackGraphBuilder
from .analyzer import AttackGraphAnalyzer, AttackPath, AttackResult, MitigationAction
from .visualization import AttackGraphVisualizer
