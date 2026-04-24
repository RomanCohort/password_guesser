"""
Knowledge Graph Module

Provides knowledge dependency graph for penetration testing:
- CVE/NVD database integration
- MITRE ATT&CK framework
- Exploit database
- Dependency relationships between knowledge items
"""

from .cve_db import CVEDatabase, CVEEntry
from .attack_db import ATTACKDatabase, ATTACKTechnique
from .dependency_graph import KnowledgeDependencyGraph, KnowledgeNode, DependencyEdge
from .exploit_db import ExploitDatabase, ExploitEntry
