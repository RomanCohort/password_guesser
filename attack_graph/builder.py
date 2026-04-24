"""
Attack Graph Builder

Constructs attack graphs from various scan result formats
(Nmap, Nessus, manual input) and infers potential attack edges.
"""

import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Optional
from urllib.parse import urlparse

from .graph import AttackGraph, AttackNode, AttackEdge, NodeType, EdgeType


class AttackGraphBuilder:
    """Builds attack graphs from scan results and manual input."""

    def from_nmap_scan(self, nmap_xml: str) -> AttackGraph:
        """Build attack graph from Nmap XML output."""
        graph = AttackGraph()

        try:
            root = ET.fromstring(nmap_xml)
        except ET.ParseError:
            raise ValueError("Invalid Nmap XML format")

        for host in root.findall(".//host"):
            if host.get("state") != "up":
                continue

            # Extract host address
            addr_elem = host.find("address[@addrtype='ipv4']")
            if addr_elem is None:
                addr_elem = host.find("address[@addrtype='ipv6']")
            if addr_elem is None:
                continue

            host_addr = addr_elem.get("addr")
            host_id = f"host_{host_addr}"

            # Get hostname
            hostname = host_addr
            hostnames = host.find("hostnames")
            if hostnames is not None:
                hn = hostnames.find("hostname")
                if hn is not None:
                    hostname = hn.get("name", host_addr)

            # Add host node
            os_name = self._detect_os(host)
            host_node = AttackNode(
                id=host_id,
                type=NodeType.HOST,
                name=hostname,
                properties={
                    "ip": host_addr,
                    "os": os_name,
                    "hostname": hostname,
                },
            )
            graph.add_node(host_node)

            # Add network node if not exists
            network_id = self._get_network_id(host_addr)
            if network_id not in graph.nodes:
                graph.add_node(AttackNode(
                    id=network_id,
                    type=NodeType.NETWORK,
                    name=network_id,
                    properties={"subnet": network_id},
                ))

            # Add scan edge from network to host
            graph.add_edge(AttackEdge(
                source=network_id,
                target=host_id,
                type=EdgeType.SCAN,
                probability=1.0,
            ))

            # Process ports
            ports_elem = host.find("ports")
            if ports_elem is not None:
                for port in ports_elem.findall("port"):
                    state = port.find("state")
                    if state is not None and state.get("state") != "open":
                        continue

                    port_id_attr = port.get("portid")
                    protocol = port.get("protocol", "tcp")
                    service_elem = port.find("service")
                    service_name = service_elem.get("name", "unknown") if service_elem is not None else "unknown"
                    service_version = service_elem.get("version", "") if service_elem is not None else ""

                    service_id = f"svc_{host_addr}_{port_id_attr}"
                    service_node = AttackNode(
                        id=service_id,
                        type=NodeType.SERVICE,
                        name=f"{service_name}:{port_id_attr}",
                        properties={
                            "port": int(port_id_attr),
                            "protocol": protocol,
                            "service": service_name,
                            "version": service_version,
                            "host": host_addr,
                        },
                    )
                    graph.add_node(service_node)
                    graph.add_edge(AttackEdge(
                        source=host_id,
                        target=service_id,
                        type=EdgeType.DEPENDS,
                    ))

        self.infer_edges(graph)
        return graph

    def from_nessus_scan(self, nessus_xml: str) -> AttackGraph:
        """Build attack graph from Nessus scan results."""
        graph = AttackGraph()

        try:
            root = ET.fromstring(nessus_xml)
        except ET.ParseError:
            raise ValueError("Invalid Nessus XML format")

        host_vulns: Dict[str, List[dict]] = {}

        for report_host in root.findall(".//ReportHost"):
            host_name = report_host.get("name", "")
            host_id = f"host_{host_name}"

            if host_id not in graph.nodes:
                graph.add_node(AttackNode(
                    id=host_id,
                    type=NodeType.HOST,
                    name=host_name,
                    properties={"ip": host_name},
                ))

            for item in report_host.findall("ReportItem"):
                port = item.get("port", "0")
                svc_name = item.get("svc_name", "unknown")
                severity = int(item.get("severity", "0"))
                plugin_name = item.get("pluginName", "")
                cve = item.get("cve", "")

                if severity >= 3:  # High/Critical
                    vuln_id = f"vuln_{host_name}_{item.get('pluginID', port)}"
                    vuln_node = AttackNode(
                        id=vuln_id,
                        type=NodeType.VULNERABILITY,
                        name=plugin_name,
                        properties={
                            "severity": severity,
                            "port": int(port),
                            "service": svc_name,
                            "cve": cve,
                            "host": host_name,
                        },
                        confidence=min(1.0, severity / 5.0),
                    )
                    graph.add_node(vuln_node)
                    graph.add_edge(AttackEdge(
                        source=host_id,
                        target=vuln_id,
                        type=EdgeType.DEPENDS,
                    ))
                    if cve:
                        host_vulns.setdefault(host_name, []).append({
                            "cve_id": cve,
                            "vuln_node_id": vuln_id,
                            "severity": severity,
                        })

        self.infer_edges(graph)
        return graph

    def from_manual_input(self, hosts: List[dict]) -> AttackGraph:
        """
        Build attack graph from manual host descriptions.

        Each host dict:
        {
            "ip": "192.168.1.100",
            "hostname": "web-server",
            "os": "Linux",
            "ports": [
                {"port": 80, "service": "http", "version": "nginx 1.18"},
                {"port": 22, "service": "ssh", "version": "OpenSSH 8.2"}
            ],
            "vulnerabilities": [
                {"cve_id": "CVE-2021-44228", "name": "Log4Shell", "severity": 10.0}
            ]
        }
        """
        graph = AttackGraph()

        for host_data in hosts:
            ip = host_data.get("ip", "")
            hostname = host_data.get("hostname", ip)
            host_id = f"host_{ip}"

            host_node = AttackNode(
                id=host_id,
                type=NodeType.HOST,
                name=hostname,
                properties={
                    "ip": ip,
                    "os": host_data.get("os", "unknown"),
                    "hostname": hostname,
                },
            )
            graph.add_node(host_node)

            # Add network node
            network_id = self._get_network_id(ip)
            if network_id not in graph.nodes:
                graph.add_node(AttackNode(
                    id=network_id,
                    type=NodeType.NETWORK,
                    name=network_id,
                    properties={"subnet": network_id},
                ))
            graph.add_edge(AttackEdge(
                source=network_id,
                target=host_id,
                type=EdgeType.SCAN,
            ))

            # Add service nodes
            for port_data in host_data.get("ports", []):
                port = port_data.get("port", 0)
                svc_name = port_data.get("service", "unknown")
                svc_id = f"svc_{ip}_{port}"

                graph.add_node(AttackNode(
                    id=svc_id,
                    type=NodeType.SERVICE,
                    name=f"{svc_name}:{port}",
                    properties={
                        "port": port,
                        "protocol": port_data.get("protocol", "tcp"),
                        "service": svc_name,
                        "version": port_data.get("version", ""),
                        "host": ip,
                    },
                ))
                graph.add_edge(AttackEdge(
                    source=host_id,
                    target=svc_id,
                    type=EdgeType.DEPENDS,
                ))

            # Add vulnerability nodes
            for vuln_data in host_data.get("vulnerabilities", []):
                cve_id = vuln_data.get("cve_id", "")
                vuln_name = vuln_data.get("name", cve_id)
                vuln_id = f"vuln_{ip}_{cve_id.replace('-', '_')}" if cve_id else f"vuln_{ip}_{len(graph.nodes)}"

                graph.add_node(AttackNode(
                    id=vuln_id,
                    type=NodeType.VULNERABILITY,
                    name=vuln_name,
                    properties={
                        "cve_id": cve_id,
                        "severity": vuln_data.get("severity", 0),
                        "host": ip,
                    },
                    confidence=min(1.0, vuln_data.get("severity", 0) / 10.0),
                ))
                graph.add_edge(AttackEdge(
                    source=host_id,
                    target=vuln_id,
                    type=EdgeType.DEPENDS,
                    cve_id=cve_id,
                ))

        self.infer_edges(graph)
        return graph

    def add_vulnerability(self, graph: AttackGraph, host_id: str, vuln: dict) -> None:
        """Add a vulnerability node to an existing graph."""
        if host_id not in graph.nodes:
            raise ValueError(f"Host node '{host_id}' not found")

        host_ip = graph.nodes[host_id].properties.get("ip", host_id)
        cve_id = vuln.get("cve_id", "")
        vuln_name = vuln.get("name", cve_id)
        vuln_id = f"vuln_{host_ip}_{cve_id.replace('-', '_')}" if cve_id else f"vuln_{len(graph.nodes)}"

        vuln_node = AttackNode(
            id=vuln_id,
            type=NodeType.VULNERABILITY,
            name=vuln_name,
            properties={
                "cve_id": cve_id,
                "severity": vuln.get("severity", 0),
                "host": host_ip,
                "description": vuln.get("description", ""),
            },
            confidence=min(1.0, vuln.get("severity", 0) / 10.0),
        )
        graph.add_node(vuln_node)
        graph.add_edge(AttackEdge(
            source=host_id,
            target=vuln_id,
            type=EdgeType.DEPENDS,
            cve_id=cve_id,
        ))

    def infer_edges(self, graph: AttackGraph) -> None:
        """
        Automatically infer potential attack edges based on graph topology.

        Rules:
        - Vulnerability nodes on host -> exploit edge from network to host
        - Compromised host -> lateral move edges to other hosts in same network
        - Credentials found -> enable edges to services
        """
        hosts = [n for n in graph.nodes.values() if n.type == NodeType.HOST]
        vulns = [n for n in graph.nodes.values() if n.type == NodeType.VULNERABILITY]

        # Vuln -> Exploit edge (from network to vuln's host)
        for vuln_node in vulns:
            host_ip = vuln_node.properties.get("host", "")
            host_id = f"host_{host_ip}"
            if host_id in graph.nodes:
                network_id = self._get_network_id(host_ip)
                if network_id in graph.nodes:
                    existing = graph.get_edge_by_nodes(network_id, host_id)
                    if existing is None or existing.type != EdgeType.EXPLOIT:
                        severity = vuln_node.properties.get("severity", 0)
                        prob = min(1.0, severity / 10.0)
                        graph.add_edge(AttackEdge(
                            source=network_id,
                            target=host_id,
                            type=EdgeType.EXPLOIT,
                            cve_id=vuln_node.properties.get("cve_id"),
                            probability=max(prob, 0.1),
                        ))

        # Lateral movement between hosts in same network
        network_groups: Dict[str, List[str]] = {}
        for host in hosts:
            ip = host.properties.get("ip", "")
            nid = self._get_network_id(ip)
            network_groups.setdefault(nid, []).append(host.id)

        for nid, host_ids in network_groups.items():
            for i, h1 in enumerate(host_ids):
                for h2 in host_ids[i + 1:]:
                    graph.add_edge(AttackEdge(
                        source=h1,
                        target=h2,
                        type=EdgeType.LATERAL,
                        probability=0.3,
                    ))
                    graph.add_edge(AttackEdge(
                        source=h2,
                        target=h1,
                        type=EdgeType.LATERAL,
                        probability=0.3,
                    ))

    @staticmethod
    def _detect_os(host_elem) -> str:
        """Detect OS from Nmap host element."""
        os_elem = host_elem.find(".//osmatch")
        if os_elem is not None:
            return os_elem.get("name", "unknown")
        return "unknown"

    @staticmethod
    def _get_network_id(ip: str) -> str:
        """Get network ID from IP address (assumes /24 subnet)."""
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{'.'.join(parts[:3])}.0/24"
        return ip
