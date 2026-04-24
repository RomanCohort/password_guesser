"""
Attack Graph Visualization

Provides visualization utilities for attack graphs:
- Mermaid diagram output
- D3.js compatible JSON
- Graphviz DOT format
"""

from typing import Dict, List, Optional
from .graph import AttackGraph, AttackNode, AttackEdge, NodeType, EdgeType


class AttackGraphVisualizer:
    """Visualize attack graphs in various formats."""

    # Node type to color mapping
    NODE_COLORS = {
        NodeType.HOST: "#4A90D9",
        NodeType.NETWORK: "#7B68EE",
        NodeType.SERVICE: "#50C878",
        NodeType.VULNERABILITY: "#FF6B6B",
        NodeType.CREDENTIAL: "#FFD700",
        NodeType.PRIVILEGE: "#9370DB",
    }

    # Edge type to style mapping
    EDGE_STYLES = {
        EdgeType.EXPLOIT: "stroke:#FF0000;stroke-width:2px",
        EdgeType.SCAN: "stroke:#888888;stroke-dasharray:5,5",
        EdgeType.LATERAL: "stroke:#FFA500;stroke-width:2px",
        EdgeType.ESCALATE: "stroke:#FF00FF;stroke-width:2px",
        EdgeType.EXFILTRATE: "stroke:#00CED1;stroke-dasharray:10,5",
        EdgeType.DEPENDS: "stroke:#CCCCCC",
        EdgeType.ENABLES: "stroke:#32CD32",
    }

    def to_mermaid(
        self,
        graph: AttackGraph,
        title: str = "Attack Graph",
        show_labels: bool = True,
    ) -> str:
        """
        Generate Mermaid diagram code.

        Usage:
            graph TD
            A[Host 1] --> B[Service 22]
            B --> C[Vuln CVE-2021]
        """
        lines = [f"```mermaid", f"graph TD"]
        lines.append(f"    %% {title}")

        # Add subgraphs for hosts
        host_services: Dict[str, List[str]] = {}
        for node in graph.nodes.values():
            if node.type == NodeType.HOST:
                host_services[node.id] = []

        # Add nodes
        for node in graph.nodes.values():
            color = self.NODE_COLORS.get(node.type, "#888888")
            label = node.name if show_labels else node.id
            label = label.replace('"', "'")

            # Shape based on type
            if node.type == NodeType.HOST:
                shape_start, shape_end = "[", "]"
            elif node.type == NodeType.VULNERABILITY:
                shape_start, shape_end = "((", "))"
            elif node.type == NodeType.NETWORK:
                shape_start, shape_end = "{{", "}}"
            else:
                shape_start, shape_end = "[", "]"

            compromised_marker = " ⚠️" if node.compromised else ""
            lines.append(
                f'    {node.id}{shape_start}"{label}{compromised_marker}"{shape_end}:::{self._type_to_class(node.type)}'
            )

        # Add edges
        for edge in graph.edges:
            if edge is None:
                continue
            edge_style = "=>" if edge.type == EdgeType.EXPLOIT else "-->"
            label = f'|{edge.type.value}|' if show_labels else ""
            lines.append(f"    {edge.source} {edge_style} {label} {edge.target}")

        # Add class definitions
        lines.append("")
        lines.append("    %% Class definitions")
        for ntype in NodeType:
            color = self.NODE_COLORS.get(ntype, "#888888")
            lines.append(f"    classDef {ntype.value} fill:{color},color:white")

        lines.append("```")
        return "\n".join(lines)

    def _type_to_class(self, ntype: NodeType) -> str:
        return ntype.value

    def to_d3_json(self, graph: AttackGraph) -> dict:
        """
        Generate D3.js compatible JSON.

        Structure:
        {
            "nodes": [{"id": "...", "name": "...", "type": "...", "compromised": false}],
            "links": [{"source": "...", "target": "...", "type": "exploit"}]
        }
        """
        nodes = []
        for node in graph.nodes.values():
            nodes.append({
                "id": node.id,
                "name": node.name,
                "type": node.type.value,
                "compromised": node.compromised,
                "confidence": node.confidence,
                "properties": node.properties,
                "color": self.NODE_COLORS.get(node.type, "#888888"),
            })

        links = []
        for edge in graph.edges:
            if edge is None:
                continue
            links.append({
                "source": edge.source,
                "target": edge.target,
                "type": edge.type.value,
                "cve_id": edge.cve_id,
                "technique_id": edge.technique_id,
                "probability": edge.probability,
            })

        return {"nodes": nodes, "links": links}

    def export_dot(
        self,
        graph: AttackGraph,
        title: str = "Attack Graph",
    ) -> str:
        """
        Export to Graphviz DOT format.

        Usage:
            dot -Tpng attack_graph.dot -o attack_graph.png
        """
        lines = [
            'digraph "Attack Graph" {',
            f'    label="{title}";',
            '    labelloc="t";',
            '    fontsize=20;',
            '    rankdir=TB;',
            '    node [shape=box, style=filled];',
            '',
        ]

        # Node definitions
        for node in graph.nodes.values():
            color = self.NODE_COLORS.get(node.type, "#888888")
            shape = "box" if node.type == NodeType.HOST else \
                    "ellipse" if node.type == NodeType.VULNERABILITY else \
                    "diamond" if node.type == NodeType.NETWORK else "box"
            label = node.name.replace('"', '\\"')
            style = "filled,bold" if node.compromised else "filled"
            lines.append(
                f'    "{node.id}" [label="{label}", fillcolor="{color}", '
                f'shape={shape}, style="{style}"];'
            )

        lines.append("")

        # Edge definitions
        for edge in graph.edges:
            if edge is None:
                continue
            style = "solid" if edge.type == EdgeType.EXPLOIT else \
                    "dashed" if edge.type == EdgeType.SCAN else \
                    "bold" if edge.type == EdgeType.LATERAL else "solid"
            color = "#FF0000" if edge.type == EdgeType.EXPLOIT else \
                    "#888888" if edge.type == EdgeType.SCAN else \
                    "#FFA500" if edge.type == EdgeType.LATERAL else "#000000"
            label = edge.type.value
            lines.append(
                f'    "{edge.source}" -> "{edge.target}" '
                f'[label="{label}", style={style}, color="{color}"];'
            )

        lines.append("}")
        return "\n".join(lines)

    def to_html(
        self,
        graph: AttackGraph,
        title: str = "Attack Graph Visualization",
    ) -> str:
        """Generate a standalone HTML file with D3.js visualization."""
        d3_data = self.to_d3_json(graph)

        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a2e;
            color: white;
        }}
        h1 {{
            text-align: center;
        }}
        #graph {{
            width: 100%;
            height: 800px;
            border: 1px solid #333;
            background: #16213e;
        }}
        .node {{
            cursor: pointer;
        }}
        .node text {{
            font-size: 12px;
            fill: white;
        }}
        .link {{
            fill: none;
            stroke-opacity: 0.6;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="graph"></div>
    <div class="tooltip" id="tooltip" style="display:none"></div>

    <script>
        const data = {d3_data};

        const width = document.getElementById('graph').clientWidth;
        const height = 800;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.selectAll(".link")
            .data(data.links)
            .enter().append("path")
            .attr("class", "link")
            .attr("stroke", d => d.type === "exploit" ? "#FF0000" :
                               d.type === "lateral" ? "#FFA500" : "#888888")
            .attr("stroke-width", d => d.type === "exploit" ? 3 : 1.5);

        const node = svg.selectAll(".node")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 20)
            .attr("fill", d => d.color)
            .attr("stroke", d => d.compromised ? "#FF0000" : "#fff")
            .attr("stroke-width", d => d.compromised ? 3 : 1);

        node.append("text")
            .attr("dx", 25)
            .attr("dy", 5)
            .text(d => d.name);

        simulation.on("tick", () => {{
            link.attr("d", d => "M" + d.source.x + "," + d.source.y +
                                "L" + d.target.x + "," + d.target.y);
            node.attr("transform", d => "translate(" + d.x + "," + d.y + ")");
        }});

        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}

        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}

        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>'''
        return html
