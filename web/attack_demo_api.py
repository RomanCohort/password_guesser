"""
Attack Demo API Router

Provides endpoints for the LLM + MOE + Attack Team demonstration.
"""

import asyncio
import json
import sys
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/attack-demo", tags=["attack-demo"])

# ============== Global State ==============

class DemoState:
    """Global state for the demo."""
    def __init__(self):
        self.llm_provider = None
        self.expert_router = None
        self.attack_team = None
        self.rag_retriever = None
        self.attack_history = []
        self.current_phase = "idle"
        self.team_memory = {}

state = DemoState()


# ============== Pydantic Models ==============

class LLMConfigRequest(BaseModel):
    """LLM configuration request."""
    provider: str = "deepseek"
    api_key: str = ""
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2000


class TargetConfigRequest(BaseModel):
    """Target configuration for simulation."""
    ip: str = "192.168.1.100"
    os: str = "Linux"
    ports: Dict[int, str] = {22: "ssh", 80: "http", 443: "https"}
    vulnerabilities: List[str] = []


class AttackStepRequest(BaseModel):
    """Request for a single attack step."""
    phase: str
    state: Dict[str, Any]


class TeamMeetingRequest(BaseModel):
    """Request for team meeting."""
    meeting_type: str = "planning"
    state: Dict[str, Any]


# ============== Initialization ==============

def init_components(llm_api_key: str = None):
    """Initialize all components."""
    from models.llm_provider import create_provider
    from models.expert_router import create_default_router
    from models.attack_team import create_attack_team

    # LLM Provider
    if llm_api_key:
        try:
            state.llm_provider = create_provider(provider="deepseek", api_key=llm_api_key)
        except Exception as e:
            print(f"LLM init failed: {e}")
            state.llm_provider = None
    else:
        state.llm_provider = None

    # Expert Router
    state.expert_router = create_default_router(
        llm_provider=state.llm_provider,
        rag_retriever=state.rag_retriever,
    )

    # Attack Team
    state.attack_team = create_attack_team(
        llm_provider=state.llm_provider,
        rag_retriever=state.rag_retriever,
    )


# ============== Endpoints ==============

@router.get("/status")
async def get_status():
    """Get current system status."""
    return {
        "llm_available": state.llm_provider is not None,
        "expert_router_ready": state.expert_router is not None,
        "attack_team_ready": state.attack_team is not None,
        "current_phase": state.current_phase,
        "history_count": len(state.attack_history),
    }


@router.post("/init")
async def initialize_system(config: LLMConfigRequest):
    """Initialize the attack system with configuration."""
    init_components(config.api_key if config.api_key else None)

    return {
        "status": "initialized",
        "llm_available": state.llm_provider is not None,
        "components": {
            "llm": state.llm_provider is not None,
            "expert_router": state.expert_router is not None,
            "attack_team": state.attack_team is not None,
        }
    }


@router.get("/experts")
async def get_experts():
    """Get list of all registered experts."""
    if not state.expert_router:
        init_components()

    experts = []
    expert_info = {
        "reconnaissance": {
            "name": "Scout",
            "role": "Reconnaissance Expert",
            "description": "Network discovery, port scanning, service identification",
            "tools": ["nmap", "masscan", "shodan", "dnsrecon"],
            "techniques": ["T1046", "T1595", "T1592"],
        },
        "vulnerability": {
            "name": "Analyst",
            "role": "Vulnerability Expert",
            "description": "CVE analysis, vulnerability assessment, risk prioritization",
            "tools": ["nmap", "nuclei", "nikto", "openvas"],
            "techniques": ["T1082", "T1083", "T1195"],
        },
        "exploitation": {
            "name": "Striker",
            "role": "Exploitation Expert",
            "description": "Exploit development, payload generation, bypass techniques",
            "tools": ["metasploit", "exploit-db", "searchsploit", "msfvenom"],
            "techniques": ["T1190", "T1203", "T1068"],
        },
        "post_exploitation": {
            "name": "Ghost",
            "role": "Post-Exploitation Expert",
            "description": "Privilege escalation, persistence, data collection",
            "tools": ["mimikatz", "bloodhound", "powerview", "winpeas"],
            "techniques": ["T1068", "T1098", "T1003"],
        },
        "credential": {
            "name": "Hunter",
            "role": "Credential Expert",
            "description": "Password cracking, hash attacks, credential harvesting",
            "tools": ["hydra", "john", "hashcat", "mimikatz"],
            "techniques": ["T1110", "T1555", "T1552"],
        },
        "lateral_movement": {
            "name": "Phantom",
            "role": "Lateral Movement Expert",
            "description": "Network pivoting, credential reuse, session hijacking",
            "tools": ["crackmapexec", "psexec", "wmi", "winrm"],
            "techniques": ["T1021", "T1078", "T1563"],
        },
    }

    if state.expert_router:
        for expert_type, expert in state.expert_router.experts.items():
            info = expert_info.get(expert_type.value, {})
            experts.append({
                "type": expert_type.value,
                "name": info.get("name", "?"),
                "role": info.get("role", "?"),
                "description": info.get("description", ""),
                "tools": info.get("tools", []),
                "techniques": info.get("techniques", []),
                "success_rate": getattr(expert, '_success_rate', 0.5),
            })

    return {"experts": experts}


@router.get("/team")
async def get_team():
    """Get attack team members."""
    if not state.attack_team:
        init_components()

    members = []
    role_names = {
        "leader": "Commander",
        "recon": "Scout",
        "vuln_analyst": "Analyst",
        "exploiter": "Striker",
        "post_ex": "Ghost",
        "cred_hunter": "Hunter",
        "mover": "Phantom",
    }

    if state.attack_team:
        for name, member in state.attack_team.members.items():
            members.append({
                "name": name,
                "role": member.role.value,
                "role_display": role_names.get(member.role.value, name),
                "expert_type": member.expert_type.value,
                "confidence": member.confidence,
                "tasks_completed": member.tasks_completed,
                "success_rate": member.success_rate,
            })

    return {"members": members}


@router.post("/route")
async def route_expert(request: AttackStepRequest):
    """Route to appropriate expert based on state."""
    if not state.expert_router:
        init_components()

    decision = state.expert_router.analyze_situation(request.state)

    expert_names = {
        "reconnaissance": "Scout",
        "vulnerability": "Analyst",
        "exploitation": "Striker",
        "post_exploitation": "Ghost",
        "credential": "Hunter",
        "lateral_movement": "Phantom",
    }

    result = {
        "primary_expert": {
            "type": decision.primary_expert.value,
            "name": expert_names.get(decision.primary_expert.value, "?"),
        },
        "supporting_experts": [
            {
                "type": e.value,
                "name": expert_names.get(e.value, "?"),
            }
            for e in decision.supporting_experts
        ],
        "confidence": decision.confidence,
        "reasoning": decision.reasoning,
        "keywords_matched": decision.keywords_matched,
    }

    # Get expert advice
    primary = state.expert_router.experts.get(decision.primary_expert)
    if primary:
        try:
            advice = primary.analyze(request.state)
            result["advice"] = {
                "summary": advice.summary,
                "tools": advice.tools_to_use,
                "confidence": advice.confidence,
                "actions": [
                    {
                        "action": a.get("action", "?"),
                        "tool": a.get("tool"),
                        "priority": a.get("priority", 5),
                    }
                    for a in advice.recommended_actions[:5]
                ],
            }
        except Exception as e:
            result["advice"] = {"error": str(e)}

    return result


@router.post("/meeting")
async def hold_meeting(request: TeamMeetingRequest):
    """Hold a team meeting."""
    if not state.attack_team:
        init_components()

    from models.attack_team import MeetingType

    meeting_types = {
        "briefing": MeetingType.BRIEFING,
        "planning": MeetingType.PLANNING,
        "review": MeetingType.REVIEW,
        "debrief": MeetingType.DEBRIEF,
        "emergency": MeetingType.EMERGENCY,
    }

    meeting_type = meeting_types.get(request.meeting_type.lower(), MeetingType.PLANNING)

    result = state.attack_team.hold_meeting(
        meeting_type=meeting_type,
        state=request.state,
    )

    return {
        "meeting_type": result.meeting_type.value,
        "participants": result.participants,
        "discussion": result.discussion[:500] if result.discussion else "",
        "decisions": result.decisions[:5],
        "action_plan": result.action_plan[:5],
        "consensus_level": result.consensus_level,
    }


@router.post("/attack-step")
async def execute_attack_step(request: AttackStepRequest):
    """Execute a single attack step with full visualization."""
    if not state.expert_router:
        init_components()

    # Step 1: Route to expert
    routing = state.expert_router.analyze_situation(request.state)

    expert_names = {
        "reconnaissance": "Scout",
        "vulnerability": "Analyst",
        "exploitation": "Striker",
        "post_exploitation": "Ghost",
        "credential": "Hunter",
        "lateral_movement": "Phantom",
    }

    step_result = {
        "phase": request.phase,
        "timestamp": datetime.now().isoformat(),
        "routing": {
            "primary": {
                "type": routing.primary_expert.value,
                "name": expert_names.get(routing.primary_expert.value, "?"),
            },
            "supporting": [
                {"type": e.value, "name": expert_names.get(e.value, "?")}
                for e in routing.supporting_experts
            ],
            "confidence": routing.confidence,
        },
    }

    # Step 2: Get expert advice
    primary_expert = state.expert_router.experts.get(routing.primary_expert)
    if primary_expert:
        try:
            advice = primary_expert.analyze(request.state)
            step_result["advice"] = {
                "summary": advice.summary,
                "tools": advice.tools_to_use,
                "confidence": advice.confidence,
                "actions": advice.recommended_actions[:3] if advice.recommended_actions else [],
            }
        except Exception as e:
            import traceback
            step_result["advice"] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # Step 3: Team meeting (if team available)
    if state.attack_team:
        from models.attack_team import MeetingType
        try:
            meeting = state.attack_team.hold_meeting(
                meeting_type=MeetingType.PLANNING,
                state=request.state,
            )
            step_result["meeting"] = {
                "participants": meeting.participants,
                "decisions": meeting.decisions[:3] if meeting.decisions else [],
                "consensus": meeting.consensus_level,
            }
        except Exception as e:
            step_result["meeting"] = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    # Step 4: Determine action
    if step_result.get("advice", {}).get("actions"):
        top_action = step_result["advice"]["actions"][0]
        step_result["selected_action"] = {
            "action": top_action.get("action", "unknown"),
            "tool": top_action.get("tool"),
            "priority": top_action.get("priority", 5),
        }
    else:
        step_result["selected_action"] = {
            "action": "wait",
            "tool": None,
            "priority": 5,
        }

    # Record in history
    state.attack_history.append(step_result)

    return step_result


@router.get("/history")
async def get_attack_history():
    """Get attack history."""
    return {"history": state.attack_history[-20:]}


@router.post("/simulate-full-attack")
async def simulate_full_attack(target: TargetConfigRequest):
    """Simulate a full attack chain."""
    if not state.expert_router:
        init_components()

    attack_phases = [
        {
            "phase": "reconnaissance",
            "state": {
                "discovered_hosts": [],
                "open_ports": {},
                "compromised_hosts": [],
                "phase": "reconnaissance",
            },
            "query": "Discover hosts on the network",
        },
        {
            "phase": "scanning",
            "state": {
                "discovered_hosts": [target.ip],
                "open_ports": {},
                "compromised_hosts": [],
                "phase": "scanning",
            },
            "query": "Scan target for open ports",
        },
        {
            "phase": "enumeration",
            "state": {
                "discovered_hosts": [target.ip],
                "open_ports": {target.ip: list(target.ports.keys())},
                "services": {target.ip: target.ports},
                "compromised_hosts": [],
                "phase": "enumeration",
            },
            "query": "Enumerate services on open ports",
        },
        {
            "phase": "vulnerability_analysis",
            "state": {
                "discovered_hosts": [target.ip],
                "open_ports": {target.ip: list(target.ports.keys())},
                "services": list(target.ports.values()),
                "vulnerabilities": target.vulnerabilities,
                "target": target.ip,
                "compromised_hosts": [],
                "phase": "vulnerability_analysis",
            },
            "query": "Analyze vulnerabilities",
        },
        {
            "phase": "exploitation",
            "state": {
                "discovered_hosts": [target.ip],
                "vulnerabilities": target.vulnerabilities,
                "target": target.ip,
                "os": target.os,
                "compromised_hosts": [],
                "phase": "exploitation",
            },
            "query": "Exploit discovered vulnerabilities",
        },
        {
            "phase": "post_exploitation",
            "state": {
                "discovered_hosts": [target.ip],
                "compromised_hosts": [target.ip],
                "privileges": {target.ip: "user"},
                "phase": "post_exploitation",
            },
            "query": "Escalate privileges",
        },
        {
            "phase": "credential_harvesting",
            "state": {
                "discovered_hosts": [target.ip],
                "compromised_hosts": [target.ip],
                "privileges": {target.ip: "root"},
                "phase": "credential_harvesting",
            },
            "query": "Harvest credentials",
        },
        {
            "phase": "lateral_movement",
            "state": {
                "discovered_hosts": [target.ip, "192.168.1.101", "192.168.1.102"],
                "compromised_hosts": [target.ip],
                "privileges": {target.ip: "root"},
                "credentials": [{"user": "admin", "password": "password123"}],
                "phase": "lateral_movement",
            },
            "query": "Move laterally to other hosts",
        },
    ]

    results = []
    total_reward = 0

    for phase_data in attack_phases:
        step_result = await execute_attack_step(
            AttackStepRequest(phase=phase_data["phase"], state=phase_data["state"])
        )

        # Calculate simulated reward
        reward_map = {
            "reconnaissance": 0.5,
            "scanning": 0.3,
            "enumeration": 0.2,
            "vulnerability_analysis": 0.5,
            "exploitation": 5.0,
            "post_exploitation": 3.0,
            "credential_harvesting": 2.0,
            "lateral_movement": 4.0,
        }
        reward = reward_map.get(phase_data["phase"], 0.1)
        total_reward += reward

        step_result["reward"] = reward
        step_result["total_reward"] = total_reward
        step_result["query"] = phase_data["query"]

        results.append(step_result)

        state.current_phase = phase_data["phase"]

    return {
        "target": target.dict(),
        "phases": results,
        "total_reward": total_reward,
        "total_steps": len(results),
        "final_phase": state.current_phase,
    }


@router.websocket("/ws")
async def websocket_attack_demo(websocket: WebSocket):
    """WebSocket for real-time attack updates."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "init":
                api_key = message.get("api_key")
                init_components(api_key)

                await websocket.send_json({
                    "type": "init_complete",
                    "llm_available": state.llm_provider is not None,
                    "expert_count": len(state.expert_router.experts) if state.expert_router else 0,
                    "team_size": len(state.attack_team.members) if state.attack_team else 0,
                })

            elif message.get("type") == "step":
                result = await execute_attack_step(
                    AttackStepRequest(
                        phase=message.get("phase", "unknown"),
                        state=message.get("state", {}),
                    )
                )
                await websocket.send_json({
                    "type": "step_result",
                    "data": result,
                })

            elif message.get("type") == "meeting":
                result = await hold_meeting(
                    TeamMeetingRequest(
                        meeting_type=message.get("meeting_type", "planning"),
                        state=message.get("state", {}),
                    )
                )
                await websocket.send_json({
                    "type": "meeting_result",
                    "data": result,
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
