"""
WebSocket Support for Real-time Updates

Provides WebSocket connection management for streaming
generation progress and real-time notifications.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Set, Dict, Optional, Any
import asyncio
import json
import time
import logging

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._connection_metadata: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self._connection_metadata[websocket] = {
            "connected_at": time.time(),
            "last_ping": time.time()
        }
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        self._connection_metadata.pop(websocket, None)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_personal(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all connections"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_progress(
        self,
        websocket: WebSocket,
        task_id: str,
        progress: float,
        message: str = ""
    ):
        """Send a progress update"""
        await self.send_personal({
            "type": "progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }, websocket)

    async def send_generation_update(
        self,
        websocket: WebSocket,
        passwords: list,
        count: int,
        total: int
    ):
        """Send a generation progress update with new passwords"""
        await self.send_personal({
            "type": "generation_update",
            "passwords": passwords,
            "count": count,
            "total": total,
            "timestamp": time.time()
        }, websocket)

    async def send_task_complete(
        self,
        websocket: WebSocket,
        task_id: str,
        result: Any
    ):
        """Send task completion notification"""
        await self.send_personal({
            "type": "task_complete",
            "task_id": task_id,
            "result": result,
            "timestamp": time.time()
        }, websocket)

    async def send_error(
        self,
        websocket: WebSocket,
        error: str,
        task_id: str = None
    ):
        """Send an error notification"""
        await self.send_personal({
            "type": "error",
            "task_id": task_id,
            "error": error,
            "timestamp": time.time()
        }, websocket)

    async def ping_all(self):
        """Send ping to all connections to keep them alive"""
        message = {"type": "ping", "timestamp": time.time()}
        await self.broadcast(message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self) -> list:
        """Get info about all connections"""
        return [
            {
                "connected_at": meta.get("connected_at"),
                "last_ping": meta.get("last_ping"),
            }
            for meta in self._connection_metadata.values()
        ]


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, task_id: str = None):
    """
    WebSocket endpoint handler.

    Handles:
    - Connection lifecycle
    - Heartbeat/ping-pong
    - Message routing

    Usage:
        @app.websocket("/ws/{task_id}")
        async def websocket_route(websocket: WebSocket, task_id: str):
            await websocket_endpoint(websocket, task_id)
    """
    await manager.connect(websocket)

    try:
        # Send welcome message
        await manager.send_personal({
            "type": "connected",
            "task_id": task_id,
            "message": "WebSocket connection established",
            "timestamp": time.time()
        }, websocket)

        # Message loop
        while True:
            # Wait for messages from client
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0  # 30 second timeout
                )

                # Parse message
                try:
                    message = json.loads(data)
                    msg_type = message.get("type", "unknown")

                    if msg_type == "ping":
                        await manager.send_personal({
                            "type": "pong",
                            "timestamp": time.time()
                        }, websocket)

                    elif msg_type == "subscribe":
                        # Subscribe to task updates
                        subscribed_task = message.get("task_id")
                        if subscribed_task:
                            await manager.send_personal({
                                "type": "subscribed",
                                "task_id": subscribed_task,
                                "timestamp": time.time()
                            }, websocket)

                    elif msg_type == "cancel":
                        # Cancel a task
                        cancel_task_id = message.get("task_id")
                        # Would integrate with TaskManager here
                        await manager.send_personal({
                            "type": "cancelled",
                            "task_id": cancel_task_id,
                            "timestamp": time.time()
                        }, websocket)

                except json.JSONDecodeError:
                    await manager.send_error(
                        websocket,
                        "Invalid JSON message"
                    )

            except asyncio.TimeoutError:
                # Send periodic ping to keep connection alive
                await manager.send_personal({
                    "type": "ping",
                    "timestamp": time.time()
                }, websocket)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        manager.disconnect(websocket)


async def broadcast_generation_progress(
    task_id: str,
    progress: float,
    current_passwords: list,
    total_generated: int,
    target_count: int
):
    """
    Broadcast generation progress to all connected clients.

    Call this from the generation loop to provide real-time updates.
    """
    await manager.broadcast({
        "type": "generation_progress",
        "task_id": task_id,
        "progress": progress,
        "current_passwords": current_passwords[-5:] if current_passwords else [],
        "total_generated": total_generated,
        "target_count": target_count,
        "timestamp": time.time()
    })


async def broadcast_task_status(
    task_id: str,
    status: str,
    result: Any = None,
    error: str = None
):
    """Broadcast task status change to all clients"""
    message = {
        "type": "task_status",
        "task_id": task_id,
        "status": status,
        "timestamp": time.time()
    }
    if result is not None:
        message["result"] = result
    if error:
        message["error"] = error

    await manager.broadcast(message)
