"""
WebSocket handler for real-time updates in LEXICON.
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Any, Set
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manager for WebSocket connections"""
    
    def __init__(self):
        """Initialize the WebSocket manager"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for the client
        """
        await websocket.accept()
        
        async with self._lock:
            self.active_connections[client_id] = websocket
            self.subscriptions[client_id] = set()
        
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
    
    async def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client.
        
        Args:
            client_id: Unique identifier for the client
        """
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]
            
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
        
        logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
    
    async def subscribe(self, client_id: str, topic: str) -> None:
        """
        Subscribe a client to a topic.
        
        Args:
            client_id: Unique identifier for the client
            topic: Topic to subscribe to
        """
        async with self._lock:
            if client_id in self.subscriptions:
                self.subscriptions[client_id].add(topic)
                logger.debug(f"Client {client_id} subscribed to {topic}")
    
    async def unsubscribe(self, client_id: str, topic: str) -> None:
        """
        Unsubscribe a client from a topic.
        
        Args:
            client_id: Unique identifier for the client
            topic: Topic to unsubscribe from
        """
        async with self._lock:
            if client_id in self.subscriptions and topic in self.subscriptions[client_id]:
                self.subscriptions[client_id].remove(topic)
                logger.debug(f"Client {client_id} unsubscribed from {topic}")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str) -> None:
        """
        Send a message to a specific client.
        
        Args:
            message: Message to send
            client_id: Unique identifier for the client
        """
        if client_id not in self.active_connections:
            logger.warning(f"Attempted to send message to non-existent client {client_id}")
            return
        
        websocket = self.active_connections[client_id]
        await websocket.send_text(json.dumps(message))
    
    async def broadcast(self, message: Dict[str, Any], topic: str = None) -> None:
        """
        Broadcast a message to all clients or clients subscribed to a topic.
        
        Args:
            message: Message to broadcast
            topic: Optional topic to filter recipients
        """
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            # Skip clients not subscribed to the topic
            if topic and (client_id not in self.subscriptions or 
                         topic not in self.subscriptions[client_id]):
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except WebSocketDisconnect:
                disconnected_clients.append(client_id)
            except Exception as e:
                logger.error(f"Error sending message to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

class WebSocketHandler:
    """Handler for WebSocket connections"""
    
    def __init__(self):
        """Initialize the WebSocket handler"""
        self.manager = WebSocketManager()
    
    async def handle_connection(self, websocket: WebSocket, app_state: Dict[str, Any]) -> None:
        """
        Handle a WebSocket connection.
        
        Args:
            websocket: The WebSocket connection
            app_state: Application state
        """
        # Generate a unique client ID
        client_id = f"client-{id(websocket)}"
        
        # Store WebSocket manager in app state for other components to use
        app_state["websocket_manager"] = self.manager
        
        try:
            # Accept the connection
            await self.manager.connect(websocket, client_id)
            
            # Send welcome message
            await self.manager.send_personal_message(
                {
                    "type": "connection_established",
                    "client_id": client_id,
                    "message": "Connected to LEXICON real-time updates"
                },
                client_id
            )
            
            # Handle messages
            while True:
                try:
                    # Receive message
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process message
                    await self.process_message(message, client_id, app_state)
                    
                except WebSocketDisconnect:
                    await self.manager.disconnect(client_id)
                    break
                except json.JSONDecodeError:
                    await self.manager.send_personal_message(
                        {"type": "error", "message": "Invalid JSON format"},
                        client_id
                    )
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await self.manager.send_personal_message(
                        {"type": "error", "message": str(e)},
                        client_id
                    )
        
        except WebSocketDisconnect:
            await self.manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            try:
                await self.manager.disconnect(client_id)
            except:
                pass
    
    async def process_message(self, message: Dict[str, Any], client_id: str, app_state: Dict[str, Any]) -> None:
        """
        Process a WebSocket message.
        
        Args:
            message: The message to process
            client_id: Unique identifier for the client
            app_state: Application state
        """
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Subscribe to a topic
            topic = message.get("topic")
            if topic:
                await self.manager.subscribe(client_id, topic)
                await self.manager.send_personal_message(
                    {"type": "subscribed", "topic": topic},
                    client_id
                )
        
        elif message_type == "unsubscribe":
            # Unsubscribe from a topic
            topic = message.get("topic")
            if topic:
                await self.manager.unsubscribe(client_id, topic)
                await self.manager.send_personal_message(
                    {"type": "unsubscribed", "topic": topic},
                    client_id
                )
        
        elif message_type == "ping":
            # Ping-pong for connection health check
            await self.manager.send_personal_message(
                {"type": "pong", "timestamp": message.get("timestamp")},
                client_id
            )
        
        elif message_type == "get_vector_updates":
            # Subscribe to vector updates
            await self.manager.subscribe(client_id, "vector_updates")
            
            # Send initial vector space snapshot
            from src.services.visualization import VisualizationService
            service = VisualizationService(app_state)
            
            # Get top concepts
            concept_ids = await service.get_top_concepts(50)
            
            # Create visualization
            viz_data = await service.create_visualization(
                concept_ids=concept_ids,
                method="tsne",
                dimensions=3
            )
            
            # Send visualization data
            await self.manager.send_personal_message(
                {
                    "type": "vector_space_snapshot",
                    "data": viz_data
                },
                client_id
            )
        
        else:
            # Unknown message type
            await self.manager.send_personal_message(
                {"type": "error", "message": f"Unknown message type: {message_type}"},
                client_id
            )

# Create singleton handler
websocket_handler = WebSocketHandler()
