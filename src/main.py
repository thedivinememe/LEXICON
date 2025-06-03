from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import uvicorn

from src.config import settings
from src.api import rest, graphql, websocket, spherical_rest
from src.api.websocket import websocket_handler
from src.neural.vectorizer import VectorizedObjectGenerator
from src.storage.database import Database
from src.storage.cache import RedisCache
from src.storage.vector_store import FAISSStore

# Global instances
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    print("Starting LEXICON...")
    
    # Initialize database
    app_state['db'] = Database(settings.database_url)
    await app_state['db'].connect()
    
    # Initialize cache (optional)
    try:
        app_state['cache'] = RedisCache(settings.redis_url)
        await app_state['cache'].connect()
        print("Redis cache connected successfully")
    except Exception as e:
        print(f"Warning: Redis cache not available - {e}")
        # Use a dummy cache implementation
        from src.storage.cache import DummyCache
        app_state['cache'] = DummyCache()
    
    # Initialize vector store
    app_state['vector_store'] = FAISSStore(
        dimension=settings.vector_dimension,
        index_path=settings.vector_index_path
    )
    
    # Initialize neural models
    app_state['vectorizer'] = VectorizedObjectGenerator(
        device=settings.device
    )
    app_state['vectorizer'].eval()  # Set to evaluation mode
    
    # Start background tasks
    app_state['tasks'] = []
    if settings.enable_meme_evolution:
        from src.services.evolution import start_evolution_loop
        task = asyncio.create_task(start_evolution_loop(app_state))
        app_state['tasks'].append(task)
    
    yield
    
    # Shutdown
    print("Shutting down LEXICON...")
    
    # Cancel background tasks
    for task in app_state['tasks']:
        task.cancel()
    
    # Close connections
    await app_state['db'].disconnect()
    await app_state['cache'].disconnect()
    app_state['vector_store'].save()

# Create FastAPI app
app = FastAPI(
    title="LEXICON",
    description="Memetic Atomic Dictionary with Vectorized Objects",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rest.router, prefix=settings.api_prefix)
app.include_router(spherical_rest.router, prefix=settings.api_prefix)
app.mount("/graphql", graphql.graphql_app)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket_handler.handle_connection(websocket, app_state)

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "device": settings.device,
        "features": {
            "gpu": settings.enable_gpu,
            "meme_evolution": settings.enable_meme_evolution,
            "real_time": settings.enable_real_time_updates,
            "spherical_universe": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
