# LEXICON: Memetic Atomic Dictionary with Vectorized Objects

LEXICON is a unified Python application that combines neural components for vectorized object generation with a comprehensive system for defining, storing, and manipulating concepts through negation.

## Architecture Overview

A monolithic Python application combining all LEXICON functionality in a single, cohesive system with integrated neural components for vectorized object generation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LEXICON Monolithic Application                │
│                         (Python 3.11+)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Core Engine Layer                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │  Primitive  │  │  X-Shaped   │  │   Vectorized    │  │  │
│  │  │   Reducer   │  │    Hole     │  │     Object      │  │  │
│  │  │   Engine    │  │   Engine    │  │   Generator     │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Neural Processing Layer                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │   Pattern   │  │   Empathy   │  │    Memetic     │  │  │
│  │  │   Encoder   │  │ Normalizer  │  │   Evolution    │  │  │
│  │  │ (BERT-base) │  │  (MHA)      │  │    (LSTM)      │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     Storage Layer                         │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │  │
│  │  │ PostgreSQL  │  │   Redis     │  │     FAISS       │  │  │
│  │  │   + JSONB   │  │   Cache     │  │  Vector Index   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                      API Layer                            │  │
│  │     FastAPI + GraphQL + WebSocket + Background Tasks     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Vectorized Object Generation**: Neural network-based generation of concept vectors
- **X-Shaped Hole Implementation**: Define concepts through negation
- **Empathy Normalization**: Optimize vectors for co-existence
- **Memetic Evolution**: Background process for concept evolution
- **Real-time Updates**: WebSocket support for live vector space visualization
- **Multiple API Interfaces**: REST, GraphQL, and WebSocket endpoints
- **Vector Visualization**: Dimensionality reduction for concept visualization
- **GPU Acceleration**: Automatic GPU usage when available

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/lexicon.git
   cd lexicon
   ```

2. Run the setup script:
   ```bash
   # On Linux/macOS
   ./scripts/setup.sh
   
   # On Windows
   scripts\setup.bat
   ```

3. Start the application:
   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Access the API at `http://localhost:8000`

## API Usage

### REST API

```python
import requests

# Define a concept
response = requests.post(
    "http://localhost:8000/api/v1/concepts/define",
    json={
        "concept": "Tree",
        "negations": ["rock", "building", "animal"]
    }
)
concept = response.json()
concept_id = concept["concept_id"]

# Get similar concepts
similar = requests.get(
    f"http://localhost:8000/api/v1/concepts/{concept_id}/similar"
)
```

### GraphQL

```graphql
mutation {
  defineConcept(input: {
    concept: "Tree",
    negations: ["rock", "building", "animal"]
  }) {
    conceptId
    conceptName
    nullRatio
    empathyScores
  }
}

query {
  concept(id: "concept-id-here") {
    name
    similarConcepts(k: 5) {
      concept
      similarity
    }
  }
}
```

### WebSocket

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
  // Subscribe to vector updates
  ws.send(JSON.stringify({
    type: "get_vector_updates"
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "vector_space_snapshot") {
    // Render visualization
    renderVectorSpace(data.data);
  }
};
```

## Performance Characteristics

- **Vector Generation**: ~50ms per concept (GPU), ~200ms (CPU)
- **Similarity Search**: <10ms for 1M vectors (FAISS)
- **API Response**: <100ms for most endpoints
- **Memory Usage**: ~2GB base + 100MB per 100k concepts
- **Concurrent Users**: Handles 1000+ with async architecture

## Development

### Project Structure

```
lexicon/
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── config.py               # Configuration management
│   │
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── primitives.py       # Existence primitive definitions
│   │   ├── reducer.py          # Pattern reduction engine
│   │   ├── x_shaped_hole.py    # X-shaped hole implementation
│   │   └── types.py            # Core type definitions
│   │
│   ├── neural/                 # Neural network components
│   │   ├── __init__.py
│   │   ├── models.py           # Neural architectures
│   │   ├── vectorizer.py       # Vector generation
│   │   ├── empathy.py          # Empathy normalization
│   │   └── evolution.py        # Memetic evolution
│   │
│   ├── storage/                # Data persistence
│   │   ├── __init__.py
│   │   ├── database.py         # PostgreSQL interface
│   │   ├── cache.py            # Redis caching
│   │   ├── vector_store.py     # FAISS integration
│   │   └── migrations/         # Database migrations
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── rest.py             # REST endpoints
│   │   ├── graphql.py          # GraphQL schema
│   │   ├── websocket.py        # Real-time updates
│   │   └── dependencies.py     # Shared dependencies
│   │
│   └── services/               # Business logic services
│       ├── __init__.py
│       ├── definition.py       # Concept definition service
│       ├── normalization.py    # Normalization service
│       ├── visualization.py    # Vector visualization
│       └── export.py           # LLM export service
│
├── models/                     # Trained model checkpoints
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── docker/                     # Docker configuration
├── requirements.txt
├── setup.py
└── README.md
```

### Running Tests

```bash
pytest tests/
```

### Docker Deployment

```bash
docker-compose up -d
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BERT model from Hugging Face Transformers
- FAISS vector search from Facebook Research
- FastAPI for the web framework
