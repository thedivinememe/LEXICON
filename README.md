# LEXICON: Memetic Atomic Dictionary with Vectorized Objects and Spherical Universal Set

LEXICON is a unified Python application that combines neural components for vectorized object generation with a comprehensive system for defining, storing, and manipulating concepts through negation. The system now features a spherical universal set (Bloch sphere topology) where antipodal points represent perfect negations.

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
- **Spherical Universal Set**: Bloch sphere topology where antipodal points represent perfect negations
- **Relative Type System**: Every concept is a bottom type with its negation as the top type
- **Existence Type Hierarchy**: 11 levels from VOID to TRANSCENDENT

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL
- Redis
- CUDA-compatible GPU (optional, for faster processing)
- Matplotlib and NumPy for visualizations

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

### Visualizing the Spherical Universal Set

To visualize the Bloch sphere with core concepts:

```bash
# Generate all visualizations and open in browser
python scripts/run_spherical_visualization.py --all --open

# Visualize a specific concept
python scripts/run_spherical_visualization.py --concept concept --open

# Visualize interpolation between two concepts
python scripts/run_spherical_visualization.py --interpolate abstraction,concrete --open

# Visualize null gradient
python scripts/run_spherical_visualization.py --null-gradient --open

# Or use the batch scripts
# On Windows
scripts\run_spherical_visualization.bat

# On Linux/macOS
chmod +x scripts/run_spherical_visualization.sh
./scripts/run_spherical_visualization.sh
```

The visualization system generates several interactive 3D visualizations:
- **Sphere Visualization**: Shows all concepts positioned in the spherical universe
- **Null Gradient Visualization**: Displays the null field intensity gradient
- **Relationship Visualization**: Shows AND/OR/NOT relationships between concepts
- **Type Hierarchy Visualization**: Displays the relative type hierarchy for a concept
- **Concept Cluster Visualization**: Shows clusters of related concepts
- **Nearest Concepts Visualization**: Displays the nearest concepts to a given concept
- **Concept Interpolation Visualization**: Shows the interpolation path between two concepts

For more details on the spherical system, see [SPHERICAL_SYSTEM.md](docs/SPHERICAL_SYSTEM.md).

## API Usage

### REST API (Including Spherical Endpoints)

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

# Define a concept in spherical space
response = requests.post(
    "http://localhost:8000/api/v1/concepts/define-spherical",
    json={
        "concept": "Tree",
        "negations": ["rock", "building", "animal"],
        "growth_pattern": "radial",
        "existence_type": "BIOLOGICAL"
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
# Define a concept in spherical space
mutation {
  defineConceptSpherical(input: {
    concept: "Tree",
    negations: ["rock", "building", "animal"],
    growthPattern: "radial",
    existenceType: "BIOLOGICAL"
  }) {
    conceptId
    conceptName
    sphericalPosition {
      r
      theta
      phi
    }
    nullDistance
    antipodalNegation {
      r
      theta
      phi
    }
  }
}

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
│   │   ├── types.py            # Core type definitions
│   │   ├── spherical_universe.py # Bloch sphere universe
│   │   ├── null_gradient.py    # Null gradient manager
│   │   ├── centroid_builder.py # Centroid concept builder
│   │   ├── existence_types.py  # Existence type hierarchy
│   │   ├── relative_type_system.py # Relative type system
│   │   ├── empathy_memeplex.py # Empathy memeplex
│   │   └── empathetic_golden_loop.py # Golden Loop processor
│   │
│   ├── neural/                 # Neural network components
│   │   ├── __init__.py
│   │   ├── models.py           # Neural architectures
│   │   ├── vectorizer.py       # Vector generation
│   │   ├── empathy.py          # Empathy normalization
│   │   ├── evolution.py        # Memetic evolution
│   │   └── spherical_vectorizer.py # Spherical relationship vectorizer
│   │
│   ├── storage/                # Data persistence
│   │   ├── __init__.py
│   │   ├── database.py         # PostgreSQL interface
│   │   ├── cache.py            # Redis caching
│   │   ├── vector_store.py     # FAISS integration
│   │   └── migrations/         # Database migrations
│   │       ├── 001_initial.sql # Initial schema
│   │       └── 003_spherical.sql # Spherical system schema
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── rest.py             # REST endpoints
│   │   ├── graphql.py          # GraphQL schema
│   │   ├── websocket.py        # Real-time updates
│   │   ├── dependencies.py     # Shared dependencies
│   │   └── spherical_rest.py   # Spherical API endpoints
│   │
│   ├── services/               # Business logic services
│   │   ├── __init__.py
│   │   ├── definition.py       # Concept definition service
│   │   ├── normalization.py    # Normalization service
│   │   ├── visualization.py    # Vector visualization
│   │   ├── export.py           # LLM export service
│   │   └── sphere_visualization.py # Spherical visualization
│   │
│   ├── data/                   # Data definitions
│   │   └── core_definitions.py # Core concept definitions
│   │
│   └── examples/               # Example code
│       └── spherical_integration_example.py # Spherical integration
│
├── models/                     # Trained model checkpoints
├── tests/                      # Test suite
│   └── test_spherical_system.py # Spherical system tests
├── scripts/                    # Utility scripts
│   ├── run_spherical_visualization.py # Run visualization
│   ├── run_spherical_visualization.bat # Windows script
│   └── run_spherical_visualization.sh # Unix script
├── docs/                       # Documentation
│   └── SPHERICAL_SYSTEM.md     # Spherical system docs
├── visualizations/             # Visualization outputs
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

## Security

For information on managing API keys and other sensitive credentials, see [SECURITY.md](docs/SECURITY.md).

## Deployment Guide

### Deploying to Heroku and GitHub Pages

This project is set up for deployment with:
- Backend: Heroku
- Frontend: GitHub Pages

For detailed deployment instructions, see [DEPLOYMENT.md](docs/DEPLOYMENT.md).

#### Quick Start

1. Set up GitHub repository secrets for automated deployment
2. Push to the `main` branch to trigger deployment
3. Backend will be deployed to Heroku
4. Frontend will be deployed to GitHub Pages

#### Deployment Scripts

The project includes deployment scripts to simplify the deployment process:

- **Backend Deployment**:
  ```bash
  # Windows
  scripts\deploy_to_heroku.bat
  
  # Linux/macOS
  chmod +x scripts/deploy_to_heroku.sh
  ./scripts/deploy_to_heroku.sh
  ```

- **Frontend Deployment**:
  ```bash
  # Windows
  scripts\deploy_to_github_pages.bat
  
  # Linux/macOS
  chmod +x scripts/deploy_to_github_pages.sh
  ./scripts/deploy_to_github_pages.sh
  ```

These scripts handle the entire deployment process, including:
- Creating a temporary directory with only necessary files (for backend)
- Setting up the correct API URL for the frontend
- Building and deploying the application

#### Deployment Process

The deployment process is automated using GitHub Actions:

1. **Backend Deployment (Heroku)**:
   - Triggered when changes are pushed to the `main` branch (excluding frontend files)
   - Runs tests before deployment
   - Deploys the application to Heroku
   - Runs database migrations after deployment
   - Sets necessary environment variables

2. **Frontend Deployment (GitHub Pages)**:
   - Triggered when changes are pushed to the `main` branch in the `frontend` directory
   - Builds the React application with the correct API URL
   - Deploys the built files to the `gh-pages` branch

#### Manual Deployment

To manually deploy:

1. **Backend (Heroku)**:
   ```bash
   # Login to Heroku
   heroku login

   # Create a new Heroku app if you don't have one
   heroku create your-app-name

   # Add PostgreSQL addon
   heroku addons:create heroku-postgresql:hobby-dev

   # Add Redis addon (optional)
   heroku addons:create heroku-redis:hobby-dev

   # Set environment variables
   heroku config:set ENVIRONMENT=production
   heroku config:set GITHUB_ORG=your-github-username
   heroku config:set GITHUB_REPO=your-repo-name

   # Deploy
   git push heroku main

   # Run database migrations
   heroku run python -m scripts.init_db
   ```

2. **Frontend (GitHub Pages)**:
   ```bash
   # Set the API URL in .env.local
   echo "REACT_APP_API_URL=https://your-heroku-app.herokuapp.com/api/v1" > frontend/.env.local

   # Build the frontend
   cd frontend
   npm install
   npm run build

   # Deploy to GitHub Pages
   npx gh-pages -d build
   ```

### Environment Configuration

Make sure to set the following environment variables in your Heroku application:

- `DATABASE_URL`: Set automatically by Heroku PostgreSQL addon
- `REDIS_URL`: Set automatically by Heroku Redis addon (if used)
- `SECRET_KEY`: A secure random string for JWT signing
- `ENVIRONMENT`: Set to "production"
- `GITHUB_ORG`: Your GitHub username or organization
- `GITHUB_REPO`: Your GitHub repository name
- `GITHUB_PAGES_URL`: The URL to your GitHub Pages site

## Acknowledgments

- BERT model from Hugging Face Transformers
- FAISS vector search from Facebook Research
- FastAPI for the web framework
