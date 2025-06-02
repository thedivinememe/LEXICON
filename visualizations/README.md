# LEXICON Visualization Tools

This directory contains visualizations of the LEXICON concept space. These visualizations help to understand the relationships between different concepts and how they cluster together in the vector space.

## Getting Started

### Step 1: Install Dependencies

First, install the required dependencies:

```bash
# On Windows
scripts\install_visualization_dependencies.bat

# On Unix/Linux/Mac
bash scripts/install_visualization_dependencies.sh

# Or directly with Python
python scripts/install_visualization_dependencies.py
```

This will install the necessary packages:
- matplotlib
- plotly
- scikit-learn
- numpy
- requests

### Step 2: Generate Visualizations

After installing dependencies, generate the visualizations:

```bash
# On Windows
scripts\generate_visualizations.bat

# On Unix/Linux/Mac
bash scripts/generate_visualizations.sh

# Or directly with Python
python scripts/generate_visualizations.py
```

This will:
1. Generate 2D and 3D visualizations of the concept space using t-SNE and PCA
2. Create an HTML dashboard to view the visualizations

### Step 3: Run All Tools (Optional)

Alternatively, you can run all visualization tools at once:

```bash
# On Windows
scripts\run_all_visualization_tools.bat

# On Unix/Linux/Mac
bash scripts/run_all_visualization_tools.sh

# Or directly with Python
python scripts/run_all_visualization_tools.py
```

## Visualization Types

### 2D Visualizations

- **concepts_2d_tsne.png**: A 2D visualization using t-SNE, which is good at preserving local structure and showing which concepts are similar to each other.
- **concepts_2d_pca.png**: A 2D visualization using PCA, which preserves global structure and shows the principal directions of variation in the data.

### 3D Visualizations

- **concepts_3d_tsne.html**: An interactive 3D visualization using t-SNE. You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.
- **concepts_3d_pca.html**: An interactive 3D visualization using PCA. You can rotate, zoom, and pan to explore the concept space. Hover over points to see concept names.

### Dashboard

- **index.html**: An HTML dashboard that displays all the visualizations in one place. Open this file in a web browser to view the dashboard.

## Understanding the Visualizations

The visualizations show the relationships between different concepts in the LEXICON system. Concepts that are similar to each other will be closer together in the visualization, while dissimilar concepts will be farther apart.

The concepts are colored by cluster, with each cluster representing a different category of concepts:

- **philosophical_cluster**: Core philosophical concepts like existence, knowledge, and empathy
- **ethical_cluster**: Ethical concepts like cooperation, golden rule, and mutual benefit
- **cooperative_cluster**: Concepts related to cooperation, sharing, and altruism
- **competitive_cluster**: Concepts related to competition, rivalry, and conflict
- **existence_level**: Concepts at the existence level, like rock and energy
- **life_level**: Concepts at the life level, like organism and growth
- **consciousness_level**: Concepts at the consciousness level, like awareness and thought

## API Access

You can also access the visualizations through the LEXICON API. Start the API server with:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit:
- **http://localhost:8000/docs**: API documentation
- **http://localhost:8000/api/v1/vectors/visualize**: Get visualization data for the top 50 concepts
- **http://localhost:8000/api/v1/vectors/visualize?concept_ids=concept1,concept2,concept3**: Get visualization data for specific concepts

## Customizing Visualizations

To customize the visualizations, you can modify the following files:

- **scripts/visualize_test_data.py**: The main script that generates the visualizations
- **scripts/create_dashboard.py**: The script that creates the HTML dashboard
- **tests/test_data.py**: The test data used for the visualizations

## Requirements

The visualization tools require the following Python packages:

- numpy
- matplotlib
- plotly
- scikit-learn

These packages should be installed as part of the LEXICON project requirements.
