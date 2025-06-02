"""
Visualize test data from LEXICON.
This script generates visualizations of the test data concepts.
"""

import asyncio
import numpy as np
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from tests.test_data import (
    PHILOSOPHICAL_CONCEPTS,
    ETHICAL_CONCEPTS,
    HIERARCHICAL_CONCEPTS,
    CONCEPT_CLUSTERS,
    get_test_vectors,
    get_all_test_concepts
)

def generate_mock_vectors(concepts, dim=768):
    """Generate mock vectors for concepts"""
    vectors = {}
    for concept, negations in concepts:
        # Create a random unit vector
        vector = np.random.randn(dim)
        vector = vector / np.linalg.norm(vector)
        vectors[concept] = vector
    return vectors

def reduce_dimensions(vectors, method='tsne', dimensions=3):
    """Reduce dimensions of vectors"""
    # Convert to numpy array
    concept_names = list(vectors.keys())
    vectors_array = np.array([vectors[name] for name in concept_names])
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=dimensions)
    else:  # default to t-SNE
        # Calculate appropriate perplexity (should be less than n_samples)
        n_samples = vectors_array.shape[0]
        perplexity = min(30, n_samples - 1)  # Default is 30, but ensure it's less than n_samples
        print(f"Using perplexity of {perplexity} for {n_samples} samples")
        reducer = TSNE(n_components=dimensions, random_state=42, perplexity=perplexity)
    
    reduced = reducer.fit_transform(vectors_array)
    
    # Create result dictionary
    result = {}
    for i, name in enumerate(concept_names):
        result[name] = reduced[i]
    
    return result

def visualize_2d(reduced_vectors, title, output_file=None, clusters=None):
    """Create a 2D visualization"""
    # Extract data
    concept_names = list(reduced_vectors.keys())
    x = [reduced_vectors[name][0] for name in concept_names]
    y = [reduced_vectors[name][1] for name in concept_names]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot points
    if clusters:
        # Color by cluster
        cluster_names = list(clusters.keys())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(cluster_names)))
        
        for i, cluster_name in enumerate(cluster_names):
            cluster_concepts = clusters[cluster_name]
            cluster_x = []
            cluster_y = []
            
            for concept in cluster_concepts:
                if concept in reduced_vectors:
                    cluster_x.append(reduced_vectors[concept][0])
                    cluster_y.append(reduced_vectors[concept][1])
            
            plt.scatter(cluster_x, cluster_y, color=colors[i], label=cluster_name, s=100)
    else:
        # No clusters, use a single color
        plt.scatter(x, y, color='blue', s=100)
    
    # Add labels
    for i, name in enumerate(concept_names):
        plt.annotate(name, (x[i], y[i]), fontsize=9)
    
    # Add title and legend
    plt.title(title, fontsize=16)
    if clusters:
        plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved 2D visualization to {output_file}")
    else:
        plt.show()

def visualize_3d(reduced_vectors, title, output_file=None, clusters=None):
    """Create a 3D visualization using Plotly"""
    # Extract data
    concept_names = list(reduced_vectors.keys())
    x = [reduced_vectors[name][0] for name in concept_names]
    y = [reduced_vectors[name][1] for name in concept_names]
    z = [reduced_vectors[name][2] for name in concept_names]
    
    # Create figure
    if clusters:
        # Color by cluster
        cluster_names = list(clusters.keys())
        colors = px.colors.qualitative.Plotly
        
        fig = go.Figure()
        
        for i, cluster_name in enumerate(cluster_names):
            cluster_concepts = clusters[cluster_name]
            cluster_x = []
            cluster_y = []
            cluster_z = []
            cluster_text = []
            
            for concept in cluster_concepts:
                if concept in reduced_vectors:
                    cluster_x.append(reduced_vectors[concept][0])
                    cluster_y.append(reduced_vectors[concept][1])
                    cluster_z.append(reduced_vectors[concept][2])
                    cluster_text.append(concept)
            
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter3d(
                x=cluster_x,
                y=cluster_y,
                z=cluster_z,
                text=cluster_text,
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=color,
                ),
                name=cluster_name
            ))
    else:
        # No clusters, use a single color
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            text=concept_names,
            mode='markers+text',
            marker=dict(
                size=8,
                color='blue',
            )
        )])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save or show
    if output_file:
        fig.write_html(output_file)
        print(f"Saved 3D visualization to {output_file}")
    else:
        fig.show()

def create_cluster_mapping():
    """Create a mapping of concepts to clusters"""
    clusters = {}
    
    # Add concept clusters
    for cluster_name, concepts in CONCEPT_CLUSTERS.items():
        clusters[cluster_name] = concepts
    
    # Add philosophical concepts
    clusters['philosophical'] = [concept for concept, _ in PHILOSOPHICAL_CONCEPTS]
    
    # Add ethical concepts
    clusters['ethical'] = [concept for concept, _ in ETHICAL_CONCEPTS]
    
    # Add hierarchical concepts by level
    for level, concepts in HIERARCHICAL_CONCEPTS.items():
        clusters[level] = [concept for concept, _ in concepts]
    
    return clusters

def main():
    """Main function"""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations of LEXICON test data...")
    
    # Get all test concepts
    all_concepts = get_all_test_concepts()
    print(f"Found {len(all_concepts)} test concepts")
    
    # Try to use pre-generated test vectors if available
    try:
        vectors = get_test_vectors()
        print("Using pre-generated test vectors")
    except:
        # Generate mock vectors
        print("Generating mock vectors")
        vectors = generate_mock_vectors(all_concepts)
    
    # Create cluster mapping
    clusters = create_cluster_mapping()
    
    # Reduce dimensions
    print("Reducing dimensions with t-SNE...")
    reduced_2d = reduce_dimensions(vectors, method='tsne', dimensions=2)
    reduced_3d = reduce_dimensions(vectors, method='tsne', dimensions=3)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 2D visualization
    visualize_2d(
        reduced_2d,
        "LEXICON Concept Space (2D t-SNE)",
        output_file=str(output_dir / 'concepts_2d_tsne.png'),
        clusters=clusters
    )
    
    # 3D visualization
    visualize_3d(
        reduced_3d,
        "LEXICON Concept Space (3D t-SNE)",
        output_file=str(output_dir / 'concepts_3d_tsne.html'),
        clusters=clusters
    )
    
    # Also create PCA visualizations
    print("Reducing dimensions with PCA...")
    reduced_2d_pca = reduce_dimensions(vectors, method='pca', dimensions=2)
    reduced_3d_pca = reduce_dimensions(vectors, method='pca', dimensions=3)
    
    visualize_2d(
        reduced_2d_pca,
        "LEXICON Concept Space (2D PCA)",
        output_file=str(output_dir / 'concepts_2d_pca.png'),
        clusters=clusters
    )
    
    visualize_3d(
        reduced_3d_pca,
        "LEXICON Concept Space (3D PCA)",
        output_file=str(output_dir / 'concepts_3d_pca.html'),
        clusters=clusters
    )
    
    print("\nVisualization complete!")
    print(f"Output files are in the {output_dir} directory")
    print("Open the HTML files in a web browser to view interactive 3D visualizations")

if __name__ == "__main__":
    main()
