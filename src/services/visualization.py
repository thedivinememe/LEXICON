"""
Visualization service for LEXICON.
Provides methods for visualizing vectors in lower dimensions.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

class VisualizationService:
    """Service for visualizing vectors in lower dimensions"""
    
    def __init__(self, app_state: Dict[str, Any]):
        """Initialize the visualization service"""
        self.app_state = app_state
        self.db = app_state["db"]
        self.vector_store = app_state["vector_store"]
        self.cache = app_state["cache"]
    
    async def get_top_concepts(self, limit: int = 50) -> List[str]:
        """
        Get the IDs of the top concepts based on access frequency.
        
        Args:
            limit: Maximum number of concepts to return
        
        Returns:
            List of concept IDs
        """
        # Check cache first
        cache_key = f"top_concepts:{limit}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get concepts accessed in the last week
        one_week_ago = datetime.utcnow() - timedelta(days=7)
        
        # Query for most accessed concepts
        query = """
        SELECT concept_id, COUNT(*) as access_count
        FROM concept_access
        WHERE accessed_at > $1
        GROUP BY concept_id
        ORDER BY access_count DESC
        LIMIT $2
        """
        
        results = await self.db.fetch(query, one_week_ago, limit)
        
        # Extract concept IDs
        concept_ids = [row["concept_id"] for row in results]
        
        # If we don't have enough concepts from access data, get the most recent ones
        if len(concept_ids) < limit:
            remaining = limit - len(concept_ids)
            
            query = """
            SELECT id
            FROM concepts
            WHERE id NOT IN (SELECT unnest($1::text[]))
            ORDER BY created_at DESC
            LIMIT $2
            """
            
            additional = await self.db.fetch(query, concept_ids, remaining)
            concept_ids.extend([row["id"] for row in additional])
        
        # Cache the result
        await self.cache.set(cache_key, concept_ids, expire=3600)  # 1 hour
        
        return concept_ids
    
    async def create_visualization(
        self,
        concept_ids: List[str],
        method: str = "tsne",
        dimensions: int = 3
    ) -> Dict[str, Any]:
        """
        Create a visualization of vectors in lower dimensions.
        
        Args:
            concept_ids: List of concept IDs to visualize
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            dimensions: Number of dimensions (2 or 3)
        
        Returns:
            Dictionary with visualization data
        """
        # Validate inputs
        if dimensions not in (2, 3):
            raise ValueError("Dimensions must be 2 or 3")
        
        if method not in ("tsne", "pca", "umap"):
            raise ValueError("Method must be 'tsne', 'pca', or 'umap'")
        
        # Check cache
        cache_key = f"viz:{method}:{dimensions}:{','.join(concept_ids)}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        # Get concept data
        concepts = []
        vectors = []
        
        for concept_id in concept_ids:
            # Get concept from database
            concept = await self.db.concepts.find_one({"id": concept_id})
            if not concept:
                continue
            
            # Get vector from vector store
            vector = self.vector_store.get_vector(concept_id)
            if vector is None:
                continue
            
            concepts.append(concept)
            vectors.append(vector)
        
        if not vectors:
            return {"points": []}
        
        # Convert to numpy array
        vectors_array = np.array(vectors)
        
        # Apply dimensionality reduction
        coordinates = self._reduce_dimensions(vectors_array, method, dimensions)
        
        # Create result
        points = []
        for i, concept in enumerate(concepts):
            points.append({
                "concept_id": concept["id"],
                "concept_name": concept["name"],
                "coordinates": coordinates[i].tolist(),
                "null_ratio": concept.get("null_ratio", 0.0)
            })
        
        # Optionally add clustering
        if len(points) >= 5:  # Only cluster if we have enough points
            clusters = self._cluster_points(coordinates)
            for i, cluster in enumerate(clusters):
                points[i]["cluster"] = int(cluster)
        
        result = {
            "method": method,
            "dimensions": dimensions,
            "points": points
        }
        
        # Cache the result
        await self.cache.set(cache_key, result, expire=3600)  # 1 hour
        
        return result
    
    def _reduce_dimensions(
        self,
        vectors: np.ndarray,
        method: str,
        dimensions: int
    ) -> np.ndarray:
        """
        Reduce the dimensionality of vectors.
        
        Args:
            vectors: Array of vectors
            method: Dimensionality reduction method
            dimensions: Number of dimensions
        
        Returns:
            Array of reduced vectors
        """
        if method == "pca":
            return self._reduce_pca(vectors, dimensions)
        elif method == "tsne":
            return self._reduce_tsne(vectors, dimensions)
        elif method == "umap":
            return self._reduce_umap(vectors, dimensions)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _reduce_pca(self, vectors: np.ndarray, dimensions: int) -> np.ndarray:
        """Reduce dimensions using PCA"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=dimensions)
        return pca.fit_transform(vectors)
    
    def _reduce_tsne(self, vectors: np.ndarray, dimensions: int) -> np.ndarray:
        """Reduce dimensions using t-SNE"""
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=dimensions, random_state=42)
        return tsne.fit_transform(vectors)
    
    def _reduce_umap(self, vectors: np.ndarray, dimensions: int) -> np.ndarray:
        """Reduce dimensions using UMAP"""
        try:
            import umap
            
            reducer = umap.UMAP(n_components=dimensions, random_state=42)
            return reducer.fit_transform(vectors)
        except ImportError:
            # Fall back to t-SNE if UMAP is not available
            return self._reduce_tsne(vectors, dimensions)
    
    def _cluster_points(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Cluster points using K-means.
        
        Args:
            coordinates: Array of coordinates
        
        Returns:
            Array of cluster labels
        """
        from sklearn.cluster import KMeans
        
        # Determine number of clusters based on data size
        n_samples = coordinates.shape[0]
        n_clusters = min(max(2, n_samples // 10), 10)  # Between 2 and 10 clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(coordinates)
