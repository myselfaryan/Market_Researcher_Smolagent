"""
Embedding model for text similarity and clustering
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer

from ..config.settings import DEFAULT_EMBEDDING_MODEL

class EmbeddingModel:
    """Provides text embedding functionality for semantic analysis"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding model
        
        Args:
            model_name: Name of the pretrained model to use
        """
        self.logger = logging.getLogger("models.embedding")
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.model = None
    
    def _load_model(self):
        """Load the embedding model if not already loaded"""
        if self.model is None:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            try:
                self.model = SentenceTransformer(self.model_name)
                self.logger.info("Embedding model loaded successfully")
            except Exception as e:
                self.logger.error(f"Error loading embedding model: {str(e)}")
                raise
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        self._load_model()
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.model.encode(texts, show_progress_bar=len(texts) > 10)
        
        return embeddings
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        self._load_model()
        
        # Generate embeddings
        embedding1 = self.model.encode(text1)
        embedding2 = self.model.encode(text2)
        
        # Calculate cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        else:
            similarity = 0
            
        return float(similarity)
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster texts into groups based on semantic similarity
        
        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with cluster information
        """
        from sklearn.cluster import KMeans
        
        if len(texts) < n_clusters:
            n_clusters = max(1, len(texts) // 2)
        
        # Generate embeddings
        embeddings = self.get_embeddings(texts)
        
        # Perform clustering
        self.logger.info(f"Clustering {len(texts)} texts into {n_clusters} clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize texts by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            label_str = str(label)
            if label_str not in clusters:
                clusters[label_str] = []
            clusters[label_str].append(texts[i])
        
        # Find central text for each cluster
        central_texts = {}
        for label, cluster_texts in clusters.items():
            if not cluster_texts:
                continue
                
            # Get cluster centroid
            cluster_indices = [i for i, l in enumerate(cluster_labels) if str(l) == label]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find text closest to centroid
            distances = [np.linalg.norm(centroid - embeddings[i]) for i in cluster_indices]
            closest_idx = cluster_indices[np.argmin(distances)]
            central_texts[label] = texts[closest_idx]
        
        return {
            "clusters": clusters,
            "central_texts": central_texts,
            "cluster_labels": cluster_labels.tolist(),
            "n_clusters": n_clusters
        }
    
    def find_similar_texts(self, query: str, texts: List[str], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find texts most similar to a query
        
        Args:
            query: Query text
            texts: List of texts to search
            top_n: Number of similar texts to return
            
        Returns:
            List of dictionaries with text and similarity score
        """
        self._load_model()
        
        # Generate query embedding
        query_embedding = self.model.encode(query)
        
        # Generate embeddings for all texts
        text_embeddings = self.get_embeddings(texts)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(text_embeddings):
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(embedding)
            
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(query_embedding, embedding) / (norm1 * norm2)
            else:
                similarity = 0
                
            similarities.append({
                "text": texts[i],
                "similarity": float(similarity),
                "index": i
            })
        
        # Sort by similarity (descending)
        sorted_similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
        
        # Return top N
        return sorted_similarities[:top_n]
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract keywords from text using embedding-based approach
        
        Args:
            text: Text to extract keywords from
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        import re
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Clean and tokenize text
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Extract candidate keywords (n-grams)
        n_gram_range = (1, 1)  # Unigrams
        stop_words = "english"
        
        # Extract candidate words
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([text])
        candidates = count.get_feature_names_out()
        
        if len(candidates) == 0:
            return []
        
        # Get embeddings for candidates and the document
        self._load_model()
        doc_embedding = self.model.encode([text])[0]
        candidate_embeddings = self.model.encode(candidates)
        
        # Calculate similarities
        similarities = cosine_similarity(doc_embedding.reshape(1, -1), candidate_embeddings)[0]
        
        # Rank candidates by similarity
        keywords = [(candidates[i], similarities[i]) for i in range(len(candidates))]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return [kw[0] for kw in keywords[:top_n]]