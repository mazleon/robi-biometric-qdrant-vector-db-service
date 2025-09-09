"""
High-performance Qdrant client with GPU-accelerated vector operations.
Maintains compatibility with existing FAISS interface while providing superior performance.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CollectionInfo, PointStruct, 
    SearchRequest, Filter, FieldCondition, MatchValue,
    CreateCollection, UpdateCollection, OptimizersConfigDiff,
    HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, QuantizationSearchParams
)
from qdrant_client.models import ScoredPoint

from ..config.settings import QdrantSettings
from .gpu_optimizer import GPUVectorOptimizer


class QdrantVectorStore:
    """
    High-performance Qdrant vector store with GPU acceleration.
    
    Provides a drop-in replacement for FAISS with enhanced performance
    and better scalability for face recognition embeddings.
    """
    
    def __init__(self, settings: QdrantSettings):
        self.settings = settings
        self.gpu_optimizer = GPUVectorOptimizer(
            device_id=settings.gpu_device_id,
            memory_fraction=settings.gpu_memory_fraction
        )
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            api_key=settings.qdrant_api_key,
            timeout=settings.search_timeout
        )
        
        self._collection_initialized = False
        self._stats = {
            "total_vectors": 0,
            "search_count": 0,
            "add_count": 0,
            "total_search_time": 0.0,
            "total_add_time": 0.0
        }
    
    async def initialize_collection(self) -> bool:
        """
        Initialize Qdrant collection with optimized settings for face embeddings.
        
        Returns:
            True if collection was created/updated successfully
        """
        try:
            collection_name = self.settings.collection_name
            
            # Check if collection exists
            collections = await asyncio.to_thread(self.client.get_collections)
            existing_collection = None
            
            for collection in collections.collections:
                if collection.name == collection_name:
                    existing_collection = collection
                    break
            
            if existing_collection:
                logger.info(f"Collection '{collection_name}' already exists")
                await self._update_collection_if_needed()
            else:
                await self._create_collection()
            
            # Verify collection
            collection_info = await asyncio.to_thread(
                self.client.get_collection, collection_name
            )
            
            self._stats["total_vectors"] = collection_info.points_count
            self._collection_initialized = True
            
            logger.info(f"Collection '{collection_name}' initialized with {self._stats['total_vectors']} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False
    
    async def _create_collection(self):
        """Create new collection with optimized configuration."""
        # Prepare vector configuration
        vectors_config = VectorParams(
            size=self.settings.vector_size,
            distance=Distance.COSINE
        )
        
        # Prepare HNSW configuration
        hnsw_config = HnswConfigDiff(
            ef_construct=self.settings.hnsw_ef_construct,
            m=self.settings.hnsw_m,
            full_scan_threshold=self.settings.full_scan_threshold,
            max_indexing_threads=self.settings.hnsw_max_indexing_threads,
            on_disk=self.settings.hnsw_on_disk
        )
        
        # Prepare optimizers configuration
        optimizers_config = OptimizersConfigDiff(
            deleted_threshold=0.1,  # More aggressive cleanup for large datasets
            vacuum_min_vector_number=self.settings.vacuum_min_vector_number,
            default_segment_number=0,  # Auto-calculate optimal segments
            max_segment_size=self.settings.segment_size_mb * 1024 * 1024,
            memmap_threshold=self.settings.memmap_threshold_mb * 1024 * 1024,
            indexing_threshold=self.settings.indexing_threshold,
            flush_interval_sec=1,  # Faster flushing for real-time performance
            max_optimization_threads=0  # Use all available threads
        )
        
        # Prepare quantization configuration if enabled
        quantization_config = None
        if self.settings.quantization_enabled:
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8 if self.settings.quantization_type == "int8" else ScalarType.BINARY,
                    quantile=self.settings.quantization_quantile,
                    always_ram=self.settings.quantization_always_ram
                )
            )
        
        # Create collection with direct parameters
        await asyncio.to_thread(
            self.client.create_collection,
            collection_name=self.settings.collection_name,
            vectors_config=vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            quantization_config=quantization_config,
            replication_factor=self.settings.replication_factor
        )
        
        logger.info(f"Created collection '{self.settings.collection_name}' with optimized settings")
    
    async def _update_collection_if_needed(self):
        """Update existing collection configuration if needed."""
        try:
            update_config = UpdateCollection(
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=20000,
                    flush_interval_sec=5
                )
            )
            
            await asyncio.to_thread(
                self.client.update_collection,
                collection_name=self.settings.collection_name,
                optimizer_config=update_config.optimizers_config
            )
            
            logger.info("Updated collection configuration")
            
        except Exception as e:
            logger.warning(f"Failed to update collection: {e}")
    
    async def add_vector(self, vector: Union[List[float], np.ndarray], 
                        user_id: str, metadata: Optional[Dict[str, Any]] = None,
                        point_id: Optional[str] = None) -> str:
        """
        Add a single vector to the collection with GPU optimization.
        
        Args:
            vector: Embedding vector
            user_id: User identifier
            metadata: Additional metadata
            point_id: Optional custom point ID
            
        Returns:
            Point ID of the added vector
        """
        if not self._collection_initialized:
            await self.initialize_collection()
        
        start_time = time.perf_counter()
        
        try:
            # Convert to numpy array and normalize using GPU
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            
            # GPU-accelerated normalization
            normalized_vector = self.gpu_optimizer.normalize_embeddings_gpu(
                vector.reshape(1, -1)
            )[0]
            
            # Generate point ID if not provided
            if point_id is None:
                point_id = str(uuid4())
            
            # Prepare payload
            payload = {
                "user_id": user_id,
                "timestamp": time.time(),
                **(metadata or {})
            }
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=normalized_vector.tolist(),
                payload=payload
            )
            
            # Insert point
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.settings.collection_name,
                points=[point]
            )
            
            # Update stats
            self._stats["add_count"] += 1
            self._stats["total_vectors"] += 1
            duration = time.perf_counter() - start_time
            self._stats["total_add_time"] += duration
            
            logger.debug(f"Added vector for user {user_id} in {duration*1000:.2f}ms")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to add vector: {e}")
            raise
    
    async def add_vectors_batch(self, vectors: List[np.ndarray], 
                               user_ids: List[str],
                               metadata_list: Optional[List[Dict[str, Any]]] = None,
                               point_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add multiple vectors in batch with GPU optimization.
        
        Args:
            vectors: List of embedding vectors
            user_ids: List of user identifiers
            metadata_list: List of metadata dictionaries
            point_ids: Optional list of custom point IDs
            
        Returns:
            List of point IDs for added vectors
        """
        if not vectors:
            return []
        
        if not self._collection_initialized:
            await self.initialize_collection()
        
        start_time = time.perf_counter()
        
        try:
            # GPU-accelerated batch normalization
            batch_array = np.stack(vectors)
            normalized_vectors = self.gpu_optimizer.normalize_embeddings_gpu(batch_array)
            
            # Prepare points
            points = []
            result_point_ids = []
            
            for i, (vector, user_id) in enumerate(zip(normalized_vectors, user_ids)):
                point_id = point_ids[i] if point_ids else str(uuid4())
                result_point_ids.append(point_id)
                
                payload = {
                    "user_id": user_id,
                    "timestamp": time.time(),
                    **(metadata_list[i] if metadata_list else {})
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                ))
            
            # Batch insert
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.settings.collection_name,
                points=points
            )
            
            # Update stats
            self._stats["add_count"] += len(vectors)
            self._stats["total_vectors"] += len(vectors)
            duration = time.perf_counter() - start_time
            self._stats["total_add_time"] += duration
            
            logger.info(f"Added {len(vectors)} vectors in batch in {duration*1000:.2f}ms")
            return result_point_ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors batch: {e}")
            raise
    
    async def search(self, query_vector: Union[List[float], np.ndarray],
                    k: int = 10, score_threshold: Optional[float] = None,
                    user_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Search for similar vectors with GPU acceleration.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score
            user_filter: Optional user ID filter
            
        Returns:
            Search results with metadata
        """
        if not self._collection_initialized:
            await self.initialize_collection()
        
        start_time = time.perf_counter()
        
        try:
            # Convert to numpy array and normalize using GPU
            if isinstance(query_vector, list):
                query_vector = np.array(query_vector, dtype=np.float32)
            
            # GPU-accelerated normalization
            normalized_query = self.gpu_optimizer.normalize_embeddings_gpu(
                query_vector.reshape(1, -1)
            )[0]
            
            # Prepare search request with optimized parameters
            search_params = None
            if self.settings.quantization_enabled:
                search_params = QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=3.0  # Higher oversampling for better accuracy
                )
            
            # Add HNSW search parameters for better performance
            hnsw_ef = min(max(k * 4, self.settings.hnsw_ef), 512)  # Dynamic ef based on k
            
            # Prepare filter
            query_filter = None
            if user_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_filter)
                        )
                    ]
                )
            
            # Perform search
            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=self.settings.collection_name,
                query_vector=normalized_query.tolist(),
                query_filter=query_filter,
                limit=k,
                score_threshold=score_threshold or self.settings.similarity_threshold,
                params=search_params
            )
            
            # Process results
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "user_id": scored_point.payload.get("user_id"),
                    "metadata": {k: v for k, v in scored_point.payload.items() 
                               if k not in ["user_id", "timestamp"]},
                    "timestamp": scored_point.payload.get("timestamp")
                }
                results.append(result)
            
            # Update stats
            self._stats["search_count"] += 1
            duration = time.perf_counter() - start_time
            self._stats["total_search_time"] += duration
            
            response = {
                "results": results,
                "query_time_ms": duration * 1000,
                "total_results": len(results)
            }
            
            logger.debug(f"Search completed in {duration*1000:.2f}ms, found {len(results)} results")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def delete_vector(self, point_id: str) -> bool:
        """
        Delete a vector by point ID.
        
        Args:
            point_id: Point ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.settings.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            
            self._stats["total_vectors"] = max(0, self._stats["total_vectors"] - 1)
            logger.debug(f"Deleted vector with ID: {point_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector {point_id}: {e}")
            return False
    
    async def delete_user_vectors(self, user_id: str) -> int:
        """
        Delete all vectors for a specific user.
        
        Args:
            user_id: User ID to delete vectors for
            
        Returns:
            Number of vectors deleted
        """
        try:
            # Delete with filter
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
            
            result = await asyncio.to_thread(
                self.client.delete,
                collection_name=self.settings.collection_name,
                points_selector=delete_filter
            )
            
            deleted_count = getattr(result, 'operation_id', 0)  # Approximate
            self._stats["total_vectors"] = max(0, self._stats["total_vectors"] - deleted_count)
            
            logger.info(f"Deleted vectors for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete vectors for user {user_id}: {e}")
            return 0
    
    async def get_vector_count(self) -> int:
        """Get total number of vectors in collection."""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.settings.collection_name
            )
            return collection_info.points_count
            
        except Exception as e:
            logger.error(f"Failed to get vector count: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get performance and usage statistics."""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.settings.collection_name
            )
            
            gpu_info = self.gpu_optimizer.get_memory_info()
            
            avg_search_time = (
                self._stats["total_search_time"] / max(1, self._stats["search_count"])
            ) * 1000
            
            avg_add_time = (
                self._stats["total_add_time"] / max(1, self._stats["add_count"])
            ) * 1000
            
            return {
                "collection_info": {
                    "name": collection_info.config.params.vectors.size,
                    "vector_count": collection_info.points_count,
                    "indexed_vectors": collection_info.indexed_vectors_count,
                    "segments_count": collection_info.segments_count
                },
                "performance_stats": {
                    "total_searches": self._stats["search_count"],
                    "total_additions": self._stats["add_count"],
                    "avg_search_time_ms": avg_search_time,
                    "avg_add_time_ms": avg_add_time
                },
                "gpu_info": gpu_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the vector store."""
        try:
            # Check Qdrant connection
            collections = await asyncio.to_thread(self.client.get_collections)
            qdrant_healthy = True
            
            # Check collection
            collection_exists = any(
                c.name == self.settings.collection_name 
                for c in collections.collections
            )
            
            # Check GPU
            gpu_info = self.gpu_optimizer.get_memory_info()
            
            return {
                "status": "healthy" if qdrant_healthy and collection_exists else "unhealthy",
                "qdrant_connection": qdrant_healthy,
                "collection_exists": collection_exists,
                "collection_name": self.settings.collection_name,
                "gpu_available": gpu_info.get("gpu_available", False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            self.gpu_optimizer.cleanup()
            logger.info("Qdrant vector store cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'gpu_optimizer'):
                self.gpu_optimizer.cleanup()
        except:
            pass
