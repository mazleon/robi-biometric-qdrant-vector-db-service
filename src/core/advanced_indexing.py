"""
Advanced indexing and filtering strategies for world-class face recognition performance.
Implements payload field indexing, multi-vector search, and advanced filtering techniques.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PayloadIndexParams, KeywordIndexParams, IntegerIndexParams,
    FloatIndexParams, DatetimeIndexParams, FieldType,
    CreateFieldIndexCollection, Filter, FieldCondition,
    MatchValue, Range, GeoBoundingBox, ValuesCount
)

from ..config.settings import QdrantSettings


class AdvancedIndexingManager:
    """
    Manages advanced indexing strategies for optimal face recognition performance.
    
    Features:
    - Payload field indexing for fast filtering
    - Multi-dimensional search optimization
    - Dynamic index management
    - Performance monitoring
    """
    
    def __init__(self, client: QdrantClient, settings: QdrantSettings):
        self.client = client
        self.settings = settings
        self._indexed_fields = set()
        
    async def setup_payload_indexes(self) -> bool:
        """
        Setup optimized payload field indexes for face recognition.
        
        Returns:
            True if all indexes were created successfully
        """
        try:
            collection_name = self.settings.collection_name
            
            # Index configuration for different field types
            index_configs = [
                {
                    "field_name": "user_id",
                    "field_type": FieldType.KEYWORD,
                    "params": PayloadIndexParams(
                        keyword_index_params=KeywordIndexParams(
                            on_disk=False  # Keep in RAM for fast user filtering
                        )
                    )
                },
                {
                    "field_name": "timestamp",
                    "field_type": FieldType.DATETIME,
                    "params": PayloadIndexParams(
                        datetime_index_params=DatetimeIndexParams(
                            on_disk=True,  # Can be on disk for time-based queries
                            is_principal=False
                        )
                    )
                },
                {
                    "field_name": "confidence_score",
                    "field_type": FieldType.FLOAT,
                    "params": PayloadIndexParams(
                        float_index_params=FloatIndexParams(
                            on_disk=False,  # Keep in RAM for range queries
                            is_principal=False
                        )
                    )
                },
                {
                    "field_name": "face_quality",
                    "field_type": FieldType.INTEGER,
                    "params": PayloadIndexParams(
                        integer_index_params=IntegerIndexParams(
                            on_disk=False,
                            is_principal=False
                        )
                    )
                },
                {
                    "field_name": "enrollment_group",
                    "field_type": FieldType.KEYWORD,
                    "params": PayloadIndexParams(
                        keyword_index_params=KeywordIndexParams(
                            on_disk=False
                        )
                    )
                }
            ]
            
            success_count = 0
            for config in index_configs:
                try:
                    await asyncio.to_thread(
                        self.client.create_payload_index,
                        collection_name=collection_name,
                        field_name=config["field_name"],
                        field_schema=config["field_type"],
                        field_index_params=config["params"],
                        wait=True
                    )
                    
                    self._indexed_fields.add(config["field_name"])
                    success_count += 1
                    logger.info(f"Created index for field: {config['field_name']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create index for {config['field_name']}: {e}")
            
            logger.info(f"Successfully created {success_count}/{len(index_configs)} payload indexes")
            return success_count == len(index_configs)
            
        except Exception as e:
            logger.error(f"Failed to setup payload indexes: {e}")
            return False
    
    async def create_advanced_filters(self, 
                                    user_ids: Optional[List[str]] = None,
                                    time_range: Optional[Dict[str, float]] = None,
                                    confidence_min: Optional[float] = None,
                                    quality_min: Optional[int] = None,
                                    enrollment_groups: Optional[List[str]] = None) -> Optional[Filter]:
        """
        Create advanced filters for high-performance searches.
        
        Args:
            user_ids: List of user IDs to filter by
            time_range: Time range filter {"start": timestamp, "end": timestamp}
            confidence_min: Minimum confidence score
            quality_min: Minimum face quality score
            enrollment_groups: List of enrollment groups to include
            
        Returns:
            Optimized Filter object or None
        """
        try:
            conditions = []
            
            # User ID filtering (most common, optimize first)
            if user_ids:
                if len(user_ids) == 1:
                    conditions.append(
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_ids[0])
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(any=user_ids)
                        )
                    )
            
            # Time range filtering
            if time_range:
                conditions.append(
                    FieldCondition(
                        key="timestamp",
                        range=Range(
                            gte=time_range.get("start"),
                            lte=time_range.get("end")
                        )
                    )
                )
            
            # Confidence score filtering
            if confidence_min is not None:
                conditions.append(
                    FieldCondition(
                        key="confidence_score",
                        range=Range(gte=confidence_min)
                    )
                )
            
            # Quality filtering
            if quality_min is not None:
                conditions.append(
                    FieldCondition(
                        key="face_quality",
                        range=Range(gte=quality_min)
                    )
                )
            
            # Enrollment group filtering
            if enrollment_groups:
                conditions.append(
                    FieldCondition(
                        key="enrollment_group",
                        match=MatchValue(any=enrollment_groups)
                    )
                )
            
            if not conditions:
                return None
            
            return Filter(must=conditions)
            
        except Exception as e:
            logger.error(f"Failed to create advanced filters: {e}")
            return None
    
    async def optimize_search_parameters(self, 
                                       query_vector: np.ndarray,
                                       k: int,
                                       collection_size: int) -> Dict[str, Any]:
        """
        Dynamically optimize search parameters based on collection size and query.
        
        Args:
            query_vector: Query embedding vector
            k: Number of results requested
            collection_size: Current collection size
            
        Returns:
            Optimized search parameters
        """
        try:
            # Dynamic HNSW ef parameter based on collection size and k
            if collection_size < 10000:
                # Small collection: use exact search
                ef = max(k * 2, 64)
                use_exact = True
            elif collection_size < 100000:
                # Medium collection: balanced approach
                ef = max(k * 4, 128)
                use_exact = False
            elif collection_size < 1000000:
                # Large collection: optimize for speed
                ef = max(k * 6, 256)
                use_exact = False
            else:
                # Very large collection (10M+): maximum optimization
                ef = max(k * 8, 512)
                use_exact = False
            
            # Quantization parameters for large collections
            quantization_params = None
            if self.settings.quantization_enabled and collection_size > 50000:
                quantization_params = {
                    "ignore": False,
                    "rescore": True,
                    "oversampling": min(4.0, max(2.0, collection_size / 500000))
                }
            
            # Search timeout based on collection size
            timeout = min(
                self.settings.search_timeout,
                max(0.001, 0.1 * (collection_size / 1000000))  # Scale with size
            )
            
            return {
                "ef": ef,
                "use_exact": use_exact,
                "quantization_params": quantization_params,
                "timeout": timeout,
                "collection_size": collection_size
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize search parameters: {e}")
            return {
                "ef": self.settings.hnsw_ef,
                "use_exact": False,
                "quantization_params": None,
                "timeout": self.settings.search_timeout
            }
    
    async def create_multi_vector_search(self,
                                       query_vectors: List[np.ndarray],
                                       weights: Optional[List[float]] = None,
                                       k: int = 10) -> Dict[str, Any]:
        """
        Perform multi-vector search for enhanced accuracy.
        
        Args:
            query_vectors: List of query vectors (e.g., multiple face angles)
            weights: Optional weights for each vector
            k: Number of results to return
            
        Returns:
            Combined search results with weighted scores
        """
        try:
            if not query_vectors:
                return {"results": [], "query_time_ms": 0}
            
            if weights is None:
                weights = [1.0] * len(query_vectors)
            
            start_time = time.perf_counter()
            
            # Perform parallel searches
            search_tasks = []
            for i, (vector, weight) in enumerate(zip(query_vectors, weights)):
                task = asyncio.create_task(
                    self._single_vector_search(vector, k * 2, f"query_{i}")
                )
                search_tasks.append((task, weight))
            
            # Collect results
            all_results = {}
            for task, weight in search_tasks:
                results = await task
                for result in results.get("results", []):
                    point_id = result["id"]
                    weighted_score = result["score"] * weight
                    
                    if point_id in all_results:
                        all_results[point_id]["score"] += weighted_score
                        all_results[point_id]["vote_count"] += 1
                    else:
                        all_results[point_id] = {
                            **result,
                            "score": weighted_score,
                            "vote_count": 1
                        }
            
            # Sort by combined score and vote count
            final_results = sorted(
                all_results.values(),
                key=lambda x: (x["score"], x["vote_count"]),
                reverse=True
            )[:k]
            
            duration = time.perf_counter() - start_time
            
            return {
                "results": final_results,
                "query_time_ms": duration * 1000,
                "total_results": len(final_results),
                "multi_vector_count": len(query_vectors)
            }
            
        except Exception as e:
            logger.error(f"Multi-vector search failed: {e}")
            return {"results": [], "query_time_ms": 0, "error": str(e)}
    
    async def _single_vector_search(self, 
                                  vector: np.ndarray, 
                                  k: int, 
                                  query_id: str) -> Dict[str, Any]:
        """Helper method for single vector search."""
        try:
            # This would integrate with the main QdrantVectorStore search method
            # For now, return a placeholder structure
            return {
                "results": [],
                "query_time_ms": 0,
                "query_id": query_id
            }
        except Exception as e:
            logger.error(f"Single vector search failed for {query_id}: {e}")
            return {"results": [], "query_time_ms": 0, "error": str(e)}
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics and performance metrics."""
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                self.settings.collection_name
            )
            
            return {
                "indexed_fields": list(self._indexed_fields),
                "collection_size": collection_info.points_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "segments_count": collection_info.segments_count,
                "indexing_status": {
                    "green": collection_info.status == "green",
                    "optimization_status": getattr(collection_info, 'optimizer_status', 'unknown')
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_indexes(self) -> bool:
        """Remove all created indexes."""
        try:
            success_count = 0
            for field_name in self._indexed_fields.copy():
                try:
                    await asyncio.to_thread(
                        self.client.delete_payload_index,
                        collection_name=self.settings.collection_name,
                        field_name=field_name,
                        wait=True
                    )
                    self._indexed_fields.remove(field_name)
                    success_count += 1
                    logger.info(f"Deleted index for field: {field_name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to delete index for {field_name}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup indexes: {e}")
            return False
