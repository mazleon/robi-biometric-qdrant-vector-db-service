"""
FastAPI endpoints for Qdrant vector database service.
"""
import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from loguru import logger

from .schemas import *
from ..core.qdrant_client import QdrantVectorStore
from ..config.settings import settings


# Global vector store instance
_vector_store: QdrantVectorStore = None


async def get_vector_store() -> QdrantVectorStore:
    """Dependency to get vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(settings)
        await _vector_store.initialize_collection()
    return _vector_store


# Create router
router = APIRouter()


@router.post("/vectors/add", response_model=AddVectorResponse, tags=["Vectors"])
async def add_vector(
    request: AddVectorRequest,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """
    Add a single embedding vector to the database.
    
    - **embedding**: 512-dimensional face embedding vector
    - **user_id**: User identifier
    - **metadata**: Optional additional metadata
    - **point_id**: Optional custom point ID
    """
    try:
        point_id = await vector_store.add_vector(
            vector=request.embedding,
            user_id=request.user_id,
            metadata=request.metadata,
            point_id=request.point_id
        )
        
        return AddVectorResponse(
            success=True,
            point_id=point_id,
            message="Vector added successfully",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to add vector: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add vector: {str(e)}"
        )


@router.post("/vectors/add_batch", response_model=AddVectorsBatchResponse, tags=["Vectors"])
async def add_vectors_batch(
    request: AddVectorsBatchRequest,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """
    Add multiple embedding vectors in batch for optimal performance.
    
    - **embeddings**: List of vector embeddings
    - **user_ids**: List of user identifiers
    - **metadata_list**: Optional list of metadata dictionaries
    - **point_ids**: Optional list of custom point IDs
    """
    try:
        import numpy as np
        
        # Convert embeddings to numpy arrays
        vectors = [np.array(emb, dtype=np.float32) for emb in request.embeddings]
        
        point_ids = await vector_store.add_vectors_batch(
            vectors=vectors,
            user_ids=request.user_ids,
            metadata_list=request.metadata_list,
            point_ids=request.point_ids
        )
        
        return AddVectorsBatchResponse(
            success=True,
            point_ids=point_ids,
            added_count=len(point_ids),
            message=f"Successfully added {len(point_ids)} vectors",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to add vectors batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add vectors batch: {str(e)}"
        )


@router.post("/vectors/search", response_model=SearchResponse, tags=["Vectors"])
async def search_vectors(
    request: SearchRequest,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """
    Search for similar vectors using GPU-accelerated similarity computation.
    
    - **embedding**: Query embedding vector
    - **k**: Number of results to return (1-100)
    - **threshold**: Minimum similarity threshold (0.0-1.0)
    - **user_filter**: Optional filter by specific user ID
    """
    try:
        import numpy as np
        
        query_vector = np.array(request.embedding, dtype=np.float32)
        
        result = await vector_store.search(
            query_vector=query_vector,
            k=request.k,
            score_threshold=request.threshold,
            user_filter=request.user_filter
        )
        
        # Convert results to schema format
        search_results = [
            SearchResult(
                id=r["id"],
                score=r["score"],
                user_id=r["user_id"],
                metadata=r["metadata"],
                timestamp=r["timestamp"]
            )
            for r in result["results"]
        ]
        
        return SearchResponse(
            success=True,
            results=search_results,
            query_time_ms=result["query_time_ms"],
            total_results=result["total_results"],
            message=f"Found {len(search_results)} similar vectors",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.delete("/vectors/{point_id}", response_model=DeleteVectorResponse, tags=["Vectors"])
async def delete_vector(
    point_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """
    Delete a specific vector by point ID.
    
    - **point_id**: Point ID to delete
    """
    try:
        deleted = await vector_store.delete_vector(point_id)
        
        return DeleteVectorResponse(
            success=deleted,
            deleted=deleted,
            message=f"Vector {point_id} {'deleted' if deleted else 'not found'}",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to delete vector {point_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete vector: {str(e)}"
        )


@router.delete("/vectors/user/{user_id}", response_model=DeleteUserVectorsResponse, tags=["Vectors"])
async def delete_user_vectors(
    user_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """
    Delete all vectors for a specific user.
    
    - **user_id**: User ID to delete vectors for
    """
    try:
        deleted_count = await vector_store.delete_user_vectors(user_id)
        
        return DeleteUserVectorsResponse(
            success=True,
            deleted_count=deleted_count,
            message=f"Deleted {deleted_count} vectors for user {user_id}",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to delete vectors for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user vectors: {str(e)}"
        )




# System Endpoints

@router.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats(
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """Get database and performance statistics."""
    try:
        stats = await vector_store.get_stats()
        
        return StatsResponse(
            success=True,
            collection_info=stats.get("collection_info", {}),
            performance_stats=stats.get("performance_stats", {}),
            gpu_info=stats.get("gpu_info", {}),
            message="Statistics retrieved successfully",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check(
    vector_store: QdrantVectorStore = Depends(get_vector_store)
):
    """Perform health check on the vector database service."""
    try:
        health = await vector_store.health_check()
        
        return HealthCheckResponse(
            success=health["status"] == "healthy",
            status=health["status"],
            qdrant_connection=health["qdrant_connection"],
            collection_exists=health["collection_exists"],
            collection_name=health["collection_name"],
            gpu_available=health["gpu_available"],
            message=f"Service is {health['status']}",
            timestamp=health["timestamp"]
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            success=False,
            status="unhealthy",
            qdrant_connection=False,
            collection_exists=False,
            collection_name=settings.collection_name,
            gpu_available=False,
            message=f"Health check failed: {str(e)}",
            timestamp=time.time()
        )


@router.get("/info", tags=["System"])
async def get_service_info():
    """Get service information and configuration."""
    try:
        from ..core.gpu_optimizer import CUDA_AVAILABLE
        
        return {
            "service_name": "Qdrant Vector Database Service",
            "version": "1.0.0",
            "description": "High-performance GPU-accelerated vector database service",
            "configuration": {
                "vector_dimension": settings.vector_size,
                "similarity_metric": settings.distance_metric,
                "similarity_threshold": settings.similarity_threshold,
                "collection_name": settings.collection_name,
                "gpu_enabled": settings.use_gpu,
                "cuda_available": CUDA_AVAILABLE,
                "batch_size": settings.batch_size
            },
            "endpoints": {
                "vectors": ["/vectors/add", "/vectors/add_batch", "/vectors/search", "/vectors/{point_id}", "/vectors/user/{user_id}"],
                "system": ["/health", "/stats", "/info"]
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get service info: {str(e)}"
        )


# Note: Exception handlers moved to src/main.py (FastAPI app level)
# APIRouter does not support exception_handler decorator
