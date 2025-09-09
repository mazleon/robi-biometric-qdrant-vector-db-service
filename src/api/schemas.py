"""Pydantic schemas for Qdrant vector database API.
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = Field(description="Operation success status")
    message: Optional[str] = Field(default=None, description="Response message")
    timestamp: Optional[float] = Field(default=None, description="Response timestamp")


class AddVectorRequest(BaseModel):
    """Request model for adding a single vector."""
    embedding: List[float] = Field(description="Vector embedding")
    user_id: str = Field(description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    point_id: Optional[str] = Field(default=None, description="Custom point ID")
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if len(v) < 1:
            raise ValueError('Embedding must not be empty')
        return v


class AddVectorResponse(BaseResponse):
    """Response model for adding a single vector."""
    point_id: Optional[str] = Field(default=None, description="Generated or provided point ID")


class AddVectorsBatchRequest(BaseModel):
    """Request model for adding multiple vectors in batch."""
    embeddings: List[List[float]] = Field(description="List of vector embeddings")
    user_ids: List[str] = Field(description="List of user identifiers")
    metadata_list: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of metadata")
    point_ids: Optional[List[str]] = Field(default=None, description="List of custom point IDs")
    
    @validator('embeddings')
    def validate_embeddings(cls, v):
        for i, embedding in enumerate(v):
            if len(embedding) < 1:
                raise ValueError(f'Embedding at index {i} must not be empty')
        return v
    
    @validator('user_ids')
    def validate_user_ids_length(cls, v, values):
        if 'embeddings' in values and len(v) != len(values['embeddings']):
            raise ValueError('Number of user_ids must match number of embeddings')
        return v


class AddVectorsBatchResponse(BaseResponse):
    """Response model for batch vector addition."""
    point_ids: List[str] = Field(description="List of generated or provided point IDs")
    added_count: int = Field(description="Number of vectors successfully added")


class SearchRequest(BaseModel):
    """Request model for vector similarity search."""
    embedding: List[float] = Field(description="Query embedding vector")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum similarity threshold")
    user_filter: Optional[str] = Field(default=None, description="Filter by specific user ID")
    
    @validator('embedding')
    def validate_embedding_dimension(cls, v):
        if len(v) < 1:
            raise ValueError('Embedding must not be empty')
        return v


class SearchResult(BaseModel):
    """Individual search result."""
    id: str = Field(description="Point ID")
    score: float = Field(description="Similarity score")
    user_id: str = Field(description="User identifier")
    metadata: Dict[str, Any] = Field(description="Associated metadata")
    timestamp: Optional[float] = Field(default=None, description="Vector creation timestamp")


class SearchResponse(BaseResponse):
    """Response model for vector search."""
    results: List[SearchResult] = Field(description="Search results")
    query_time_ms: float = Field(description="Query execution time in milliseconds")
    total_results: int = Field(description="Total number of results returned")


class DeleteVectorRequest(BaseModel):
    """Request model for deleting a vector."""
    point_id: str = Field(description="Point ID to delete")


class DeleteVectorResponse(BaseResponse):
    """Response model for vector deletion."""
    deleted: bool = Field(description="Whether the vector was deleted")


class DeleteUserVectorsRequest(BaseModel):
    """Request model for deleting all vectors for a user."""
    user_id: str = Field(description="User ID to delete vectors for")


class DeleteUserVectorsResponse(BaseResponse):
    """Response model for user vector deletion."""
    deleted_count: int = Field(description="Number of vectors deleted")


class StatsResponse(BaseResponse):
    """Response model for database statistics."""
    collection_info: Dict[str, Any] = Field(description="Collection information")
    performance_stats: Dict[str, Any] = Field(description="Performance statistics")
    gpu_info: Dict[str, Any] = Field(description="GPU information")


class HealthCheckResponse(BaseResponse):
    """Response model for health check."""
    status: str = Field(description="Service health status")
    qdrant_connection: bool = Field(description="Qdrant connection status")
    collection_exists: bool = Field(description="Collection existence status")
    collection_name: str = Field(description="Collection name")
    gpu_available: bool = Field(description="GPU availability status")




class BenchmarkRequest(BaseModel):
    """Benchmark request for performance testing."""
    num_vectors: int = Field(default=1000, ge=1, le=100000, description="Number of vectors to test")
    vector_dimension: int = Field(default=512, description="Vector dimension")
    search_queries: int = Field(default=100, ge=1, le=10000, description="Number of search queries")
    batch_size: int = Field(default=32, ge=1, le=1000, description="Batch size for operations")


class BenchmarkResponse(BaseResponse):
    """Benchmark response with performance metrics."""
    test_config: Dict[str, Any] = Field(description="Test configuration")
    insertion_metrics: Dict[str, float] = Field(description="Vector insertion metrics")
    search_metrics: Dict[str, float] = Field(description="Search performance metrics")
    gpu_metrics: Dict[str, Any] = Field(description="GPU utilization metrics")
    comparison_with_target: Dict[str, Any] = Field(description="Comparison with target performance")


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: float = Field(description="Error timestamp")
