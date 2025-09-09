"""
Configuration settings for Qdrant GPU-accelerated vector database service.
"""
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class QdrantSettings(BaseSettings):
    """Qdrant service configuration with GPU optimization settings."""
    
    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant HTTP port")
    qdrant_grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    collection_name: str = Field(default="face_embeddings", description="Collection name")
    
    # Vector Configuration
    vector_size: int = Field(default=512, description="Embedding vector dimension")
    distance_metric: str = Field(default="Cosine", description="Distance metric for similarity")
    similarity_threshold: float = Field(default=0.65, description="Face verification threshold - optimized for accuracy")
    
    # GPU Configuration
    use_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    gpu_device_id: int = Field(default=0, description="CUDA device ID")
    gpu_memory_fraction: float = Field(default=0.8, description="GPU memory allocation fraction")
    enable_cuda_graphs: bool = Field(default=True, description="Enable CUDA graphs optimization")
    
    # Performance Settings (Optimized for high throughput)
    batch_size: int = Field(default=128, description="Batch size for operations - larger for GPU efficiency")
    search_timeout: float = Field(default=0.1, description="Search timeout in seconds - microsecond target")
    max_concurrent_requests: int = Field(default=1000, description="Max concurrent requests")
    
    # Memory Management
    segment_size_mb: int = Field(default=512, description="Segment size in MB")
    memmap_threshold_mb: int = Field(default=1024, description="Memory mapping threshold")
    vacuum_min_vector_number: int = Field(default=10000, description="Min vectors before vacuum")
    
    # HNSW Optimization Settings (Optimized for 10M+ vectors)
    hnsw_ef_construct: int = Field(default=512, description="HNSW ef_construct parameter - higher for better recall")
    hnsw_m: int = Field(default=64, description="HNSW M parameter - higher for 10M+ vectors")
    hnsw_ef: int = Field(default=256, description="HNSW ef parameter for search")
    hnsw_max_indexing_threads: int = Field(default=0, description="Max indexing threads (0 = auto)")
    hnsw_on_disk: bool = Field(default=False, description="Store HNSW index on disk for large datasets")
    
    # Advanced Quantization Settings
    quantization_enabled: bool = Field(default=True, description="Enable scalar quantization")
    quantization_type: str = Field(default="int8", description="Quantization type: int8, binary")
    quantization_always_ram: bool = Field(default=True, description="Keep quantized vectors in RAM")
    quantization_quantile: float = Field(default=0.99, description="Quantization quantile")
    
    # Indexing Optimization
    indexing_threshold: int = Field(default=50000, description="Vectors before HNSW indexing starts")
    full_scan_threshold: int = Field(default=20000, description="Threshold for full scan vs HNSW")
    
    replication_factor: int = Field(default=1, description="Collection replication factor")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8001, description="API server port")
    api_title: str = Field(default="Qdrant Vector Database Service", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8002, description="Metrics server port")
    
    class Config:
        env_file = ".env"
        env_prefix = "QDRANT_"
        case_sensitive = False


# Global settings instance
settings = QdrantSettings()
