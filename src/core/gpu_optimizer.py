"""
GPU-accelerated vector operations for Qdrant service.
Provides CUDA-optimized vector processing for face embeddings.
"""
import time
from typing import List, Union, Optional, Tuple
import numpy as np
from loguru import logger

try:
    import cupy as cp
    import torch
    CUDA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CUDA libraries not available: {e}")
    CUDA_AVAILABLE = False


class GPUVectorOptimizer:
    """GPU-accelerated vector operations using CuPy and PyTorch."""
    
    def __init__(self, device_id: int = 0, memory_fraction: float = 0.8):
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = f"cuda:{device_id}"
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            self._initialize_gpu()
        else:
            logger.warning("GPU optimization disabled - CUDA not available")
    
    def _initialize_gpu(self):
        """Initialize GPU settings and memory management."""
        try:
            # Set PyTorch device
            torch.cuda.set_device(self.device_id)
            
            # Set CuPy device
            cp.cuda.Device(self.device_id).use()
            
            # Configure memory pool
            self._setup_memory_pool()
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(self.device_id)
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            logger.info(f"Initialized GPU {self.device_id}: {gpu_name}")
            logger.info(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU: {e}")
            self.cuda_available = False
    
    def _setup_memory_pool(self):
        """Configure CUDA memory pool for optimal performance."""
        if not self.cuda_available:
            return
            
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device_id)
            
            # Configure CuPy memory pool
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            
            # Set memory pool limits
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            pool_limit = int(total_memory * self.memory_fraction)
            mempool.set_limit(size=pool_limit)
            
            logger.info(f"GPU memory pool configured: {pool_limit / 1024**3:.2f} GB")
            
        except Exception as e:
            logger.error(f"Failed to setup memory pool: {e}")
    
    def normalize_embeddings_gpu(self, embeddings: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated L2 normalization of embeddings.
        
        Args:
            embeddings: Input embeddings array (N, D)
            
        Returns:
            Normalized embeddings array
        """
        if not self.cuda_available or embeddings.size == 0:
            return self._normalize_embeddings_cpu(embeddings)
        
        try:
            start_time = time.perf_counter()
            
            # Transfer to GPU
            gpu_embeddings = cp.asarray(embeddings, dtype=cp.float32)
            
            # L2 normalization
            norms = cp.linalg.norm(gpu_embeddings, axis=1, keepdims=True)
            norms = cp.maximum(norms, 1e-12)  # Avoid division by zero
            normalized = gpu_embeddings / norms
            
            # Transfer back to CPU
            result = cp.asnumpy(normalized)
            
            duration = time.perf_counter() - start_time
            logger.debug(f"GPU normalization completed in {duration*1000:.2f}ms for {embeddings.shape[0]} vectors")
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU normalization failed, falling back to CPU: {e}")
            return self._normalize_embeddings_cpu(embeddings)
    
    def _normalize_embeddings_cpu(self, embeddings: np.ndarray) -> np.ndarray:
        """CPU fallback for embedding normalization."""
        if embeddings.size == 0:
            return embeddings
            
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return embeddings / norms
    
    def batch_cosine_similarity_gpu(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated batch cosine similarity computation.
        
        Args:
            query: Query embedding (D,)
            candidates: Candidate embeddings (N, D)
            
        Returns:
            Similarity scores array (N,)
        """
        if not self.cuda_available or candidates.size == 0:
            return self._batch_cosine_similarity_cpu(query, candidates)
        
        try:
            start_time = time.perf_counter()
            
            # Transfer to GPU
            query_gpu = cp.asarray(query, dtype=cp.float32)
            candidates_gpu = cp.asarray(candidates, dtype=cp.float32)
            
            # Ensure query is 1D
            if query_gpu.ndim > 1:
                query_gpu = query_gpu.flatten()
            
            # Compute cosine similarity
            similarities = cp.dot(candidates_gpu, query_gpu)
            
            # Transfer back to CPU
            result = cp.asnumpy(similarities)
            
            duration = time.perf_counter() - start_time
            logger.debug(f"GPU similarity computation completed in {duration*1000:.2f}ms for {candidates.shape[0]} vectors")
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU similarity computation failed, falling back to CPU: {e}")
            return self._batch_cosine_similarity_cpu(query, candidates)
    
    def _batch_cosine_similarity_cpu(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """CPU fallback for cosine similarity computation."""
        if candidates.size == 0:
            return np.array([])
            
        if query.ndim > 1:
            query = query.flatten()
            
        return np.dot(candidates, query)
    
    def optimize_batch_processing(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """
        Optimize batch processing of embeddings on GPU.
        
        Args:
            embeddings: List of embedding arrays
            
        Returns:
            List of normalized embeddings
        """
        if not embeddings:
            return []
        
        if not self.cuda_available:
            return [self._normalize_embeddings_cpu(emb) for emb in embeddings]
        
        try:
            start_time = time.perf_counter()
            
            # Stack embeddings into batch tensor
            batch_array = np.stack(embeddings)
            
            # Process on GPU
            normalized_batch = self.normalize_embeddings_gpu(batch_array)
            
            # Split back into list
            result = [normalized_batch[i] for i in range(len(embeddings))]
            
            duration = time.perf_counter() - start_time
            logger.debug(f"GPU batch processing completed in {duration*1000:.2f}ms for {len(embeddings)} vectors")
            
            return result
            
        except Exception as e:
            logger.warning(f"GPU batch processing failed, falling back to CPU: {e}")
            return [self._normalize_embeddings_cpu(emb) for emb in embeddings]
    
    def prefetch_to_gpu(self, data: np.ndarray) -> Optional[cp.ndarray]:
        """
        Prefetch data to GPU memory for faster processing.
        
        Args:
            data: Input data array
            
        Returns:
            GPU array or None if GPU not available
        """
        if not self.cuda_available:
            return None
        
        try:
            return cp.asarray(data, dtype=cp.float32)
        except Exception as e:
            logger.warning(f"Failed to prefetch data to GPU: {e}")
            return None
    
    def get_memory_info(self) -> dict:
        """Get GPU memory usage information."""
        if not self.cuda_available:
            return {"gpu_available": False}
        
        try:
            # PyTorch memory info
            allocated = torch.cuda.memory_allocated(self.device_id)
            reserved = torch.cuda.memory_reserved(self.device_id)
            total = torch.cuda.get_device_properties(self.device_id).total_memory
            
            # CuPy memory pool info
            mempool = cp.get_default_memory_pool()
            pool_used = mempool.used_bytes()
            pool_total = mempool.total_bytes()
            
            return {
                "gpu_available": True,
                "device_id": self.device_id,
                "device_name": torch.cuda.get_device_name(self.device_id),
                "total_memory_gb": total / 1024**3,
                "allocated_memory_gb": allocated / 1024**3,
                "reserved_memory_gb": reserved / 1024**3,
                "memory_utilization": allocated / total,
                "pool_used_gb": pool_used / 1024**3,
                "pool_total_gb": pool_total / 1024**3
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def cleanup(self):
        """Cleanup GPU resources."""
        if not self.cuda_available:
            return
        
        try:
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Clear CuPy memory pools
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
            logger.info("GPU resources cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup GPU resources: {e}")


class CUDAMemoryManager:
    """Advanced CUDA memory management for optimal performance."""
    
    def __init__(self, memory_fraction: float = 0.8):
        self.memory_fraction = memory_fraction
        self.cuda_available = CUDA_AVAILABLE
        
        if self.cuda_available:
            self._setup_memory_management()
    
    def _setup_memory_management(self):
        """Configure advanced CUDA memory management."""
        try:
            # Configure PyTorch memory allocator
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory pool
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            
            logger.info("Advanced CUDA memory management configured")
            
        except Exception as e:
            logger.error(f"Failed to setup advanced memory management: {e}")
    
    def get_memory_stats(self) -> dict:
        """Get detailed memory statistics."""
        if not self.cuda_available:
            return {"cuda_available": False}
        
        try:
            stats = torch.cuda.memory_stats()
            return {
                "cuda_available": True,
                "allocated_current": stats.get("allocated_bytes.all.current", 0),
                "allocated_peak": stats.get("allocated_bytes.all.peak", 0),
                "reserved_current": stats.get("reserved_bytes.all.current", 0),
                "reserved_peak": stats.get("reserved_bytes.all.peak", 0),
                "num_alloc_retries": stats.get("num_alloc_retries", 0),
                "num_ooms": stats.get("num_ooms", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"cuda_available": False, "error": str(e)}
