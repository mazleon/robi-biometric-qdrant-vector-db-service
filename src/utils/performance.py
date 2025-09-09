"""
Performance monitoring and benchmarking utilities for Qdrant service.
Provides comprehensive metrics collection and analysis for GPU-accelerated operations.
"""
import time
import asyncio
import statistics
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not available - metrics disabled")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_used_mb: float
    gpu_utilization: float
    batch_size: int
    vector_count: int
    timestamp: float


class MetricsCollector:
    """Prometheus metrics collector for Qdrant service."""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enabled = enable_prometheus and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            self._setup_metrics()
        
        self.performance_history: List[PerformanceMetrics] = []
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        # Counters
        self.search_requests = Counter(
            'qdrant_search_requests_total', 
            'Total number of search requests'
        )
        self.add_requests = Counter(
            'qdrant_add_requests_total', 
            'Total number of add requests'
        )
        self.errors_total = Counter(
            'qdrant_errors_total', 
            'Total number of errors',
            ['error_type']
        )
        
        # Histograms
        self.search_duration = Histogram(
            'qdrant_search_duration_seconds',
            'Search request duration',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        self.add_duration = Histogram(
            'qdrant_add_duration_seconds',
            'Add request duration',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Gauges
        self.gpu_memory_usage = Gauge(
            'qdrant_gpu_memory_usage_bytes',
            'GPU memory usage in bytes'
        )
        self.gpu_utilization = Gauge(
            'qdrant_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        self.vector_count = Gauge(
            'qdrant_vector_count_total',
            'Total number of vectors in database'
        )
        self.active_connections = Gauge(
            'qdrant_active_connections',
            'Number of active connections'
        )
        
        # Summary
        self.throughput = Summary(
            'qdrant_throughput_ops_per_second',
            'Operations throughput per second'
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_search(self, duration: float, vector_count: int = 1):
        """Record search operation metrics."""
        if self.enabled:
            self.search_requests.inc()
            self.search_duration.observe(duration)
            self.throughput.observe(vector_count / duration if duration > 0 else 0)
    
    def record_add(self, duration: float, vector_count: int = 1):
        """Record add operation metrics."""
        if self.enabled:
            self.add_requests.inc()
            self.add_duration.observe(duration)
            self.throughput.observe(vector_count / duration if duration > 0 else 0)
    
    def record_error(self, error_type: str):
        """Record error occurrence."""
        if self.enabled:
            self.errors_total.labels(error_type=error_type).inc()
    
    def update_gpu_metrics(self, memory_bytes: int, utilization_percent: float):
        """Update GPU metrics."""
        if self.enabled:
            self.gpu_memory_usage.set(memory_bytes)
            self.gpu_utilization.set(utilization_percent)
    
    def update_vector_count(self, count: int):
        """Update total vector count."""
        if self.enabled:
            self.vector_count.set(count)
    
    def add_performance_record(self, metrics: PerformanceMetrics):
        """Add performance record to history."""
        self.performance_history.append(metrics)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        # Filter by operation if specified
        records = self.performance_history
        if operation:
            records = [r for r in records if r.operation == operation]
        
        if not records:
            return {"error": f"No data for operation: {operation}"}
        
        durations = [r.duration_ms for r in records]
        throughputs = [r.throughput_ops_per_sec for r in records]
        gpu_utils = [r.gpu_utilization for r in records if r.gpu_utilization > 0]
        
        return {
            "operation": operation or "all",
            "total_operations": len(records),
            "duration_stats": {
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "std_dev_ms": statistics.stdev(durations) if len(durations) > 1 else 0
            },
            "throughput_stats": {
                "mean_ops_per_sec": statistics.mean(throughputs),
                "median_ops_per_sec": statistics.median(throughputs),
                "max_ops_per_sec": max(throughputs)
            },
            "gpu_stats": {
                "mean_utilization": statistics.mean(gpu_utils) if gpu_utils else 0,
                "max_utilization": max(gpu_utils) if gpu_utils else 0
            } if gpu_utils else {"gpu_data_available": False}
        }


# Global metrics collector
metrics_collector = MetricsCollector()


@contextmanager
def measure_performance(operation_name: str, batch_size: int = 1, 
                       vector_count: int = 1, gpu_optimizer=None):
    """
    Context manager for measuring operation performance.
    
    Args:
        operation_name: Name of the operation being measured
        batch_size: Size of the batch being processed
        vector_count: Number of vectors involved
        gpu_optimizer: GPU optimizer instance for memory info
    """
    start_time = time.perf_counter()
    start_memory = 0
    gpu_util_start = 0
    
    # Get initial GPU metrics if available
    if gpu_optimizer:
        try:
            gpu_info = gpu_optimizer.get_memory_info()
            start_memory = gpu_info.get("allocated_memory_gb", 0) * 1024  # Convert to MB
            gpu_util_start = gpu_info.get("memory_utilization", 0) * 100
        except:
            pass
    
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000
        
        # Calculate throughput
        throughput = vector_count / duration if duration > 0 else 0
        
        # Get final GPU metrics
        end_memory = start_memory
        gpu_util_end = gpu_util_start
        if gpu_optimizer:
            try:
                gpu_info = gpu_optimizer.get_memory_info()
                end_memory = gpu_info.get("allocated_memory_gb", 0) * 1024
                gpu_util_end = gpu_info.get("memory_utilization", 0) * 100
            except:
                pass
        
        # Create performance metrics
        perf_metrics = PerformanceMetrics(
            operation=operation_name,
            duration_ms=duration_ms,
            throughput_ops_per_sec=throughput,
            memory_used_mb=max(end_memory - start_memory, 0),
            gpu_utilization=max(gpu_util_end, gpu_util_start),
            batch_size=batch_size,
            vector_count=vector_count,
            timestamp=time.time()
        )
        
        # Record metrics
        if operation_name.startswith("search"):
            metrics_collector.record_search(duration, vector_count)
        elif operation_name.startswith("add"):
            metrics_collector.record_add(duration, vector_count)
        
        metrics_collector.add_performance_record(perf_metrics)
        
        # Update GPU metrics
        if gpu_util_end > 0:
            metrics_collector.update_gpu_metrics(
                int(end_memory * 1024 * 1024),  # Convert to bytes
                gpu_util_end
            )
        
        logger.debug(f"{operation_name} completed in {duration_ms:.2f}ms "
                    f"(throughput: {throughput:.1f} ops/sec)")


class BenchmarkSuite:
    """Comprehensive benchmarking suite for Qdrant service."""
    
    def __init__(self, vector_store, gpu_optimizer=None):
        self.vector_store = vector_store
        self.gpu_optimizer = gpu_optimizer
        self.results = {}
    
    async def run_insertion_benchmark(self, num_vectors: int = 1000, 
                                    vector_dim: int = 512, 
                                    batch_size: int = 32) -> Dict[str, Any]:
        """Benchmark vector insertion performance."""
        logger.info(f"Running insertion benchmark: {num_vectors} vectors, batch_size={batch_size}")
        
        # Generate random vectors
        vectors = [
            np.random.randn(vector_dim).astype(np.float32) 
            for _ in range(num_vectors)
        ]
        user_ids = [f"bench_user_{i}" for i in range(num_vectors)]
        
        # Single insertion benchmark
        single_times = []
        for i in range(min(100, num_vectors)):  # Test first 100 for single insertion
            with measure_performance("add_single", 1, 1, self.gpu_optimizer):
                await self.vector_store.add_vector(
                    vector=vectors[i],
                    user_id=f"single_{user_ids[i]}"
                )
                single_times.append(time.perf_counter())
        
        # Batch insertion benchmark
        batch_times = []
        batch_vectors = []
        batch_user_ids = []
        
        start_idx = 100  # Start after single insertion test vectors
        for i in range(start_idx, num_vectors, batch_size):
            end_idx = min(i + batch_size, num_vectors)
            batch_vectors = vectors[i:end_idx]
            batch_user_ids = user_ids[i:end_idx]
            
            with measure_performance("add_batch", len(batch_vectors), len(batch_vectors), self.gpu_optimizer):
                await self.vector_store.add_vectors_batch(
                    vectors=batch_vectors,
                    user_ids=batch_user_ids
                )
                batch_times.append(time.perf_counter())
        
        return {
            "total_vectors": num_vectors,
            "vector_dimension": vector_dim,
            "batch_size": batch_size,
            "single_insertion": {
                "count": len(single_times),
                "avg_time_ms": statistics.mean(single_times) * 1000 if single_times else 0
            },
            "batch_insertion": {
                "batches": len(batch_times),
                "avg_batch_time_ms": statistics.mean(batch_times) * 1000 if batch_times else 0,
                "avg_vectors_per_sec": batch_size / (statistics.mean(batch_times) if batch_times else 1)
            }
        }
    
    async def run_search_benchmark(self, num_queries: int = 100, 
                                 k: int = 10, vector_dim: int = 512) -> Dict[str, Any]:
        """Benchmark search performance."""
        logger.info(f"Running search benchmark: {num_queries} queries, k={k}")
        
        # Generate random query vectors
        query_vectors = [
            np.random.randn(vector_dim).astype(np.float32) 
            for _ in range(num_queries)
        ]
        
        search_times = []
        result_counts = []
        
        for i, query_vector in enumerate(query_vectors):
            with measure_performance("search", 1, 1, self.gpu_optimizer):
                result = await self.vector_store.search(
                    query_vector=query_vector,
                    k=k
                )
                search_times.append(result["query_time_ms"])
                result_counts.append(len(result["results"]))
        
        return {
            "total_queries": num_queries,
            "k": k,
            "vector_dimension": vector_dim,
            "search_performance": {
                "avg_time_ms": statistics.mean(search_times),
                "median_time_ms": statistics.median(search_times),
                "min_time_ms": min(search_times),
                "max_time_ms": max(search_times),
                "std_dev_ms": statistics.stdev(search_times) if len(search_times) > 1 else 0,
                "queries_per_sec": 1000 / statistics.mean(search_times) if search_times else 0
            },
            "result_stats": {
                "avg_results": statistics.mean(result_counts),
                "total_results": sum(result_counts)
            }
        }
    
    async def run_concurrent_benchmark(self, concurrent_requests: int = 10,
                                     operations_per_request: int = 10) -> Dict[str, Any]:
        """Benchmark concurrent operation performance."""
        logger.info(f"Running concurrent benchmark: {concurrent_requests} concurrent, "
                   f"{operations_per_request} ops each")
        
        async def worker(worker_id: int):
            """Worker function for concurrent operations."""
            times = []
            for i in range(operations_per_request):
                query_vector = np.random.randn(512).astype(np.float32)
                
                start_time = time.perf_counter()
                await self.vector_store.search(query_vector=query_vector, k=5)
                duration = time.perf_counter() - start_time
                times.append(duration * 1000)
            
            return {
                "worker_id": worker_id,
                "avg_time_ms": statistics.mean(times),
                "total_operations": len(times)
            }
        
        # Run concurrent workers
        start_time = time.perf_counter()
        tasks = [worker(i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        
        # Aggregate results
        all_times = []
        total_ops = 0
        for result in results:
            all_times.append(result["avg_time_ms"])
            total_ops += result["total_operations"]
        
        return {
            "concurrent_requests": concurrent_requests,
            "operations_per_request": operations_per_request,
            "total_operations": total_ops,
            "total_time_sec": total_time,
            "overall_throughput_ops_per_sec": total_ops / total_time,
            "avg_response_time_ms": statistics.mean(all_times),
            "worker_results": results
        }
    
    async def run_full_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("Starting full benchmark suite")
        
        benchmark_results = {
            "config": config,
            "timestamp": time.time(),
            "gpu_info": self.gpu_optimizer.get_memory_info() if self.gpu_optimizer else {}
        }
        
        try:
            # Insertion benchmark
            benchmark_results["insertion"] = await self.run_insertion_benchmark(
                num_vectors=config.get("num_vectors", 1000),
                batch_size=config.get("batch_size", 32)
            )
            
            # Search benchmark
            benchmark_results["search"] = await self.run_search_benchmark(
                num_queries=config.get("search_queries", 100),
                k=config.get("k", 10)
            )
            
            # Concurrent benchmark
            benchmark_results["concurrent"] = await self.run_concurrent_benchmark(
                concurrent_requests=config.get("concurrent_requests", 10),
                operations_per_request=config.get("operations_per_request", 10)
            )
            
            # Performance comparison with targets
            benchmark_results["performance_analysis"] = self._analyze_performance(benchmark_results)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            benchmark_results["error"] = str(e)
        
        logger.info("Benchmark suite completed")
        return benchmark_results
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance against targets."""
        target_search_time_ms = 20  # Target from memory
        target_throughput_ops_per_sec = 1000
        
        analysis = {
            "targets": {
                "search_time_ms": target_search_time_ms,
                "throughput_ops_per_sec": target_throughput_ops_per_sec
            }
        }
        
        # Search performance analysis
        if "search" in results:
            actual_search_time = results["search"]["search_performance"]["avg_time_ms"]
            search_improvement = ((target_search_time_ms - actual_search_time) / target_search_time_ms) * 100
            
            analysis["search_analysis"] = {
                "actual_avg_time_ms": actual_search_time,
                "target_time_ms": target_search_time_ms,
                "improvement_percent": search_improvement,
                "meets_target": actual_search_time <= target_search_time_ms
            }
        
        # Throughput analysis
        if "concurrent" in results:
            actual_throughput = results["concurrent"]["overall_throughput_ops_per_sec"]
            throughput_improvement = ((actual_throughput - target_throughput_ops_per_sec) / target_throughput_ops_per_sec) * 100
            
            analysis["throughput_analysis"] = {
                "actual_ops_per_sec": actual_throughput,
                "target_ops_per_sec": target_throughput_ops_per_sec,
                "improvement_percent": throughput_improvement,
                "meets_target": actual_throughput >= target_throughput_ops_per_sec
            }
        
        return analysis


def start_metrics_server(port: int = 8002):
    """Start Prometheus metrics server."""
    if PROMETHEUS_AVAILABLE:
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    else:
        logger.warning("Prometheus not available - metrics server not started")


async def cleanup_old_metrics():
    """Cleanup old performance metrics to prevent memory leaks."""
    global metrics_collector
    
    # Keep only last 24 hours of data
    cutoff_time = time.time() - (24 * 60 * 60)
    
    metrics_collector.performance_history = [
        record for record in metrics_collector.performance_history
        if record.timestamp > cutoff_time
    ]
    
    logger.debug(f"Cleaned up old metrics, kept {len(metrics_collector.performance_history)} records")
