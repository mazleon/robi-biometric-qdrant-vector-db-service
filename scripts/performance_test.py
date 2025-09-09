#!/usr/bin/env python3
"""
Performance testing script for Qdrant Vector Database Service.

This script generates realistic 512-dimensional embeddings and performs comprehensive
performance testing including add, search, and delete operations with detailed logging.
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from loguru import logger
import aiohttp
import argparse
from datetime import datetime
import uuid

# Configure logger
logger.remove()
logger.add(
    "logs/performance_test_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level="INFO",
    rotation="10 MB"
)
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)


class VectorGenerator:
    """Generate realistic 512-dimensional embeddings for testing."""
    
    @staticmethod
    def generate_normalized_embedding(seed: int = None) -> List[float]:
        """Generate a normalized 512-dimensional embedding vector."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random vector with normal distribution
        vector = np.random.normal(0, 1, 512).astype(np.float32)
        
        # Normalize to unit length (common for embeddings)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    @staticmethod
    def generate_similar_embedding(base_embedding: List[float], similarity: float = 0.8) -> List[float]:
        """Generate an embedding similar to the base embedding."""
        base_vector = np.array(base_embedding, dtype=np.float32)
        
        # Generate noise vector
        noise = np.random.normal(0, 1, 512).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        
        # Mix base vector with noise
        mixed_vector = similarity * base_vector + (1 - similarity) * noise
        
        # Normalize
        mixed_vector = mixed_vector / np.linalg.norm(mixed_vector)
        
        return mixed_vector.tolist()


class PerformanceTester:
    """Comprehensive performance testing for vector database operations."""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session = None
        self.test_results = {
            "test_config": {},
            "add_operations": [],
            "search_operations": [],
            "delete_operations": [],
            "batch_operations": [],
            "summary": {}
        }
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> bool:
        """Check if the service is healthy."""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Service health: {data.get('status', 'unknown')}")
                    return data.get("success", False)
                else:
                    logger.error(f"Health check failed with status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def add_single_vector(self, embedding: List[float], user_id: str, 
                               metadata: Dict[str, Any] = None) -> Tuple[bool, float, str]:
        """Add a single vector and measure performance."""
        start_time = time.perf_counter()
        
        payload = {
            "embedding": embedding,
            "user_id": user_id,
            "metadata": metadata or {}
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/vectors/add",
                json=payload
            ) as response:
                duration = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    point_id = data.get("point_id")
                    logger.debug(f"Added vector for user {user_id} in {duration*1000:.2f}ms, point_id: {point_id}")
                    return True, duration, point_id
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to add vector: {response.status} - {error_text}")
                    return False, duration, None
                    
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Exception during vector addition: {e}")
            return False, duration, None
    
    async def add_batch_vectors(self, embeddings: List[List[float]], user_ids: List[str],
                               metadata_list: List[Dict[str, Any]] = None) -> Tuple[bool, float, List[str]]:
        """Add multiple vectors in batch and measure performance."""
        start_time = time.perf_counter()
        
        payload = {
            "embeddings": embeddings,
            "user_ids": user_ids,
            "metadata_list": metadata_list or [{}] * len(embeddings)
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/vectors/add_batch",
                json=payload
            ) as response:
                duration = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    point_ids = data.get("point_ids", [])
                    logger.info(f"Added {len(embeddings)} vectors in batch in {duration*1000:.2f}ms")
                    return True, duration, point_ids
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to add batch vectors: {response.status} - {error_text}")
                    return False, duration, []
                    
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Exception during batch vector addition: {e}")
            return False, duration, []
    
    async def search_vectors(self, query_embedding: List[float], k: int = 10,
                            threshold: float = None, user_filter: str = None) -> Tuple[bool, float, List[Dict]]:
        """Search for similar vectors and measure performance."""
        start_time = time.perf_counter()
        
        payload = {
            "embedding": query_embedding,
            "k": k,
            "threshold": threshold,
            "user_filter": user_filter
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/vectors/search",
                json=payload
            ) as response:
                duration = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    logger.debug(f"Search completed in {duration*1000:.2f}ms, found {len(results)} results")
                    return True, duration, results
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to search vectors: {response.status} - {error_text}")
                    return False, duration, []
                    
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Exception during vector search: {e}")
            return False, duration, []
    
    async def delete_vector(self, point_id: str) -> Tuple[bool, float]:
        """Delete a vector by point ID and measure performance."""
        start_time = time.perf_counter()
        
        try:
            async with self.session.delete(
                f"{self.base_url}/api/v1/vectors/{point_id}"
            ) as response:
                duration = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    deleted = data.get("deleted", False)
                    logger.debug(f"Deleted vector {point_id} in {duration*1000:.2f}ms")
                    return deleted, duration
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to delete vector: {response.status} - {error_text}")
                    return False, duration
                    
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Exception during vector deletion: {e}")
            return False, duration
    
    async def delete_user_vectors(self, user_id: str) -> Tuple[bool, float, int]:
        """Delete all vectors for a user and measure performance."""
        start_time = time.perf_counter()
        
        try:
            async with self.session.delete(
                f"{self.base_url}/api/v1/vectors/user/{user_id}"
            ) as response:
                duration = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    deleted_count = data.get("deleted_count", 0)
                    logger.info(f"Deleted {deleted_count} vectors for user {user_id} in {duration*1000:.2f}ms")
                    return True, duration, deleted_count
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to delete user vectors: {response.status} - {error_text}")
                    return False, duration, 0
                    
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(f"Exception during user vector deletion: {e}")
            return False, duration, 0
    
    async def run_comprehensive_test(self, num_vectors: int = 1000, num_users: int = 100,
                                   batch_size: int = 32, search_queries: int = 100) -> Dict[str, Any]:
        """Run comprehensive performance test."""
        logger.info(f"Starting comprehensive performance test")
        logger.info(f"Config: {num_vectors} vectors, {num_users} users, batch_size={batch_size}, {search_queries} searches")
        
        # Store test configuration
        self.test_results["test_config"] = {
            "num_vectors": num_vectors,
            "num_users": num_users,
            "batch_size": batch_size,
            "search_queries": search_queries,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check service health
        if not await self.health_check():
            logger.error("Service health check failed. Aborting test.")
            return self.test_results
        
        # Generate test data
        logger.info("Generating test embeddings...")
        embeddings = []
        user_ids = []
        metadata_list = []
        
        for i in range(num_vectors):
            user_id = f"test_user_{i % num_users}"
            embedding = VectorGenerator.generate_normalized_embedding(seed=i)
            metadata = {
                "test_id": i,
                "category": f"category_{i % 10}",
                "timestamp": time.time()
            }
            
            embeddings.append(embedding)
            user_ids.append(user_id)
            metadata_list.append(metadata)
        
        # Test 1: Single vector additions
        logger.info("Testing single vector additions...")
        single_add_times = []
        added_point_ids = []
        
        for i in range(min(100, num_vectors)):  # Test first 100 for single adds
            success, duration, point_id = await self.add_single_vector(
                embeddings[i], user_ids[i], metadata_list[i]
            )
            if success:
                single_add_times.append(duration)
                added_point_ids.append(point_id)
                self.test_results["add_operations"].append({
                    "operation": "single_add",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "point_id": point_id
                })
        
        # Test 2: Batch vector additions
        logger.info("Testing batch vector additions...")
        batch_add_times = []
        batch_point_ids = []
        
        remaining_embeddings = embeddings[100:]  # Use remaining embeddings for batch
        remaining_user_ids = user_ids[100:]
        remaining_metadata = metadata_list[100:]
        
        for i in range(0, len(remaining_embeddings), batch_size):
            batch_emb = remaining_embeddings[i:i+batch_size]
            batch_users = remaining_user_ids[i:i+batch_size]
            batch_meta = remaining_metadata[i:i+batch_size]
            
            success, duration, point_ids = await self.add_batch_vectors(
                batch_emb, batch_users, batch_meta
            )
            if success:
                batch_add_times.append(duration)
                batch_point_ids.extend(point_ids)
                self.test_results["batch_operations"].append({
                    "operation": "batch_add",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "batch_size": len(batch_emb),
                    "vectors_per_second": len(batch_emb) / duration
                })
        
        # Wait for indexing to complete
        logger.info("Waiting for indexing to complete...")
        await asyncio.sleep(2)
        
        # Test 3: Vector searches
        logger.info("Testing vector searches...")
        search_times = []
        
        for i in range(search_queries):
            # Use random embeddings for search queries
            query_embedding = VectorGenerator.generate_normalized_embedding(seed=i+10000)
            
            success, duration, results = await self.search_vectors(
                query_embedding, k=10, threshold=0.1
            )
            if success:
                search_times.append(duration)
                self.test_results["search_operations"].append({
                    "operation": "search",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "results_count": len(results),
                    "query_id": i
                })
        
        # Test 4: Similar vector searches (should find matches)
        logger.info("Testing similar vector searches...")
        similar_search_times = []
        
        for i in range(min(50, len(embeddings))):
            # Generate similar embedding
            similar_embedding = VectorGenerator.generate_similar_embedding(embeddings[i], similarity=0.9)
            
            success, duration, results = await self.search_vectors(
                similar_embedding, k=5, threshold=0.5
            )
            if success:
                similar_search_times.append(duration)
                self.test_results["search_operations"].append({
                    "operation": "similar_search",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "results_count": len(results),
                    "base_vector_index": i
                })
        
        # Test 5: Vector deletions
        logger.info("Testing vector deletions...")
        delete_times = []
        
        # Delete some individual vectors
        for point_id in added_point_ids[:50]:  # Delete first 50 single-added vectors
            success, duration = await self.delete_vector(point_id)
            if success:
                delete_times.append(duration)
                self.test_results["delete_operations"].append({
                    "operation": "single_delete",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "point_id": point_id
                })
        
        # Test 6: User vector deletions
        logger.info("Testing user vector deletions...")
        user_delete_times = []
        
        # Delete vectors for some users
        test_users = [f"test_user_{i}" for i in range(0, min(10, num_users))]
        for user_id in test_users:
            success, duration, deleted_count = await self.delete_user_vectors(user_id)
            if success:
                user_delete_times.append(duration)
                self.test_results["delete_operations"].append({
                    "operation": "user_delete",
                    "success": True,
                    "duration_ms": duration * 1000,
                    "user_id": user_id,
                    "deleted_count": deleted_count
                })
        
        # Calculate summary statistics
        self.test_results["summary"] = {
            "single_add_stats": self._calculate_stats(single_add_times, "Single Add"),
            "batch_add_stats": self._calculate_stats(batch_add_times, "Batch Add"),
            "search_stats": self._calculate_stats(search_times, "Search"),
            "similar_search_stats": self._calculate_stats(similar_search_times, "Similar Search"),
            "delete_stats": self._calculate_stats(delete_times, "Delete"),
            "user_delete_stats": self._calculate_stats(user_delete_times, "User Delete"),
            "total_operations": {
                "single_adds": len(single_add_times),
                "batch_adds": len(batch_add_times),
                "searches": len(search_times),
                "similar_searches": len(similar_search_times),
                "deletes": len(delete_times),
                "user_deletes": len(user_delete_times)
            }
        }
        
        logger.info("Performance test completed successfully!")
        return self.test_results
    
    def _calculate_stats(self, times: List[float], operation_name: str) -> Dict[str, float]:
        """Calculate statistics for operation times."""
        if not times:
            return {"count": 0}
        
        times_ms = [t * 1000 for t in times]  # Convert to milliseconds
        
        stats = {
            "count": len(times),
            "min_ms": min(times_ms),
            "max_ms": max(times_ms),
            "mean_ms": statistics.mean(times_ms),
            "median_ms": statistics.median(times_ms),
            "std_dev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            "p95_ms": np.percentile(times_ms, 95),
            "p99_ms": np.percentile(times_ms, 99)
        }
        
        logger.info(f"{operation_name} Performance:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Mean: {stats['mean_ms']:.2f}ms")
        logger.info(f"  Median: {stats['median_ms']:.2f}ms")
        logger.info(f"  P95: {stats['p95_ms']:.2f}ms")
        logger.info(f"  P99: {stats['p99_ms']:.2f}ms")
        
        return stats
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_results_{timestamp}.json"
        
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to: {filepath}")
        return filepath


async def main():
    """Main function to run performance tests."""
    parser = argparse.ArgumentParser(description="Qdrant Vector Database Performance Test")
    parser.add_argument("--vectors", type=int, default=1000, help="Number of vectors to test")
    parser.add_argument("--users", type=int, default=100, help="Number of unique users")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for operations")
    parser.add_argument("--searches", type=int, default=100, help="Number of search queries")
    parser.add_argument("--base-url", default="http://localhost:8001", help="Base URL of the service")
    parser.add_argument("--output", help="Output filename for results")
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("=== Qdrant Vector Database Performance Test ===")
    logger.info(f"Service URL: {args.base_url}")
    logger.info(f"Test Configuration:")
    logger.info(f"  Vectors: {args.vectors}")
    logger.info(f"  Users: {args.users}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Search Queries: {args.searches}")
    
    async with PerformanceTester(args.base_url) as tester:
        try:
            results = await tester.run_comprehensive_test(
                num_vectors=args.vectors,
                num_users=args.users,
                batch_size=args.batch_size,
                search_queries=args.searches
            )
            
            # Save results
            output_file = tester.save_results(args.output)
            
            # Print summary
            logger.info("\n=== PERFORMANCE TEST SUMMARY ===")
            summary = results.get("summary", {})
            
            for operation, stats in summary.items():
                if isinstance(stats, dict) and "mean_ms" in stats:
                    logger.info(f"{operation.replace('_', ' ').title()}:")
                    logger.info(f"  Operations: {stats['count']}")
                    logger.info(f"  Average: {stats['mean_ms']:.2f}ms")
                    logger.info(f"  P95: {stats['p95_ms']:.2f}ms")
            
            logger.info(f"\nDetailed results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
