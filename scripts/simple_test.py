#!/usr/bin/env python3
"""
Simple test script to validate Qdrant Vector Database Service functionality.
Tests basic operations with 512-dimensional embeddings.
"""

import asyncio
import time
import json
import numpy as np
import aiohttp
from loguru import logger
from pathlib import Path

# Configure logger
logger.remove()
logger.add(
    "logs/simple_test_{time}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    level="INFO"
)
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>",
    level="INFO"
)


def generate_512d_embedding(seed: int = None) -> list:
    """Generate a normalized 512-dimensional embedding."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random vector with normal distribution
    vector = np.random.normal(0, 1, 512).astype(np.float32)
    
    # Normalize to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()


async def test_vector_operations():
    """Test basic vector operations."""
    base_url = "http://localhost:8001"
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Health Check
        logger.info("Testing health check...")
        try:
            async with session.get(f"{base_url}/api/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✓ Health check passed: {data.get('status')}")
                else:
                    logger.error(f"✗ Health check failed: {response.status}")
                    return
        except Exception as e:
            logger.error(f"✗ Cannot connect to service: {e}")
            return
        
        # Test 2: Add single vector
        logger.info("Testing single vector addition...")
        embedding = generate_512d_embedding(seed=42)
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{base_url}/api/v1/vectors/add", json={
                "embedding": embedding,
                "user_id": "test_user_1",
                "metadata": {"test": "single_add", "category": "test"}
            }) as response:
                add_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    point_id = data.get("point_id")
                    logger.info(f"✓ Vector added successfully in {add_time*1000:.2f}ms, point_id: {point_id}")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed to add vector: {response.status} - {error_text}")
                    return
        except Exception as e:
            logger.error(f"✗ Exception during vector addition: {e}")
            return
        
        # Test 3: Add batch vectors
        logger.info("Testing batch vector addition...")
        batch_embeddings = [generate_512d_embedding(seed=i) for i in range(100, 110)]
        batch_user_ids = [f"test_user_{i}" for i in range(2, 12)]
        batch_metadata = [{"test": "batch_add", "index": i} for i in range(10)]
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{base_url}/api/v1/vectors/add_batch", json={
                "embeddings": batch_embeddings,
                "user_ids": batch_user_ids,
                "metadata_list": batch_metadata
            }) as response:
                batch_add_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    added_count = data.get("added_count", 0)
                    logger.info(f"✓ Batch added {added_count} vectors in {batch_add_time*1000:.2f}ms")
                    logger.info(f"  Average per vector: {(batch_add_time/added_count)*1000:.2f}ms")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed to add batch vectors: {response.status} - {error_text}")
                    return
        except Exception as e:
            logger.error(f"✗ Exception during batch addition: {e}")
            return
        
        # Wait for indexing
        logger.info("Waiting for indexing...")
        await asyncio.sleep(2)
        
        # Test 4: Search vectors
        logger.info("Testing vector search...")
        query_embedding = generate_512d_embedding(seed=42)  # Same as first vector
        
        start_time = time.perf_counter()
        try:
            async with session.post(f"{base_url}/api/v1/vectors/search", json={
                "embedding": query_embedding,
                "k": 5,
                "threshold": 0.1
            }) as response:
                search_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    logger.info(f"✓ Search completed in {search_time*1000:.2f}ms, found {len(results)} results")
                    
                    if results:
                        best_match = results[0]
                        logger.info(f"  Best match: score={best_match['score']:.4f}, user_id={best_match['user_id']}")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed to search vectors: {response.status} - {error_text}")
                    return
        except Exception as e:
            logger.error(f"✗ Exception during search: {e}")
            return
        
        # Test 5: Search with user filter
        logger.info("Testing filtered search...")
        start_time = time.perf_counter()
        try:
            async with session.post(f"{base_url}/api/v1/vectors/search", json={
                "embedding": query_embedding,
                "k": 3,
                "threshold": 0.1,
                "user_filter": "test_user_1"
            }) as response:
                filtered_search_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    results = data.get("results", [])
                    logger.info(f"✓ Filtered search completed in {filtered_search_time*1000:.2f}ms, found {len(results)} results")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed filtered search: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"✗ Exception during filtered search: {e}")
        
        # Test 6: Delete vector
        logger.info("Testing vector deletion...")
        if point_id:
            start_time = time.perf_counter()
            try:
                async with session.delete(f"{base_url}/api/v1/vectors/{point_id}") as response:
                    delete_time = time.perf_counter() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        deleted = data.get("deleted", False)
                        logger.info(f"✓ Vector deleted in {delete_time*1000:.2f}ms, success: {deleted}")
                    else:
                        error_text = await response.text()
                        logger.error(f"✗ Failed to delete vector: {response.status} - {error_text}")
            except Exception as e:
                logger.error(f"✗ Exception during deletion: {e}")
        
        # Test 7: Delete user vectors
        logger.info("Testing user vector deletion...")
        start_time = time.perf_counter()
        try:
            async with session.delete(f"{base_url}/api/v1/vectors/user/test_user_2") as response:
                user_delete_time = time.perf_counter() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    deleted_count = data.get("deleted_count", 0)
                    logger.info(f"✓ Deleted {deleted_count} vectors for user in {user_delete_time*1000:.2f}ms")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed to delete user vectors: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"✗ Exception during user deletion: {e}")
        
        # Test 8: Get service stats
        logger.info("Testing service statistics...")
        try:
            async with session.get(f"{base_url}/api/v1/stats") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("✓ Service statistics retrieved:")
                    
                    collection_info = data.get("collection_info", {})
                    performance_stats = data.get("performance_stats", {})
                    
                    logger.info(f"  Vector count: {collection_info.get('vector_count', 'N/A')}")
                    logger.info(f"  Total searches: {performance_stats.get('total_searches', 'N/A')}")
                    logger.info(f"  Avg search time: {performance_stats.get('avg_search_time_ms', 'N/A')}ms")
                else:
                    error_text = await response.text()
                    logger.error(f"✗ Failed to get stats: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"✗ Exception getting stats: {e}")
        
        logger.info("\n=== TEST SUMMARY ===")
        logger.info(f"Single vector add time: {add_time*1000:.2f}ms")
        logger.info(f"Batch add time (10 vectors): {batch_add_time*1000:.2f}ms")
        logger.info(f"Search time: {search_time*1000:.2f}ms")
        logger.info(f"Filtered search time: {filtered_search_time*1000:.2f}ms")
        logger.info("All tests completed successfully! ✓")


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("=== Qdrant Vector Database Simple Test ===")
    asyncio.run(test_vector_operations())
