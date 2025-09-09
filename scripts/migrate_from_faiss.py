"""
Migration script to transfer data from FAISS to Qdrant vector database.
Maintains data integrity while upgrading to GPU-accelerated performance.
"""
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

# Import Qdrant components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.qdrant_client import QdrantVectorStore
from src.config.settings import QdrantSettings


class FAISSToQdrantMigrator:
    """Migrates data from FAISS-based system to Qdrant."""
    
    def __init__(self, faiss_data_path: str, metadata_path: str, 
                 qdrant_settings: QdrantSettings):
        self.faiss_data_path = Path(faiss_data_path)
        self.metadata_path = Path(metadata_path)
        self.qdrant_settings = qdrant_settings
        self.qdrant_store = None
        
        self.migration_stats = {
            "total_vectors": 0,
            "migrated_vectors": 0,
            "failed_vectors": 0,
            "start_time": None,
            "end_time": None,
            "batch_size": 32
        }
    
    async def initialize_qdrant(self) -> bool:
        """Initialize Qdrant vector store."""
        try:
            self.qdrant_store = QdrantVectorStore(self.qdrant_settings)
            success = await self.qdrant_store.initialize_collection()
            
            if success:
                logger.info("Qdrant vector store initialized successfully")
                return True
            else:
                logger.error("Failed to initialize Qdrant vector store")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            return False
    
    def load_faiss_data(self) -> tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
        """
        Load data from FAISS files.
        
        Returns:
            Tuple of (embeddings, user_ids, metadata_list)
        """
        try:
            # Load embeddings (assuming .npy format)
            if self.faiss_data_path.suffix == '.npy':
                embeddings = np.load(self.faiss_data_path)
            else:
                # Try to load FAISS index and extract vectors
                import faiss
                index = faiss.read_index(str(self.faiss_data_path))
                embeddings = index.reconstruct_n(0, index.ntotal)
            
            logger.info(f"Loaded {embeddings.shape[0]} embeddings from FAISS")
            
            # Load metadata
            user_ids = []
            metadata_list = []
            
            if self.metadata_path.suffix == '.json':
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if isinstance(metadata, list):
                    # List format
                    for item in metadata:
                        user_ids.append(item.get('user_id', f'user_{len(user_ids)}'))
                        metadata_list.append({
                            k: v for k, v in item.items() 
                            if k not in ['user_id', 'embedding']
                        })
                elif isinstance(metadata, dict):
                    # Dict format with user_id as keys
                    for user_id, meta in metadata.items():
                        user_ids.append(user_id)
                        metadata_list.append(meta)
            else:
                # Generate default metadata
                for i in range(len(embeddings)):
                    user_ids.append(f'migrated_user_{i}')
                    metadata_list.append({'migrated': True, 'original_index': i})
            
            logger.info(f"Loaded metadata for {len(user_ids)} users")
            
            # Ensure consistent lengths
            min_length = min(len(embeddings), len(user_ids))
            embeddings = embeddings[:min_length]
            user_ids = user_ids[:min_length]
            metadata_list = metadata_list[:min_length] if metadata_list else [{}] * min_length
            
            return embeddings, user_ids, metadata_list
            
        except Exception as e:
            logger.error(f"Error loading FAISS data: {e}")
            raise
    
    async def migrate_batch(self, embeddings: np.ndarray, user_ids: List[str], 
                           metadata_list: List[Dict[str, Any]]) -> int:
        """
        Migrate a batch of vectors to Qdrant.
        
        Returns:
            Number of successfully migrated vectors
        """
        try:
            # Convert embeddings to list of arrays
            vectors = [emb.astype(np.float32) for emb in embeddings]
            
            # Add migration timestamp to metadata
            enhanced_metadata = []
            for meta in metadata_list:
                enhanced_meta = meta.copy()
                enhanced_meta.update({
                    'migrated_from_faiss': True,
                    'migration_timestamp': time.time(),
                    'migration_batch': True
                })
                enhanced_metadata.append(enhanced_meta)
            
            # Perform batch insertion
            point_ids = await self.qdrant_store.add_vectors_batch(
                vectors=vectors,
                user_ids=user_ids,
                metadata_list=enhanced_metadata
            )
            
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Error migrating batch: {e}")
            return 0
    
    async def run_migration(self, batch_size: int = 32, 
                           verify_migration: bool = True) -> Dict[str, Any]:
        """
        Run the complete migration process.
        
        Args:
            batch_size: Number of vectors to migrate per batch
            verify_migration: Whether to verify migrated data
            
        Returns:
            Migration results and statistics
        """
        self.migration_stats["start_time"] = time.time()
        self.migration_stats["batch_size"] = batch_size
        
        try:
            logger.info("Starting FAISS to Qdrant migration")
            
            # Initialize Qdrant
            if not await self.initialize_qdrant():
                raise RuntimeError("Failed to initialize Qdrant")
            
            # Load FAISS data
            logger.info("Loading FAISS data...")
            embeddings, user_ids, metadata_list = self.load_faiss_data()
            
            self.migration_stats["total_vectors"] = len(embeddings)
            logger.info(f"Total vectors to migrate: {self.migration_stats['total_vectors']}")
            
            # Migrate in batches
            migrated_count = 0
            failed_count = 0
            
            for i in range(0, len(embeddings), batch_size):
                end_idx = min(i + batch_size, len(embeddings))
                
                batch_embeddings = embeddings[i:end_idx]
                batch_user_ids = user_ids[i:end_idx]
                batch_metadata = metadata_list[i:end_idx]
                
                logger.info(f"Migrating batch {i//batch_size + 1}: "
                           f"vectors {i}-{end_idx-1}")
                
                try:
                    batch_migrated = await self.migrate_batch(
                        batch_embeddings, batch_user_ids, batch_metadata
                    )
                    migrated_count += batch_migrated
                    
                    if batch_migrated < len(batch_embeddings):
                        failed_count += len(batch_embeddings) - batch_migrated
                    
                    # Progress update
                    progress = (migrated_count / self.migration_stats["total_vectors"]) * 100
                    logger.info(f"Migration progress: {progress:.1f}% "
                               f"({migrated_count}/{self.migration_stats['total_vectors']})")
                    
                except Exception as e:
                    logger.error(f"Batch migration failed: {e}")
                    failed_count += len(batch_embeddings)
                
                # Small delay between batches to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            self.migration_stats["migrated_vectors"] = migrated_count
            self.migration_stats["failed_vectors"] = failed_count
            self.migration_stats["end_time"] = time.time()
            
            # Verification
            verification_results = {}
            if verify_migration and migrated_count > 0:
                logger.info("Verifying migration...")
                verification_results = await self.verify_migration(embeddings[:10])
            
            # Final statistics
            duration = self.migration_stats["end_time"] - self.migration_stats["start_time"]
            
            results = {
                "migration_successful": migrated_count > 0,
                "statistics": {
                    **self.migration_stats,
                    "duration_seconds": duration,
                    "migration_rate_vectors_per_sec": migrated_count / duration if duration > 0 else 0,
                    "success_rate": (migrated_count / self.migration_stats["total_vectors"]) * 100
                },
                "verification": verification_results
            }
            
            logger.info(f"Migration completed: {migrated_count}/{self.migration_stats['total_vectors']} vectors migrated")
            logger.info(f"Migration duration: {duration:.2f} seconds")
            logger.info(f"Migration rate: {migrated_count/duration:.1f} vectors/second")
            
            return results
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.migration_stats["end_time"] = time.time()
            
            return {
                "migration_successful": False,
                "error": str(e),
                "statistics": self.migration_stats
            }
    
    async def verify_migration(self, sample_embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Verify migration by testing search functionality.
        
        Args:
            sample_embeddings: Sample embeddings to test search
            
        Returns:
            Verification results
        """
        try:
            verification_results = {
                "search_tests": 0,
                "successful_searches": 0,
                "average_search_time_ms": 0,
                "sample_results": []
            }
            
            search_times = []
            
            for i, embedding in enumerate(sample_embeddings):
                try:
                    start_time = time.perf_counter()
                    
                    result = await self.qdrant_store.search(
                        query_vector=embedding,
                        k=5,
                        score_threshold=0.1
                    )
                    
                    search_time = (time.perf_counter() - start_time) * 1000
                    search_times.append(search_time)
                    
                    verification_results["search_tests"] += 1
                    
                    if len(result["results"]) > 0:
                        verification_results["successful_searches"] += 1
                        
                        # Store sample result
                        if len(verification_results["sample_results"]) < 3:
                            verification_results["sample_results"].append({
                                "query_index": i,
                                "results_found": len(result["results"]),
                                "top_score": result["results"][0]["score"] if result["results"] else 0,
                                "search_time_ms": search_time
                            })
                    
                except Exception as e:
                    logger.warning(f"Search verification failed for sample {i}: {e}")
            
            if search_times:
                verification_results["average_search_time_ms"] = sum(search_times) / len(search_times)
            
            verification_results["verification_success_rate"] = (
                verification_results["successful_searches"] / 
                max(1, verification_results["search_tests"])
            ) * 100
            
            logger.info(f"Verification completed: {verification_results['successful_searches']}/{verification_results['search_tests']} searches successful")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {"verification_error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.qdrant_store:
            await self.qdrant_store.cleanup()


async def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate from FAISS to Qdrant")
    parser.add_argument("--faiss-data", required=True, help="Path to FAISS data file")
    parser.add_argument("--metadata", required=True, help="Path to metadata file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for migration")
    parser.add_argument("--verify", action="store_true", help="Verify migration")
    parser.add_argument("--output", help="Output file for migration results")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.add("migration.log", rotation="10 MB", retention="30 days")
    
    # Create Qdrant settings
    settings = QdrantSettings()
    
    # Run migration
    migrator = FAISSToQdrantMigrator(
        faiss_data_path=args.faiss_data,
        metadata_path=args.metadata,
        qdrant_settings=settings
    )
    
    try:
        results = await migrator.run_migration(
            batch_size=args.batch_size,
            verify_migration=args.verify
        )
        
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Migration results saved to {args.output}")
        
        # Print summary
        print("\n" + "="*50)
        print("MIGRATION SUMMARY")
        print("="*50)
        
        if results["migration_successful"]:
            stats = results["statistics"]
            print(f"✅ Migration completed successfully")
            print(f"📊 Vectors migrated: {stats['migrated_vectors']}/{stats['total_vectors']}")
            print(f"⏱️  Duration: {stats['duration_seconds']:.2f} seconds")
            print(f"🚀 Rate: {stats['migration_rate_vectors_per_sec']:.1f} vectors/second")
            print(f"✅ Success rate: {stats['success_rate']:.1f}%")
            
            if "verification" in results and results["verification"]:
                ver = results["verification"]
                print(f"🔍 Verification: {ver.get('successful_searches', 0)}/{ver.get('search_tests', 0)} searches successful")
                print(f"⚡ Avg search time: {ver.get('average_search_time_ms', 0):.2f}ms")
        else:
            print(f"❌ Migration failed: {results.get('error', 'Unknown error')}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        print(f"❌ Migration script failed: {e}")
    finally:
        await migrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
