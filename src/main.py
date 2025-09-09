"""
Main FastAPI application for Qdrant GPU-accelerated vector database service.
High-performance face recognition backend with GPU optimization.
"""
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

from .config.settings import settings
from .api.endpoints import router
from .api.schemas import ErrorResponse
from .utils.logging_config import configure_service_logging, RequestLogger
from .utils.performance import start_metrics_server, cleanup_old_metrics
from .core.qdrant_client import QdrantVectorStore
from .core.gpu_optimizer import GPUVectorOptimizer

from loguru import logger


# Global instances
vector_store: QdrantVectorStore = None
gpu_optimizer: GPUVectorOptimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown."""
    # Startup
    logger.info("Starting Qdrant Vector Database Service")
    
    try:
        # Initialize GPU optimizer
        global gpu_optimizer
        gpu_optimizer = GPUVectorOptimizer(
            device_id=settings.gpu_device_id,
            memory_fraction=settings.gpu_memory_fraction
        )
        
        # Initialize vector store
        global vector_store
        vector_store = QdrantVectorStore(settings)
        
        # Initialize collection
        collection_ready = await vector_store.initialize_collection()
        if not collection_ready:
            logger.error("Failed to initialize Qdrant collection")
            raise RuntimeError("Collection initialization failed")
        
        # Start metrics server if enabled
        if settings.enable_metrics:
            start_metrics_server(settings.metrics_port)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(periodic_cleanup())
        
        logger.info("Service startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Service startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Qdrant Vector Database Service")
    
    try:
        # Cleanup resources
        if vector_store:
            await vector_store.cleanup()
        
        if gpu_optimizer:
            gpu_optimizer.cleanup()
        
        # Cancel cleanup task
        if 'cleanup_task' in locals():
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


async def periodic_cleanup():
    """Periodic cleanup task for memory and metrics."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            await cleanup_old_metrics()
            
            if gpu_optimizer:
                # Periodic GPU memory cleanup
                gpu_optimizer.cleanup()
            
            logger.debug("Periodic cleanup completed")
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic cleanup error: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="High-performance GPU-accelerated vector database service for face recognition",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging and performance tracking middleware."""
    start_time = time.perf_counter()
    correlation_id = f"req_{int(time.time() * 1000)}_{hash(str(request.url)) % 10000}"
    
    # Create request logger
    request_logger = RequestLogger(correlation_id)
    
    # Log request
    client_ip = request.client.host if request.client else "unknown"
    request_logger.log_request(request.method, str(request.url.path), client_ip)
    
    # Add correlation ID to request state
    request.state.correlation_id = correlation_id
    request.state.request_logger = request_logger
    
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Log response
        request_logger.log_response(response.status_code, duration_ms)
        
        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response
        
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        request_logger.log_error(e, "request_processing")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "correlation_id": correlation_id,
                "timestamp": time.time()
            },
            headers={
                "X-Correlation-ID": correlation_id,
                "X-Response-Time": f"{duration_ms:.2f}ms"
            }
        )


# Include API routes
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Qdrant Vector Database Service",
        "version": settings.api_version,
        "status": "running",
        "description": "High-performance GPU-accelerated vector database for face recognition",
        "endpoints": {
            "health": "/api/v1/health",
            "docs": "/docs",
            "metrics": f"http://localhost:{settings.metrics_port}" if settings.enable_metrics else None
        },
        "timestamp": time.time()
    }


# Additional system endpoints
@app.get("/metrics", tags=["System"])
async def metrics_endpoint():
    """Prometheus metrics endpoint (if metrics server is not running separately)."""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        raise HTTPException(status_code=500, detail="Prometheus client not available")


@app.get("/benchmark", tags=["System"])
async def run_benchmark():
    """Run performance benchmark (development/testing only)."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        from .utils.performance import BenchmarkSuite
        
        benchmark_suite = BenchmarkSuite(vector_store, gpu_optimizer)
        
        # Run lightweight benchmark
        config = {
            "num_vectors": 100,
            "search_queries": 50,
            "batch_size": 16,
            "concurrent_requests": 5,
            "operations_per_request": 5
        }
        
        results = await benchmark_suite.run_full_benchmark(config)
        
        return {
            "success": True,
            "benchmark_results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with additional information."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.api_title,
        version=settings.api_version,
        description="""
        # Qdrant GPU-Accelerated Vector Database Service
        
        High-performance vector database service optimized for face recognition embeddings.
        
        ## Features
        - **GPU Acceleration**: CUDA-optimized vector operations using RTX 4090
        - **High Performance**: <20ms search latency for 10K vectors
        - **Scalable**: Supports batch operations and concurrent requests
        - **Compatible**: Drop-in replacement for FAISS-based systems
        
        ## Performance Targets
        - Search latency: <15ms (25% improvement over FAISS)
        - Throughput: >1000 searches/second
        - GPU utilization: >80% during peak loads
        - Memory efficiency: <8GB VRAM for 100K vectors
        
        ## API Compatibility
        Maintains full compatibility with existing face recognition APIs:
        - `/enroll` - Face enrollment
        - `/verify` - Face verification  
        - `/detect` - Face identification
        
        ## Vector Operations
        Advanced vector database operations:
        - `/vectors/add` - Add single vector
        - `/vectors/add_batch` - Batch vector addition
        - `/vectors/search` - Similarity search
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://qdrant.tech/images/logo.svg"
    }
    
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{settings.api_port}",
            "description": "Development server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "available_endpoints": [
                "/api/v1/health",
                "/api/v1/stats", 
                "/api/v1/vectors/search",
                "/api/v1/enroll",
                "/api/v1/verify",
                "/api/v1/detect"
            ],
            "timestamp": time.time()
        }
    )


# Additional exception handlers for API endpoints
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_code="VALIDATION_ERROR",
            timestamp=time.time()
        ).dict(),
        headers={"X-Correlation-ID": correlation_id}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle internal server errors."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"exception": str(exc)},
            timestamp=time.time()
        ).dict(),
        headers={"X-Correlation-ID": correlation_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    correlation_id = getattr(request.state, 'correlation_id', 'unknown')
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            details={"exception": str(exc)},
            timestamp=time.time()
        ).dict(),
        headers={"X-Correlation-ID": correlation_id}
    )


# Main entry point
if __name__ == "__main__":
    # Configure logging
    configure_service_logging(settings)
    
    # Run the application
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,  # Disable reload in production
        workers=1,     # Single worker for GPU memory management
        loop="asyncio",
        log_config=None,  # Use our custom logging
        access_log=False  # Disable default access log (we have custom middleware)
    )
