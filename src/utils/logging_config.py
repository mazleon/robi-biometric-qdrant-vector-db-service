"""
Logging configuration for Qdrant vector database service.
Provides structured logging with JSON format and performance tracking.
"""
import sys
import json
from typing import Dict, Any
from loguru import logger
from pythonjsonlogger import jsonlogger


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record: Dict[str, Any], record, message_dict: Dict[str, Any]):
        super(StructuredFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['service'] = 'qdrant-vector-service'
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_record['correlation_id'] = record.correlation_id
        
        # Add performance metrics if available
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        if hasattr(record, 'operation'):
            log_record['operation'] = record.operation


def setup_logging(log_level: str = "INFO", log_format: str = "json", 
                 enable_file_logging: bool = True):
    """
    Setup logging configuration for the service.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Log format (json, text)
        enable_file_logging: Enable file logging
    """
    # Remove default logger
    logger.remove()
    
    if log_format.lower() == "json":
        # JSON format for production
        log_format_str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}"
        
        # Console handler with JSON format
        logger.add(
            sys.stdout,
            format=log_format_str,
            level=log_level,
            serialize=True,  # Enable JSON serialization
            backtrace=True,
            diagnose=True
        )
        
        if enable_file_logging:
            # File handler with JSON format
            logger.add(
                "logs/qdrant_service.log",
                format=log_format_str,
                level=log_level,
                serialize=True,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                backtrace=True,
                diagnose=True
            )
            
            # Error file handler
            logger.add(
                "logs/qdrant_service_errors.log",
                format=log_format_str,
                level="ERROR",
                serialize=True,
                rotation="50 MB",
                retention="90 days",
                compression="gz",
                backtrace=True,
                diagnose=True
            )
    else:
        # Human-readable format for development
        log_format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Console handler with colored format
        logger.add(
            sys.stdout,
            format=log_format_str,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        if enable_file_logging:
            # File handler without colors
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
            
            logger.add(
                "logs/qdrant_service.log",
                format=file_format,
                level=log_level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                backtrace=True,
                diagnose=True
            )
    
    logger.info(f"Logging configured: level={log_level}, format={log_format}")


def get_correlation_logger(correlation_id: str):
    """
    Get a logger with correlation ID for request tracking.
    
    Args:
        correlation_id: Unique identifier for request correlation
        
    Returns:
        Logger instance with correlation ID
    """
    return logger.bind(correlation_id=correlation_id)


def log_performance(operation: str, duration_ms: float, **kwargs):
    """
    Log performance metrics for operations.
    
    Args:
        operation: Operation name
        duration_ms: Duration in milliseconds
        **kwargs: Additional metrics
    """
    logger.bind(
        operation=operation,
        duration_ms=duration_ms,
        **kwargs
    ).info(f"Performance: {operation} completed in {duration_ms:.2f}ms")


def log_gpu_metrics(gpu_info: Dict[str, Any]):
    """
    Log GPU utilization metrics.
    
    Args:
        gpu_info: GPU information dictionary
    """
    if gpu_info.get("gpu_available"):
        logger.bind(
            gpu_device_id=gpu_info.get("device_id"),
            gpu_memory_used_gb=gpu_info.get("allocated_memory_gb"),
            gpu_memory_total_gb=gpu_info.get("total_memory_gb"),
            gpu_utilization=gpu_info.get("memory_utilization")
        ).debug("GPU metrics updated")


def log_vector_operation(operation: str, vector_count: int, user_id: str = None, 
                        duration_ms: float = None, **kwargs):
    """
    Log vector database operations with structured data.
    
    Args:
        operation: Operation type (add, search, delete)
        vector_count: Number of vectors involved
        user_id: User ID if applicable
        duration_ms: Operation duration
        **kwargs: Additional operation data
    """
    log_data = {
        "operation": operation,
        "vector_count": vector_count,
        **kwargs
    }
    
    if user_id:
        log_data["user_id"] = user_id
    if duration_ms:
        log_data["duration_ms"] = duration_ms
    
    logger.bind(**log_data).info(f"Vector operation: {operation}")


class RequestLogger:
    """Request logging middleware for FastAPI."""
    
    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.logger = get_correlation_logger(correlation_id)
    
    def log_request(self, method: str, path: str, client_ip: str):
        """Log incoming request."""
        self.logger.info(f"Request: {method} {path} from {client_ip}")
    
    def log_response(self, status_code: int, duration_ms: float):
        """Log response with timing."""
        self.logger.bind(
            status_code=status_code,
            duration_ms=duration_ms
        ).info(f"Response: {status_code} in {duration_ms:.2f}ms")
    
    def log_error(self, error: Exception, context: str = None):
        """Log error with context."""
        self.logger.bind(
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        ).error(f"Error in {context or 'request'}: {error}")


# Global logger configuration
def configure_service_logging(settings):
    """Configure logging based on service settings."""
    setup_logging(
        log_level=settings.log_level,
        log_format=settings.log_format,
        enable_file_logging=True
    )
    
    # Log service startup
    logger.info("Qdrant Vector Database Service starting up")
    logger.info(f"Configuration: GPU={settings.use_gpu}, Collection={settings.collection_name}")
    logger.info(f"Vector dimension: {settings.vector_size}, Threshold: {settings.similarity_threshold}")
