"""
Centralized logging module for GraphRAG Engine
"""
import logging
import sys
from config import LOG_LEVEL, ENABLE_DEBUG_LOGGING

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        log_level = logging.DEBUG if ENABLE_DEBUG_LOGGING else getattr(logging, LOG_LEVEL, logging.INFO)
        logger.setLevel(log_level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with context."""
    error_msg = f"{context}: {str(error)}" if context else str(error)
    logger.error(error_msg, exc_info=ENABLE_DEBUG_LOGGING)
    return error_msg

def log_info(logger: logging.Logger, message: str):
    """Log info message."""
    logger.info(message)

def log_warning(logger: logging.Logger, message: str):
    """Log warning message."""
    logger.warning(message)

def log_debug(logger: logging.Logger, message: str):
    """Log debug message."""
    logger.debug(message)
