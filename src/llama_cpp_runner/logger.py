"""
Enhanced logging system for llama-cpp-runner with emoji support

This module provides a standardized logging system with pretty emoji formatting,
log rotation, and flexible output options.
"""

import os
import sys
import logging
import time
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

# Define emoji prefixes for log levels
LEVEL_EMOJIS = {
    'DEBUG': '🔍',
    'INFO': '📝',
    'WARNING': '⚠️',
    'ERROR': '❌',
    'CRITICAL': '🚨'
}

# Configure default settings
DEFAULT_LOG_FORMAT = '%(asctime)s | %(levelprefix)s | %(name)s | %(message)s'
DEFAULT_LOG_DIR = os.path.expanduser('~/.llama_cpp_runner/logs')
DEFAULT_LOG_LEVEL = logging.INFO

# Registry of configured loggers
_LOGGERS = {}

class EmojiFormatter(logging.Formatter):
    """Formatter that adds emoji prefixes to log levels"""
    
    def format(self, record):
        """Format the log record with emoji prefix"""
        # Add emoji prefix to level name
        levelname = record.levelname
        emoji = LEVEL_EMOJIS.get(levelname, '')
        record.levelprefix = f"{emoji} {levelname}"
        
        # Call the parent formatter
        return super().format(record)

def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Configure and get a logger with emoji support.
    
    Args:
        name: Logger name
        level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        console: Whether to log to console
        file: Whether to log to file
        format_str: Custom log format string
        
    Returns:
        Configured logger
    """
    # Return existing logger if already configured
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    # Set up default values
    level = getattr(logging, (level or 'INFO').upper())
    log_dir = log_dir or DEFAULT_LOG_DIR
    format_str = format_str or DEFAULT_LOG_FORMAT
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter with emoji support
    formatter = EmojiFormatter(format_str)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file:
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store logger in registry
    _LOGGERS[name] = logger
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    
    return setup_logger(name)

def log_method_call(logger):
    """
    Decorator to log method entry and exit with timing.
    
    Args:
        logger: Logger instance
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get method name and class name
            name = func.__name__
            class_name = args[0].__class__.__name__ if args else ""
            qualified_name = f"{class_name}.{name}" if class_name else name
            
            # Log method entry
            logger.debug(f"▶️ Starting {qualified_name}")
            start_time = time.time()
            
            try:
                # Call the method
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"✅ Completed {qualified_name} in {elapsed:.3f}s")
                return result
            except Exception as e:
                # Log exception
                elapsed = time.time() - start_time
                logger.error(f"❌ Failed {qualified_name} after {elapsed:.3f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator

# Configure root logger with reasonable defaults
root_logger = setup_logger(
    "llama_cpp_runner",
    level="INFO",
    console=True,
    file=True
)