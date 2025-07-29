"""
Retry utilities for RIPPLe data access operations.

This module provides decorators and utilities for implementing robust retry logic
with exponential backoff, jitter, and circuit breaker patterns.
"""

import time
import random
import logging
from typing import Any, Callable, Dict, Optional, Type, Tuple, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum

# LSST imports
from lsst.daf.butler import DataIdValueError
try:
    from lsst.daf.butler import DatasetNotFoundError
except ImportError:
    DatasetNotFoundError = Exception

# RIPPLe imports
from .exceptions import DataAccessError, ButlerConnectionError

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies for different types of operations."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Exception types that should trigger retries
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        ButlerConnectionError,
        LookupError
    )
    
    # Exception types that should NOT trigger retries
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (
        DataIdValueError,
        DatasetNotFoundError,
        ValueError,
        TypeError
    )


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation for failure protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise DataAccessError("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker transitioning to CLOSED")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker transitioning to OPEN")


class RetryManager:
    """Manages retry operations with backoff and circuit breaker patterns."""
    
    def __init__(self, config: RetryConfig, circuit_breaker: Optional[CircuitBreaker] = None):
        self.config = config
        self.circuit_breaker = circuit_breaker
        self.operation_stats: Dict[str, Dict] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def retry_operation(self, func: Callable, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        # Initialize stats for this operation
        if operation_name not in self.operation_stats:
            self.operation_stats[operation_name] = {
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'total_delay': 0.0
            }
        
        stats = self.operation_stats[operation_name]
        
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            stats['attempts'] += 1
            
            try:
                # Use circuit breaker if available
                if self.circuit_breaker:
                    result = self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                stats['successes'] += 1
                
                if attempt > 0:
                    self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should trigger a retry
                if not self._should_retry(e):
                    self.logger.error(f"Non-retryable exception in {operation_name}: {e}")
                    stats['failures'] += 1
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                stats['total_delay'] += delay
                
                self.logger.warning(
                    f"Operation {operation_name} failed on attempt {attempt + 1}/{self.config.max_attempts}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                time.sleep(delay)
        
        # All attempts failed
        stats['failures'] += 1
        raise DataAccessError(
            f"Operation {operation_name} failed after {self.config.max_attempts} attempts: {last_exception}"
        )
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Check non-retryable exceptions first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        if isinstance(exception, self.config.retryable_exceptions):
            return True
        
        # Default to not retrying unknown exceptions
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * (attempt + 1)
        else:  # EXPONENTIAL_BACKOFF
            delay = self.config.initial_delay * (self.config.backoff_factor ** attempt)
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure positive delay
        
        return delay
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            'operations': dict(self.operation_stats),
            'circuit_breaker_state': self.circuit_breaker.state.value if self.circuit_breaker else None,
            'config': {
                'max_attempts': self.config.max_attempts,
                'initial_delay': self.config.initial_delay,
                'max_delay': self.config.max_delay,
                'backoff_factor': self.config.backoff_factor,
                'strategy': self.config.strategy.value
            }
        }
    
    def reset_statistics(self):
        """Reset retry statistics."""
        self.operation_stats.clear()


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = None,
    operation_name: Optional[str] = None
):
    """
    Decorator for adding retry logic with exponential backoff.
    
    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts
    initial_delay : float
        Initial delay between retries in seconds
    backoff_factor : float
        Multiplier for exponential backoff
    max_delay : float
        Maximum delay between retries
    jitter : bool
        Whether to add random jitter to delays
    retryable_exceptions : Tuple[Type[Exception], ...]
        Exception types that should trigger retries
    operation_name : str, optional
        Name for logging and statistics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set up configuration
            config = RetryConfig(
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                backoff_factor=backoff_factor,
                max_delay=max_delay,
                jitter=jitter,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            )
            
            if retryable_exceptions:
                config.retryable_exceptions = retryable_exceptions
            
            # Create retry manager
            retry_manager = RetryManager(config)
            
            # Execute with retry logic
            op_name = operation_name or func.__name__
            return retry_manager.retry_operation(func, op_name, *args, **kwargs)
        
        return wrapper
    return decorator


def butler_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
):
    """
    Decorator specifically for Butler operations with appropriate exception handling.
    """
    return retry_with_backoff(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            ButlerConnectionError,
            KeyError
        )
    )


def cutout_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 1.5
):
    """
    Decorator specifically for cutout operations with faster retry.
    """
    return retry_with_backoff(
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=10.0,  # Shorter max delay for cutouts
        retryable_exceptions=(
            ConnectionError,
            TimeoutError,
            ButlerConnectionError
        )
    )