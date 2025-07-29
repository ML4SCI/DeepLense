# RIPPLe Data Access Layer Architecture

## Overview

The RIPPLe Data Access Layer provides a comprehensive interface for LSST data retrieval with optimized performance, robust error handling, and support for modern Butler Gen3 architecture patterns.

## Component Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LsstDataFetcher                                   │
│                      (Main Public Interface)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │ButlerClient │    │CoordinateConverter│  │CacheManager │    │PerformanceMonitor│ │
│  │             │    │              │    │             │    │             │ │
│  │- Connection │    │- RA/Dec to   │    │- LRU Cache  │    │- Metrics    │ │
│  │- Queries    │    │  tract/patch │    │- Statistics │    │- Timing     │ │
│  │- Retries    │    │- Spatial ops │    │- Cleanup    │    │- Logging    │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Exception Handling                                  │
│  DataAccessError → ButlerConnectionError, DataIdValidationError,            │
│                    CutoutExtractionError, CoordinateConversionError         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Input Request → Validation → Coordinate Conversion → Cache Check → Butler Query → 
   ↓                                                                              
Processing → Quality Check → Caching → Performance Logging → Result Return
```

## Component Specifications

### 1. LsstDataFetcher (Main Interface)

**Purpose**: Primary public interface for all data access operations

**Key Methods**:
- `fetch_cutout()`: Single/multi-band cutout retrieval
- `fetch_batch_cutouts()`: Batch processing with parallel execution
- `get_available_data()`: Query data availability
- `fetch_catalog()`: Catalog retrieval with spatial filtering

**Configuration**:
- `ButlerConfig`: Centralized configuration management
- `CutoutRequest`: Structured request parameters
- `PerformanceMetrics`: Performance monitoring data

**Features**:
- Context manager support (`with` statement)
- Comprehensive error handling
- Progress callbacks for batch operations
- Automatic resource cleanup

### 2. ButlerClient (Butler Wrapper)

**Purpose**: Abstraction layer over LSST Butler with enhanced functionality

**Key Responsibilities**:
- Butler initialization (direct/client-server)
- Connection management and health checking
- Registry query optimization
- Retry logic implementation
- Collection management

**Interface Design**:
```python
class ButlerClient:
    def __init__(self, config: ButlerConfig)
    def test_connection(self) -> bool
    def get_skymap(self) -> BaseSkyMap
    def query_available_datasets(self, tract: int, patch: str) -> Dict[str, Any]
    def get_exposure(self, data_id: Dict, bbox: Optional[Box2I] = None) -> Exposure
    def get_catalog(self, tract: int, patch: str, catalog_type: str) -> SourceCatalog
    def cleanup(self) -> None
```

**2025 Architecture Support**:
- Automatic detection of client/server vs direct Butler
- Authentication handling for remote Butler
- Connection pooling for multiple simultaneous requests
- Fallback mechanisms for server failures

### 3. CoordinateConverter (Spatial Operations)

**Purpose**: Convert between coordinate systems and perform spatial operations

**Key Methods**:
- `radec_to_tract_patch()`: Convert sky coordinates to tract/patch
- `radec_to_tract_patch_radius()`: Find all tract/patch in radius
- `tract_patch_to_radec()`: Convert tract/patch to sky coordinates
- `pixel_to_sky()`: Pixel coordinate conversion
- `sky_to_pixel()`: Sky coordinate conversion

**Spatial Indexing**:
- Efficient tract/patch lookup
- Boundary checking and validation
- Multi-point query optimization
- Geometric operations for regions

### 4. CacheManager (Performance Optimization)

**Purpose**: LRU cache for frequently accessed data

**Cached Data Types**:
- Small cutouts (< 128x128 pixels)
- Metadata (WCS, PSF models)
- Coordinate conversions
- Dataset availability queries

**Features**:
- Configurable cache size
- Automatic eviction policy
- Cache statistics and monitoring
- Memory usage tracking

### 5. Exception Hierarchy

**Base Exception**: `RippleError`
- Includes error context and details dictionary
- Supports error chaining and recovery information

**Specific Exceptions**:
- `ButlerConnectionError`: Connection/authentication failures
- `DataIdValidationError`: Invalid parameters or coordinates
- `CutoutExtractionError`: Cutout processing failures
- `CoordinateConversionError`: Coordinate transformation failures
- `CollectionError`: Collection access/permission issues
- `PerformanceError`: Performance threshold violations

## Configuration System

### ButlerConfig Structure

```python
@dataclass
class ButlerConfig:
    # Connection settings
    repo_path: Optional[str] = None
    server_url: Optional[str] = None
    collections: List[str] = field(default_factory=list)
    instrument: str = "LSSTCam-imSim"
    
    # Performance settings
    max_connections: int = 10
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Optimization settings
    cache_size: int = 1000
    enable_performance_monitoring: bool = True
    batch_size: int = 32
    max_workers: int = 4
```

### Usage Patterns

```python
# Basic usage
config = ButlerConfig(
    repo_path="/path/to/repo",
    collections=["2.2i/runs/DP0.2"],
    cache_size=500
)

with LsstDataFetcher(config) as fetcher:
    cutout = fetcher.fetch_cutout(ra=150.0, dec=2.5, size=64)
```

## Performance Optimization Strategy

### 1. Memory Management
- Streaming processing for large datasets
- Automatic garbage collection triggers
- Memory usage monitoring and alerts
- Efficient data structures for metadata

### 2. Parallel Processing
- ThreadPoolExecutor for I/O-bound operations
- Batch processing with optimal chunk sizes
- Asynchronous operations where applicable
- Resource pooling for Butler connections

### 3. Caching Strategy
- Multi-level caching (memory, disk)
- Intelligent cache warming
- Cache invalidation policies
- Statistics-driven optimization

### 4. Butler Optimization
- Registry query result caching
- Deferred loading for large datasets
- Batch dataset reference retrieval
- Connection reuse and pooling

## Error Handling Strategy

### 1. Retry Logic
- Exponential backoff with jitter
- Operation-specific retry policies
- Circuit breaker pattern for persistent failures
- Partial failure handling in batch operations

### 2. Graceful Degradation
- Fallback to alternative data sources
- Reduced functionality mode
- Quality threshold adaptation
- User notification of limitations

### 3. Error Context
- Detailed error messages with context
- Error aggregation for batch operations
- Recovery suggestions and alternatives
- Comprehensive logging for debugging

## Integration Points

### 1. Phase 0 Integration
- Builds on established LSST environment
- Uses documented Butler best practices
- Integrates with existing MockButler testing
- Leverages Phase 0 error handling patterns

### 2. Phase 2 Preparation
- Clean interface for preprocessing integration
- Standardized data formats
- Performance metrics for optimization
- Configuration hooks for preprocessing parameters

### 3. DeepLense Compatibility
- Tensor-ready output formats
- Batch processing alignment
- Memory management coordination
- Quality assurance integration

## Testing Strategy

### 1. Unit Testing
- Individual component testing
- Mock Butler integration
- Error scenario validation
- Performance benchmark testing

### 2. Integration Testing
- End-to-end workflow validation
- Real Butler testing (when available)
- Multi-component interaction testing
- Configuration validation

### 3. Performance Testing
- Throughput benchmarking
- Memory usage profiling
- Concurrent access testing
- Stress testing with large datasets

## Monitoring and Observability

### 1. Performance Metrics
- Operation timing and throughput
- Memory usage and allocation
- Cache hit rates and efficiency
- Error rates and recovery times

### 2. Logging Strategy
- Structured logging with context
- Configurable log levels
- Performance logging
- Error tracking and aggregation

### 3. Health Checks
- Butler connection monitoring
- Cache health assessment
- Performance threshold monitoring
- Resource utilization tracking

This architecture provides a solid foundation for the LSST Data Access Layer, ensuring scalability, reliability, and performance for the RIPPLe pipeline.