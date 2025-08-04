# Tutorial 08: Advanced Error Handling

## Learning Objectives
- Master Rust's sophisticated error handling patterns and best practices
- Understand when to use `Result`, `Option`, panic, and custom error types
- Learn advanced error propagation techniques with `?` operator and combinators
- Implement domain-specific error types with `thiserror` and `anyhow`
- Apply error handling strategies in concurrent and async contexts
- Create resilient space simulation systems with comprehensive error recovery
- Learn error handling in FFI and unsafe code contexts

## Lesson: Advanced Error Handling in Rust

### What Makes Error Handling "Advanced"?

Advanced error handling goes beyond basic `Result` and `Option` usage to encompass:
- **Error type design**: Creating meaningful, composable error hierarchies
- **Error propagation**: Elegant error bubbling across system boundaries
- **Error recovery**: Graceful degradation and fault tolerance
- **Error context**: Preserving diagnostic information across call stacks
- **Performance**: Zero-cost error handling where possible

### Rust's Error Handling Philosophy

#### No Exceptions, No Hidden Control Flow
Unlike languages with exceptions, Rust makes errors explicit:
- **Visible in function signatures**: `Result<T, E>` shows fallibility
- **Forced handling**: Compiler ensures errors are addressed
- **No unwinding**: Errors don't bypass intermediate code
- **Predictable performance**: No hidden exception handling costs

#### Zero-Cost Error Handling
Rust's error handling is designed for performance:
- **No allocation**: Errors are stack-allocated by default
- **Compile-time optimization**: `Result` branches optimize away
- **Inlining**: Error handling code gets inlined aggressively
- **LLVM optimization**: Advanced optimizations eliminate overhead

### Error Handling Patterns

#### 1. Early Return Pattern
```rust
fn validate_ship_status(ship: &Ship) -> Result<(), ShipError> {
    if ship.fuel_level < 0.0 {
        return Err(ShipError::InvalidFuelLevel);
    }
    if ship.cargo_weight > ship.max_cargo {
        return Err(ShipError::OverCapacity);
    }
    Ok(())
}
```

#### 2. Error Chaining with `?`
```rust
fn transfer_cargo(from: &mut Ship, to: &mut Ship, amount: f32) -> Result<(), TransferError> {
    validate_ship_status(from)?;
    validate_ship_status(to)?;
    from.remove_cargo(amount)?;
    to.add_cargo(amount)?;
    Ok(())
}
```

#### 3. Error Mapping and Context
```rust
fn load_ship_config(path: &str) -> Result<ShipConfig, ConfigError> {
    std::fs::read_to_string(path)
        .map_err(|e| ConfigError::FileRead { path: path.to_string(), source: e })?
        .parse()
        .map_err(|e| ConfigError::ParseError { source: e })
}
```

### Custom Error Types

#### Simple Enum Errors
```rust
#[derive(Debug, Clone)]
pub enum NavigationError {
    NoRoute,
    ObstacleDetected,
    FuelInsufficient,
    SystemFailure,
}
```

#### Rich Error Types with Context
```rust
#[derive(Debug)]
pub enum SimulationError {
    EntityNotFound { entity_id: u64 },
    ResourceExhausted { resource: String, required: f32, available: f32 },
    NetworkTimeout { operation: String, timeout_ms: u64 },
    ConfigurationError { field: String, value: String, reason: String },
}
```

### Error Handling Libraries

#### `thiserror` for Custom Errors
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpaceError {
    #[error("Ship {ship_id} not found")]
    ShipNotFound { ship_id: u64 },
    
    #[error("Insufficient fuel: need {required}, have {available}")]
    InsufficientFuel { required: f32, available: f32 },
    
    #[error("Network error")]
    Network(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Config(String),
}
```

#### `anyhow` for Application Errors
```rust
use anyhow::{Context, Result};

fn complex_operation() -> Result<()> {
    load_configuration()
        .context("Failed to load configuration")?;
    
    initialize_systems()
        .context("Failed to initialize simulation systems")?;
        
    Ok(())
}
```

### Error Handling in Async Context

#### Async Error Propagation
```rust
async fn fetch_ship_data(ship_id: u64) -> Result<ShipData, NetworkError> {
    let response = http_client.get(&format!("/ships/{}", ship_id))
        .await
        .map_err(NetworkError::RequestFailed)?;
        
    if !response.status().is_success() {
        return Err(NetworkError::ServerError {
            status: response.status().as_u16(),
            ship_id,
        });
    }
    
    response.json()
        .await
        .map_err(NetworkError::DeserializationFailed)
}
```

#### Concurrent Error Handling
```rust
use tokio::try_join;

async fn update_fleet(ships: Vec<u64>) -> Result<Vec<ShipStatus>, FleetError> {
    let updates: Vec<_> = ships.into_iter()
        .map(|ship_id| update_single_ship(ship_id))
        .collect();
    
    // All must succeed or all fail
    try_join_all(updates).await
        .map_err(FleetError::UpdateFailed)
}
```

### Error Recovery Strategies

#### Retry with Backoff
```rust
async fn robust_network_call<T, F, Fut>(
    operation: F,
    max_retries: usize,
) -> Result<T, NetworkError>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T, NetworkError>>,
{
    let mut delay = Duration::from_millis(100);
    
    for attempt in 0..max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt == max_retries - 1 => return Err(e),
            Err(NetworkError::Timeout | NetworkError::ServerError { .. }) => {
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
            Err(e) => return Err(e), // Don't retry permanent errors
        }
    }
    
    unreachable!()
}
```

#### Circuit Breaker Pattern
```rust
pub struct CircuitBreaker {
    failure_count: Arc<AtomicUsize>,
    last_failure: Arc<Mutex<Option<Instant>>>,
    failure_threshold: usize,
    timeout: Duration,
}

impl CircuitBreaker {
    pub async fn call<T, E, F, Fut>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        if self.is_open() {
            return Err(CircuitBreakerError::CircuitOpen);
        }
        
        match operation().await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(CircuitBreakerError::OperationFailed(e))
            }
        }
    }
}
```

### Why Advanced Error Handling Matters

#### System Reliability:
1. **Fault tolerance**: Systems continue operating despite component failures
2. **Graceful degradation**: Reduced functionality rather than complete failure
3. **Error isolation**: Failures don't cascade across system boundaries
4. **Recovery mechanisms**: Automatic retry and fallback strategies

#### Developer Experience:
1. **Clear error messages**: Actionable diagnostic information
2. **Error traceability**: Stack traces and error chains
3. **Type safety**: Compile-time verification of error handling
4. **Performance**: No hidden costs or allocations

### Space Simulation Applications

In our space simulation, advanced error handling enables:
- **Network resilience**: Handle communication failures between ships and stations
- **Resource management**: Graceful handling of resource exhaustion
- **System failures**: Component failures don't crash the entire simulation
- **Data integrity**: Ensure simulation state remains consistent during errors
- **User experience**: Meaningful error messages and recovery options

## Key Concepts

### 1. Custom Error Types with Rich Context

Creating domain-specific errors that provide meaningful diagnostic information.

```rust
use std::fmt;
use thiserror::Error;

// Base error types for our space simulation
#[derive(Error, Debug)]
pub enum SimulationError {
    #[error("Entity error: {0}")]
    Entity(#[from] EntityError),
    
    #[error("Resource error: {0}")]
    Resource(#[from] ResourceError),
    
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
    
    #[error("System error: {0}")]
    System(#[from] SystemError),
}

#[derive(Error, Debug)]
pub enum EntityError {
    #[error("Entity {entity_id} not found in sector {sector}")]
    NotFound { entity_id: u64, sector: String },
    
    #[error("Entity {entity_id} is in invalid state: {state}")]
    InvalidState { entity_id: u64, state: String },
    
    #[error("Entity {entity_id} collision with {other_id} at position ({x}, {y}, {z})")]
    Collision { entity_id: u64, other_id: u64, x: f32, y: f32, z: f32 },
    
    #[error("Entity {entity_id} component {component} missing")]
    MissingComponent { entity_id: u64, component: String },
}

#[derive(Error, Debug)]
pub enum ResourceError {
    #[error("Insufficient {resource}: requested {requested}, available {available}")]
    Insufficient { resource: String, requested: f32, available: f32 },
    
    #[error("Resource {resource} not found in container {container_id}")]
    NotFound { resource: String, container_id: u64 },
    
    #[error("Container {container_id} at capacity: {current}/{max}")]
    ContainerFull { container_id: u64, current: f32, max: f32 },
    
    #[error("Invalid resource transfer: {reason}")]
    InvalidTransfer { reason: String },
}

#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection timeout after {timeout_ms}ms to {endpoint}")]
    Timeout { endpoint: String, timeout_ms: u64 },
    
    #[error("Protocol error: expected {expected}, got {actual}")]
    ProtocolMismatch { expected: String, actual: String },
    
    #[error("Message too large: {size} bytes (max: {max_size})")]
    MessageTooLarge { size: usize, max_size: usize },
    
    #[error("Authentication failed for entity {entity_id}")]
    AuthenticationFailed { entity_id: u64 },
}

#[derive(Error, Debug)]
pub enum SystemError {
    #[error("System {system_name} failed to initialize: {reason}")]
    InitializationFailed { system_name: String, reason: String },
    
    #[error("Configuration error in {section}: {field} = {value} ({reason})")]
    Configuration { section: String, field: String, value: String, reason: String },
    
    #[error("Critical system failure: {details}")]
    Critical { details: String },
    
    #[error("Resource exhaustion: {resource} ({current}/{limit})")]
    ResourceExhaustion { resource: String, current: u64, limit: u64 },
}

// Helper trait for adding context to errors
pub trait ErrorContext<T> {
    fn with_context<F>(self, f: F) -> Result<T, SimulationError>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<SimulationError>,
{
    fn with_context<F>(self, f: F) -> Result<T, SimulationError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base_error = e.into();
            SimulationError::System(SystemError::Critical {
                details: format!("{}: {}", f(), base_error),
            })
        })
    }
}
```

### 2. Error Recovery and Resilience Patterns

Implementing robust error recovery mechanisms for simulation systems.

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::timeout;

// Resilient operation executor with multiple recovery strategies
pub struct ResilientExecutor {
    max_retries: usize,
    base_delay: Duration,
    max_delay: Duration,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl ResilientExecutor {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            circuit_breaker: Arc::new(CircuitBreaker::new(5, Duration::from_secs(30))),
        }
    }
    
    // Execute operation with retry, timeout, and circuit breaker
    pub async fn execute<T, E, F, Fut>(
        &self,
        operation: F,
        operation_timeout: Duration,
    ) -> Result<T, ExecutionError<E>>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, E>> + Send,
        E: Send + Sync + 'static,
        T: Send,
    {
        if self.circuit_breaker.is_open().await {
            return Err(ExecutionError::CircuitBreakerOpen);
        }
        
        let mut current_delay = self.base_delay;
        let mut last_error = None;
        
        for attempt in 0..=self.max_retries {
            // Execute with timeout
            let result = timeout(operation_timeout, operation()).await;
            
            match result {
                Ok(Ok(value)) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(value);
                }
                Ok(Err(e)) => {
                    last_error = Some(e);
                    self.circuit_breaker.record_failure().await;
                    
                    if attempt < self.max_retries && self.should_retry(&last_error) {
                        tokio::time::sleep(current_delay).await;
                        current_delay = (current_delay * 2).min(self.max_delay);
                        continue;
                    }
                }
                Err(_) => {
                    self.circuit_breaker.record_failure().await;
                    if attempt < self.max_retries {
                        tokio::time::sleep(current_delay).await;
                        current_delay = (current_delay * 2).min(self.max_delay);
                        continue;
                    }
                    return Err(ExecutionError::Timeout);
                }
            }
        }
        
        Err(ExecutionError::MaxRetriesExceeded {
            attempts: self.max_retries + 1,
            last_error: last_error.map(|e| format!("{:?}", e)),
        })
    }
    
    fn should_retry<E>(&self, _error: &Option<E>) -> bool {
        // Implement retry logic based on error type
        // For now, retry all errors
        true
    }
}

#[derive(Debug)]
pub enum ExecutionError<E> {
    CircuitBreakerOpen,
    Timeout,
    MaxRetriesExceeded { attempts: usize, last_error: Option<String> },
    Other(E),
}

// Circuit breaker implementation
pub struct CircuitBreaker {
    failure_threshold: usize,
    recovery_timeout: Duration,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug, Clone)]
enum CircuitBreakerState {
    Closed { failure_count: usize },
    Open { opened_at: Instant },
    HalfOpen,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, recovery_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            recovery_timeout,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed { failure_count: 0 })),
        }
    }
    
    pub async fn is_open(&self) -> bool {
        let mut state = self.state.write().await;
        
        match *state {
            CircuitBreakerState::Open { opened_at } => {
                if opened_at.elapsed() >= self.recovery_timeout {
                    *state = CircuitBreakerState::HalfOpen;
                    false
                } else {
                    true
                }
            }
            _ => false,
        }
    }
    
    pub async fn record_success(&self) {
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed { failure_count: 0 };
    }
    
    pub async fn record_failure(&self) {
        let mut state = self.state.write().await;
        
        match *state {
            CircuitBreakerState::Closed { failure_count } => {
                let new_count = failure_count + 1;
                if new_count >= self.failure_threshold {
                    *state = CircuitBreakerState::Open { opened_at: Instant::now() };
                } else {
                    *state = CircuitBreakerState::Closed { failure_count: new_count };
                }
            }
            CircuitBreakerState::HalfOpen => {
                *state = CircuitBreakerState::Open { opened_at: Instant::now() };
            }
            CircuitBreakerState::Open { .. } => {
                // Already open, do nothing
            }
        }
    }
}

// Fallback mechanisms for critical operations
pub struct FallbackChain<T> {
    primary: Box<dyn Fn() -> Result<T, SimulationError> + Send + Sync>,
    fallbacks: Vec<Box<dyn Fn() -> Result<T, SimulationError> + Send + Sync>>,
}

impl<T> FallbackChain<T> {
    pub fn new<F>(primary: F) -> Self
    where
        F: Fn() -> Result<T, SimulationError> + Send + Sync + 'static,
    {
        Self {
            primary: Box::new(primary),
            fallbacks: Vec::new(),
        }
    }
    
    pub fn with_fallback<F>(mut self, fallback: F) -> Self
    where
        F: Fn() -> Result<T, SimulationError> + Send + Sync + 'static,
    {
        self.fallbacks.push(Box::new(fallback));
        self
    }
    
    pub fn execute(&self) -> Result<T, SimulationError> {
        match (self.primary)() {
            Ok(result) => Ok(result),
            Err(primary_error) => {
                for (index, fallback) in self.fallbacks.iter().enumerate() {
                    match fallback() {
                        Ok(result) => {
                            log::warn!("Primary operation failed, used fallback {}: {}", 
                                     index + 1, primary_error);
                            return Ok(result);
                        }
                        Err(fallback_error) => {
                            log::debug!("Fallback {} failed: {}", index + 1, fallback_error);
                            continue;
                        }
                    }
                }
                
                Err(SimulationError::System(SystemError::Critical {
                    details: format!("All fallback mechanisms failed. Primary error: {}", primary_error),
                }))
            }
        }
    }
}
```

### 3. Async Error Handling and Cancellation

Advanced error handling patterns for asynchronous operations in the simulation.

```rust
use tokio::sync::{mpsc, oneshot};
use tokio::time::{interval, Duration};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

// Request tracking for async operations
#[derive(Debug)]
pub struct RequestTracker {
    pending_requests: Arc<Mutex<HashMap<u64, PendingRequest>>>,
    next_request_id: Arc<Mutex<u64>>,
}

#[derive(Debug)]
struct PendingRequest {
    sender: oneshot::Sender<Result<String, NetworkError>>,
    started_at: Instant,
    timeout: Duration,
    operation: String,
}

impl RequestTracker {
    pub fn new() -> Self {
        let tracker = Self {
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
            next_request_id: Arc::new(Mutex::new(1)),
        };
        
        // Start timeout cleanup task
        let cleanup_tracker = tracker.clone();
        tokio::spawn(async move {
            cleanup_tracker.cleanup_expired_requests().await;
        });
        
        tracker
    }
    
    pub async fn track_request(
        &self,
        operation: String,
        timeout: Duration,
    ) -> (u64, oneshot::Receiver<Result<String, NetworkError>>) {
        let request_id = {
            let mut next_id = self.next_request_id.lock().await;
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        let (sender, receiver) = oneshot::channel();
        
        let request = PendingRequest {
            sender,
            started_at: Instant::now(),
            timeout,
            operation,
        };
        
        self.pending_requests.lock().await.insert(request_id, request);
        
        (request_id, receiver)
    }
    
    pub async fn complete_request(
        &self,
        request_id: u64,
        result: Result<String, NetworkError>,
    ) -> Result<(), CompletionError> {
        let mut pending = self.pending_requests.lock().await;
        
        if let Some(request) = pending.remove(&request_id) {
            request.sender.send(result).map_err(|_| CompletionError::ReceiverDropped)?;
            Ok(())
        } else {
            Err(CompletionError::RequestNotFound { request_id })
        }
    }
    
    async fn cleanup_expired_requests(&self) {
        let mut cleanup_interval = interval(Duration::from_secs(1));
        
        loop {
            cleanup_interval.tick().await;
            
            let mut expired_requests = Vec::new();
            let now = Instant::now();
            
            {
                let mut pending = self.pending_requests.lock().await;
                let mut to_remove = Vec::new();
                
                for (&request_id, request) in pending.iter() {
                    if now.duration_since(request.started_at) >= request.timeout {
                        to_remove.push(request_id);
                        expired_requests.push((request_id, request.operation.clone()));
                    }
                }
                
                for request_id in to_remove {
                    if let Some(request) = pending.remove(&request_id) {
                        let _ = request.sender.send(Err(NetworkError::Timeout {
                            endpoint: request.operation.clone(),
                            timeout_ms: request.timeout.as_millis() as u64,
                        }));
                    }
                }
            }
            
            for (request_id, operation) in expired_requests {
                log::warn!("Request {} ({}) timed out", request_id, operation);
            }
        }
    }
}

impl Clone for RequestTracker {
    fn clone(&self) -> Self {
        Self {
            pending_requests: self.pending_requests.clone(),
            next_request_id: self.next_request_id.clone(),
        }
    }
}

#[derive(Error, Debug)]
pub enum CompletionError {
    #[error("Request {request_id} not found")]
    RequestNotFound { request_id: u64 },
    
    #[error("Receiver dropped")]
    ReceiverDropped,
}

// Graceful shutdown handler for async systems
pub struct ShutdownHandler {
    shutdown_tx: Option<mpsc::Sender<()>>,
    shutdown_rx: Arc<Mutex<Option<mpsc::Receiver<()>>>>,
    active_tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl ShutdownHandler {
    pub fn new() -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);
        
        Self {
            shutdown_tx: Some(shutdown_tx),
            shutdown_rx: Arc::new(Mutex::new(Some(shutdown_rx))),
            active_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub async fn register_task<F, Fut>(&self, task_factory: F) -> Result<(), ShutdownError>
    where
        F: FnOnce(mpsc::Receiver<()>) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let shutdown_rx = {
            let mut rx_guard = self.shutdown_rx.lock().await;
            rx_guard.take().ok_or(ShutdownError::AlreadyStarted)?
        };
        
        let handle = tokio::spawn(task_factory(shutdown_rx));
        self.active_tasks.lock().await.push(handle);
        
        Ok(())
    }
    
    pub async fn initiate_shutdown(&mut self) -> Result<(), ShutdownError> {
        if let Some(tx) = self.shutdown_tx.take() {
            // Signal shutdown to all tasks
            drop(tx);
            
            // Wait for all tasks to complete with timeout
            let tasks = {
                let mut active_tasks = self.active_tasks.lock().await;
                std::mem::take(&mut *active_tasks)
            };
            
            let shutdown_timeout = Duration::from_secs(10);
            
            match timeout(shutdown_timeout, futures::future::join_all(tasks)).await {
                Ok(results) => {
                    for result in results {
                        if let Err(e) = result {
                            log::error!("Task failed during shutdown: {}", e);
                        }
                    }
                    Ok(())
                }
                Err(_) => {
                    log::error!("Shutdown timeout exceeded");
                    Err(ShutdownError::TimeoutExceeded)
                }
            }
        } else {
            Err(ShutdownError::AlreadyShutdown)
        }
    }
}

#[derive(Error, Debug)]
pub enum ShutdownError {
    #[error("Shutdown already initiated")]
    AlreadyStarted,
    
    #[error("System already shut down")]
    AlreadyShutdown,
    
    #[error("Shutdown timeout exceeded")]
    TimeoutExceeded,
}

// Error aggregation for batch operations
pub struct ErrorCollector<E> {
    errors: Vec<(usize, E)>,
    max_errors: Option<usize>,
}

impl<E> ErrorCollector<E> {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            max_errors: None,
        }
    }
    
    pub fn with_max_errors(max_errors: usize) -> Self {
        Self {
            errors: Vec::new(),
            max_errors: Some(max_errors),
        }
    }
    
    pub fn add_error(&mut self, index: usize, error: E) -> Result<(), CollectionError> {
        self.errors.push((index, error));
        
        if let Some(max) = self.max_errors {
            if self.errors.len() >= max {
                return Err(CollectionError::TooManyErrors { 
                    count: self.errors.len(),
                    max,
                });
            }
        }
        
        Ok(())
    }
    
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    pub fn into_result<T>(self, success_value: T) -> Result<T, BatchError<E>> {
        if self.errors.is_empty() {
            Ok(success_value)
        } else {
            Err(BatchError::PartialFailures {
                errors: self.errors,
                total_operations: self.errors.len(), // This would be passed in real implementation
            })
        }
    }
}

#[derive(Error, Debug)]
pub enum CollectionError {
    #[error("Too many errors: {count} >= {max}")]
    TooManyErrors { count: usize, max: usize },
}

#[derive(Error, Debug)]
pub enum BatchError<E: std::fmt::Debug> {
    #[error("Partial failures: {errors:?} out of {total_operations} operations failed")]
    PartialFailures { errors: Vec<(usize, E)>, total_operations: usize },
}
```

### 4. Error Handling in Unsafe and FFI Contexts

Safe error handling when interfacing with unsafe code and external systems.

```rust
use std::ffi::{CStr, CString, c_char, c_int};
use std::ptr;

// Safe wrapper for unsafe FFI operations
pub struct SafeFFIWrapper;

impl SafeFFIWrapper {
    // Safe wrapper for C function that might return error codes
    pub fn call_external_function(input: &str) -> Result<String, FFIError> {
        // Convert Rust string to C string safely
        let c_input = CString::new(input)
            .map_err(|e| FFIError::StringConversion { 
                source: e,
                input: input.to_string() 
            })?;
        
        // Allocate buffer for output
        const BUFFER_SIZE: usize = 1024;
        let mut buffer = vec![0u8; BUFFER_SIZE];
        
        // Call unsafe C function
        let result_code = unsafe {
            external_c_function(
                c_input.as_ptr(),
                buffer.as_mut_ptr() as *mut c_char,
                BUFFER_SIZE as c_int,
            )
        };
        
        // Handle C error codes
        match result_code {
            0 => {
                // Success - convert C string back to Rust string
                let c_str = unsafe { CStr::from_ptr(buffer.as_ptr() as *const c_char) };
                c_str.to_str()
                    .map(|s| s.to_string())
                    .map_err(|e| FFIError::StringDecoding { source: e })
            }
            -1 => Err(FFIError::BufferTooSmall { required: result_code as usize }),
            -2 => Err(FFIError::InvalidInput { input: input.to_string() }),
            -3 => Err(FFIError::ExternalSystemError),
            code => Err(FFIError::UnknownErrorCode { code }),
        }
    }
    
    // Safe memory management for external allocations
    pub fn allocate_external_resource(size: usize) -> Result<ExternalResource, FFIError> {
        let ptr = unsafe { external_malloc(size) };
        
        if ptr.is_null() {
            Err(FFIError::AllocationFailed { size })
        } else {
            Ok(ExternalResource {
                ptr,
                size,
                _phantom: std::marker::PhantomData,
            })
        }
    }
}

// RAII wrapper for external resources
pub struct ExternalResource {
    ptr: *mut u8,
    size: usize,
    _phantom: std::marker::PhantomData<u8>,
}

impl ExternalResource {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for ExternalResource {
    fn drop(&mut self) {
        unsafe {
            external_free(self.ptr);
        }
    }
}

// Send and Sync are not automatically implemented for raw pointers
unsafe impl Send for ExternalResource {}
unsafe impl Sync for ExternalResource {}

#[derive(Error, Debug)]
pub enum FFIError {
    #[error("String conversion failed: {source}")]
    StringConversion { 
        #[source] source: std::ffi::NulError,
        input: String,
    },
    
    #[error("String decoding failed: {source}")]
    StringDecoding { 
        #[source] source: std::str::Utf8Error,
    },
    
    #[error("Buffer too small, required: {required} bytes")]
    BufferTooSmall { required: usize },
    
    #[error("Invalid input: {input}")]
    InvalidInput { input: String },
    
    #[error("External system error")]
    ExternalSystemError,
    
    #[error("Memory allocation failed for {size} bytes")]
    AllocationFailed { size: usize },
    
    #[error("Unknown error code: {code}")]
    UnknownErrorCode { code: c_int },
}

// Mock external C functions (would be provided by actual C library)
extern "C" {
    fn external_c_function(input: *const c_char, output: *mut c_char, output_size: c_int) -> c_int;
    fn external_malloc(size: usize) -> *mut u8;
    fn external_free(ptr: *mut u8);
}

// Panic-safe wrapper for potentially panicking operations
pub fn panic_safe_operation<T, F>(operation: F) -> Result<T, PanicError>
where
    F: FnOnce() -> T + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(operation) {
        Ok(result) => Ok(result),
        Err(panic_info) => {
            let message = if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            
            Err(PanicError::Caught { message })
        }
    }
}

#[derive(Error, Debug)]
pub enum PanicError {
    #[error("Panic caught: {message}")]
    Caught { message: String },
}

// Error boundary for isolating subsystem failures
pub struct ErrorBoundary<T> {
    name: String,
    fallback_value: T,
    error_count: std::sync::atomic::AtomicUsize,
    max_errors: usize,
}

impl<T: Clone> ErrorBoundary<T> {
    pub fn new(name: String, fallback_value: T, max_errors: usize) -> Self {
        Self {
            name,
            fallback_value,
            error_count: std::sync::atomic::AtomicUsize::new(0),
            max_errors,
        }
    }
    
    pub fn execute<F, E>(&self, operation: F) -> Result<T, BoundaryError<E>>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Debug,
    {
        let current_errors = self.error_count.load(std::sync::atomic::Ordering::Relaxed);
        
        if current_errors >= self.max_errors {
            log::error!("Error boundary {} has exceeded max errors ({}), using fallback", 
                       self.name, self.max_errors);
            return Ok(self.fallback_value.clone());
        }
        
        match operation() {
            Ok(result) => {
                // Reset error count on success
                self.error_count.store(0, std::sync::atomic::Ordering::Relaxed);
                Ok(result)
            }
            Err(e) => {
                let new_count = self.error_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                
                log::warn!("Error in boundary {}: {:?} (error count: {})", 
                          self.name, e, new_count);
                
                if new_count >= self.max_errors {
                    log::error!("Error boundary {} switching to fallback mode", self.name);
                    Ok(self.fallback_value.clone())
                } else {
                    Err(BoundaryError::OperationFailed { 
                        boundary: self.name.clone(),
                        error: format!("{:?}", e),
                        error_count: new_count,
                    })
                }
            }
        }
    }
}

#[derive(Error, Debug)]
pub enum BoundaryError<E: std::fmt::Debug> {
    #[error("Operation failed in boundary {boundary}: {error} (errors: {error_count})")]
    OperationFailed { 
        boundary: String,
        error: String,
        error_count: usize,
    },
}
```

## Practical Application: Resilient Space Station Communication System

```rust
use tokio::sync::mpsc;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

// Complete example: Resilient communication system for space stations
pub struct SpaceStationNetwork {
    executor: ResilientExecutor,
    request_tracker: RequestTracker,
    stations: Arc<RwLock<HashMap<u64, StationInfo>>>,
    message_handlers: Arc<RwLock<HashMap<MessageType, MessageHandler>>>,
}

#[derive(Debug, Clone)]
pub struct StationInfo {
    pub id: u64,
    pub name: String,
    pub position: (f32, f32, f32),
    pub status: StationStatus,
    pub last_contact: Instant,
}

#[derive(Debug, Clone)]
pub enum StationStatus {
    Online,
    Degraded,
    Offline,
    Maintenance,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MessageType {
    ResourceRequest,
    StatusUpdate,
    NavigationData,
    EmergencyAlert,
}

type MessageHandler = Box<dyn Fn(StationMessage) -> Result<StationMessage, MessageError> + Send + Sync>;

#[derive(Debug, Clone)]
pub struct StationMessage {
    pub from: u64,
    pub to: u64,
    pub message_type: MessageType,
    pub payload: String,
    pub timestamp: Instant,
    pub priority: MessagePriority,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Error, Debug)]
pub enum MessageError {
    #[error("Station {station_id} not found")]
    StationNotFound { station_id: u64 },
    
    #[error("Message handler not found for type {message_type:?}")]
    HandlerNotFound { message_type: MessageType },
    
    #[error("Message processing failed: {reason}")]
    ProcessingFailed { reason: String },
    
    #[error("Network error: {0}")]
    Network(#[from] NetworkError),
}

impl SpaceStationNetwork {
    pub fn new() -> Self {
        Self {
            executor: ResilientExecutor::new(),
            request_tracker: RequestTracker::new(),
            stations: Arc::new(RwLock::new(HashMap::new())),
            message_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_station(&self, station: StationInfo) -> Result<(), SimulationError> {
        let mut stations = self.stations.write().await;
        stations.insert(station.id, station);
        Ok(())
    }
    
    pub async fn send_message(&self, message: StationMessage) -> Result<StationMessage, MessageError> {
        // Create fallback chain for message delivery
        let fallback_chain = FallbackChain::new({
            let msg = message.clone();
            move || self.send_direct_message(msg.clone())
        })
        .with_fallback({
            let msg = message.clone();
            move || self.send_relay_message(msg.clone())
        })
        .with_fallback({
            let msg = message.clone();
            move || self.queue_message_for_later(msg.clone())
        });
        
        // Execute with resilient executor
        let result = self.executor.execute(
            || async { fallback_chain.execute() },
            Duration::from_secs(5),
        ).await;
        
        match result {
            Ok(response) => Ok(response),
            Err(ExecutionError::CircuitBreakerOpen) => {
                // Circuit breaker is open, queue message
                self.queue_message_for_later(message)
            }
            Err(ExecutionError::Timeout) => {
                Err(MessageError::Network(NetworkError::Timeout {
                    endpoint: format!("station_{}", message.to),
                    timeout_ms: 5000,
                }))
            }
            Err(ExecutionError::MaxRetriesExceeded { attempts, last_error }) => {
                Err(MessageError::ProcessingFailed {
                    reason: format!("Failed after {} attempts. Last error: {:?}", attempts, last_error),
                })
            }
            Err(ExecutionError::Other(e)) => Err(e),
        }
    }
    
    fn send_direct_message(&self, message: StationMessage) -> Result<StationMessage, SimulationError> {
        // Simulate direct message sending
        // In real implementation, this would use actual network communication
        
        // Check if destination station exists and is online
        let stations = futures::executor::block_on(self.stations.read());
        let station = stations.get(&message.to)
            .ok_or_else(|| SimulationError::Entity(EntityError::NotFound {
                entity_id: message.to,
                sector: "unknown".to_string(),
            }))?;
        
        match station.status {
            StationStatus::Online => {
                // Simulate message processing
                let response = StationMessage {
                    from: message.to,
                    to: message.from,
                    message_type: message.message_type.clone(),
                    payload: format!("ACK: {}", message.payload),
                    timestamp: Instant::now(),
                    priority: MessagePriority::Normal,
                };
                Ok(response)
            }
            StationStatus::Offline => {
                Err(SimulationError::Network(NetworkError::Timeout {
                    endpoint: format!("station_{}", message.to),
                    timeout_ms: 1000,
                }))
            }
            StationStatus::Degraded => {
                // 50% chance of success in degraded mode
                if rand::random::<f32>() > 0.5 {
                    let response = StationMessage {
                        from: message.to,
                        to: message.from,
                        message_type: message.message_type.clone(),
                        payload: format!("DEGRADED_ACK: {}", message.payload),
                        timestamp: Instant::now(),
                        priority: MessagePriority::Normal,
                    };
                    Ok(response)
                } else {
                    Err(SimulationError::System(SystemError::Critical {
                        details: "Station in degraded mode, message failed".to_string(),
                    }))
                }
            }
            StationStatus::Maintenance => {
                Err(SimulationError::System(SystemError::Critical {
                    details: format!("Station {} is under maintenance", message.to),
                }))
            }
        }
    }
    
    fn send_relay_message(&self, message: StationMessage) -> Result<StationMessage, SimulationError> {
        // Simulate relay through another station
        log::info!("Attempting relay delivery for message to station {}", message.to);
        
        // In a real implementation, this would find intermediate stations
        // For simulation, we'll just add a delay and try again
        std::thread::sleep(Duration::from_millis(100));
        
        let response = StationMessage {
            from: message.to,
            to: message.from,
            message_type: message.message_type.clone(),
            payload: format!("RELAY_ACK: {}", message.payload),
            timestamp: Instant::now(),
            priority: MessagePriority::Normal,
        };
        
        Ok(response)
    }
    
    fn queue_message_for_later(&self, message: StationMessage) -> Result<StationMessage, SimulationError> {
        // Queue message for later delivery
        log::warn!("Queueing message for later delivery: {:?}", message);
        
        // In real implementation, this would add to a persistent queue
        let response = StationMessage {
            from: 0, // System message
            to: message.from,
            message_type: MessageType::StatusUpdate,
            payload: "Message queued for later delivery".to_string(),
            timestamp: Instant::now(),
            priority: MessagePriority::Low,
        };
        
        Ok(response)
    }
    
    pub async fn process_messages(&self, mut message_rx: mpsc::Receiver<StationMessage>) {
        let mut error_collector = ErrorCollector::with_max_errors(10);
        let mut message_count = 0;
        
        while let Some(message) = message_rx.recv().await {
            message_count += 1;
            
            let result = self.process_single_message(message.clone()).await;
            
            if let Err(e) = result {
                if let Err(collection_error) = error_collector.add_error(message_count, e) {
                    log::error!("Too many message processing errors: {:?}", collection_error);
                    break;
                }
            }
        }
        
        if error_collector.has_errors() {
            match error_collector.into_result(()) {
                Ok(_) => unreachable!(),
                Err(batch_error) => {
                    log::error!("Batch message processing completed with errors: {:?}", batch_error);
                }
            }
        }
    }
    
    async fn process_single_message(&self, message: StationMessage) -> Result<(), MessageError> {
        let handlers = self.message_handlers.read().await;
        
        let handler = handlers.get(&message.message_type)
            .ok_or_else(|| MessageError::HandlerNotFound {
                message_type: message.message_type.clone(),
            })?;
        
        match handler(message) {
            Ok(response) => {
                log::debug!("Message processed successfully: {:?}", response);
                Ok(())
            }
            Err(e) => {
                log::error!("Message processing failed: {:?}", e);
                Err(e)
            }
        }
    }
}

// Usage example
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_resilient_message_delivery() {
        let network = SpaceStationNetwork::new();
        
        // Register test stations
        let station1 = StationInfo {
            id: 1,
            name: "Alpha Station".to_string(),
            position: (0.0, 0.0, 0.0),
            status: StationStatus::Online,
            last_contact: Instant::now(),
        };
        
        let station2 = StationInfo {
            id: 2,
            name: "Beta Station".to_string(),
            position: (100.0, 0.0, 0.0),
            status: StationStatus::Degraded,
            last_contact: Instant::now(),
        };
        
        network.register_station(station1).await.unwrap();
        network.register_station(station2).await.unwrap();
        
        // Test message delivery
        let message = StationMessage {
            from: 1,
            to: 2,
            message_type: MessageType::ResourceRequest,
            payload: "Request 100 units of fuel".to_string(),
            timestamp: Instant::now(),
            priority: MessagePriority::High,
        };
        
        let result = network.send_message(message).await;
        
        match result {
            Ok(response) => {
                println!("Message delivered successfully: {:?}", response);
                assert!(response.payload.contains("ACK") || response.payload.contains("RELAY"));
            }
            Err(e) => {
                println!("Message delivery failed with graceful fallback: {:?}", e);
                // In degraded mode, some failures are expected
            }
        }
    }
}
```

## Key Takeaways

1. **Error Type Design**: Create rich, domain-specific error types that provide actionable diagnostic information
2. **Error Recovery**: Implement multiple fallback strategies including retry, circuit breakers, and graceful degradation
3. **Async Error Handling**: Use proper error propagation and cancellation patterns in async contexts
4. **Resilience Patterns**: Build systems that continue operating despite component failures
5. **Performance**: Leverage Rust's zero-cost error handling for high-performance applications
6. **Safety**: Handle unsafe code and FFI operations with proper error boundaries

## Best Practices

- Use `thiserror` for library errors and `anyhow` for application errors
- Implement retry logic with exponential backoff for transient failures
- Use circuit breakers to prevent cascading failures
- Provide meaningful error messages with context
- Design error types to be composable and extensible
- Test error handling paths as thoroughly as success paths
- Use error boundaries to isolate subsystem failures
- Log errors appropriately without overwhelming logs

## Performance Considerations

- Error handling in Rust has minimal runtime overhead
- Avoid allocating in error paths where possible
- Use static error messages when dynamic context isn't needed
- Consider error aggregation for batch operations
- Profile error-heavy code paths to ensure performance
- Use `Result` over `Option` when you need error information

## Next Steps

In the next tutorial, we'll explore Unsafe Rust and FFI, learning how to safely interface with C/C++ code and manage memory manually when performance demands it, building on the error handling patterns we've established here.