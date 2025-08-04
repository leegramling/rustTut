// Tutorial 08: Advanced Error Handling - Complete Implementation
// 
// This file contains the complete implementation of advanced error handling patterns
// for a resilient space simulation system.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use thiserror::Error;
use std::sync::atomic::{AtomicUsize, Ordering};

// ================================
// Exercise 1: Custom Error Types
// ================================

#[derive(Error, Debug)]
pub enum SpaceSimulationError {
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

pub trait ErrorContext<T> {
    fn with_context<F>(self, f: F) -> Result<T, SpaceSimulationError>
    where
        F: FnOnce() -> String;
}

impl<T, E> ErrorContext<T> for Result<T, E>
where
    E: Into<SpaceSimulationError>,
{
    fn with_context<F>(self, f: F) -> Result<T, SpaceSimulationError>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let base_error = e.into();
            SpaceSimulationError::System(SystemError::Critical {
                details: format!("{}: {}", f(), base_error),
            })
        })
    }
}

// ================================
// Exercise 2: Resilient Executor
// ================================

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
            let result = tokio::time::timeout(operation_timeout, operation()).await;
            
            match result {
                Ok(Ok(value)) => {
                    self.circuit_breaker.record_success().await;
                    return Ok(value);
                }
                Ok(Err(e)) => {
                    last_error = Some(e);
                    self.circuit_breaker.record_failure().await;
                    
                    if attempt < self.max_retries {
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
}

#[derive(Error, Debug)]
pub enum ExecutionError<E> {
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
    
    #[error("Operation timed out")]
    Timeout,
    
    #[error("Max retries exceeded ({attempts} attempts). Last error: {last_error:?}")]
    MaxRetriesExceeded { attempts: usize, last_error: Option<String> },
    
    #[error("Other error: {0:?}")]
    Other(E),
}

// ================================
// Exercise 3: Circuit Breaker
// ================================

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

// ================================
// Exercise 4: Fallback Chain
// ================================

pub struct FallbackChain<T> {
    primary: Box<dyn Fn() -> Result<T, SpaceSimulationError> + Send + Sync>,
    fallbacks: Vec<Box<dyn Fn() -> Result<T, SpaceSimulationError> + Send + Sync>>,
}

impl<T> FallbackChain<T> {
    pub fn new<F>(primary: F) -> Self
    where
        F: Fn() -> Result<T, SpaceSimulationError> + Send + Sync + 'static,
    {
        Self {
            primary: Box::new(primary),
            fallbacks: Vec::new(),
        }
    }
    
    pub fn with_fallback<F>(mut self, fallback: F) -> Self
    where
        F: Fn() -> Result<T, SpaceSimulationError> + Send + Sync + 'static,
    {
        self.fallbacks.push(Box::new(fallback));
        self
    }
    
    pub fn execute(&self) -> Result<T, SpaceSimulationError> {
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
                
                Err(SpaceSimulationError::System(SystemError::Critical {
                    details: format!("All fallback mechanisms failed. Primary error: {}", primary_error),
                }))
            }
        }
    }
}

// ================================
// Exercise 5: Request Tracking
// ================================

pub struct RequestTracker {
    pending_requests: Arc<Mutex<HashMap<u64, PendingRequest>>>,
    next_request_id: Arc<Mutex<u64>>,
}

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
        
        // Start cleanup task
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
        let mut cleanup_interval = tokio::time::interval(Duration::from_secs(1));
        
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

// ================================
// Exercise 6: Error Aggregation
// ================================

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
                total_operations: 100, // This would be passed in real implementation
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

// ================================
// Exercise 7: Space Station Network
// ================================

pub struct SpaceStationNetwork {
    executor: ResilientExecutor,
    request_tracker: RequestTracker,
    stations: Arc<RwLock<HashMap<u64, StationInfo>>>,
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

#[derive(Debug, Clone)]
pub struct StationMessage {
    pub from: u64,
    pub to: u64,
    pub message_type: MessageType,
    pub payload: String,
    pub timestamp: Instant,
    pub priority: MessagePriority,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum MessageType {
    ResourceRequest,
    StatusUpdate,
    NavigationData,
    EmergencyAlert,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl SpaceStationNetwork {
    pub fn new() -> Self {
        Self {
            executor: ResilientExecutor::new(),
            request_tracker: RequestTracker::new(),
            stations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn register_station(&self, station: StationInfo) -> Result<(), SpaceSimulationError> {
        let mut stations = self.stations.write().await;
        stations.insert(station.id, station);
        Ok(())
    }
    
    pub async fn send_message(&self, message: StationMessage) -> Result<StationMessage, SpaceSimulationError> {
        let fallback_chain = FallbackChain::new({
            let msg = message.clone();
            let network = self;
            move || network.send_direct_message(msg.clone())
        })
        .with_fallback({
            let msg = message.clone();
            let network = self;
            move || network.send_relay_message(msg.clone())
        })
        .with_fallback({
            let msg = message.clone();
            let network = self;
            move || network.queue_message_for_later(msg.clone())
        });
        
        let result = self.executor.execute(
            || async { fallback_chain.execute() },
            Duration::from_secs(5),
        ).await;
        
        match result {
            Ok(response) => Ok(response),
            Err(ExecutionError::CircuitBreakerOpen) => {
                self.queue_message_for_later(message)
            }
            Err(ExecutionError::Timeout) => {
                Err(SpaceSimulationError::Network(NetworkError::Timeout {
                    endpoint: format!("station_{}", message.to),
                    timeout_ms: 5000,
                }))
            }
            Err(ExecutionError::MaxRetriesExceeded { attempts, last_error }) => {
                Err(SpaceSimulationError::System(SystemError::Critical {
                    details: format!("Failed after {} attempts. Last error: {:?}", attempts, last_error),
                }))
            }
            Err(ExecutionError::Other(e)) => Err(e),
        }
    }
    
    fn send_direct_message(&self, message: StationMessage) -> Result<StationMessage, SpaceSimulationError> {
        let stations = futures::executor::block_on(self.stations.read());
        let station = stations.get(&message.to)
            .ok_or_else(|| SpaceSimulationError::Entity(EntityError::NotFound {
                entity_id: message.to,
                sector: "unknown".to_string(),
            }))?;
        
        match station.status {
            StationStatus::Online => {
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
                Err(SpaceSimulationError::Network(NetworkError::Timeout {
                    endpoint: format!("station_{}", message.to),
                    timeout_ms: 1000,
                }))
            }
            StationStatus::Degraded => {
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
                    Err(SpaceSimulationError::System(SystemError::Critical {
                        details: "Station in degraded mode, message failed".to_string(),
                    }))
                }
            }
            StationStatus::Maintenance => {
                Err(SpaceSimulationError::System(SystemError::Critical {
                    details: format!("Station {} is under maintenance", message.to),
                }))
            }
        }
    }
    
    fn send_relay_message(&self, message: StationMessage) -> Result<StationMessage, SpaceSimulationError> {
        log::info!("Attempting relay delivery for message to station {}", message.to);
        
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
    
    fn queue_message_for_later(&self, message: StationMessage) -> Result<StationMessage, SpaceSimulationError> {
        log::warn!("Queueing message for later delivery: {:?}", message);
        
        let response = StationMessage {
            from: 0,
            to: message.from,
            message_type: MessageType::StatusUpdate,
            payload: "Message queued for later delivery".to_string(),
            timestamp: Instant::now(),
            priority: MessagePriority::Low,
        };
        
        Ok(response)
    }
}

// ================================
// Exercise 8: Testing Error Paths
// ================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;
    
    #[tokio::test]
    async fn test_circuit_breaker_behavior() {
        let circuit_breaker = CircuitBreaker::new(3, Duration::from_millis(100));
        
        // Start in Closed state
        assert!(!circuit_breaker.is_open().await);
        
        // Record failures until threshold reached
        circuit_breaker.record_failure().await;
        assert!(!circuit_breaker.is_open().await);
        
        circuit_breaker.record_failure().await;
        assert!(!circuit_breaker.is_open().await);
        
        circuit_breaker.record_failure().await;
        assert!(circuit_breaker.is_open().await);
        
        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;
        assert!(!circuit_breaker.is_open().await); // Should be HalfOpen now
        
        // Test success resets to Closed
        circuit_breaker.record_success().await;
        assert!(!circuit_breaker.is_open().await);
    }
    
    #[tokio::test]
    async fn test_resilient_executor_retry() {
        let executor = ResilientExecutor::new();
        let mut attempt_count = 0;
        
        let result = executor.execute(
            || {
                attempt_count += 1;
                async move {
                    if attempt_count < 3 {
                        Err("Simulated failure")
                    } else {
                        Ok("Success")
                    }
                }
            },
            Duration::from_secs(1),
        ).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(attempt_count, 3);
    }
    
    #[tokio::test]
    async fn test_fallback_chain() {
        let mut primary_called = false;
        let mut fallback1_called = false;
        let mut fallback2_called = false;
        
        let chain = FallbackChain::new(move || {
            primary_called = true;
            Err(SpaceSimulationError::System(SystemError::Critical {
                details: "Primary failed".to_string(),
            }))
        })
        .with_fallback(move || {
            fallback1_called = true;
            Err(SpaceSimulationError::System(SystemError::Critical {
                details: "Fallback 1 failed".to_string(),
            }))
        })
        .with_fallback(move || {
            fallback2_called = true;
            Ok("Fallback 2 success".to_string())
        });
        
        let result = chain.execute();
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Fallback 2 success");
    }
    
    #[tokio::test]
    async fn test_error_aggregation() {
        let mut collector = ErrorCollector::with_max_errors(3);
        
        collector.add_error(1, "Error 1").unwrap();
        collector.add_error(2, "Error 2").unwrap();
        
        assert!(collector.has_errors());
        
        let result = collector.into_result("Success");
        assert!(result.is_err());
        
        match result {
            Err(BatchError::PartialFailures { errors, .. }) => {
                assert_eq!(errors.len(), 2);
            }
        }
    }
    
    #[tokio::test]
    async fn test_message_delivery_resilience() {
        let network = SpaceStationNetwork::new();
        
        // Register stations with different statuses
        let online_station = create_test_station(1, StationStatus::Online);
        let offline_station = create_test_station(2, StationStatus::Offline);
        
        network.register_station(online_station).await.unwrap();
        network.register_station(offline_station).await.unwrap();
        
        // Test message delivery to online station
        let message = create_test_message(1, 1, MessageType::StatusUpdate);
        let result = network.send_message(message).await;
        assert!(result.is_ok());
        
        // Test fallback for offline station
        let message = create_test_message(1, 2, MessageType::StatusUpdate);
        let result = network.send_message(message).await;
        // Should succeed due to fallback mechanisms
        assert!(result.is_ok());
    }
}

// ================================
// Helper Functions
// ================================

fn create_test_station(id: u64, status: StationStatus) -> StationInfo {
    StationInfo {
        id,
        name: format!("Test Station {}", id),
        position: (id as f32 * 100.0, 0.0, 0.0),
        status,
        last_contact: Instant::now(),
    }
}

fn create_test_message(from: u64, to: u64, message_type: MessageType) -> StationMessage {
    StationMessage {
        from,
        to,
        message_type,
        payload: "Test message".to_string(),
        timestamp: Instant::now(),
        priority: MessagePriority::Normal,
    }
}

// Additional helper functions for testing
fn simulate_network_failure() -> NetworkError {
    NetworkError::Timeout {
        endpoint: "test_endpoint".to_string(),
        timeout_ms: 1000,
    }
}

fn create_resource_error(resource: &str, requested: f32, available: f32) -> ResourceError {
    ResourceError::Insufficient {
        resource: resource.to_string(),
        requested,
        available,
    }
}

fn create_entity_error(entity_id: u64, sector: &str) -> EntityError {
    EntityError::NotFound {
        entity_id,
        sector: sector.to_string(),
    }
}