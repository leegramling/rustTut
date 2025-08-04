// Tutorial 04: Concurrency Fundamentals
// Complete the following exercises to practice concurrent programming in Rust

use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use tokio::sync::mpsc as async_mpsc;

// Exercise 1: Thread-Safe Resource Pool
#[derive(Debug, Clone)]
pub struct Resource {
    pub id: u32,
    pub resource_type: ResourceType,
    pub amount: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceType {
    Fuel,
    Water,
    Food,
    Minerals,
}

// TODO: Implement ResourcePool with thread-safe operations
#[derive(Debug, Default)]
pub struct ResourcePool {
    // TODO: Add fields for resources (Vec<Resource>) and total_allocated (usize)
    // Both should be wrapped in Mutex for thread safety
}

impl ResourcePool {
    // TODO: Implement new() that returns Arc<Self>
    pub fn new() -> Arc<Self> {
        todo!("Create new ResourcePool wrapped in Arc")
    }
    
    // TODO: Implement add_resource that safely adds a resource
    // Should update both resources vector and total_allocated counter
    pub fn add_resource(&self, resource: Resource) {
        todo!("Safely add resource and update total")
    }
    
    // TODO: Implement get_resource that removes and returns first resource of given type
    // Return None if no resource of that type exists
    pub fn get_resource(&self, resource_type: ResourceType) -> Option<Resource> {
        todo!("Find and remove resource of specified type")
    }
    
    // TODO: Implement get_total_allocated that returns current total
    pub fn get_total_allocated(&self) -> usize {
        todo!("Return total allocated amount")
    }
}

// TODO: Implement demonstrate_thread_safety function
// Should spawn 3 producer threads (each adding 5 resources) and 2 consumer threads
// Producers should add different resource types based on index
// Consumers should try to get resources of different types
pub fn demonstrate_thread_safety() {
    todo!("Create producer and consumer threads with shared ResourcePool")
}

// Exercise 2: Configuration Management with RwLock
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub max_ships: usize,
    pub resource_generation_rate: f32,
    pub market_volatility: f32,
    pub physics_time_step: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            max_ships: 100,
            resource_generation_rate: 1.0,
            market_volatility: 0.1,
            physics_time_step: 0.016,
        }
    }
}

// TODO: Implement ConfigurationManager using RwLock
pub struct ConfigurationManager {
    // TODO: Add config field with RwLock<SimulationConfig>
    // TODO: Add stats field with RwLock<HashMap<String, f32>>
}

impl ConfigurationManager {
    // TODO: Implement new() constructor
    pub fn new() -> Self {
        todo!("Create ConfigurationManager with default config and empty stats")
    }
    
    // TODO: Implement get_config() that returns a clone of current config
    // Should use read lock
    pub fn get_config(&self) -> SimulationConfig {
        todo!("Read and return current configuration")
    }
    
    // TODO: Implement update_config() that takes a closure to modify config
    // Should use write lock
    pub fn update_config<F>(&self, updater: F) 
    where 
        F: FnOnce(&mut SimulationConfig),
    {
        todo!("Apply updater function to configuration")
    }
    
    // TODO: Implement record_stat() that adds to a statistic
    pub fn record_stat(&self, key: String, value: f32) {
        todo!("Add value to existing stat or create new entry")
    }
    
    // TODO: Implement get_stat() that returns current value for a key
    pub fn get_stat(&self, key: &str) -> f32 {
        todo!("Return stat value or 0.0 if not found")
    }
    
    // TODO: Implement get_all_stats() that returns clone of all stats
    pub fn get_all_stats(&self) -> HashMap<String, f32> {
        todo!("Return clone of all statistics")
    }
}

// TODO: Implement demonstrate_rwlock_usage function
// Should spawn 5 reader threads (each reading config 10 times and recording stats)
// Should spawn 1 writer thread (updating config 3 times)
// Print final statistics at the end
pub fn demonstrate_rwlock_usage() {
    todo!("Demonstrate concurrent reads and occasional writes with RwLock")
}

// Exercise 3: Message Passing with Channels
#[derive(Debug, Clone)]
pub enum SimulationMessage {
    ShipStatusUpdate {
        ship_id: u32,
        position: (f32, f32, f32),
        fuel_level: f32,
        cargo_amount: f32,
    },
    ResourceDiscovered {
        position: (f32, f32, f32),
        resource_type: ResourceType,
        amount: f32,
    },
    TradeRequest {
        from_ship: u32,
        to_station: u32,
        resource: ResourceType,
        amount: f32,
    },
    SystemAlert {
        severity: AlertSeverity,
        message: String,
        timestamp: Instant,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

// TODO: Implement MessageDispatcher
pub struct MessageDispatcher {
    // TODO: Add sender field with mpsc::Sender<SimulationMessage>
}

impl MessageDispatcher {
    // TODO: Implement new() that returns (MessageDispatcher, mpsc::Receiver<SimulationMessage>)
    pub fn new() -> (Self, mpsc::Receiver<SimulationMessage>) {
        todo!("Create channel and return dispatcher and receiver")
    }
    
    // TODO: Implement send_message()
    pub fn send_message(&self, message: SimulationMessage) -> Result<(), mpsc::SendError<SimulationMessage>> {
        todo!("Send message through channel")
    }
    
    // TODO: Implement clone_sender() to create additional senders
    pub fn clone_sender(&self) -> mpsc::Sender<SimulationMessage> {
        todo!("Clone the sender for use in other threads")
    }
}

// TODO: Implement MessageProcessor
pub struct MessageProcessor;

impl MessageProcessor {
    // TODO: Implement process_messages() that processes messages until Shutdown
    // Should count different message types and print statistics
    // Should check for low fuel alerts (<20%) in ship status updates
    pub fn process_messages(receiver: mpsc::Receiver<SimulationMessage>) {
        todo!("Process messages and maintain statistics until shutdown")
    }
}

// TODO: Implement demonstrate_message_passing function
// Should spawn message processor thread
// Should spawn 3 ship simulation threads (each sending 5 status updates)
// Should spawn alert system thread (sending 2 alerts)
// Ships should occasionally discover resources and send trade requests
pub fn demonstrate_message_passing() {
    todo!("Demonstrate message passing between multiple threads")
}

// Exercise 4: Async Resource Management
#[derive(Debug, Clone)]
pub struct NetworkResource {
    pub id: String,
    pub url: String,
    pub data: Vec<u8>,
    pub last_updated: Instant,
}

// TODO: Implement AsyncResourceManager
pub struct AsyncResourceManager {
    // TODO: Add resources field with Arc<tokio::sync::Mutex<HashMap<String, NetworkResource>>>
    // TODO: Add update_sender field with async_mpsc::UnboundedSender<String>
}

impl AsyncResourceManager {
    // TODO: Implement new() that returns (Self, async_mpsc::UnboundedReceiver<String>)
    pub fn new() -> (Self, async_mpsc::UnboundedReceiver<String>) {
        todo!("Create async resource manager with update channel")
    }
    
    // TODO: Implement async fetch_resource() that simulates network fetch
    // Should sleep for 100ms + (id.len() * 50)ms to simulate network delay
    // Should return error if id contains "error"
    // Should store result in resources HashMap and send notification
    pub async fn fetch_resource(&self, id: String, url: String) -> Result<NetworkResource, String> {
        todo!("Simulate async network fetch and cache result")
    }
    
    // TODO: Implement async get_resource()
    pub async fn get_resource(&self, id: &str) -> Option<NetworkResource> {
        todo!("Get resource from cache")
    }
    
    // TODO: Implement async get_all_resources()
    pub async fn get_all_resources(&self) -> Vec<NetworkResource> {
        todo!("Return all cached resources")
    }
    
    // TODO: Implement async fetch_batch() that fetches multiple resources concurrently
    // Use futures::future::join_all for concurrent execution
    pub async fn fetch_batch(&self, requests: Vec<(String, String)>) -> Vec<Result<NetworkResource, String>> {
        todo!("Fetch multiple resources concurrently using join_all")
    }
    
    // TODO: Implement async cleanup_old_resources()
    // Remove resources older than max_age
    pub async fn cleanup_old_resources(&self, max_age: Duration) {
        todo!("Remove resources older than max_age from cache")
    }
}

// TODO: Implement async demonstrate_async_operations function
// Should create AsyncResourceManager and demonstrate:
// - Fetching individual resources
// - Batch fetching with some failures
// - Processing update notifications
// - Background cleanup task
pub async fn demonstrate_async_operations() {
    todo!("Demonstrate async resource management patterns")
}

// Exercise 5: Lock-Free Performance Counter
// TODO: Implement PerformanceCounter using atomic operations
pub struct PerformanceCounter {
    // TODO: Add value field with AtomicUsize
    // TODO: Add started field with AtomicBool  
    // TODO: Add start_time field with Mutex<Option<Instant>>
}

impl PerformanceCounter {
    // TODO: Implement new() that returns Arc<Self>
    pub fn new() -> Arc<Self> {
        todo!("Create new performance counter")
    }
    
    // TODO: Implement start() that sets started flag and records start time
    // Use compare_exchange to ensure only started once
    pub fn start(&self) {
        todo!("Start the counter if not already started")
    }
    
    // TODO: Implement increment() that atomically increments and returns new value
    pub fn increment(&self) -> usize {
        todo!("Atomically increment counter")
    }
    
    // TODO: Implement add() that atomically adds amount and returns new value
    pub fn add(&self, amount: usize) -> usize {
        todo!("Atomically add amount to counter")
    }
    
    // TODO: Implement get() that returns current value
    pub fn get(&self) -> usize {
        todo!("Get current counter value")
    }
    
    // TODO: Implement get_rate_per_second() that calculates rate based on elapsed time
    pub fn get_rate_per_second(&self) -> f64 {
        todo!("Calculate operations per second since start")
    }
}

// Exercise 6: Thread-Safe Work Queue
// TODO: Implement WorkQueue using Mutex<VecDeque<T>> and AtomicUsize
pub struct WorkQueue<T> {
    // TODO: Add queue field with Mutex<VecDeque<T>>
    // TODO: Add pending_count field with AtomicUsize for lock-free length queries
}

impl<T> WorkQueue<T> {
    // TODO: Implement new() that returns Arc<Self>
    pub fn new() -> Arc<Self> {
        todo!("Create new work queue")
    }
    
    // TODO: Implement push() that adds item to back of queue
    pub fn push(&self, item: T) {
        todo!("Add item to back of queue and increment counter")
    }
    
    // TODO: Implement pop() that removes item from front of queue
    pub fn pop(&self) -> Option<T> {
        todo!("Remove and return item from front of queue")
    }
    
    // TODO: Implement len() that returns current queue length
    pub fn len(&self) -> usize {
        todo!("Return current queue length using atomic counter")
    }
    
    // TODO: Implement is_empty()
    pub fn is_empty(&self) -> bool {
        todo!("Check if queue is empty")
    }
}

// Work item for simulation tasks
#[derive(Debug)]
pub struct SimulationTask {
    pub task_id: usize,
    pub task_type: TaskType,
    pub priority: u8,
    pub estimated_duration: Duration,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    UpdateShipPosition,
    ProcessTradeRequest,
    CalculatePhysics,
    UpdateMarketPrices,
    GenerateResources,
}

impl SimulationTask {
    pub fn new(task_id: usize, task_type: TaskType, priority: u8) -> Self {
        let estimated_duration = match task_type {
            TaskType::UpdateShipPosition => Duration::from_millis(5),
            TaskType::ProcessTradeRequest => Duration::from_millis(20),
            TaskType::CalculatePhysics => Duration::from_millis(15),
            TaskType::UpdateMarketPrices => Duration::from_millis(50),
            TaskType::GenerateResources => Duration::from_millis(10),
        };
        
        Self {
            task_id,
            task_type,
            priority,
            estimated_duration,
            created_at: Instant::now(),
        }
    }
    
    pub fn execute(&self) {
        thread::sleep(self.estimated_duration);
    }
}

// TODO: Implement WorkerPool that manages multiple worker threads
pub struct WorkerPool {
    // TODO: Add work_queue field with Arc<WorkQueue<SimulationTask>>
    // TODO: Add task_counter and completed_counter fields with Arc<PerformanceCounter>
    // TODO: Add worker_handles field with Vec<thread::JoinHandle<()>>
}

impl WorkerPool {
    // TODO: Implement new() that creates worker threads
    // Each worker should pop tasks from queue and execute them
    // Workers should increment completed_counter after each task
    pub fn new(num_workers: usize) -> Self {
        todo!("Create worker pool with specified number of worker threads")
    }
    
    // TODO: Implement submit_task() that adds task to queue
    pub fn submit_task(&self, task: SimulationTask) {
        todo!("Submit task to work queue")
    }
    
    // TODO: Implement get_queue_length()
    pub fn get_queue_length(&self) -> usize {
        todo!("Return current queue length")
    }
    
    // TODO: Implement get_task_stats() that returns (submitted, completed, submit_rate, completion_rate)
    pub fn get_task_stats(&self) -> (usize, usize, f64, f64) {
        todo!("Return task statistics")
    }
    
    // TODO: Implement shutdown() that waits for all worker threads
    pub fn shutdown(self) {
        todo!("Shutdown all worker threads")
    }
}

// TODO: Implement demonstrate_concurrent_processing function
// Should create WorkerPool with 4 workers
// Should submit 100 initial tasks, then 20 more tasks every second for 3 seconds
// Should monitor progress and print statistics
// Should wait for all tasks to complete before shutting down
pub fn demonstrate_concurrent_processing() {
    todo!("Demonstrate concurrent task processing with worker pool")
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_resource_pool_basic() {
        let pool = ResourcePool::new();
        
        let resource = Resource {
            id: 1,
            resource_type: ResourceType::Fuel,
            amount: 100.0,
        };
        
        pool.add_resource(resource.clone());
        assert_eq!(pool.get_total_allocated(), 100);
        
        let retrieved = pool.get_resource(ResourceType::Fuel);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 1);
        
        let empty = pool.get_resource(ResourceType::Water);
        assert!(empty.is_none());
    }

    #[test]
    fn test_configuration_manager() {
        let config_manager = ConfigurationManager::new();
        
        let initial_config = config_manager.get_config();
        assert_eq!(initial_config.max_ships, 100);
        
        config_manager.update_config(|config| {
            config.max_ships = 200;
        });
        
        let updated_config = config_manager.get_config();
        assert_eq!(updated_config.max_ships, 200);
        
        config_manager.record_stat("test_stat".to_string(), 42.0);
        assert_eq!(config_manager.get_stat("test_stat"), 42.0);
    }

    #[test]
    fn test_message_dispatcher() {
        let (dispatcher, receiver) = MessageDispatcher::new();
        
        let message = SimulationMessage::SystemAlert {
            severity: AlertSeverity::Info,
            message: "Test alert".to_string(),
            timestamp: Instant::now(),
        };
        
        dispatcher.send_message(message).unwrap();
        
        let received = receiver.recv().unwrap();
        match received {
            SimulationMessage::SystemAlert { message, .. } => {
                assert_eq!(message, "Test alert");
            }
            _ => panic!("Wrong message type received"),
        }
    }

    #[tokio::test]
    async fn test_async_resource_manager() {
        let (manager, mut receiver) = AsyncResourceManager::new();
        
        let result = manager.fetch_resource(
            "test".to_string(),
            "http://example.com".to_string()
        ).await;
        
        assert!(result.is_ok());
        let resource = result.unwrap();
        assert_eq!(resource.id, "test");
        
        // Should receive update notification
        let notification = receiver.recv().await;
        assert!(notification.is_some());
        assert_eq!(notification.unwrap(), "test");
        
        // Should be able to retrieve from cache
        let cached = manager.get_resource("test").await;
        assert!(cached.is_some());
    }

    #[test]
    fn test_performance_counter() {
        let counter = PerformanceCounter::new();
        
        counter.start();
        assert_eq!(counter.get(), 0);
        
        let new_value = counter.increment();
        assert_eq!(new_value, 1);
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        std::thread::sleep(Duration::from_millis(100));
        assert!(counter.get_rate_per_second() > 0.0);
    }

    #[test] 
    fn test_work_queue() {
        let queue = WorkQueue::new();
        
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        
        queue.push("item1");
        queue.push("item2");
        
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);
        
        let item = queue.pop();
        assert_eq!(item, Some("item1"));
        assert_eq!(queue.len(), 1);
        
        let item = queue.pop();
        assert_eq!(item, Some("item2"));
        assert!(queue.is_empty());
        
        let empty = queue.pop();
        assert_eq!(empty, None);
    }
}