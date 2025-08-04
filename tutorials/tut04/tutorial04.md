# Tutorial 04: Concurrency Fundamentals

## Learning Objectives
- Master Rust's ownership-based concurrency model
- Understand threads, async/await, and the differences between them
- Learn safe data sharing with Arc, Mutex, and RwLock
- Implement producer-consumer patterns with channels
- Build concurrent resource management systems
- Apply async programming for I/O-bound operations
- Handle concurrent access patterns without data races

## Lesson: Concurrency in Rust

### What is Concurrency?

Concurrency is about dealing with multiple tasks at the same time. In programming, this means:
- **Multithreading**: Multiple threads executing simultaneously
- **Asynchronous programming**: Non-blocking operations that yield control
- **Parallelism**: Actually running tasks simultaneously on multiple cores

### Rust's Unique Approach: Fearless Concurrency

Rust's ownership system provides "fearless concurrency" - the ability to write concurrent code confidently without common pitfalls:

#### Traditional Concurrency Problems:
1. **Data races**: Multiple threads accessing data simultaneously
2. **Deadlocks**: Threads waiting for each other indefinitely
3. **Memory corruption**: Invalid memory access in concurrent contexts
4. **Use-after-free**: Accessing freed memory

#### Rust's Solutions:
1. **Ownership rules** prevent data races at compile time
2. **Type system** ensures thread safety
3. **Borrow checker** catches lifetime issues
4. **Send and Sync traits** control thread safety

### Core Concurrency Concepts

#### Send and Sync Traits
```rust
// Send: Type can be moved between threads
// Sync: Type can be shared between threads (via references)
pub unsafe trait Send {} // Most types implement this
pub unsafe trait Sync {} // Types safe to share references
```

#### Ownership in Concurrent Contexts
- **Move semantics**: Transfer ownership to threads
- **Arc<T>**: Atomic reference counting for shared ownership
- **Mutex<T>**: Mutual exclusion for mutable access
- **RwLock<T>**: Reader-writer lock for read-heavy workloads

### Threading vs Async Programming

#### When to Use Threads:
- **CPU-intensive tasks**
- **Parallel computation**
- **Independent processing**
- **When you need true parallelism**

#### When to Use Async:
- **I/O-bound operations**
- **Network requests**
- **File operations**
- **When you need high concurrency with low overhead**

### Memory Safety Guarantees

#### Compile-Time Checks:
```rust
// This won't compile - data race detected!
// let mut data = vec![1, 2, 3];
// thread::spawn(move || data.push(4)); // Moves data
// data.push(5); // Error: use after move
```

#### Safe Sharing Patterns:
```rust
// Safe sharing with Arc<Mutex<T>>
let data = Arc::new(Mutex::new(vec![1, 2, 3]));
let data_clone = data.clone();
thread::spawn(move || {
    let mut guard = data_clone.lock().unwrap();
    guard.push(4);
});
```

### Channel-Based Communication

Rust encourages "Don't communicate by sharing memory; share memory by communicating":

```rust
use std::sync::mpsc;

let (sender, receiver) = mpsc::channel();
thread::spawn(move || {
    sender.send("Hello from thread!").unwrap();
});
let message = receiver.recv().unwrap();
```

### Error Handling in Concurrent Code

#### Thread Panics:
- Threads can panic independently
- Use `JoinHandle` to detect panics
- Consider panic recovery strategies

#### Async Error Handling:
- Errors propagate through `Result` types
- Use `?` operator for error propagation
- Handle errors at appropriate levels

### Performance Considerations

#### Thread Overhead:
- Thread creation/destruction cost
- Context switching overhead
- Memory usage per thread (~2MB stack)

#### Async Overhead:
- State machine generation
- Polling mechanism
- Runtime scheduler overhead

#### Synchronization Costs:
- Mutex contention
- Cache coherency
- Memory barriers

### Space Simulation Applications

In our space simulation, concurrency enables:
- **Parallel entity processing**: Update entities simultaneously
- **I/O operations**: Network communication, file loading
- **Background tasks**: AI computation, pathfinding
- **Real-time response**: Handle user input while simulating
- **Resource management**: Concurrent access to shared resources

## Key Concepts

### 1. Rust's Fearless Concurrency

Rust's ownership system eliminates data races at compile time, enabling "fearless concurrency" where concurrent code is both safe and performant.

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

// Safe shared state using Arc<Mutex<T>>
#[derive(Debug, Default)]
pub struct ResourcePool {
    resources: Mutex<Vec<Resource>>,
    total_allocated: Mutex<usize>,
}

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

impl ResourcePool {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    
    pub fn add_resource(&self, resource: Resource) {
        let mut resources = self.resources.lock().unwrap();
        let mut total = self.total_allocated.lock().unwrap();
        
        resources.push(resource.clone());
        *total += resource.amount as usize;
        
        println!("Added resource {:?}, total allocated: {}", resource, *total);
    }
    
    pub fn get_resource(&self, resource_type: ResourceType) -> Option<Resource> {
        let mut resources = self.resources.lock().unwrap();
        
        if let Some(pos) = resources.iter().position(|r| r.resource_type == resource_type) {
            let resource = resources.remove(pos);
            println!("Retrieved resource: {:?}", resource);
            Some(resource)
        } else {
            None
        }
    }
    
    pub fn get_total_allocated(&self) -> usize {
        *self.total_allocated.lock().unwrap()
    }
}

// Multi-threaded resource management
pub fn demonstrate_thread_safety() {
    let pool = ResourcePool::new();
    let mut handles = vec![];
    
    // Spawn producer threads
    for i in 0..3 {
        let pool = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            for j in 0..5 {
                let resource = Resource {
                    id: i * 5 + j,
                    resource_type: match j % 4 {
                        0 => ResourceType::Fuel,
                        1 => ResourceType::Water,
                        2 => ResourceType::Food,
                        _ => ResourceType::Minerals,
                    },
                    amount: (i + j + 1) as f32 * 10.0,
                };
                
                pool.add_resource(resource);
                thread::sleep(Duration::from_millis(100));
            }
        });
        handles.push(handle);
    }
    
    // Spawn consumer threads
    for i in 0..2 {
        let pool = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            let resource_types = [ResourceType::Fuel, ResourceType::Water, ResourceType::Food, ResourceType::Minerals];
            
            for _ in 0..8 {
                let resource_type = resource_types[i % resource_types.len()].clone();
                if let Some(resource) = pool.get_resource(resource_type) {
                    println!("Consumer {} got: {:?}", i, resource);
                }
                thread::sleep(Duration::from_millis(150));
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final total allocated: {}", pool.get_total_allocated());
}
```

### 2. Reader-Writer Locks for Concurrent Access

RwLock enables multiple readers or single writer access patterns, perfect for data that's read frequently but written occasionally.

```rust
use std::sync::{Arc, RwLock};
use std::thread;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// Shared configuration that many threads read, few threads write
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
            physics_time_step: 0.016, // ~60 FPS
        }
    }
}

pub struct ConfigurationManager {
    config: Arc<RwLock<SimulationConfig>>,
    stats: Arc<RwLock<HashMap<String, f32>>>,
}

impl ConfigurationManager {
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(SimulationConfig::default())),
            stats: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    // Many threads can read simultaneously
    pub fn get_config(&self) -> SimulationConfig {
        let config = self.config.read().unwrap();
        config.clone()
    }
    
    // Only one thread can write at a time
    pub fn update_config<F>(&self, updater: F) 
    where 
        F: FnOnce(&mut SimulationConfig),
    {
        let mut config = self.config.write().unwrap();
        updater(&mut config);
        println!("Configuration updated: {:?}", *config);
    }
    
    // Collect statistics from multiple threads
    pub fn record_stat(&self, key: String, value: f32) {
        let mut stats = self.stats.write().unwrap();
        *stats.entry(key).or_insert(0.0) += value;
    }
    
    pub fn get_stat(&self, key: &str) -> f32 {
        let stats = self.stats.read().unwrap();
        stats.get(key).copied().unwrap_or(0.0)
    }
    
    pub fn get_all_stats(&self) -> HashMap<String, f32> {
        let stats = self.stats.read().unwrap();
        stats.clone()
    }
}

// Demonstrate concurrent access patterns
pub fn demonstrate_rwlock_usage() {
    let config_manager = Arc::new(ConfigurationManager::new());
    let mut handles = vec![];
    
    // Spawn reader threads (frequent reads)
    for i in 0..5 {
        let config_manager = Arc::clone(&config_manager);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let config = config_manager.get_config();
                
                // Simulate work using configuration
                let work_duration = Duration::from_millis((config.physics_time_step * 1000.0) as u64);
                thread::sleep(work_duration);
                
                // Record some statistics
                config_manager.record_stat(format!("thread_{}_iterations", i), 1.0);
                config_manager.record_stat("total_reads".to_string(), 1.0);
                
                if j % 3 == 0 {
                    println!("Thread {} read config: max_ships = {}", i, config.max_ships);
                }
            }
        });
        handles.push(handle);
    }
    
    // Spawn writer thread (occasional writes)
    let config_manager_writer = Arc::clone(&config_manager);
    let writer_handle = thread::spawn(move || {
        for i in 0..3 {
            thread::sleep(Duration::from_millis(200));
            
            config_manager_writer.update_config(|config| {
                config.max_ships += 10;
                config.resource_generation_rate *= 1.1;
                config.market_volatility = (config.market_volatility + 0.05).min(1.0);
            });
            
            config_manager_writer.record_stat("config_updates".to_string(), 1.0);
        }
    });
    handles.push(writer_handle);
    
    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Print final statistics
    println!("Final statistics: {:?}", config_manager.get_all_stats());
}
```

### 3. Channels for Message Passing

Channels enable safe communication between threads following the "Don't communicate by sharing memory; share memory by communicating" principle.

```rust
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

// Message types for space simulation
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

// Central message dispatcher
pub struct MessageDispatcher {
    sender: mpsc::Sender<SimulationMessage>,
}

impl MessageDispatcher {
    pub fn new() -> (Self, mpsc::Receiver<SimulationMessage>) {
        let (sender, receiver) = mpsc::channel();
        (Self { sender }, receiver)
    }
    
    pub fn send_message(&self, message: SimulationMessage) -> Result<(), mpsc::SendError<SimulationMessage>> {
        self.sender.send(message)
    }
    
    pub fn clone_sender(&self) -> mpsc::Sender<SimulationMessage> {
        self.sender.clone()
    }
}

// Message processing system
pub struct MessageProcessor;

impl MessageProcessor {
    pub fn process_messages(receiver: mpsc::Receiver<SimulationMessage>) {
        let mut message_count = 0;
        let mut ship_updates = 0;
        let mut trade_requests = 0;
        let mut alerts = 0;
        
        while let Ok(message) = receiver.recv() {
            match message {
                SimulationMessage::ShipStatusUpdate { ship_id, position, fuel_level, cargo_amount } => {
                    ship_updates += 1;
                    if ship_updates % 10 == 0 {
                        println!("Processed {} ship updates", ship_updates);
                    }
                    
                    // Check for low fuel alerts
                    if fuel_level < 0.2 {
                        println!("ALERT: Ship {} has low fuel: {:.1}%", ship_id, fuel_level * 100.0);
                    }
                }
                
                SimulationMessage::ResourceDiscovered { position, resource_type, amount } => {
                    println!("Resource discovered: {:?} at {:?}, amount: {}", 
                            resource_type, position, amount);
                }
                
                SimulationMessage::TradeRequest { from_ship, to_station, resource, amount } => {
                    trade_requests += 1;
                    println!("Trade request #{}: Ship {} wants to trade {} {:?} with Station {}", 
                            trade_requests, from_ship, amount, resource, to_station);
                }
                
                SimulationMessage::SystemAlert { severity, message, timestamp } => {
                    alerts += 1;
                    println!("[{:?}] Alert #{}: {} (at {:?})", 
                            severity, alerts, message, timestamp);
                }
                
                SimulationMessage::Shutdown => {
                    println!("Shutdown signal received. Processed {} messages total.", message_count);
                    break;
                }
            }
            
            message_count += 1;
        }
        
        println!("Final stats - Ships: {}, Trades: {}, Alerts: {}", 
                ship_updates, trade_requests, alerts);
    }
}

// Demonstrate channel-based communication
pub fn demonstrate_message_passing() {
    let (dispatcher, receiver) = MessageDispatcher::new();
    
    // Start message processor in separate thread
    let processor_handle = thread::spawn(move || {
        MessageProcessor::process_messages(receiver);
    });
    
    // Spawn ship simulation threads
    let mut ship_handles = vec![];
    
    for ship_id in 0..3 {
        let sender = dispatcher.clone_sender();
        let handle = thread::spawn(move || {
            let mut position = (ship_id as f32 * 100.0, 0.0, 0.0);
            let mut fuel_level = 1.0;
            let mut cargo_amount = 0.0;
            
            for update in 0..5 {
                // Simulate ship movement and resource consumption
                position.0 += 10.0;
                position.1 += (update as f32) * 5.0;
                fuel_level -= 0.1;
                cargo_amount += 15.0;
                
                // Send status update
                sender.send(SimulationMessage::ShipStatusUpdate {
                    ship_id,
                    position,
                    fuel_level,
                    cargo_amount,
                }).unwrap();
                
                // Occasionally discover resources
                if update % 2 == 0 {
                    sender.send(SimulationMessage::ResourceDiscovered {
                        position: (position.0 + 50.0, position.1, position.2),
                        resource_type: ResourceType::Minerals,
                        amount: 25.0 + (update as f32) * 5.0,
                    }).unwrap();
                }
                
                // Send trade requests
                if cargo_amount > 40.0 {
                    sender.send(SimulationMessage::TradeRequest {
                        from_ship: ship_id,
                        to_station: 1,
                        resource: ResourceType::Minerals,
                        amount: cargo_amount,
                    }).unwrap();
                    cargo_amount = 0.0; // Reset after trade
                }
                
                thread::sleep(Duration::from_millis(100));
            }
        });
        ship_handles.push(handle);
    }
    
    // Spawn alert system thread
    let alert_sender = dispatcher.clone_sender();
    let alert_handle = thread::spawn(move || {
        thread::sleep(Duration::from_millis(150));
        alert_sender.send(SimulationMessage::SystemAlert {
            severity: AlertSeverity::Warning,
            message: "High traffic detected in sector 7".to_string(),
            timestamp: Instant::now(),
        }).unwrap();
        
        thread::sleep(Duration::from_millis(300));
        alert_sender.send(SimulationMessage::SystemAlert {
            severity: AlertSeverity::Critical,
            message: "Asteroid collision imminent!".to_string(),
            timestamp: Instant::now(),
        }).unwrap();
    });
    
    // Wait for all ship threads
    for handle in ship_handles {
        handle.join().unwrap();
    }
    alert_handle.join().unwrap();
    
    // Send shutdown signal
    dispatcher.send_message(SimulationMessage::Shutdown).unwrap();
    
    // Wait for processor to finish
    processor_handle.join().unwrap();
}
```

### 4. Async/Await for I/O-Bound Operations

Async programming is ideal for I/O-bound operations like network communication and file handling.

```rust
use tokio::time::{sleep, Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use std::sync::Arc;
use std::collections::HashMap;

// Async resource manager for network-based operations
#[derive(Debug, Clone)]
pub struct NetworkResource {
    pub id: String,
    pub url: String,
    pub data: Vec<u8>,
    pub last_updated: Instant,
}

pub struct AsyncResourceManager {
    resources: Arc<Mutex<HashMap<String, NetworkResource>>>,
    update_sender: mpsc::UnboundedSender<String>,
}

impl AsyncResourceManager {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<String>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        (Self {
            resources: Arc::new(Mutex::new(HashMap::new())),
            update_sender: sender,
        }, receiver)
    }
    
    // Simulate fetching resource data from network
    pub async fn fetch_resource(&self, id: String, url: String) -> Result<NetworkResource, String> {
        println!("Fetching resource {} from {}", id, url);
        
        // Simulate network delay
        sleep(Duration::from_millis(100 + (id.len() * 50) as u64)).await;
        
        // Simulate occasional network failures
        if id.contains("error") {
            return Err(format!("Network error fetching {}", id));
        }
        
        let resource = NetworkResource {
            id: id.clone(),
            url,
            data: format!("Data for resource {}", id).into_bytes(),
            last_updated: Instant::now(),
        };
        
        // Store in cache
        {
            let mut resources = self.resources.lock().await;
            resources.insert(id.clone(), resource.clone());
        }
        
        // Notify update channel
        let _ = self.update_sender.send(id);
        
        Ok(resource)
    }
    
    pub async fn get_resource(&self, id: &str) -> Option<NetworkResource> {
        let resources = self.resources.lock().await;
        resources.get(id).cloned()
    }
    
    pub async fn get_all_resources(&self) -> Vec<NetworkResource> {
        let resources = self.resources.lock().await;
        resources.values().cloned().collect()
    }
    
    // Batch fetch multiple resources concurrently
    pub async fn fetch_batch(&self, requests: Vec<(String, String)>) -> Vec<Result<NetworkResource, String>> {
        let futures = requests.into_iter().map(|(id, url)| {
            self.fetch_resource(id, url)
        });
        
        // Execute all fetches concurrently
        futures::future::join_all(futures).await
    }
    
    // Periodic cleanup of old resources
    pub async fn cleanup_old_resources(&self, max_age: Duration) {
        let mut resources = self.resources.lock().await;
        let now = Instant::now();
        
        let old_resources: Vec<String> = resources
            .iter()
            .filter(|(_, resource)| now.duration_since(resource.last_updated) > max_age)
            .map(|(id, _)| id.clone())
            .collect();
        
        for id in &old_resources {
            resources.remove(id);
        }
        
        if !old_resources.is_empty() {
            println!("Cleaned up {} old resources", old_resources.len());
        }
    }
}

// Async simulation coordinator
pub struct SimulationCoordinator {
    resource_manager: AsyncResourceManager,
    update_receiver: mpsc::UnboundedReceiver<String>,
}

impl SimulationCoordinator {
    pub fn new() -> Self {
        let (resource_manager, update_receiver) = AsyncResourceManager::new();
        
        Self {
            resource_manager,
            update_receiver,
        }
    }
    
    pub async fn run_simulation(&mut self) {
        println!("Starting async simulation...");
        
        // Start background tasks
        let resource_manager = self.resource_manager.clone();
        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(2));
            
            loop {
                interval.tick().await;
                resource_manager.cleanup_old_resources(Duration::from_secs(5)).await;
            }
        });
        
        // Start resource fetching tasks
        let resource_manager = self.resource_manager.clone();
        let fetch_task = tokio::spawn(async move {
            // Simulate fetching ship configuration data
            let ship_configs = vec![
                ("ship_001".to_string(), "https://api.space.sim/ships/001".to_string()),
                ("ship_002".to_string(), "https://api.space.sim/ships/002".to_string()),
                ("ship_error".to_string(), "https://api.space.sim/ships/error".to_string()),
                ("station_alpha".to_string(), "https://api.space.sim/stations/alpha".to_string()),
            ];
            
            let results = resource_manager.fetch_batch(ship_configs).await;
            
            for (i, result) in results.iter().enumerate() {
                match result {
                    Ok(resource) => println!("Successfully fetched: {}", resource.id),
                    Err(e) => println!("Failed to fetch resource {}: {}", i, e),
                }
            }
            
            sleep(Duration::from_millis(500)).await;
            
            // Fetch market data
            let market_data = vec![
                ("market_prices".to_string(), "https://api.space.sim/market/prices".to_string()),
                ("trade_routes".to_string(), "https://api.space.sim/market/routes".to_string()),
            ];
            
            let _market_results = resource_manager.fetch_batch(market_data).await;
        });
        
        // Process update notifications
        let mut update_count = 0;
        let timeout = sleep(Duration::from_secs(3));
        tokio::pin!(timeout);
        
        loop {
            tokio::select! {
                Some(resource_id) = self.update_receiver.recv() => {
                    update_count += 1;
                    println!("Resource updated: {} (total updates: {})", resource_id, update_count);
                    
                    if update_count >= 6 { // Expect 6 updates (4 ships + 2 market)
                        break;
                    }
                }
                _ = &mut timeout => {
                    println!("Simulation timeout reached");
                    break;
                }
            }
        }
        
        // Wait for background tasks to complete
        cleanup_task.abort();
        let _ = fetch_task.await;
        
        // Print final state
        let all_resources = self.resource_manager.get_all_resources().await;
        println!("Simulation completed. Total resources cached: {}", all_resources.len());
        
        for resource in all_resources {
            println!("  - {}: {} bytes", resource.id, resource.data.len());
        }
    }
}

// Demonstrate async patterns
pub async fn demonstrate_async_operations() {
    let mut coordinator = SimulationCoordinator::new();
    coordinator.run_simulation().await;
}
```

### 5. Concurrent Data Structures and Patterns

Building thread-safe data structures for high-performance concurrent access.

```rust
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;

// Lock-free counter for performance metrics
pub struct PerformanceCounter {
    value: AtomicUsize,
    started: AtomicBool,
    start_time: Mutex<Option<Instant>>,
}

impl PerformanceCounter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            value: AtomicUsize::new(0),
            started: AtomicBool::new(false),
            start_time: Mutex::new(None),
        })
    }
    
    pub fn start(&self) {
        if self.started.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
            *self.start_time.lock().unwrap() = Some(Instant::now());
        }
    }
    
    pub fn increment(&self) -> usize {
        self.value.fetch_add(1, Ordering::SeqCst) + 1
    }
    
    pub fn add(&self, amount: usize) -> usize {
        self.value.fetch_add(amount, Ordering::SeqCst) + amount
    }
    
    pub fn get(&self) -> usize {
        self.value.load(Ordering::SeqCst)
    }
    
    pub fn get_rate_per_second(&self) -> f64 {
        if let Some(start_time) = *self.start_time.lock().unwrap() {
            let elapsed = start_time.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                self.get() as f64 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}

// Thread-safe work queue
pub struct WorkQueue<T> {
    queue: Mutex<VecDeque<T>>,
    pending_count: AtomicUsize,
}

impl<T> WorkQueue<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            queue: Mutex::new(VecDeque::new()),
            pending_count: AtomicUsize::new(0),
        })
    }
    
    pub fn push(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(item);
        self.pending_count.fetch_add(1, Ordering::SeqCst);
    }
    
    pub fn pop(&self) -> Option<T> {
        let mut queue = self.queue.lock().unwrap();
        if let Some(item) = queue.pop_front() {
            self.pending_count.fetch_sub(1, Ordering::SeqCst);
            Some(item)
        } else {
            None
        }
    }
    
    pub fn len(&self) -> usize {
        self.pending_count.load(Ordering::SeqCst)
    }
    
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        // Simulate work
        thread::sleep(self.estimated_duration);
    }
}

// Worker thread pool for concurrent task processing
pub struct WorkerPool {
    work_queue: Arc<WorkQueue<SimulationTask>>,
    task_counter: Arc<PerformanceCounter>,
    completed_counter: Arc<PerformanceCounter>,
    worker_handles: Vec<thread::JoinHandle<()>>,
}

impl WorkerPool {
    pub fn new(num_workers: usize) -> Self {
        let work_queue = WorkQueue::new();
        let task_counter = PerformanceCounter::new();
        let completed_counter = PerformanceCounter::new();
        
        task_counter.start();
        completed_counter.start();
        
        let mut worker_handles = Vec::new();
        
        for worker_id in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let completed_counter = Arc::clone(&completed_counter);
            
            let handle = thread::spawn(move || {
                println!("Worker {} started", worker_id);
                
                loop {
                    if let Some(task) = work_queue.pop() {
                        let start_time = Instant::now();
                        task.execute();
                        let execution_time = start_time.elapsed();
                        
                        completed_counter.increment();
                        
                        if task.task_id % 20 == 0 {
                            println!("Worker {} completed task {} ({:?}) in {:?}", 
                                   worker_id, task.task_id, task.task_type, execution_time);
                        }
                    } else {
                        // No work available, sleep briefly
                        thread::sleep(Duration::from_millis(10));
                        
                        // Check for shutdown condition
                        if work_queue.is_empty() {
                            // In a real system, you'd have a proper shutdown signal
                            break;
                        }
                    }
                }
                
                println!("Worker {} shutting down", worker_id);
            });
            
            worker_handles.push(handle);
        }
        
        Self {
            work_queue,
            task_counter,
            completed_counter,
            worker_handles,
        }
    }
    
    pub fn submit_task(&self, task: SimulationTask) {
        self.task_counter.increment();
        self.work_queue.push(task);
    }
    
    pub fn get_queue_length(&self) -> usize {
        self.work_queue.len()
    }
    
    pub fn get_task_stats(&self) -> (usize, usize, f64, f64) {
        (
            self.task_counter.get(),
            self.completed_counter.get(),
            self.task_counter.get_rate_per_second(),
            self.completed_counter.get_rate_per_second(),
        )
    }
    
    pub fn shutdown(self) {
        // In a real implementation, you'd send shutdown signals to workers
        for handle in self.worker_handles {
            let _ = handle.join();
        }
    }
}

// Demonstrate concurrent processing patterns
pub fn demonstrate_concurrent_processing() {
    println!("Starting concurrent processing demonstration...");
    
    let worker_pool = WorkerPool::new(4);
    
    // Generate tasks
    let task_types = [
        TaskType::UpdateShipPosition,
        TaskType::ProcessTradeRequest,
        TaskType::CalculatePhysics,
        TaskType::UpdateMarketPrices,
        TaskType::GenerateResources,
    ];
    
    // Submit initial batch of tasks
    for i in 0..100 {
        let task_type = task_types[i % task_types.len()].clone();
        let priority = if i % 10 == 0 { 3 } else { 1 }; // High priority every 10th task
        let task = SimulationTask::new(i, task_type, priority);
        worker_pool.submit_task(task);
    }
    
    // Monitor progress
    for second in 1..=5 {
        thread::sleep(Duration::from_secs(1));
        
        let (total_submitted, completed, submit_rate, completion_rate) = worker_pool.get_task_stats();
        let queue_length = worker_pool.get_queue_length();
        
        println!("Second {}: Submitted: {}, Completed: {}, Queue: {}, Rates: {:.1}/s submitted, {:.1}/s completed",
                second, total_submitted, completed, queue_length, submit_rate, completion_rate);
        
        // Submit more tasks periodically
        if second <= 3 {
            for i in 0..20 {
                let task_id = 100 + (second - 1) * 20 + i;
                let task_type = task_types[task_id % task_types.len()].clone();
                let task = SimulationTask::new(task_id, task_type, 1);
                worker_pool.submit_task(task);
            }
        }
    }
    
    // Wait for remaining tasks to complete
    while worker_pool.get_queue_length() > 0 {
        thread::sleep(Duration::from_millis(100));
        let (total_submitted, completed, _, completion_rate) = worker_pool.get_task_stats();
        println!("Finishing up... {} tasks remaining, completion rate: {:.1}/s", 
                total_submitted - completed, completion_rate);
    }
    
    let (total_submitted, completed, _, _) = worker_pool.get_task_stats();
    println!("All tasks completed! Total: {}, Completed: {}", total_submitted, completed);
    
    // Shutdown worker pool
    thread::sleep(Duration::from_millis(500)); // Give workers time to finish
    worker_pool.shutdown();
}
```

## Key Takeaways

1. **Ownership-Based Safety**: Rust's ownership system prevents data races at compile time
2. **Arc<Mutex<T>>**: Safe shared ownership with interior mutability for concurrent access
3. **RwLock**: Efficient read-heavy, write-occasional access patterns
4. **Channels**: Message passing for safe thread communication
5. **Async/Await**: Non-blocking I/O operations with cooperative multitasking
6. **Atomic Types**: Lock-free operations for high-performance counters and flags
7. **Thread Pools**: Efficient work distribution across multiple threads

## Best Practices

- Use `Arc<Mutex<T>>` for shared mutable state across threads
- Prefer `RwLock` when reads significantly outnumber writes
- Use channels for thread communication instead of shared memory when possible
- Choose async/await for I/O-bound operations, threads for CPU-bound work
- Atomic types for simple counters and flags to avoid lock overhead
- Design for minimal lock contention and short critical sections

## Performance Considerations

- Lock granularity affects concurrency: finer locks enable more parallelism
- Reader-writer locks improve performance for read-heavy workloads
- Atomic operations are faster than mutexes for simple operations
- Async tasks have lower overhead than threads for I/O-bound work
- Consider lock-free data structures for high-performance scenarios

## Next Steps

In the next tutorial, we'll explore message passing patterns in depth, building sophisticated actor-based systems and implementing the communication protocols needed for our distributed space simulation engine.