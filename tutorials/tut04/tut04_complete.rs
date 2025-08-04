// Tutorial 04: Concurrency Fundamentals - Complete Solutions
// This file contains the complete implementations for all exercises

use std::sync::{Arc, Mutex, RwLock, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use tokio::sync::mpsc as async_mpsc;

// Exercise 1: Thread-Safe Resource Pool - Complete Implementation
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

#[derive(Debug, Default)]
pub struct ResourcePool {
    resources: Mutex<Vec<Resource>>,
    total_allocated: Mutex<usize>,
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
    
    // Bonus: Get resource count by type
    pub fn get_resource_count_by_type(&self, resource_type: ResourceType) -> usize {
        let resources = self.resources.lock().unwrap();
        resources.iter().filter(|r| r.resource_type == resource_type).count()
    }
    
    // Bonus: Get all resources of a type without removing them
    pub fn peek_resources_by_type(&self, resource_type: ResourceType) -> Vec<Resource> {
        let resources = self.resources.lock().unwrap();
        resources.iter()
                 .filter(|r| r.resource_type == resource_type)
                 .cloned()
                 .collect()
    }
}

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

// Exercise 2: Configuration Management with RwLock - Complete Implementation
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
    
    pub fn get_config(&self) -> SimulationConfig {
        let config = self.config.read().unwrap();
        config.clone()
    }
    
    pub fn update_config<F>(&self, updater: F) 
    where 
        F: FnOnce(&mut SimulationConfig),
    {
        let mut config = self.config.write().unwrap();
        updater(&mut config);
        println!("Configuration updated: {:?}", *config);
    }
    
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
    
    // Bonus: Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        stats.clear();
    }
    
    // Bonus: Get config field safely
    pub fn get_max_ships(&self) -> usize {
        let config = self.config.read().unwrap();
        config.max_ships
    }
    
    // Bonus: Conditional config update
    pub fn update_config_if<F, P>(&self, predicate: P, updater: F) -> bool
    where 
        F: FnOnce(&mut SimulationConfig),
        P: FnOnce(&SimulationConfig) -> bool,
    {
        let config = self.config.read().unwrap();
        if predicate(&config) {
            drop(config); // Release read lock
            let mut config = self.config.write().unwrap();
            updater(&mut config);
            println!("Conditional configuration update applied: {:?}", *config);
            true
        } else {
            false
        }
    }
}

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

// Exercise 3: Message Passing with Channels - Complete Implementation
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

pub struct MessageProcessor;

impl MessageProcessor {
    pub fn process_messages(receiver: mpsc::Receiver<SimulationMessage>) {
        let mut message_count = 0;
        let mut ship_updates = 0;
        let mut trade_requests = 0;
        let mut alerts = 0;
        let mut resources_discovered = 0;
        
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
                    
                    // Check for cargo capacity alerts
                    if cargo_amount > 80.0 {
                        println!("INFO: Ship {} cargo nearly full: {:.1} units", ship_id, cargo_amount);
                    }
                }
                
                SimulationMessage::ResourceDiscovered { position, resource_type, amount } => {
                    resources_discovered += 1;
                    println!("Resource #{} discovered: {:?} at {:?}, amount: {}", 
                            resources_discovered, resource_type, position, amount);
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
        
        println!("Final stats - Ships: {}, Trades: {}, Alerts: {}, Resources: {}", 
                ship_updates, trade_requests, alerts, resources_discovered);
    }
}

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
        
        thread::sleep(Duration::from_millis(200));
        alert_sender.send(SimulationMessage::SystemAlert {
            severity: AlertSeverity::Info,
            message: "New trade route established".to_string(),
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

// Exercise 4: Async Resource Management - Complete Implementation
#[derive(Debug, Clone)]
pub struct NetworkResource {
    pub id: String,
    pub url: String,
    pub data: Vec<u8>,
    pub last_updated: Instant,
}

pub struct AsyncResourceManager {
    resources: Arc<tokio::sync::Mutex<HashMap<String, NetworkResource>>>,
    update_sender: async_mpsc::UnboundedSender<String>,
}

impl Clone for AsyncResourceManager {
    fn clone(&self) -> Self {
        Self {
            resources: Arc::clone(&self.resources),
            update_sender: self.update_sender.clone(),
        }
    }
}

impl AsyncResourceManager {
    pub fn new() -> (Self, async_mpsc::UnboundedReceiver<String>) {
        let (sender, receiver) = async_mpsc::unbounded_channel();
        
        (Self {
            resources: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            update_sender: sender,
        }, receiver)
    }
    
    pub async fn fetch_resource(&self, id: String, url: String) -> Result<NetworkResource, String> {
        println!("Fetching resource {} from {}", id, url);
        
        // Simulate network delay
        tokio::time::sleep(Duration::from_millis(100 + (id.len() * 50) as u64)).await;
        
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
    
    pub async fn fetch_batch(&self, requests: Vec<(String, String)>) -> Vec<Result<NetworkResource, String>> {
        let futures = requests.into_iter().map(|(id, url)| {
            self.fetch_resource(id, url)
        });
        
        // Execute all fetches concurrently
        futures::future::join_all(futures).await
    }
    
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
    
    // Bonus: Update resource data
    pub async fn update_resource_data(&self, id: &str, new_data: Vec<u8>) -> bool {
        let mut resources = self.resources.lock().await;
        if let Some(resource) = resources.get_mut(id) {
            resource.data = new_data;
            resource.last_updated = Instant::now();
            true
        } else {
            false
        }
    }
    
    // Bonus: Get resource metadata
    pub async fn get_resource_metadata(&self, id: &str) -> Option<(String, usize, Instant)> {
        let resources = self.resources.lock().await;
        resources.get(id).map(|r| (r.url.clone(), r.data.len(), r.last_updated))
    }
}

pub async fn demonstrate_async_operations() {
    println!("Starting async resource management demonstration...");
    
    let (resource_manager, mut update_receiver) = AsyncResourceManager::new();
    
    // Start background cleanup task
    let cleanup_manager = resource_manager.clone();
    let cleanup_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        let mut cleanup_count = 0;
        
        loop {
            interval.tick().await;
            cleanup_manager.cleanup_old_resources(Duration::from_secs(5)).await;
            cleanup_count += 1;
            
            if cleanup_count >= 3 { // Run cleanup 3 times then exit
                break;
            }
        }
    });
    
    // Start resource fetching tasks
    let fetch_manager = resource_manager.clone();
    let fetch_task = tokio::spawn(async move {
        // Simulate fetching ship configuration data
        let ship_configs = vec![
            ("ship_001".to_string(), "https://api.space.sim/ships/001".to_string()),
            ("ship_002".to_string(), "https://api.space.sim/ships/002".to_string()),
            ("ship_error".to_string(), "https://api.space.sim/ships/error".to_string()),
            ("station_alpha".to_string(), "https://api.space.sim/stations/alpha".to_string()),
        ];
        
        println!("Fetching ship configurations...");
        let results = fetch_manager.fetch_batch(ship_configs).await;
        
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(resource) => println!("Successfully fetched: {} ({} bytes)", resource.id, resource.data.len()),
                Err(e) => println!("Failed to fetch resource {}: {}", i, e),
            }
        }
        
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        // Fetch market data
        let market_data = vec![
            ("market_prices".to_string(), "https://api.space.sim/market/prices".to_string()),
            ("trade_routes".to_string(), "https://api.space.sim/market/routes".to_string()),
        ];
        
        println!("Fetching market data...");
        let market_results = fetch_manager.fetch_batch(market_data).await;
        
        for result in market_results {
            match result {
                Ok(resource) => println!("Market data fetched: {} ({} bytes)", resource.id, resource.data.len()),
                Err(e) => println!("Failed to fetch market data: {}", e),
            }
        }
    });
    
    // Process update notifications
    let notification_task = tokio::spawn(async move {
        let mut update_count = 0;
        let timeout = tokio::time::sleep(Duration::from_secs(4));
        tokio::pin!(timeout);
        
        loop {
            tokio::select! {
                Some(resource_id) = update_receiver.recv() => {
                    update_count += 1;
                    println!("Resource updated: {} (total updates: {})", resource_id, update_count);
                    
                    if update_count >= 6 { // Expect 6 updates (4 ships + 2 market)
                        break;
                    }
                }
                _ = &mut timeout => {
                    println!("Notification timeout reached");
                    break;
                }
            }
        }
        
        update_count
    });
    
    // Wait for all tasks to complete
    let (_, _, update_count) = tokio::join!(cleanup_task, fetch_task, notification_task);
    
    // Print final state
    let all_resources = resource_manager.get_all_resources().await;
    println!("Async operations completed. Updates processed: {}", update_count);
    println!("Total resources cached: {}", all_resources.len());
    
    for resource in all_resources {
        println!("  - {}: {} bytes (from {})", resource.id, resource.data.len(), resource.url);
    }
}

// Exercise 5: Lock-Free Performance Counter - Complete Implementation
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
    
    // Bonus: Reset counter
    pub fn reset(&self) {
        self.value.store(0, Ordering::SeqCst);
        self.started.store(false, Ordering::SeqCst);
        *self.start_time.lock().unwrap() = None;
    }
    
    // Bonus: Compare and swap
    pub fn compare_and_swap(&self, current: usize, new: usize) -> Result<usize, usize> {
        match self.value.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(prev) => Ok(prev),
            Err(actual) => Err(actual),
        }
    }
}

// Exercise 6: Thread-Safe Work Queue - Complete Implementation
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
    
    // Bonus: Push to front (priority)
    pub fn push_front(&self, item: T) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_front(item);
        self.pending_count.fetch_add(1, Ordering::SeqCst);
    }
    
    // Bonus: Peek at front item without removing
    pub fn peek(&self) -> Option<T> where T: Clone {
        let queue = self.queue.lock().unwrap();
        queue.front().cloned()
    }
    
    // Bonus: Drain all items
    pub fn drain(&self) -> Vec<T> {
        let mut queue = self.queue.lock().unwrap();
        let items: Vec<T> = queue.drain(..).collect();
        self.pending_count.store(0, Ordering::SeqCst);
        items
    }
}

// Work item implementation
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
    
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

pub struct WorkerPool {
    work_queue: Arc<WorkQueue<SimulationTask>>,
    task_counter: Arc<PerformanceCounter>,
    completed_counter: Arc<PerformanceCounter>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    shutdown_flag: Arc<AtomicBool>,
}

impl WorkerPool {
    pub fn new(num_workers: usize) -> Self {
        let work_queue = WorkQueue::new();
        let task_counter = PerformanceCounter::new();
        let completed_counter = PerformanceCounter::new();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        
        task_counter.start();
        completed_counter.start();
        
        let mut worker_handles = Vec::new();
        
        for worker_id in 0..num_workers {
            let work_queue = Arc::clone(&work_queue);
            let completed_counter = Arc::clone(&completed_counter);
            let shutdown_flag = Arc::clone(&shutdown_flag);
            
            let handle = thread::spawn(move || {
                println!("Worker {} started", worker_id);
                
                while !shutdown_flag.load(Ordering::SeqCst) {
                    if let Some(task) = work_queue.pop() {
                        let start_time = Instant::now();
                        let task_age = task.age();
                        task.execute();
                        let execution_time = start_time.elapsed();
                        
                        completed_counter.increment();
                        
                        if task.task_id % 20 == 0 {
                            println!("Worker {} completed task {} ({:?}) in {:?} (age: {:?})", 
                                   worker_id, task.task_id, task.task_type, execution_time, task_age);
                        }
                    } else {
                        // No work available, sleep briefly
                        thread::sleep(Duration::from_millis(10));
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
            shutdown_flag,
        }
    }
    
    pub fn submit_task(&self, task: SimulationTask) {
        self.task_counter.increment();
        if task.priority > 2 {
            self.work_queue.push_front(task); // High priority to front
        } else {
            self.work_queue.push(task);
        }
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
        self.shutdown_flag.store(true, Ordering::SeqCst);
        
        for handle in self.worker_handles {
            let _ = handle.join();
        }
    }
}

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
    println!("Submitting initial batch of 100 tasks...");
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
                let priority = if task_id % 15 == 0 { 3 } else { 1 }; // Some high priority tasks
                let task = SimulationTask::new(task_id, task_type, priority);
                worker_pool.submit_task(task);
            }
            println!("  Submitted 20 additional tasks");
        }
    }
    
    // Wait for remaining tasks to complete
    println!("Waiting for remaining tasks to complete...");
    while worker_pool.get_queue_length() > 0 {
        thread::sleep(Duration::from_millis(100));
        let (total_submitted, completed, _, completion_rate) = worker_pool.get_task_stats();
        let remaining = total_submitted - completed;
        if remaining > 0 && remaining % 10 == 0 {
            println!("  {} tasks remaining, completion rate: {:.1}/s", remaining, completion_rate);
        }
    }
    
    // Final statistics
    let (total_submitted, completed, submit_rate, completion_rate) = worker_pool.get_task_stats();
    println!("All tasks completed!");
    println!("Final statistics:");
    println!("  Total submitted: {}", total_submitted);
    println!("  Total completed: {}", completed);
    println!("  Average submit rate: {:.1}/s", submit_rate);
    println!("  Average completion rate: {:.1}/s", completion_rate);
    
    // Shutdown worker pool
    thread::sleep(Duration::from_millis(100)); // Give workers time to finish current tasks
    worker_pool.shutdown();
    println!("Worker pool shut down successfully");
}

// Comprehensive test suite
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_resource_pool_complete() {
        let pool = ResourcePool::new();
        
        let resource = Resource {
            id: 1,
            resource_type: ResourceType::Fuel,
            amount: 100.0,
        };
        
        pool.add_resource(resource.clone());
        assert_eq!(pool.get_total_allocated(), 100);
        assert_eq!(pool.get_resource_count_by_type(ResourceType::Fuel), 1);
        
        let retrieved = pool.get_resource(ResourceType::Fuel);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, 1);
        assert_eq!(pool.get_resource_count_by_type(ResourceType::Fuel), 0);
        
        let empty = pool.get_resource(ResourceType::Water);
        assert!(empty.is_none());
    }

    #[test]
    fn test_resource_pool_concurrent() {
        let pool = ResourcePool::new();
        let mut handles = vec![];
        
        // Spawn producer threads
        for i in 0..2 {
            let pool = Arc::clone(&pool);
            let handle = thread::spawn(move || {
                for j in 0..3 {
                    let resource = Resource {
                        id: i * 3 + j,
                        resource_type: ResourceType::Fuel,
                        amount: 10.0,
                    };
                    pool.add_resource(resource);
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(pool.get_resource_count_by_type(ResourceType::Fuel), 6);
        assert_eq!(pool.get_total_allocated(), 60);
    }

    #[test]
    fn test_configuration_manager_complete() {
        let config_manager = ConfigurationManager::new();
        
        let initial_config = config_manager.get_config();
        assert_eq!(initial_config.max_ships, 100);
        
        config_manager.update_config(|config| {
            config.max_ships = 200;
        });
        
        let updated_config = config_manager.get_config();
        assert_eq!(updated_config.max_ships, 200);
        
        config_manager.record_stat("test_stat".to_string(), 42.0);
        config_manager.record_stat("test_stat".to_string(), 8.0);
        assert_eq!(config_manager.get_stat("test_stat"), 50.0);
        
        // Test conditional update
        let updated = config_manager.update_config_if(
            |config| config.max_ships > 150,
            |config| config.max_ships = 300
        );
        assert!(updated);
        assert_eq!(config_manager.get_max_ships(), 300);
    }

    #[test]
    fn test_message_dispatcher_complete() {
        let (dispatcher, receiver) = MessageDispatcher::new();
        
        let message = SimulationMessage::SystemAlert {
            severity: AlertSeverity::Info,
            message: "Test alert".to_string(),
            timestamp: Instant::now(),
        };
        
        dispatcher.send_message(message).unwrap();
        
        // Test cloned sender
        let cloned_sender = dispatcher.clone_sender();
        let trade_message = SimulationMessage::TradeRequest {
            from_ship: 1,
            to_station: 2,
            resource: ResourceType::Fuel,
            amount: 100.0,
        };
        cloned_sender.send(trade_message).unwrap();
        
        // Receive both messages
        let received1 = receiver.recv().unwrap();
        let received2 = receiver.recv().unwrap();
        
        match received1 {
            SimulationMessage::SystemAlert { message, .. } => {
                assert_eq!(message, "Test alert");
            }
            _ => panic!("Wrong message type received"),
        }
        
        match received2 {
            SimulationMessage::TradeRequest { from_ship, .. } => {
                assert_eq!(from_ship, 1);
            }
            _ => panic!("Wrong message type received"),
        }
    }

    #[tokio::test]
    async fn test_async_resource_manager_complete() {
        let (manager, mut receiver) = AsyncResourceManager::new();
        
        let result = manager.fetch_resource(
            "test".to_string(),
            "http://example.com".to_string()
        ).await;
        
        assert!(result.is_ok());
        let resource = result.unwrap();
        assert_eq!(resource.id, "test");
        assert_eq!(resource.url, "http://example.com");
        
        // Should receive update notification
        let notification = tokio::time::timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(notification.is_ok());
        assert_eq!(notification.unwrap().unwrap(), "test");
        
        // Should be able to retrieve from cache
        let cached = manager.get_resource("test").await;
        assert!(cached.is_some());
        
        // Test metadata
        let metadata = manager.get_resource_metadata("test").await;
        assert!(metadata.is_some());
        let (url, size, _) = metadata.unwrap();
        assert_eq!(url, "http://example.com");
        assert!(size > 0);
        
        // Test error case
        let error_result = manager.fetch_resource(
            "error_test".to_string(),
            "http://error.com".to_string()
        ).await;
        assert!(error_result.is_err());
    }

    #[tokio::test]
    async fn test_async_batch_fetch() {
        let (manager, _) = AsyncResourceManager::new();
        
        let requests = vec![
            ("resource1".to_string(), "http://api.com/1".to_string()),
            ("resource2".to_string(), "http://api.com/2".to_string()),
            ("error_resource".to_string(), "http://api.com/error".to_string()),
        ];
        
        let results = manager.fetch_batch(requests).await;
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
    }

    #[test]
    fn test_performance_counter_complete() {
        let counter = PerformanceCounter::new();
        
        counter.start();
        assert_eq!(counter.get(), 0);
        
        let new_value = counter.increment();
        assert_eq!(new_value, 1);
        assert_eq!(counter.get(), 1);
        
        counter.add(5);
        assert_eq!(counter.get(), 6);
        
        // Test compare and swap
        let result = counter.compare_and_swap(6, 10);
        assert!(result.is_ok());
        assert_eq!(counter.get(), 10);
        
        let failed_result = counter.compare_and_swap(6, 15);
        assert!(failed_result.is_err());
        assert_eq!(failed_result.unwrap_err(), 10);
        
        thread::sleep(Duration::from_millis(100));
        assert!(counter.get_rate_per_second() > 0.0);
    }

    #[test] 
    fn test_work_queue_complete() {
        let queue = WorkQueue::new();
        
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        
        queue.push("item1");
        queue.push("item2");
        
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);
        
        // Test peek
        let peeked = queue.peek();
        assert_eq!(peeked, Some("item1"));
        assert_eq!(queue.len(), 2); // Should not change length
        
        let item = queue.pop();
        assert_eq!(item, Some("item1"));
        assert_eq!(queue.len(), 1);
        
        // Test priority push
        queue.push_front("priority_item");
        let next_item = queue.pop();
        assert_eq!(next_item, Some("priority_item"));
        
        // Test drain
        queue.push("item3");
        queue.push("item4");
        let all_items = queue.drain();
        assert_eq!(all_items.len(), 3); // item2, item3, item4
        assert!(queue.is_empty());
    }

    #[test]
    fn test_worker_pool_basic() {
        let worker_pool = WorkerPool::new(2);
        
        let task = SimulationTask::new(1, TaskType::UpdateShipPosition, 1);
        worker_pool.submit_task(task);
        
        // Wait a bit for task to be processed
        thread::sleep(Duration::from_millis(100));
        
        let (submitted, completed, _, _) = worker_pool.get_task_stats();
        assert_eq!(submitted, 1);
        assert!(completed <= 1); // May or may not be completed yet
        
        worker_pool.shutdown();
    }
}