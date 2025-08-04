# Tutorial 05: Message Passing

## Learning Objectives
- Master advanced message passing patterns in Rust
- Implement actor-based systems for distributed simulation
- Build reliable communication protocols with error handling
- Design scalable message routing and dispatching systems
- Apply backpressure and flow control mechanisms
- Create type-safe message protocols for different entity types
- Implement publish-subscribe patterns for event distribution

## Lesson: Message Passing in Rust

### What is Message Passing?

Message passing is a communication paradigm where components interact by sending messages rather than sharing memory. This approach:
- **Eliminates data races** by avoiding shared mutable state
- **Provides clear communication boundaries** between components
- **Enables distributed systems** where components run on different machines
- **Simplifies reasoning** about concurrent behavior

### The Actor Model

The actor model is a conceptual model for concurrent computation where:
- **Actors** are independent computational units
- **Messages** are the only way actors communicate
- **Mailboxes** queue incoming messages for processing
- **Behavior** defines how actors respond to messages

#### Core Actor Principles:
1. **Isolation**: Actors don't share state
2. **Asynchronous**: Message sending doesn't block
3. **Location transparency**: Actors can be local or remote
4. **Fault tolerance**: Actor failures don't crash the system

### Rust's Message Passing Advantages

#### Type Safety:
- **Compile-time verification** of message types
- **Pattern matching** for exhaustive message handling
- **Enum variants** for type-safe message protocols

#### Performance:
- **Zero-copy messaging** where possible
- **Efficient channels** with minimal overhead
- **Lock-free algorithms** in channel implementations

#### Safety:
- **No data races** due to ownership transfer
- **Memory safety** without garbage collection
- **Structured concurrency** with clear lifetimes

### Channel Types in Rust

#### MPSC (Multi-Producer, Single-Consumer):
```rust
use std::sync::mpsc;
let (sender, receiver) = mpsc::channel();
// Multiple senders, one receiver
```

#### Tokio Channels:
```rust
use tokio::sync::mpsc;
let (sender, mut receiver) = mpsc::channel(100); // Bounded
// Async-friendly channels
```

#### Oneshot Channels:
```rust
use tokio::sync::oneshot;
let (sender, receiver) = oneshot::channel();
// Single message, request-response pattern
```

### Message Passing Patterns

#### 1. Fire-and-Forget:
- Send message without waiting for response
- Suitable for notifications and events

#### 2. Request-Response:
- Send message and wait for reply
- Uses oneshot channels for responses

#### 3. Publish-Subscribe:
- One sender, multiple receivers
- Broadcast channels or topic-based routing

#### 4. Pipeline:
- Chain of processing stages
- Each stage transforms and forwards messages

### Advanced Message Passing Concepts

#### Backpressure:
- **Problem**: Fast producers overwhelm slow consumers
- **Solution**: Bounded channels that block when full
- **Alternative**: Drop messages or apply flow control

#### Message Ordering:
- **FIFO**: First-in, first-out within single channel
- **Priority**: Some messages processed before others
- **Causal**: Maintain causal relationships between messages

#### Reliability:
- **At-most-once**: Message delivered zero or one time
- **At-least-once**: Message delivered one or more times
- **Exactly-once**: Message delivered exactly one time (expensive)

#### Flow Control:
- **Buffering**: Queue messages temporarily
- **Rate limiting**: Control message send rate
- **Credit-based**: Receiver grants permission to send

### Error Handling in Message Passing

#### Channel Errors:
- **Send errors**: Receiver dropped, channel full
- **Receive errors**: All senders dropped
- **Timeout errors**: Operation didn't complete in time

#### Message Errors:
- **Serialization**: Failed to encode/decode messages
- **Network**: Messages lost or corrupted in transit
- **Processing**: Actor failed to handle message

### Performance Considerations

#### Message Size:
- **Small messages**: Low latency, high throughput
- **Large messages**: Consider reference passing
- **Batching**: Group small messages together

#### Channel Capacity:
- **Unbounded**: Risk of memory exhaustion
- **Bounded**: Risk of blocking or dropping
- **Adaptive**: Adjust capacity based on load

### Space Simulation Applications

Message passing is ideal for:
- **Entity communication**: Ships talking to stations
- **Event systems**: Broadcast simulation events
- **Network protocols**: Client-server communication
- **AI systems**: Behavior trees and decision making
- **Resource management**: Coordinate resource allocation

### When to Use Message Passing

#### Good for:
- **Distributed systems**
- **Clear component boundaries**
- **Event-driven architectures**
- **Fault-tolerant systems**

#### Consider alternatives for:
- **High-frequency, low-latency communication**
- **Shared read-only data**
- **Simple sequential processing**

## Key Concepts

### 1. Actor Model Fundamentals

The actor model provides a powerful abstraction for concurrent systems where entities communicate exclusively through message passing.

```rust
use tokio::sync::{mpsc, oneshot};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

// Core message types for space simulation actors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActorMessage {
    // Ship-to-Ship communication
    ShipToShip {
        from: ActorId,
        to: ActorId,
        content: ShipMessage,
    },
    // Ship-to-Station communication  
    ShipToStation {
        from: ActorId,
        to: ActorId,
        content: StationMessage,
    },
    // System-wide broadcasts
    SystemBroadcast {
        sender: ActorId,
        content: SystemMessage,
    },
    // Actor lifecycle management
    ActorControl {
        target: ActorId,
        command: ControlCommand,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShipMessage {
    RequestDocking {
        ship_specs: ShipSpecs,
        cargo_manifest: Vec<CargoItem>,
    },
    FormationRequest {
        formation_type: FormationType,
        position: (f32, f32, f32),
    },
    TradeOffer {
        offering: Vec<CargoItem>,
        requesting: Vec<ResourceRequest>,
    },
    DistressSignal {
        emergency_type: EmergencyType,
        location: (f32, f32, f32),
        severity: u8,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StationMessage {
    DockingPermission {
        granted: bool,
        berth_assignment: Option<u32>,
        estimated_wait: Duration,
    },
    MarketPrices {
        commodities: HashMap<String, f32>,
        last_updated: Instant,
    },
    ServiceAvailability {
        repairs: bool,
        refueling: bool,
        crew_quarters: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemMessage {
    NavigationHazard {
        location: (f32, f32, f32),
        radius: f32,
        hazard_type: HazardType,
    },
    MarketUpdate {
        sector: String,
        price_changes: HashMap<String, f32>,
    },
    WeatherAlert {
        affected_regions: Vec<String>,
        severity: WeatherSeverity,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    Shutdown,
    Pause,
    Resume,
    GetStatus,
}

// Supporting types
pub type ActorId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShipSpecs {
    pub ship_class: String,
    pub length: f32,
    pub mass: f32,
    pub cargo_capacity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoItem {
    pub item_type: String,
    pub quantity: f32,
    pub unit_value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub resource_type: String,
    pub quantity: f32,
    pub max_price: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FormationType {
    Convoy,
    Defensive,
    Exploration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmergencyType {
    EngineFailure,
    HullBreach,
    CrewMedical,
    Pirate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HazardType {
    Asteroid,
    Debris,
    EnergyStorm,
    Pirates,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeatherSeverity {
    Low,
    Moderate,
    High,
    Extreme,
}
```

### 2. Actor Implementation with Typed Channels

```rust
// Base actor trait defining common behavior
#[async_trait::async_trait]
pub trait Actor {
    type Message: Send + 'static;
    type State: Send + 'static;
    
    async fn handle_message(&mut self, message: Self::Message, state: &mut Self::State) -> Result<(), ActorError>;
    async fn on_start(&mut self, _state: &mut Self::State) -> Result<(), ActorError> { Ok(()) }
    async fn on_stop(&mut self, _state: &mut Self::State) -> Result<(), ActorError> { Ok(()) }
}

#[derive(Debug, thiserror::Error)]
pub enum ActorError {
    #[error("Message handling failed: {0}")]
    MessageHandling(String),
    #[error("Actor is shutting down")]
    ShuttingDown,
    #[error("Invalid state transition: {0}")]
    InvalidState(String),
    #[error("Communication error: {0}")]
    Communication(String),
}

// Actor handle for external communication
pub struct ActorHandle<M> {
    sender: mpsc::UnboundedSender<M>,
    actor_id: ActorId,
}

impl<M> Clone for ActorHandle<M> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            actor_id: self.actor_id,
        }
    }
}

impl<M> ActorHandle<M> {
    pub fn new(sender: mpsc::UnboundedSender<M>, actor_id: ActorId) -> Self {
        Self { sender, actor_id }
    }
    
    pub fn send(&self, message: M) -> Result<(), mpsc::error::SendError<M>> {
        self.sender.send(message)
    }
    
    pub fn actor_id(&self) -> ActorId {
        self.actor_id
    }
    
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

// Actor context providing communication capabilities
pub struct ActorContext {
    pub actor_id: ActorId,
    pub message_router: MessageRouter,
}

impl ActorContext {
    pub async fn send_to_actor<M>(&self, target: ActorId, message: M) -> Result<(), ActorError>
    where
        M: Into<ActorMessage> + Send + 'static,
    {
        self.message_router
            .route_message(target, message.into())
            .await
            .map_err(|e| ActorError::Communication(e.to_string()))
    }
    
    pub async fn broadcast<M>(&self, message: M) -> Result<(), ActorError>
    where
        M: Into<ActorMessage> + Send + 'static,
    {
        self.message_router
            .broadcast_message(message.into())
            .await
            .map_err(|e| ActorError::Communication(e.to_string()))
    }
}

// Ship actor implementation
pub struct ShipActor {
    context: ActorContext,
    position: (f32, f32, f32),
    velocity: (f32, f32, f32),
    fuel_level: f32,
    cargo: Vec<CargoItem>,
    state: ShipState,
}

#[derive(Debug, Clone)]
pub enum ShipState {
    Idle,
    Flying { destination: (f32, f32, f32) },
    Docking { station_id: ActorId },
    Docked { station_id: ActorId },
    Trading,
    Emergency { reason: EmergencyType },
}

#[derive(Debug)]
pub enum ShipActorMessage {
    UpdatePosition { dt: f32 },
    IncomingMessage { message: ActorMessage },
    SetDestination { target: (f32, f32, f32) },
    InitiateDocking { station_id: ActorId },
    ProcessTradeOffer { offer: Vec<CargoItem> },
    HandleEmergency { emergency: EmergencyType },
}

#[async_trait::async_trait]
impl Actor for ShipActor {
    type Message = ShipActorMessage;
    type State = ShipState;
    
    async fn handle_message(&mut self, message: Self::Message, state: &mut Self::State) -> Result<(), ActorError> {
        match message {
            ShipActorMessage::UpdatePosition { dt } => {
                self.update_physics(dt);
                
                // Check if we've reached destination
                if let ShipState::Flying { destination } = state {
                    let distance = self.distance_to(*destination);
                    if distance < 10.0 {
                        *state = ShipState::Idle;
                        println!("Ship {} reached destination", self.context.actor_id);
                    }
                }
            }
            
            ShipActorMessage::IncomingMessage { message } => {
                self.handle_incoming_message(message, state).await?;
            }
            
            ShipActorMessage::SetDestination { target } => {
                *state = ShipState::Flying { destination: target };
                println!("Ship {} setting course to {:?}", self.context.actor_id, target);
            }
            
            ShipActorMessage::InitiateDocking { station_id } => {
                // Send docking request
                let ship_specs = ShipSpecs {
                    ship_class: "Merchant".to_string(),
                    length: 150.0,
                    mass: 2000.0,
                    cargo_capacity: 500.0,
                };
                
                let docking_request = ActorMessage::ShipToStation {
                    from: self.context.actor_id,
                    to: station_id,
                    content: StationMessage::DockingPermission {
                        granted: false, // This is a request, not a response
                        berth_assignment: None,
                        estimated_wait: Duration::from_secs(0),
                    },
                };
                
                self.context.send_to_actor(station_id, docking_request).await?;
                *state = ShipState::Docking { station_id };
            }
            
            ShipActorMessage::ProcessTradeOffer { offer } => {
                // Evaluate trade offer
                let total_value: f32 = offer.iter().map(|item| item.quantity * item.unit_value).sum();
                println!("Ship {} received trade offer worth {:.2} credits", 
                        self.context.actor_id, total_value);
                
                // Accept if valuable enough
                if total_value > 1000.0 {
                    // Add to cargo
                    self.cargo.extend(offer);
                    *state = ShipState::Trading;
                }
            }
            
            ShipActorMessage::HandleEmergency { emergency } => {
                *state = ShipState::Emergency { reason: emergency.clone() };
                
                // Broadcast distress signal
                let distress = ActorMessage::SystemBroadcast {
                    sender: self.context.actor_id,
                    content: SystemMessage::NavigationHazard {
                        location: self.position,
                        radius: 100.0,
                        hazard_type: HazardType::Pirates, // Simplified
                    },
                };
                
                self.context.broadcast(distress).await?;
                println!("Ship {} broadcasting emergency: {:?}", self.context.actor_id, emergency);
            }
        }
        
        Ok(())
    }
}

impl ShipActor {
    pub fn new(actor_id: ActorId, message_router: MessageRouter) -> Self {
        Self {
            context: ActorContext { actor_id, message_router },
            position: (0.0, 0.0, 0.0),
            velocity: (0.0, 0.0, 0.0),
            fuel_level: 1.0,
            cargo: Vec::new(),
            state: ShipState::Idle,
        }
    }
    
    fn update_physics(&mut self, dt: f32) {
        self.position.0 += self.velocity.0 * dt;
        self.position.1 += self.velocity.1 * dt;
        self.position.2 += self.velocity.2 * dt;
        
        // Consume fuel based on velocity
        let speed = (self.velocity.0.powi(2) + self.velocity.1.powi(2) + self.velocity.2.powi(2)).sqrt();
        self.fuel_level -= speed * dt * 0.01;
        self.fuel_level = self.fuel_level.max(0.0);
    }
    
    fn distance_to(&self, target: (f32, f32, f32)) -> f32 {
        let dx = target.0 - self.position.0;
        let dy = target.1 - self.position.1;
        let dz = target.2 - self.position.2;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    async fn handle_incoming_message(&mut self, message: ActorMessage, state: &mut ShipState) -> Result<(), ActorError> {
        match message {
            ActorMessage::ShipToShip { from, content, .. } => {
                match content {
                    ShipMessage::FormationRequest { formation_type, position } => {
                        println!("Ship {} received formation request from {}: {:?} at {:?}", 
                                self.context.actor_id, from, formation_type, position);
                        
                        // Accept formation and move to position
                        self.velocity = (
                            (position.0 - self.position.0) * 0.1,
                            (position.1 - self.position.1) * 0.1,
                            (position.2 - self.position.2) * 0.1,
                        );
                    }
                    
                    ShipMessage::TradeOffer { offering, .. } => {
                        // Process trade offer
                        let total_value: f32 = offering.iter().map(|item| item.quantity * item.unit_value).sum();
                        if total_value > 500.0 {
                            self.cargo.extend(offering);
                            println!("Ship {} accepted trade offer worth {:.2}", self.context.actor_id, total_value);
                        }
                    }
                    
                    ShipMessage::DistressSignal { emergency_type, location, severity } => {
                        println!("Ship {} received distress signal: {:?} at {:?} (severity {})", 
                                self.context.actor_id, emergency_type, location, severity);
                        
                        // If we're nearby and able to help
                        let distance = self.distance_to(location);
                        if distance < 200.0 && matches!(state, ShipState::Idle) {
                            *state = ShipState::Flying { destination: location };
                            println!("Ship {} responding to distress call", self.context.actor_id);
                        }
                    }
                    
                    _ => {}
                }
            }
            
            ActorMessage::SystemBroadcast { content, .. } => {
                match content {
                    SystemMessage::NavigationHazard { location, radius, hazard_type } => {
                        let distance = self.distance_to(location);
                        if distance < radius {
                            println!("Ship {} detected hazard: {:?} at distance {:.1}", 
                                    self.context.actor_id, hazard_type, distance);
                            
                            // Avoid hazard
                            self.velocity = (
                                (self.position.0 - location.0) * 0.05,
                                (self.position.1 - location.1) * 0.05,
                                (self.position.2 - location.2) * 0.05,
                            );
                        }
                    }
                    
                    SystemMessage::MarketUpdate { sector, price_changes } => {
                        println!("Ship {} received market update for {}: {} commodities", 
                                self.context.actor_id, sector, price_changes.len());
                    }
                    
                    _ => {}
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
}
```

### 3. Message Router and Dispatch System

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

// Central message routing system
pub struct MessageRouter {
    actors: Arc<RwLock<HashMap<ActorId, mpsc::UnboundedSender<ActorMessage>>>>,
    subscribers: Arc<RwLock<HashMap<String, Vec<ActorId>>>>, // Topic-based subscriptions
    message_stats: Arc<RwLock<MessageStatistics>>,
}

#[derive(Debug, Default)]
pub struct MessageStatistics {
    pub total_messages: u64,
    pub messages_by_type: HashMap<String, u64>,
    pub failed_deliveries: u64,
    pub active_actors: u64,
}

impl Clone for MessageRouter {
    fn clone(&self) -> Self {
        Self {
            actors: Arc::clone(&self.actors),
            subscribers: Arc::clone(&self.subscribers),
            message_stats: Arc::clone(&self.message_stats),
        }
    }
}

impl MessageRouter {
    pub fn new() -> Self {
        Self {
            actors: Arc::new(RwLock::new(HashMap::new())),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            message_stats: Arc::new(RwLock::new(MessageStatistics::default())),
        }
    }
    
    pub async fn register_actor(&self, actor_id: ActorId, sender: mpsc::UnboundedSender<ActorMessage>) {
        let mut actors = self.actors.write().await;
        actors.insert(actor_id, sender);
        
        let mut stats = self.message_stats.write().await;
        stats.active_actors = actors.len() as u64;
        
        println!("Registered actor {}, total actors: {}", actor_id, actors.len());
    }
    
    pub async fn unregister_actor(&self, actor_id: ActorId) {
        let mut actors = self.actors.write().await;
        actors.remove(&actor_id);
        
        let mut stats = self.message_stats.write().await;
        stats.active_actors = actors.len() as u64;
        
        println!("Unregistered actor {}, remaining actors: {}", actor_id, actors.len());
    }
    
    pub async fn route_message(&self, target: ActorId, message: ActorMessage) -> Result<(), RouterError> {
        let actors = self.actors.read().await;
        
        if let Some(sender) = actors.get(&target) {
            sender.send(message.clone())
                .map_err(|_| RouterError::ActorUnavailable(target))?;
            
            // Update statistics
            let mut stats = self.message_stats.write().await;
            stats.total_messages += 1;
            let message_type = Self::get_message_type(&message);
            *stats.messages_by_type.entry(message_type).or_insert(0) += 1;
            
            Ok(())
        } else {
            let mut stats = self.message_stats.write().await;
            stats.failed_deliveries += 1;
            Err(RouterError::ActorNotFound(target))
        }
    }
    
    pub async fn broadcast_message(&self, message: ActorMessage) -> Result<u32, RouterError> {
        let actors = self.actors.read().await;
        let mut delivered = 0;
        let mut failed = 0;
        
        for (actor_id, sender) in actors.iter() {
            match sender.send(message.clone()) {
                Ok(()) => delivered += 1,
                Err(_) => {
                    failed += 1;
                    println!("Failed to deliver broadcast to actor {}", actor_id);
                }
            }
        }
        
        // Update statistics
        let mut stats = self.message_stats.write().await;
        stats.total_messages += delivered as u64;
        stats.failed_deliveries += failed as u64;
        let message_type = Self::get_message_type(&message);
        *stats.messages_by_type.entry(message_type).or_insert(0) += delivered as u64;
        
        println!("Broadcast delivered to {} actors, {} failures", delivered, failed);
        Ok(delivered)
    }
    
    pub async fn subscribe_to_topic(&self, actor_id: ActorId, topic: String) {
        let mut subscribers = self.subscribers.write().await;
        subscribers.entry(topic.clone()).or_insert_with(Vec::new).push(actor_id);
        println!("Actor {} subscribed to topic '{}'", actor_id, topic);
    }
    
    pub async fn publish_to_topic(&self, topic: &str, message: ActorMessage) -> Result<u32, RouterError> {
        let subscribers = self.subscribers.read().await;
        let actors = self.actors.read().await;
        
        if let Some(subscriber_ids) = subscribers.get(topic) {
            let mut delivered = 0;
            
            for &actor_id in subscriber_ids {
                if let Some(sender) = actors.get(&actor_id) {
                    if sender.send(message.clone()).is_ok() {
                        delivered += 1;
                    }
                }
            }
            
            // Update statistics
            let mut stats = self.message_stats.write().await;
            stats.total_messages += delivered as u64;
            let message_type = Self::get_message_type(&message);
            *stats.messages_by_type.entry(message_type).or_insert(0) += delivered as u64;
            
            Ok(delivered)
        } else {
            Ok(0) // No subscribers
        }
    }
    
    pub async fn get_statistics(&self) -> MessageStatistics {
        let stats = self.message_stats.read().await;
        stats.clone()
    }
    
    fn get_message_type(message: &ActorMessage) -> String {
        match message {
            ActorMessage::ShipToShip { .. } => "ShipToShip".to_string(),
            ActorMessage::ShipToStation { .. } => "ShipToStation".to_string(),
            ActorMessage::SystemBroadcast { .. } => "SystemBroadcast".to_string(),
            ActorMessage::ActorControl { .. } => "ActorControl".to_string(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("Actor {0} not found")]
    ActorNotFound(ActorId),
    #[error("Actor {0} is unavailable")]
    ActorUnavailable(ActorId),
    #[error("No subscribers for topic")]
    NoSubscribers,
}
```

### 4. Reliable Message Delivery with Acknowledgments

```rust
use tokio::time::{timeout, Duration};

// Reliable message delivery system
pub struct ReliableMessenger {
    router: MessageRouter,
    pending_messages: Arc<RwLock<HashMap<MessageId, PendingMessage>>>,
    next_message_id: Arc<std::sync::atomic::AtomicU64>,
}

pub type MessageId = u64;

#[derive(Debug)]
struct PendingMessage {
    message: ActorMessage,
    target: ActorId,
    sent_at: Instant,
    retry_count: u8,
    max_retries: u8,
    timeout_duration: Duration,
}

#[derive(Debug, Clone)]
pub enum ReliableMessage {
    Regular(ActorMessage),
    Acknowledgment { message_id: MessageId },
    DeliveryConfirmation { message_id: MessageId, success: bool },
}

impl ReliableMessenger {
    pub fn new(router: MessageRouter) -> Self {
        Self {
            router,
            pending_messages: Arc::new(RwLock::new(HashMap::new())),
            next_message_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        }
    }
    
    pub async fn send_reliable(
        &self,
        target: ActorId,
        message: ActorMessage,
        max_retries: u8,
        timeout_duration: Duration,
    ) -> Result<MessageId, RouterError> {
        let message_id = self.next_message_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        let pending = PendingMessage {
            message: message.clone(),
            target,
            sent_at: Instant::now(),
            retry_count: 0,
            max_retries,
            timeout_duration,
        };
        
        // Store pending message
        {
            let mut pending_messages = self.pending_messages.write().await;
            pending_messages.insert(message_id, pending);
        }
        
        // Send initial message
        self.router.route_message(target, message).await?;
        
        Ok(message_id)
    }
    
    pub async fn handle_acknowledgment(&self, message_id: MessageId) {
        let mut pending_messages = self.pending_messages.write().await;
        if let Some(pending) = pending_messages.remove(&message_id) {
            println!("Message {} acknowledged by actor {} after {} retries", 
                    message_id, pending.target, pending.retry_count);
        }
    }
    
    // Background task to handle retries and timeouts
    pub async fn retry_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            self.process_pending_messages().await;
        }
    }
    
    async fn process_pending_messages(&self) {
        let mut to_retry = Vec::new();
        let mut to_remove = Vec::new();
        
        // Check pending messages
        {
            let pending_messages = self.pending_messages.read().await;
            let now = Instant::now();
            
            for (&message_id, pending) in pending_messages.iter() {
                if now.duration_since(pending.sent_at) > pending.timeout_duration {
                    if pending.retry_count < pending.max_retries {
                        to_retry.push((message_id, pending.clone()));
                    } else {
                        to_remove.push(message_id);
                        println!("Message {} to actor {} failed after {} retries", 
                                message_id, pending.target, pending.max_retries);
                    }
                }
            }
        }
        
        // Remove failed messages
        if !to_remove.is_empty() {
            let mut pending_messages = self.pending_messages.write().await;
            for message_id in to_remove {
                pending_messages.remove(&message_id);
            }
        }
        
        // Retry messages
        for (message_id, mut pending) in to_retry {
            pending.retry_count += 1;
            pending.sent_at = Instant::now();
            
            if let Ok(()) = self.router.route_message(pending.target, pending.message.clone()).await {
                let mut pending_messages = self.pending_messages.write().await;
                pending_messages.insert(message_id, pending);
                println!("Retrying message {} (attempt {})", message_id, pending.retry_count + 1);
            }
        }
    }
    
    pub async fn get_pending_count(&self) -> usize {
        let pending_messages = self.pending_messages.read().await;
        pending_messages.len()
    }
}
```

### 5. Flow Control and Backpressure

```rust
use tokio::sync::Semaphore;
use std::sync::atomic::{AtomicUsize, Ordering};

// Flow control system to prevent message flooding
pub struct FlowController {
    max_concurrent_messages: usize,
    semaphore: Semaphore,
    queue_size_limit: usize,
    current_queue_size: AtomicUsize,
    dropped_messages: AtomicUsize,
}

impl FlowController {
    pub fn new(max_concurrent_messages: usize, queue_size_limit: usize) -> Self {
        Self {
            max_concurrent_messages,
            semaphore: Semaphore::new(max_concurrent_messages),
            queue_size_limit,
            current_queue_size: AtomicUsize::new(0),
            dropped_messages: AtomicUsize::new(0),
        }
    }
    
    pub async fn acquire_permit(&self) -> Option<FlowControlPermit> {
        // Check queue size first
        let current_size = self.current_queue_size.load(Ordering::SeqCst);
        if current_size >= self.queue_size_limit {
            self.dropped_messages.fetch_add(1, Ordering::SeqCst);
            return None;
        }
        
        // Try to acquire semaphore permit
        if let Ok(permit) = self.semaphore.try_acquire() {
            self.current_queue_size.fetch_add(1, Ordering::SeqCst);
            Some(FlowControlPermit {
                _permit: permit,
                flow_controller: self,
            })
        } else {
            self.dropped_messages.fetch_add(1, Ordering::SeqCst);
            None
        }
    }
    
    pub async fn acquire_permit_blocking(&self) -> FlowControlPermit {
        let permit = self.semaphore.acquire().await.unwrap();
        self.current_queue_size.fetch_add(1, Ordering::SeqCst);
        FlowControlPermit {
            _permit: permit,
            flow_controller: self,
        }
    }
    
    pub fn get_stats(&self) -> FlowControlStats {
        FlowControlStats {
            max_concurrent: self.max_concurrent_messages,
            current_queue_size: self.current_queue_size.load(Ordering::SeqCst),
            queue_size_limit: self.queue_size_limit,
            dropped_messages: self.dropped_messages.load(Ordering::SeqCst),
            available_permits: self.semaphore.available_permits(),
        }
    }
}

pub struct FlowControlPermit<'a> {
    _permit: tokio::sync::SemaphorePermit<'a>,
    flow_controller: &'a FlowController,
}

impl Drop for FlowControlPermit<'_> {
    fn drop(&mut self) {
        self.flow_controller.current_queue_size.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub struct FlowControlStats {
    pub max_concurrent: usize,
    pub current_queue_size: usize,
    pub queue_size_limit: usize,
    pub dropped_messages: usize,
    pub available_permits: usize,
}

// Rate-limited message sender
pub struct RateLimitedSender {
    router: MessageRouter,
    flow_controller: FlowController,
    rate_limiter: TokenBucket,
}

pub struct TokenBucket {
    tokens: AtomicUsize,
    max_tokens: usize,
    refill_rate: usize, // tokens per second
    last_refill: std::sync::Mutex<Instant>,
}

impl TokenBucket {
    pub fn new(max_tokens: usize, refill_rate: usize) -> Self {
        Self {
            tokens: AtomicUsize::new(max_tokens),
            max_tokens,
            refill_rate,
            last_refill: std::sync::Mutex::new(Instant::now()),
        }
    }
    
    pub fn try_consume(&self, tokens: usize) -> bool {
        self.refill_tokens();
        
        let current = self.tokens.load(Ordering::SeqCst);
        if current >= tokens {
            let result = self.tokens.compare_exchange_weak(
                current,
                current - tokens,
                Ordering::SeqCst,
                Ordering::SeqCst,
            );
            result.is_ok()
        } else {
            false
        }
    }
    
    fn refill_tokens(&self) {
        let now = Instant::now();
        let mut last_refill = self.last_refill.lock().unwrap();
        let elapsed = now.duration_since(*last_refill);
        
        if elapsed >= Duration::from_secs(1) {
            let new_tokens = (elapsed.as_secs() as usize * self.refill_rate).min(self.max_tokens);
            let current = self.tokens.load(Ordering::SeqCst);
            let new_total = (current + new_tokens).min(self.max_tokens);
            
            self.tokens.store(new_total, Ordering::SeqCst);
            *last_refill = now;
        }
    }
    
    pub fn available_tokens(&self) -> usize {
        self.refill_tokens();
        self.tokens.load(Ordering::SeqCst)
    }
}

impl RateLimitedSender {
    pub fn new(
        router: MessageRouter,
        max_concurrent: usize,
        queue_limit: usize,
        rate_limit: usize,
    ) -> Self {
        Self {
            router,
            flow_controller: FlowController::new(max_concurrent, queue_limit),
            rate_limiter: TokenBucket::new(rate_limit * 10, rate_limit), // 10-second burst
        }
    }
    
    pub async fn send_with_backpressure(
        &self,
        target: ActorId,
        message: ActorMessage,
    ) -> Result<(), SendError> {
        // Check rate limit
        if !self.rate_limiter.try_consume(1) {
            return Err(SendError::RateLimited);
        }
        
        // Acquire flow control permit
        let _permit = match self.flow_controller.acquire_permit().await {
            Some(permit) => permit,
            None => return Err(SendError::QueueFull),
        };
        
        // Send message
        self.router.route_message(target, message).await
            .map_err(|e| SendError::RouterError(e))
    }
    
    pub fn get_flow_stats(&self) -> FlowControlStats {
        self.flow_controller.get_stats()
    }
    
    pub fn get_rate_stats(&self) -> RateStats {
        RateStats {
            available_tokens: self.rate_limiter.available_tokens(),
            max_tokens: self.rate_limiter.max_tokens,
            refill_rate: self.rate_limiter.refill_rate,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SendError {
    #[error("Rate limited")]
    RateLimited,
    #[error("Queue is full")]
    QueueFull,
    #[error("Router error: {0}")]
    RouterError(RouterError),
}

#[derive(Debug)]
pub struct RateStats {
    pub available_tokens: usize,
    pub max_tokens: usize,
    pub refill_rate: usize,
}
```

## Key Takeaways

1. **Actor Model**: Isolates state and enables scalable concurrent systems
2. **Typed Messages**: Prevent runtime errors through compile-time type checking
3. **Message Routing**: Central dispatch enables flexible communication patterns
4. **Reliability**: Acknowledgments and retries ensure message delivery
5. **Flow Control**: Backpressure prevents system overload
6. **Rate Limiting**: Token bucket prevents message flooding
7. **Topics/PubSub**: Enable one-to-many communication patterns

## Best Practices

- Design message types with clear semantics and minimal data copying
- Use bounded channels to apply backpressure naturally
- Implement proper error handling and timeout mechanisms
- Monitor message queue sizes and delivery statistics
- Use topic-based routing for event distribution
- Apply rate limiting to prevent system overload
- Design for graceful degradation under high load

## Performance Considerations

- Message serialization/deserialization overhead
- Channel buffer sizes affect memory usage and latency
- Actor mailbox processing affects throughput
- Network protocols impact distributed messaging
- Flow control prevents memory exhaustion
- Rate limiting balances fairness and throughput

## Next Steps

In the next tutorial, we'll explore data-oriented programming patterns, learning how to structure data for cache efficiency and SIMD operations in our space simulation engine.