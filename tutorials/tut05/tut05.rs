// Tutorial 05: Message Passing
// Complete the following exercises to practice advanced message passing patterns in Rust

use tokio::sync::{mpsc, oneshot, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicUsize, AtomicU64, Ordering}};
use serde::{Serialize, Deserialize};

// Exercise 1: Actor Message Types and System
pub type ActorId = u32;

// TODO: Define ActorMessage enum with the following variants:
// - ShipToShip { from: ActorId, to: ActorId, content: ShipMessage }
// - ShipToStation { from: ActorId, to: ActorId, content: StationMessage }
// - SystemBroadcast { sender: ActorId, content: SystemMessage }
// - ActorControl { target: ActorId, command: ControlCommand }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActorMessage {
    // TODO: Add message variants here
}

// TODO: Define ShipMessage enum with:
// - RequestDocking { ship_specs: ShipSpecs, cargo_manifest: Vec<CargoItem> }
// - FormationRequest { formation_type: FormationType, position: (f32, f32, f32) }
// - TradeOffer { offering: Vec<CargoItem>, requesting: Vec<ResourceRequest> }
// - DistressSignal { emergency_type: EmergencyType, location: (f32, f32, f32), severity: u8 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShipMessage {
    // TODO: Add ship message variants
}

// TODO: Define StationMessage enum with:
// - DockingPermission { granted: bool, berth_assignment: Option<u32>, estimated_wait: Duration }
// - MarketPrices { commodities: HashMap<String, f32>, last_updated: Instant }
// - ServiceAvailability { repairs: bool, refueling: bool, crew_quarters: bool }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StationMessage {
    // TODO: Add station message variants
}

// TODO: Define SystemMessage enum with:
// - NavigationHazard { location: (f32, f32, f32), radius: f32, hazard_type: HazardType }
// - MarketUpdate { sector: String, price_changes: HashMap<String, f32> }
// - WeatherAlert { affected_regions: Vec<String>, severity: WeatherSeverity }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystemMessage {
    // TODO: Add system message variants
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    Shutdown,
    Pause,
    Resume,
    GetStatus,
}

// Supporting types
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

// Exercise 2: Actor System Implementation
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

// TODO: Define Actor trait with associated types Message and State
// Should have async methods:
// - handle_message(&mut self, message: Self::Message, state: &mut Self::State) -> Result<(), ActorError>
// - on_start(&mut self, state: &mut Self::State) -> Result<(), ActorError> (default: Ok(()))
// - on_stop(&mut self, state: &mut Self::State) -> Result<(), ActorError> (default: Ok(()))
#[async_trait::async_trait]
pub trait Actor {
    // TODO: Add associated types and methods
}

// TODO: Implement ActorHandle<M> struct with:
// - sender: mpsc::UnboundedSender<M>
// - actor_id: ActorId
// Should be Clone and have methods: new(), send(), actor_id(), is_closed()
pub struct ActorHandle<M> {
    // TODO: Add fields
}

impl<M> Clone for ActorHandle<M> {
    fn clone(&self) -> Self {
        todo!("Clone the handle")
    }
}

impl<M> ActorHandle<M> {
    // TODO: Implement new() method
    pub fn new(sender: mpsc::UnboundedSender<M>, actor_id: ActorId) -> Self {
        todo!("Create new ActorHandle")
    }
    
    // TODO: Implement send() method
    pub fn send(&self, message: M) -> Result<(), mpsc::error::SendError<M>> {
        todo!("Send message through channel")
    }
    
    // TODO: Implement actor_id() method
    pub fn actor_id(&self) -> ActorId {
        todo!("Return actor ID")
    }
    
    // TODO: Implement is_closed() method
    pub fn is_closed(&self) -> bool {
        todo!("Check if channel is closed")
    }
}

// Exercise 3: Message Router Implementation
#[derive(Debug, Default)]
pub struct MessageStatistics {
    pub total_messages: u64,
    pub messages_by_type: HashMap<String, u64>,
    pub failed_deliveries: u64,
    pub active_actors: u64,
}

impl Clone for MessageStatistics {
    fn clone(&self) -> Self {
        Self {
            total_messages: self.total_messages,
            messages_by_type: self.messages_by_type.clone(),
            failed_deliveries: self.failed_deliveries,
            active_actors: self.active_actors,
        }
    }
}

// TODO: Implement MessageRouter struct with:
// - actors: Arc<RwLock<HashMap<ActorId, mpsc::UnboundedSender<ActorMessage>>>>
// - subscribers: Arc<RwLock<HashMap<String, Vec<ActorId>>>> (for topics)
// - message_stats: Arc<RwLock<MessageStatistics>>
pub struct MessageRouter {
    // TODO: Add fields
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

impl Clone for MessageRouter {
    fn clone(&self) -> Self {
        todo!("Clone the router")
    }
}

impl MessageRouter {
    // TODO: Implement new() constructor
    pub fn new() -> Self {
        todo!("Create new MessageRouter")
    }
    
    // TODO: Implement register_actor() method
    // Should add actor to actors map and update statistics
    pub async fn register_actor(&self, actor_id: ActorId, sender: mpsc::UnboundedSender<ActorMessage>) {
        todo!("Register actor with router")
    }
    
    // TODO: Implement unregister_actor() method
    pub async fn unregister_actor(&self, actor_id: ActorId) {
        todo!("Unregister actor from router")
    }
    
    // TODO: Implement route_message() method
    // Should send message to specific actor and update statistics
    pub async fn route_message(&self, target: ActorId, message: ActorMessage) -> Result<(), RouterError> {
        todo!("Route message to target actor")
    }
    
    // TODO: Implement broadcast_message() method
    // Should send message to all registered actors
    pub async fn broadcast_message(&self, message: ActorMessage) -> Result<u32, RouterError> {
        todo!("Broadcast message to all actors")
    }
    
    // TODO: Implement subscribe_to_topic() method
    pub async fn subscribe_to_topic(&self, actor_id: ActorId, topic: String) {
        todo!("Subscribe actor to topic")
    }
    
    // TODO: Implement publish_to_topic() method
    // Should send message to all subscribers of the topic
    pub async fn publish_to_topic(&self, topic: &str, message: ActorMessage) -> Result<u32, RouterError> {
        todo!("Publish message to topic subscribers")
    }
    
    // TODO: Implement get_statistics() method
    pub async fn get_statistics(&self) -> MessageStatistics {
        todo!("Return current message statistics")
    }
    
    // TODO: Implement helper method get_message_type() that returns message type as String
    fn get_message_type(message: &ActorMessage) -> String {
        todo!("Return message type as string for statistics")
    }
}

// Exercise 4: Actor Context for Communication
// TODO: Implement ActorContext struct with:
// - actor_id: ActorId
// - message_router: MessageRouter
pub struct ActorContext {
    // TODO: Add fields
}

impl ActorContext {
    // TODO: Implement send_to_actor() method
    pub async fn send_to_actor<M>(&self, target: ActorId, message: M) -> Result<(), ActorError>
    where
        M: Into<ActorMessage> + Send + 'static,
    {
        todo!("Send message to specific actor")
    }
    
    // TODO: Implement broadcast() method
    pub async fn broadcast<M>(&self, message: M) -> Result<(), ActorError>
    where
        M: Into<ActorMessage> + Send + 'static,
    {
        todo!("Broadcast message to all actors")
    }
}

// Exercise 5: Ship Actor Implementation
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

// TODO: Implement ShipActor struct with:
// - context: ActorContext
// - position: (f32, f32, f32)
// - velocity: (f32, f32, f32)
// - fuel_level: f32
// - cargo: Vec<CargoItem>
// - state: ShipState
pub struct ShipActor {
    // TODO: Add fields
}

// TODO: Implement Actor trait for ShipActor
// Message = ShipActorMessage, State = ShipState
#[async_trait::async_trait]
impl Actor for ShipActor {
    // TODO: Add associated types and implement handle_message
}

impl ShipActor {
    // TODO: Implement new() constructor
    pub fn new(actor_id: ActorId, message_router: MessageRouter) -> Self {
        todo!("Create new ShipActor")
    }
    
    // TODO: Implement update_physics() method
    // Should update position based on velocity and consume fuel
    fn update_physics(&mut self, dt: f32) {
        todo!("Update ship physics")
    }
    
    // TODO: Implement distance_to() method
    fn distance_to(&self, target: (f32, f32, f32)) -> f32 {
        todo!("Calculate distance to target")
    }
    
    // TODO: Implement handle_incoming_message() method
    // Should process different types of incoming messages
    async fn handle_incoming_message(&mut self, message: ActorMessage, state: &mut ShipState) -> Result<(), ActorError> {
        todo!("Handle incoming actor messages")
    }
}

// Exercise 6: Reliable Message Delivery
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

// TODO: Implement ReliableMessenger struct with:
// - router: MessageRouter
// - pending_messages: Arc<RwLock<HashMap<MessageId, PendingMessage>>>
// - next_message_id: Arc<AtomicU64>
pub struct ReliableMessenger {
    // TODO: Add fields
}

impl ReliableMessenger {
    // TODO: Implement new() constructor
    pub fn new(router: MessageRouter) -> Self {
        todo!("Create new ReliableMessenger")
    }
    
    // TODO: Implement send_reliable() method
    // Should send message and track it for retries
    pub async fn send_reliable(
        &self,
        target: ActorId,
        message: ActorMessage,
        max_retries: u8,
        timeout_duration: Duration,
    ) -> Result<MessageId, RouterError> {
        todo!("Send message with reliability guarantees")
    }
    
    // TODO: Implement handle_acknowledgment() method
    pub async fn handle_acknowledgment(&self, message_id: MessageId) {
        todo!("Handle message acknowledgment")
    }
    
    // TODO: Implement process_pending_messages() method
    // Should check for timeouts and retry failed messages
    async fn process_pending_messages(&self) {
        todo!("Process pending messages for retries")
    }
    
    // TODO: Implement get_pending_count() method
    pub async fn get_pending_count(&self) -> usize {
        todo!("Return number of pending messages")
    }
    
    // TODO: Implement retry_loop() method - background task for retries
    pub async fn retry_loop(&self) {
        todo!("Background loop for handling retries")
    }
}

// Exercise 7: Flow Control and Backpressure
use tokio::sync::Semaphore;

// TODO: Implement FlowController struct with:
// - max_concurrent_messages: usize
// - semaphore: Semaphore
// - queue_size_limit: usize
// - current_queue_size: AtomicUsize
// - dropped_messages: AtomicUsize
pub struct FlowController {
    // TODO: Add fields
}

#[derive(Debug)]
pub struct FlowControlStats {
    pub max_concurrent: usize,
    pub current_queue_size: usize,
    pub queue_size_limit: usize,
    pub dropped_messages: usize,
    pub available_permits: usize,
}

// TODO: Implement FlowControlPermit struct that releases permit on drop
pub struct FlowControlPermit<'a> {
    // TODO: Add fields
}

impl Drop for FlowControlPermit<'_> {
    fn drop(&mut self) {
        todo!("Release flow control permit")
    }
}

impl FlowController {
    // TODO: Implement new() constructor
    pub fn new(max_concurrent_messages: usize, queue_size_limit: usize) -> Self {
        todo!("Create new FlowController")
    }
    
    // TODO: Implement acquire_permit() method (non-blocking)
    // Should return None if queue is full or no permits available
    pub async fn acquire_permit(&self) -> Option<FlowControlPermit> {
        todo!("Try to acquire flow control permit")
    }
    
    // TODO: Implement acquire_permit_blocking() method
    pub async fn acquire_permit_blocking(&self) -> FlowControlPermit {
        todo!("Acquire permit with blocking")
    }
    
    // TODO: Implement get_stats() method
    pub fn get_stats(&self) -> FlowControlStats {
        todo!("Return flow control statistics")
    }
}

// Exercise 8: Rate Limiting with Token Bucket
// TODO: Implement TokenBucket struct with:
// - tokens: AtomicUsize
// - max_tokens: usize
// - refill_rate: usize (tokens per second)
// - last_refill: std::sync::Mutex<Instant>
pub struct TokenBucket {
    // TODO: Add fields
}

impl TokenBucket {
    // TODO: Implement new() constructor
    pub fn new(max_tokens: usize, refill_rate: usize) -> Self {
        todo!("Create new TokenBucket")
    }
    
    // TODO: Implement try_consume() method
    // Should consume tokens if available, return true if successful
    pub fn try_consume(&self, tokens: usize) -> bool {
        todo!("Try to consume tokens")
    }
    
    // TODO: Implement refill_tokens() helper method
    // Should add tokens based on elapsed time
    fn refill_tokens(&self) {
        todo!("Refill tokens based on time elapsed")
    }
    
    // TODO: Implement available_tokens() method
    pub fn available_tokens(&self) -> usize {
        todo!("Return number of available tokens")
    }
}

#[derive(Debug)]
pub struct RateStats {
    pub available_tokens: usize,
    pub max_tokens: usize,
    pub refill_rate: usize,
}

// TODO: Implement RateLimitedSender struct with:
// - router: MessageRouter
// - flow_controller: FlowController
// - rate_limiter: TokenBucket
pub struct RateLimitedSender {
    // TODO: Add fields
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

impl RateLimitedSender {
    // TODO: Implement new() constructor
    pub fn new(
        router: MessageRouter,
        max_concurrent: usize,
        queue_limit: usize,
        rate_limit: usize,
    ) -> Self {
        todo!("Create new RateLimitedSender")
    }
    
    // TODO: Implement send_with_backpressure() method
    // Should check rate limit, acquire flow control permit, then send
    pub async fn send_with_backpressure(
        &self,
        target: ActorId,
        message: ActorMessage,
    ) -> Result<(), SendError> {
        todo!("Send message with rate limiting and backpressure")
    }
    
    // TODO: Implement get_flow_stats() method
    pub fn get_flow_stats(&self) -> FlowControlStats {
        todo!("Return flow control statistics")
    }
    
    // TODO: Implement get_rate_stats() method
    pub fn get_rate_stats(&self) -> RateStats {
        todo!("Return rate limiting statistics")
    }
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_actor_handle() {
        let (sender, _receiver) = mpsc::unbounded_channel();
        let handle = ActorHandle::new(sender, 42);
        
        assert_eq!(handle.actor_id(), 42);
        assert!(!handle.is_closed());
        
        let cloned = handle.clone();
        assert_eq!(cloned.actor_id(), 42);
    }

    #[tokio::test]
    async fn test_message_router_basic() {
        let router = MessageRouter::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        
        router.register_actor(1, sender).await;
        
        let message = ActorMessage::SystemBroadcast {
            sender: 2,
            content: SystemMessage::MarketUpdate {
                sector: "Alpha".to_string(),
                price_changes: HashMap::new(),
            },
        };
        
        let result = router.route_message(1, message).await;
        assert!(result.is_ok());
        
        let received = receiver.recv().await;
        assert!(received.is_some());
    }

    #[tokio::test]
    async fn test_broadcast() {
        let router = MessageRouter::new();
        let (sender1, mut receiver1) = mpsc::unbounded_channel();
        let (sender2, mut receiver2) = mpsc::unbounded_channel();
        
        router.register_actor(1, sender1).await;
        router.register_actor(2, sender2).await;
        
        let message = ActorMessage::SystemBroadcast {
            sender: 0,
            content: SystemMessage::WeatherAlert {
                affected_regions: vec!["Sector 1".to_string()],
                severity: WeatherSeverity::High,
            },
        };
        
        let delivered = router.broadcast_message(message).await.unwrap();
        assert_eq!(delivered, 2);
        
        assert!(receiver1.recv().await.is_some());
        assert!(receiver2.recv().await.is_some());
    }

    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(10, 5);
        
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.available_tokens(), 5);
        
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.available_tokens(), 0);
        
        assert!(!bucket.try_consume(1)); // Should fail, no tokens left
    }

    #[test]
    fn test_flow_controller_basic() {
        let controller = FlowController::new(2, 10);
        let stats = controller.get_stats();
        
        assert_eq!(stats.max_concurrent, 2);
        assert_eq!(stats.current_queue_size, 0);
        assert_eq!(stats.available_permits, 2);
    }

    #[tokio::test]
    async fn test_reliable_messenger() {
        let router = MessageRouter::new();
        let messenger = ReliableMessenger::new(router);
        
        let message = ActorMessage::SystemBroadcast {
            sender: 1,
            content: SystemMessage::NavigationHazard {
                location: (100.0, 200.0, 300.0),
                radius: 50.0,
                hazard_type: HazardType::Asteroid,
            },
        };
        
        // This will fail since target actor doesn't exist, but should return message ID
        let result = messenger.send_reliable(
            999, // Non-existent actor
            message,
            3,
            Duration::from_millis(100)
        ).await;
        
        assert!(result.is_err()); // Should fail to send initially
        assert_eq!(messenger.get_pending_count().await, 0); // But no pending since send failed
    }
}