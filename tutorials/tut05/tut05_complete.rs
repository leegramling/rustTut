// Tutorial 05: Message Passing - Complete Solutions
// This file contains the complete implementations for all exercises

use tokio::sync::{mpsc, oneshot, RwLock, Semaphore};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicUsize, AtomicU64, Ordering}};
use serde::{Serialize, Deserialize};

// Exercise 1: Actor Message Types and System - Complete Implementation
pub type ActorId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActorMessage {
    ShipToShip {
        from: ActorId,
        to: ActorId,
        content: ShipMessage,
    },
    ShipToStation {
        from: ActorId,
        to: ActorId,
        content: StationMessage,
    },
    SystemBroadcast {
        sender: ActorId,
        content: SystemMessage,
    },
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
        #[serde(skip)] // Skip serialization for Instant
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

impl CargoItem {
    pub fn total_value(&self) -> f32 {
        self.quantity * self.unit_value
    }
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

// Exercise 2: Actor System Implementation - Complete Implementation
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

#[async_trait::async_trait]
pub trait Actor {
    type Message: Send + 'static;
    type State: Send + 'static;
    
    async fn handle_message(&mut self, message: Self::Message, state: &mut Self::State) -> Result<(), ActorError>;
    
    async fn on_start(&mut self, _state: &mut Self::State) -> Result<(), ActorError> {
        Ok(())
    }
    
    async fn on_stop(&mut self, _state: &mut Self::State) -> Result<(), ActorError> {
        Ok(())
    }
}

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
    
    // Bonus: Try send (non-blocking)
    pub fn try_send(&self, message: M) -> Result<(), mpsc::error::TrySendError<M>> {
        self.sender.send(message).map_err(|e| mpsc::error::TrySendError::Closed(e.0))
    }
}

// Exercise 3: Message Router Implementation - Complete Implementation
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

pub struct MessageRouter {
    actors: Arc<RwLock<HashMap<ActorId, mpsc::UnboundedSender<ActorMessage>>>>,
    subscribers: Arc<RwLock<HashMap<String, Vec<ActorId>>>>,
    message_stats: Arc<RwLock<MessageStatistics>>,
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
        
        // Remove from all topic subscriptions
        let mut subscribers = self.subscribers.write().await;
        for subscriber_list in subscribers.values_mut() {
            subscriber_list.retain(|&id| id != actor_id);
        }
        
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
    
    pub async fn unsubscribe_from_topic(&self, actor_id: ActorId, topic: &str) {
        let mut subscribers = self.subscribers.write().await;
        if let Some(subscriber_list) = subscribers.get_mut(topic) {
            subscriber_list.retain(|&id| id != actor_id);
            println!("Actor {} unsubscribed from topic '{}'", actor_id, topic);
        }
    }
    
    pub async fn publish_to_topic(&self, topic: &str, message: ActorMessage) -> Result<u32, RouterError> {
        let subscribers = self.subscribers.read().await;
        let actors = self.actors.read().await;
        
        if let Some(subscriber_ids) = subscribers.get(topic) {
            let mut delivered = 0;
            let mut failed = 0;
            
            for &actor_id in subscriber_ids {
                if let Some(sender) = actors.get(&actor_id) {
                    match sender.send(message.clone()) {
                        Ok(()) => delivered += 1,
                        Err(_) => failed += 1,
                    }
                }
            }
            
            // Update statistics
            let mut stats = self.message_stats.write().await;
            stats.total_messages += delivered as u64;
            stats.failed_deliveries += failed as u64;
            let message_type = Self::get_message_type(&message);
            *stats.messages_by_type.entry(message_type).or_insert(0) += delivered as u64;
            
            println!("Published to topic '{}': {} delivered, {} failed", topic, delivered, failed);
            Ok(delivered)
        } else {
            Ok(0) // No subscribers
        }
    }
    
    pub async fn get_statistics(&self) -> MessageStatistics {
        let stats = self.message_stats.read().await;
        stats.clone()
    }
    
    pub async fn get_topic_subscribers(&self, topic: &str) -> Vec<ActorId> {
        let subscribers = self.subscribers.read().await;
        subscribers.get(topic).cloned().unwrap_or_default()
    }
    
    pub async fn get_actor_topics(&self, actor_id: ActorId) -> Vec<String> {
        let subscribers = self.subscribers.read().await;
        subscribers
            .iter()
            .filter(|(_, actors)| actors.contains(&actor_id))
            .map(|(topic, _)| topic.clone())
            .collect()
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

impl Default for MessageRouter {
    fn default() -> Self {
        Self::new()
    }
}

// Exercise 4: Actor Context for Communication - Complete Implementation
pub struct ActorContext {
    pub actor_id: ActorId,
    pub message_router: MessageRouter,
}

impl ActorContext {
    pub fn new(actor_id: ActorId, message_router: MessageRouter) -> Self {
        Self { actor_id, message_router }
    }
    
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
            .map_err(|e| ActorError::Communication(e.to_string()))?;
        Ok(())
    }
    
    pub async fn subscribe_to_topic(&self, topic: String) {
        self.message_router.subscribe_to_topic(self.actor_id, topic).await;
    }
    
    pub async fn publish_to_topic<M>(&self, topic: &str, message: M) -> Result<u32, ActorError>
    where
        M: Into<ActorMessage> + Send + 'static,
    {
        self.message_router
            .publish_to_topic(topic, message.into())
            .await
            .map_err(|e| ActorError::Communication(e.to_string()))
    }
}

// Exercise 5: Ship Actor Implementation - Complete Implementation
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

pub struct ShipActor {
    context: ActorContext,
    position: (f32, f32, f32),
    velocity: (f32, f32, f32),
    fuel_level: f32,
    cargo: Vec<CargoItem>,
    state: ShipState,
    max_speed: f32,
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
                // Calculate velocity to reach target
                let direction = (
                    target.0 - self.position.0,
                    target.1 - self.position.1,
                    target.2 - self.position.2,
                );
                let distance = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
                
                if distance > 0.0 {
                    let speed = self.max_speed.min(distance * 0.1); // Adaptive speed
                    self.velocity = (
                        direction.0 / distance * speed,
                        direction.1 / distance * speed,
                        direction.2 / distance * speed,
                    );
                }
                
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
                        granted: false, // This is a request
                        berth_assignment: None,
                        estimated_wait: Duration::from_secs(0),
                    },
                };
                
                self.context.send_to_actor(station_id, docking_request).await?;
                *state = ShipState::Docking { station_id };
                println!("Ship {} requesting docking at station {}", self.context.actor_id, station_id);
            }
            
            ShipActorMessage::ProcessTradeOffer { offer } => {
                // Evaluate trade offer
                let total_value: f32 = offer.iter().map(|item| item.total_value()).sum();
                println!("Ship {} received trade offer worth {:.2} credits", 
                        self.context.actor_id, total_value);
                
                // Accept if valuable enough and we have space
                if total_value > 1000.0 && self.cargo.len() + offer.len() <= 20 {
                    self.cargo.extend(offer);
                    *state = ShipState::Trading;
                    println!("Ship {} accepted trade offer", self.context.actor_id);
                } else {
                    println!("Ship {} declined trade offer (value: {:.2}, cargo space: {})", 
                            self.context.actor_id, total_value, 20 - self.cargo.len());
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
                        hazard_type: match emergency {
                            EmergencyType::Pirate => HazardType::Pirates,
                            _ => HazardType::Debris,
                        },
                    },
                };
                
                self.context.broadcast(distress).await?;
                println!("Ship {} broadcasting emergency: {:?}", self.context.actor_id, emergency);
                
                // Stop movement during emergency
                self.velocity = (0.0, 0.0, 0.0);
            }
        }
        
        Ok(())
    }
    
    async fn on_start(&mut self, state: &mut Self::State) -> Result<(), ActorError> {
        println!("Ship {} starting at position {:?}", self.context.actor_id, self.position);
        self.context.subscribe_to_topic("navigation_hazards".to_string()).await;
        self.context.subscribe_to_topic("market_updates".to_string()).await;
        Ok(())
    }
    
    async fn on_stop(&mut self, _state: &mut Self::State) -> Result<(), ActorError> {
        println!("Ship {} shutting down", self.context.actor_id);
        Ok(())
    }
}

impl ShipActor {
    pub fn new(actor_id: ActorId, message_router: MessageRouter) -> Self {
        Self {
            context: ActorContext::new(actor_id, message_router),
            position: (0.0, 0.0, 0.0),
            velocity: (0.0, 0.0, 0.0),
            fuel_level: 1.0,
            cargo: Vec::new(),
            state: ShipState::Idle,
            max_speed: 50.0,
        }
    }
    
    pub fn new_at_position(actor_id: ActorId, message_router: MessageRouter, position: (f32, f32, f32)) -> Self {
        let mut ship = Self::new(actor_id, message_router);
        ship.position = position;
        ship
    }
    
    fn update_physics(&mut self, dt: f32) {
        self.position.0 += self.velocity.0 * dt;
        self.position.1 += self.velocity.1 * dt;
        self.position.2 += self.velocity.2 * dt;
        
        // Consume fuel based on velocity
        let speed = (self.velocity.0.powi(2) + self.velocity.1.powi(2) + self.velocity.2.powi(2)).sqrt();
        self.fuel_level -= speed * dt * 0.01;
        self.fuel_level = self.fuel_level.max(0.0);
        
        // Apply drag
        self.velocity.0 *= 0.99;
        self.velocity.1 *= 0.99;
        self.velocity.2 *= 0.99;
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
                        
                        // Accept formation if we're idle
                        if matches!(state, ShipState::Idle) {
                            let direction = (
                                position.0 - self.position.0,
                                position.1 - self.position.1,
                                position.2 - self.position.2,
                            );
                            let distance = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
                            
                            if distance > 0.0 {
                                self.velocity = (
                                    direction.0 / distance * 20.0,
                                    direction.1 / distance * 20.0,
                                    direction.2 / distance * 20.0,
                                );
                            }
                            *state = ShipState::Flying { destination: position };
                        }
                    }
                    
                    ShipMessage::TradeOffer { offering, .. } => {
                        // Process trade offer
                        let total_value: f32 = offering.iter().map(|item| item.total_value()).sum();
                        if total_value > 500.0 && self.cargo.len() + offering.len() <= 20 {
                            self.cargo.extend(offering);
                            println!("Ship {} accepted trade offer worth {:.2}", self.context.actor_id, total_value);
                        }
                    }
                    
                    ShipMessage::DistressSignal { emergency_type, location, severity } => {
                        println!("Ship {} received distress signal: {:?} at {:?} (severity {})", 
                                self.context.actor_id, emergency_type, location, severity);
                        
                        // If we're nearby and able to help
                        let distance = self.distance_to(location);
                        if distance < 200.0 && matches!(state, ShipState::Idle) && severity > 5 {
                            *state = ShipState::Flying { destination: location };
                            let direction = (
                                location.0 - self.position.0,
                                location.1 - self.position.1,
                                location.2 - self.position.2,
                            );
                            let distance = (direction.0.powi(2) + direction.1.powi(2) + direction.2.powi(2)).sqrt();
                            
                            if distance > 0.0 {
                                self.velocity = (
                                    direction.0 / distance * self.max_speed,
                                    direction.1 / distance * self.max_speed,
                                    direction.2 / distance * self.max_speed,
                                );
                            }
                            println!("Ship {} responding to distress call", self.context.actor_id);
                        }
                    }
                    
                    _ => {}
                }
            }
            
            ActorMessage::ShipToStation { content, .. } => {
                match content {
                    StationMessage::DockingPermission { granted, berth_assignment, estimated_wait } => {
                        if granted {
                            if let Some(berth) = berth_assignment {
                                println!("Ship {} granted docking at berth {}", self.context.actor_id, berth);
                                *state = ShipState::Docked { station_id: 0 }; // Would need to track station ID
                            }
                        } else {
                            println!("Ship {} docking denied, estimated wait: {:?}", 
                                    self.context.actor_id, estimated_wait);
                        }
                    }
                    
                    StationMessage::MarketPrices { commodities, .. } => {
                        println!("Ship {} received market prices for {} commodities", 
                                self.context.actor_id, commodities.len());
                        // Could use this to make trading decisions
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
                            
                            // Avoid hazard by moving away
                            let avoidance_direction = (
                                self.position.0 - location.0,
                                self.position.1 - location.1,
                                self.position.2 - location.2,
                            );
                            let avoidance_distance = (avoidance_direction.0.powi(2) + avoidance_direction.1.powi(2) + avoidance_direction.2.powi(2)).sqrt();
                            
                            if avoidance_distance > 0.0 {
                                self.velocity = (
                                    avoidance_direction.0 / avoidance_distance * 30.0,
                                    avoidance_direction.1 / avoidance_distance * 30.0,
                                    avoidance_direction.2 / avoidance_distance * 30.0,
                                );
                            }
                        }
                    }
                    
                    SystemMessage::MarketUpdate { sector, price_changes } => {
                        println!("Ship {} received market update for {}: {} commodities", 
                                self.context.actor_id, sector, price_changes.len());
                    }
                    
                    SystemMessage::WeatherAlert { affected_regions, severity } => {
                        println!("Ship {} received weather alert: {:?} regions, severity {:?}", 
                                self.context.actor_id, affected_regions.len(), severity);
                        
                        // Reduce speed in severe weather
                        if matches!(severity, WeatherSeverity::High | WeatherSeverity::Extreme) {
                            self.velocity.0 *= 0.5;
                            self.velocity.1 *= 0.5;
                            self.velocity.2 *= 0.5;
                        }
                    }
                }
            }
            
            ActorMessage::ActorControl { command, .. } => {
                match command {
                    ControlCommand::Shutdown => {
                        println!("Ship {} received shutdown command", self.context.actor_id);
                        return Err(ActorError::ShuttingDown);
                    }
                    ControlCommand::Pause => {
                        self.velocity = (0.0, 0.0, 0.0);
                        println!("Ship {} paused", self.context.actor_id);
                    }
                    ControlCommand::Resume => {
                        println!("Ship {} resumed", self.context.actor_id);
                    }
                    ControlCommand::GetStatus => {
                        println!("Ship {} status: pos={:?}, vel={:?}, fuel={:.2}, cargo={}", 
                                self.context.actor_id, self.position, self.velocity, 
                                self.fuel_level, self.cargo.len());
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Bonus methods
    pub fn get_position(&self) -> (f32, f32, f32) {
        self.position
    }
    
    pub fn get_fuel_level(&self) -> f32 {
        self.fuel_level
    }
    
    pub fn get_cargo_count(&self) -> usize {
        self.cargo.len()
    }
    
    pub fn get_cargo_value(&self) -> f32 {
        self.cargo.iter().map(|item| item.total_value()).sum()
    }
}

// Exercise 6: Reliable Message Delivery - Complete Implementation
pub type MessageId = u64;

#[derive(Debug, Clone)]
struct PendingMessage {
    message: ActorMessage,
    target: ActorId,
    sent_at: Instant,
    retry_count: u8,
    max_retries: u8,
    timeout_duration: Duration,
}

pub struct ReliableMessenger {
    router: MessageRouter,
    pending_messages: Arc<RwLock<HashMap<MessageId, PendingMessage>>>,
    next_message_id: Arc<AtomicU64>,
}

impl ReliableMessenger {
    pub fn new(router: MessageRouter) -> Self {
        Self {
            router,
            pending_messages: Arc::new(RwLock::new(HashMap::new())),
            next_message_id: Arc::new(AtomicU64::new(1)),
        }
    }
    
    pub async fn send_reliable(
        &self,
        target: ActorId,
        message: ActorMessage,
        max_retries: u8,
        timeout_duration: Duration,
    ) -> Result<MessageId, RouterError> {
        let message_id = self.next_message_id.fetch_add(1, Ordering::SeqCst);
        
        // Try to send initial message
        match self.router.route_message(target, message.clone()).await {
            Ok(()) => {
                // Store as pending for acknowledgment tracking
                let pending = PendingMessage {
                    message,
                    target,
                    sent_at: Instant::now(),
                    retry_count: 0,
                    max_retries,
                    timeout_duration,
                };
                
                let mut pending_messages = self.pending_messages.write().await;
                pending_messages.insert(message_id, pending);
                
                println!("Reliable message {} sent to actor {}", message_id, target);
                Ok(message_id)
            }
            Err(e) => Err(e),
        }
    }
    
    pub async fn handle_acknowledgment(&self, message_id: MessageId) {
        let mut pending_messages = self.pending_messages.write().await;
        if let Some(pending) = pending_messages.remove(&message_id) {
            println!("Message {} acknowledged by actor {} after {} retries", 
                    message_id, pending.target, pending.retry_count);
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
            
            match self.router.route_message(pending.target, pending.message.clone()).await {
                Ok(()) => {
                    let mut pending_messages = self.pending_messages.write().await;
                    pending_messages.insert(message_id, pending);
                    println!("Retrying message {} (attempt {})", message_id, pending.retry_count + 1);
                }
                Err(e) => {
                    println!("Failed to retry message {}: {:?}", message_id, e);
                }
            }
        }
    }
    
    pub async fn get_pending_count(&self) -> usize {
        let pending_messages = self.pending_messages.read().await;
        pending_messages.len()
    }
    
    pub async fn retry_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        
        loop {
            interval.tick().await;
            self.process_pending_messages().await;
        }
    }
    
    // Bonus: Get pending message statistics
    pub async fn get_pending_stats(&self) -> PendingStats {
        let pending_messages = self.pending_messages.read().await;
        let mut retry_counts = HashMap::new();
        let mut oldest_message_age = Duration::from_secs(0);
        let now = Instant::now();
        
        for pending in pending_messages.values() {
            *retry_counts.entry(pending.retry_count).or_insert(0) += 1;
            let age = now.duration_since(pending.sent_at);
            if age > oldest_message_age {
                oldest_message_age = age;
            }
        }
        
        PendingStats {
            total_pending: pending_messages.len(),
            retry_counts,
            oldest_message_age,
        }
    }
}

#[derive(Debug)]
pub struct PendingStats {
    pub total_pending: usize,
    pub retry_counts: HashMap<u8, usize>,
    pub oldest_message_age: Duration,
}

// Exercise 7: Flow Control and Backpressure - Complete Implementation
pub struct FlowController {
    max_concurrent_messages: usize,
    semaphore: Semaphore,
    queue_size_limit: usize,
    current_queue_size: AtomicUsize,
    dropped_messages: AtomicUsize,
}

#[derive(Debug)]
pub struct FlowControlStats {
    pub max_concurrent: usize,
    pub current_queue_size: usize,
    pub queue_size_limit: usize,
    pub dropped_messages: usize,
    pub available_permits: usize,
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
    
    pub fn reset_stats(&self) {
        self.dropped_messages.store(0, Ordering::SeqCst);
    }
}

// Exercise 8: Rate Limiting with Token Bucket - Complete Implementation
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
        
        // Try to consume tokens using compare-and-swap loop
        loop {
            let current = self.tokens.load(Ordering::SeqCst);
            if current < tokens {
                return false;
            }
            
            match self.tokens.compare_exchange_weak(
                current,
                current - tokens,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(_) => continue, // Retry if another thread modified tokens
            }
        }
    }
    
    fn refill_tokens(&self) {
        let now = Instant::now();
        if let Ok(mut last_refill) = self.last_refill.try_lock() {
            let elapsed = now.duration_since(*last_refill);
            
            if elapsed >= Duration::from_millis(100) { // Refill every 100ms for smoother distribution
                let elapsed_seconds = elapsed.as_secs_f64();
                let new_tokens = (elapsed_seconds * self.refill_rate as f64) as usize;
                
                if new_tokens > 0 {
                    let current = self.tokens.load(Ordering::SeqCst);
                    let new_total = (current + new_tokens).min(self.max_tokens);
                    self.tokens.store(new_total, Ordering::SeqCst);
                    *last_refill = now;
                }
            }
        }
    }
    
    pub fn available_tokens(&self) -> usize {
        self.refill_tokens();
        self.tokens.load(Ordering::SeqCst)
    }
    
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    
    pub fn refill_rate(&self) -> usize {
        self.refill_rate
    }
    
    // Bonus: Get time until next token refill
    pub fn time_until_refill(&self) -> Duration {
        if let Ok(last_refill) = self.last_refill.try_lock() {
            let elapsed = last_refill.elapsed();
            let refill_interval = Duration::from_millis(100);
            if elapsed < refill_interval {
                refill_interval - elapsed
            } else {
                Duration::from_secs(0)
            }
        } else {
            Duration::from_secs(0)
        }
    }
}

#[derive(Debug)]
pub struct RateStats {
    pub available_tokens: usize,
    pub max_tokens: usize,
    pub refill_rate: usize,
}

pub struct RateLimitedSender {
    router: MessageRouter,
    flow_controller: FlowController,
    rate_limiter: TokenBucket,
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
    pub fn new(
        router: MessageRouter,
        max_concurrent: usize,
        queue_limit: usize,
        rate_limit: usize,
    ) -> Self {
        Self {
            router,
            flow_controller: FlowController::new(max_concurrent, queue_limit),
            rate_limiter: TokenBucket::new(rate_limit * 10, rate_limit), // 10-second burst capacity
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
            .map_err(SendError::RouterError)
    }
    
    pub async fn send_with_backpressure_blocking(
        &self,
        target: ActorId,
        message: ActorMessage,
    ) -> Result<(), SendError> {
        // Wait for rate limit if necessary
        while !self.rate_limiter.try_consume(1) {
            tokio::time::sleep(self.rate_limiter.time_until_refill()).await;
        }
        
        // Acquire flow control permit (blocking)
        let _permit = self.flow_controller.acquire_permit_blocking().await;
        
        // Send message
        self.router.route_message(target, message).await
            .map_err(SendError::RouterError)
    }
    
    pub fn get_flow_stats(&self) -> FlowControlStats {
        self.flow_controller.get_stats()
    }
    
    pub fn get_rate_stats(&self) -> RateStats {
        RateStats {
            available_tokens: self.rate_limiter.available_tokens(),
            max_tokens: self.rate_limiter.max_tokens(),
            refill_rate: self.rate_limiter.refill_rate(),
        }
    }
    
    // Bonus: Send with priority (consumes more tokens for priority)
    pub async fn send_with_priority(
        &self,
        target: ActorId,
        message: ActorMessage,
        priority: MessagePriority,
    ) -> Result<(), SendError> {
        let token_cost = match priority {
            MessagePriority::Low => 1,
            MessagePriority::Normal => 1,
            MessagePriority::High => 2,
            MessagePriority::Critical => 3,
        };
        
        if !self.rate_limiter.try_consume(token_cost) {
            return Err(SendError::RateLimited);
        }
        
        let _permit = match self.flow_controller.acquire_permit().await {
            Some(permit) => permit,
            None => return Err(SendError::QueueFull),
        };
        
        self.router.route_message(target, message).await
            .map_err(SendError::RouterError)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

// Comprehensive test suite
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, timeout};

    #[test]
    fn test_actor_handle_complete() {
        let (sender, _receiver) = mpsc::unbounded_channel();
        let handle = ActorHandle::new(sender, 42);
        
        assert_eq!(handle.actor_id(), 42);
        assert!(!handle.is_closed());
        
        let cloned = handle.clone();
        assert_eq!(cloned.actor_id(), 42);
        
        // Test try_send
        let message = ActorMessage::SystemBroadcast {
            sender: 1,
            content: SystemMessage::NavigationHazard {
                location: (0.0, 0.0, 0.0),
                radius: 10.0,
                hazard_type: HazardType::Asteroid,
            },
        };
        assert!(handle.try_send(message).is_ok());
    }

    #[tokio::test]
    async fn test_message_router_complete() {
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
        
        let result = router.route_message(1, message.clone()).await;
        assert!(result.is_ok());
        
        let received = timeout(Duration::from_millis(100), receiver.recv()).await;
        assert!(received.is_ok());
        
        // Test statistics
        let stats = router.get_statistics().await;
        assert_eq!(stats.total_messages, 1);
        assert_eq!(stats.active_actors, 1);
    }

    #[tokio::test]
    async fn test_broadcast_complete() {
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
        
        assert!(timeout(Duration::from_millis(100), receiver1.recv()).await.is_ok());
        assert!(timeout(Duration::from_millis(100), receiver2.recv()).await.is_ok());
        
        // Test unregister
        router.unregister_actor(1).await;
        let stats = router.get_statistics().await;
        assert_eq!(stats.active_actors, 1);
    }

    #[tokio::test]
    async fn test_topic_subscription() {
        let router = MessageRouter::new();
        let (sender1, mut receiver1) = mpsc::unbounded_channel();
        let (sender2, mut receiver2) = mpsc::unbounded_channel();
        
        router.register_actor(1, sender1).await;
        router.register_actor(2, sender2).await;
        
        // Subscribe actor 1 to "alerts" topic
        router.subscribe_to_topic(1, "alerts".to_string()).await;
        
        let message = ActorMessage::SystemBroadcast {
            sender: 0,
            content: SystemMessage::NavigationHazard {
                location: (100.0, 200.0, 300.0),
                radius: 50.0,
                hazard_type: HazardType::Pirates,
            },
        };
        
        let delivered = router.publish_to_topic("alerts", message).await.unwrap();
        assert_eq!(delivered, 1); // Only actor 1 subscribed
        
        assert!(timeout(Duration::from_millis(100), receiver1.recv()).await.is_ok());
        assert!(timeout(Duration::from_millis(100), receiver2.recv()).await.is_err()); // Should timeout
        
        // Test topic subscriber queries
        let subscribers = router.get_topic_subscribers("alerts").await;
        assert_eq!(subscribers, vec![1]);
        
        let topics = router.get_actor_topics(1).await;
        assert_eq!(topics, vec!["alerts"]);
    }

    #[test]
    fn test_token_bucket_complete() {
        let bucket = TokenBucket::new(10, 5);
        
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.available_tokens(), 5);
        
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.available_tokens(), 0);
        
        assert!(!bucket.try_consume(1)); // Should fail, no tokens left
        
        // Test token properties
        assert_eq!(bucket.max_tokens(), 10);
        assert_eq!(bucket.refill_rate(), 5);
    }

    #[tokio::test]
    async fn test_flow_controller_complete() {
        let controller = FlowController::new(2, 10);
        let stats = controller.get_stats();
        
        assert_eq!(stats.max_concurrent, 2);
        assert_eq!(stats.current_queue_size, 0);
        assert_eq!(stats.available_permits, 2);
        
        // Acquire permits
        let _permit1 = controller.acquire_permit().await.unwrap();
        let _permit2 = controller.acquire_permit().await.unwrap();
        let permit3 = controller.acquire_permit().await;
        
        assert!(permit3.is_none()); // Should be None, no more permits
        
        let stats = controller.get_stats();
        assert_eq!(stats.available_permits, 0);
        assert_eq!(stats.current_queue_size, 2);
        
        // Drop permits and check stats
        drop(_permit1);
        drop(_permit2);
        
        // Give a small delay for drop to be processed
        tokio::task::yield_now().await;
        
        let stats = controller.get_stats();
        assert_eq!(stats.current_queue_size, 0);
    }

    #[tokio::test]
    async fn test_reliable_messenger_complete() {
        let router = MessageRouter::new();
        let messenger = ReliableMessenger::new(router.clone());
        
        // Register target actor
        let (sender, mut receiver) = mpsc::unbounded_channel();
        router.register_actor(1, sender).await;
        
        let message = ActorMessage::SystemBroadcast {
            sender: 0,
            content: SystemMessage::NavigationHazard {
                location: (100.0, 200.0, 300.0),
                radius: 50.0,
                hazard_type: HazardType::Asteroid,
            },
        };
        
        // Send reliable message
        let message_id = messenger.send_reliable(
            1,
            message,
            3,
            Duration::from_millis(100)
        ).await;
        
        assert!(message_id.is_ok());
        let msg_id = message_id.unwrap();
        
        // Should have message in pending
        assert_eq!(messenger.get_pending_count().await, 1);
        
        // Simulate acknowledgment
        messenger.handle_acknowledgment(msg_id).await;
        assert_eq!(messenger.get_pending_count().await, 0);
        
        // Should have received the message
        assert!(timeout(Duration::from_millis(100), receiver.recv()).await.is_ok());
    }

    #[tokio::test]
    async fn test_rate_limited_sender() {
        let router = MessageRouter::new();
        let (sender, mut receiver) = mpsc::unbounded_channel();
        router.register_actor(1, sender).await;
        
        let rate_sender = RateLimitedSender::new(router, 5, 10, 2); // 2 messages per second
        
        let message = ActorMessage::SystemBroadcast {
            sender: 0,
            content: SystemMessage::NavigationHazard {
                location: (0.0, 0.0, 0.0),
                radius: 10.0,
                hazard_type: HazardType::Debris,
            },
        };
        
        // Should be able to send initially
        assert!(rate_sender.send_with_backpressure(1, message.clone()).await.is_ok());
        assert!(rate_sender.send_with_backpressure(1, message.clone()).await.is_ok());
        
        // Check rate stats
        let rate_stats = rate_sender.get_rate_stats();
        assert!(rate_stats.available_tokens < rate_stats.max_tokens);
        
        // Check flow stats
        let flow_stats = rate_sender.get_flow_stats();
        assert_eq!(flow_stats.max_concurrent, 5);
    }

    #[tokio::test]
    async fn test_ship_actor_integration() {
        let router = MessageRouter::new();
        let mut ship = ShipActor::new_at_position(1, router.clone(), (0.0, 0.0, 0.0));
        let mut state = ShipState::Idle;
        
        // Test position update
        let update_msg = ShipActorMessage::UpdatePosition { dt: 1.0 };
        assert!(ship.handle_message(update_msg, &mut state).await.is_ok());
        
        // Test destination setting
        let dest_msg = ShipActorMessage::SetDestination { target: (100.0, 100.0, 100.0) };
        assert!(ship.handle_message(dest_msg, &mut state).await.is_ok());
        
        match state {
            ShipState::Flying { destination } => {
                assert_eq!(destination, (100.0, 100.0, 100.0));
            }
            _ => panic!("Expected Flying state"),
        }
        
        // Test actor properties
        assert_eq!(ship.get_position(), (0.0, 0.0, 0.0));
        assert_eq!(ship.get_fuel_level(), 1.0);
        assert_eq!(ship.get_cargo_count(), 0);
        assert_eq!(ship.get_cargo_value(), 0.0);
    }

    #[tokio::test]
    async fn test_message_priority() {
        let router = MessageRouter::new();
        let (sender, _receiver) = mpsc::unbounded_channel();
        router.register_actor(1, sender).await;
        
        let rate_sender = RateLimitedSender::new(router, 5, 10, 5);
        
        let message = ActorMessage::SystemBroadcast {
            sender: 0,
            content: SystemMessage::NavigationHazard {
                location: (0.0, 0.0, 0.0),
                radius: 10.0,
                hazard_type: HazardType::Pirates,
            },
        };
        
        // Critical messages consume more tokens
        assert!(rate_sender.send_with_priority(1, message, MessagePriority::Critical).await.is_ok());
        
        let rate_stats = rate_sender.get_rate_stats();
        // Should have consumed 3 tokens for critical message
        assert!(rate_stats.available_tokens <= rate_stats.max_tokens - 3);
    }
}