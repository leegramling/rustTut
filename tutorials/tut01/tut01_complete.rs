// Tutorial 01: Rust Design Patterns - Complete Solutions
// This file contains the complete implementations for all exercises

use tokio::sync::broadcast;

// Exercise 1: Builder Pattern - Complete Implementation
#[derive(Debug, Clone)]
pub struct Ship {
    pub id: u32,
    pub name: String,
    pub cargo_capacity: u32,
    pub fuel_capacity: f32,
    pub crew_capacity: u8,
}

pub struct ShipBuilder {
    id: Option<u32>,
    name: Option<String>,
    cargo_capacity: Option<u32>,
    fuel_capacity: Option<f32>,
    crew_capacity: Option<u8>,
}

impl ShipBuilder {
    pub fn new() -> Self {
        Self {
            id: None,
            name: None,
            cargo_capacity: None,
            fuel_capacity: None,
            crew_capacity: None,
        }
    }
    
    pub fn id(mut self, id: u32) -> Self {
        self.id = Some(id);
        self
    }
    
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    pub fn cargo_capacity(mut self, capacity: u32) -> Self {
        self.cargo_capacity = Some(capacity);
        self
    }
    
    pub fn fuel_capacity(mut self, capacity: f32) -> Self {
        self.fuel_capacity = Some(capacity);
        self
    }
    
    pub fn crew_capacity(mut self, capacity: u8) -> Self {
        self.crew_capacity = Some(capacity);
        self
    }
    
    pub fn build(self) -> Result<Ship, String> {
        Ok(Ship {
            id: self.id.ok_or("Ship ID is required")?,
            name: self.name.ok_or("Ship name is required")?,
            cargo_capacity: self.cargo_capacity.unwrap_or(100),
            fuel_capacity: self.fuel_capacity.unwrap_or(1000.0),
            crew_capacity: self.crew_capacity.unwrap_or(4),
        })
    }
}

impl Default for ShipBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Exercise 2: Factory Pattern - Complete Implementation
#[derive(Debug, Clone)]
pub enum ShipType {
    Cargo,
    Mining,
    Transport,
    Scout,
}

pub struct ShipFactory;

impl ShipFactory {
    pub fn create_ship(ship_type: ShipType, id: u32, name: String) -> Ship {
        match ship_type {
            ShipType::Cargo => ShipBuilder::new()
                .id(id)
                .name(name)
                .cargo_capacity(2000)
                .fuel_capacity(1500.0)
                .crew_capacity(3)
                .build()
                .expect("Failed to build cargo ship"),
                
            ShipType::Mining => ShipBuilder::new()
                .id(id)
                .name(name)
                .cargo_capacity(800)
                .fuel_capacity(2000.0)
                .crew_capacity(6)
                .build()
                .expect("Failed to build mining ship"),
                
            ShipType::Transport => ShipBuilder::new()
                .id(id)
                .name(name)
                .cargo_capacity(50)
                .fuel_capacity(800.0)
                .crew_capacity(12)
                .build()
                .expect("Failed to build transport ship"),
                
            ShipType::Scout => ShipBuilder::new()
                .id(id)
                .name(name)
                .cargo_capacity(20)
                .fuel_capacity(500.0)
                .crew_capacity(2)
                .build()
                .expect("Failed to build scout ship"),
        }
    }
    
    // Bonus: Factory method with default names
    pub fn create_ship_with_defaults(ship_type: ShipType, id: u32) -> Ship {
        let default_name = match ship_type {
            ShipType::Cargo => format!("Cargo-{}", id),
            ShipType::Mining => format!("Miner-{}", id),
            ShipType::Transport => format!("Transport-{}", id),
            ShipType::Scout => format!("Scout-{}", id),
        };
        
        Self::create_ship(ship_type, id, default_name)
    }
}

// Exercise 3: Observer Pattern with Channels - Complete Implementation
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    ShipDocked { ship_id: u32, station_id: u32 },
    ShipUndocked { ship_id: u32, station_id: u32 },
    ResourceMined { ship_id: u32, resource_type: String, amount: u32 },
    CargoTransferred { from_id: u32, to_id: u32, resource: String, amount: u32 },
}

pub struct EventBus {
    sender: broadcast::Sender<SimulationEvent>,
}

impl EventBus {
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(1000);
        Self { sender }
    }
    
    pub fn subscribe(&self) -> broadcast::Receiver<SimulationEvent> {
        self.sender.subscribe()
    }
    
    pub fn publish(&self, event: SimulationEvent) -> Result<usize, broadcast::error::SendError<SimulationEvent>> {
        self.sender.send(event)
    }
    
    // Bonus: Get subscriber count
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ResourceTracker {
    event_receiver: broadcast::Receiver<SimulationEvent>,
    total_resources_mined: u32,
    resources_by_type: std::collections::HashMap<String, u32>,
}

impl ResourceTracker {
    pub fn new(event_bus: &EventBus) -> Self {
        Self {
            event_receiver: event_bus.subscribe(),
            total_resources_mined: 0,
            resources_by_type: std::collections::HashMap::new(),
        }
    }
    
    pub async fn run(&mut self) {
        while let Ok(event) = self.event_receiver.recv().await {
            match event {
                SimulationEvent::ResourceMined { resource_type, amount, ship_id } => {
                    self.total_resources_mined += amount;
                    *self.resources_by_type.entry(resource_type.clone()).or_insert(0) += amount;
                    println!("Ship {} mined {} {}", ship_id, amount, resource_type);
                    println!("Total resources mined: {}", self.total_resources_mined);
                }
                _ => {} // Ignore other events
            }
        }
    }
    
    // Bonus: Get statistics
    pub fn get_total_mined(&self) -> u32 {
        self.total_resources_mined
    }
    
    pub fn get_resources_by_type(&self) -> &std::collections::HashMap<String, u32> {
        &self.resources_by_type
    }
}

// Exercise 4: Strategy Pattern - Complete Implementation
pub trait NavigationStrategy {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)>;
    
    // Bonus: Estimate fuel cost
    fn estimate_fuel_cost(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> f32 {
        let route = self.calculate_route(from, to);
        route.windows(2).map(|segment| {
            let (x1, y1, z1) = segment[0];
            let (x2, y2, z2) = segment[1];
            ((x2 - x1).powi(2) + (y2 - y1).powi(2) + (z2 - z1).powi(2)).sqrt()
        }).sum()
    }
}

pub struct DirectNavigation;

impl NavigationStrategy for DirectNavigation {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        vec![from, to]
    }
}

pub struct EfficientNavigation {
    fuel_efficiency_factor: f32,
}

impl EfficientNavigation {
    pub fn new(fuel_efficiency_factor: f32) -> Self {
        Self { fuel_efficiency_factor }
    }
}

impl NavigationStrategy for EfficientNavigation {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        // Simplified efficient routing - could implement actual pathfinding
        let (x1, y1, z1) = from;
        let (x2, y2, z2) = to;
        
        // Add intermediate waypoint for fuel efficiency
        let mid_x = (x1 + x2) / 2.0;
        let mid_y = (y1 + y2) / 2.0;
        let mid_z = (z1 + z2) / 2.0 + 10.0; // Slightly higher for efficiency
        
        vec![from, (mid_x, mid_y, mid_z), to]
    }
    
    fn estimate_fuel_cost(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> f32 {
        NavigationStrategy::estimate_fuel_cost(self, from, to) * self.fuel_efficiency_factor
    }
}

pub struct SafeNavigation {
    safety_margin: f32,
}

impl SafeNavigation {
    pub fn new(safety_margin: f32) -> Self {
        Self { safety_margin }
    }
}

impl NavigationStrategy for SafeNavigation {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        let (x1, y1, z1) = from;
        let (x2, y2, z2) = to;
        
        // Add multiple waypoints for safety
        let waypoint1 = (
            x1 + (x2 - x1) * 0.25,
            y1 + (y2 - y1) * 0.25 + self.safety_margin,
            z1 + (z2 - z1) * 0.25
        );
        
        let waypoint2 = (
            x1 + (x2 - x1) * 0.75,
            y1 + (y2 - y1) * 0.75 + self.safety_margin,
            z1 + (z2 - z1) * 0.75
        );
        
        vec![from, waypoint1, waypoint2, to]
    }
}

pub struct Navigator {
    strategy: Box<dyn NavigationStrategy>,
}

impl Navigator {
    pub fn new(strategy: Box<dyn NavigationStrategy>) -> Self {
        Self { strategy }
    }
    
    pub fn set_strategy(&mut self, strategy: Box<dyn NavigationStrategy>) {
        self.strategy = strategy;
    }
    
    pub fn navigate(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        self.strategy.calculate_route(from, to)
    }
    
    pub fn estimate_fuel_cost(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> f32 {
        self.strategy.estimate_fuel_cost(from, to)
    }
}

// Comprehensive test suite
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{timeout, Duration};

    #[test]
    fn test_ship_builder_complete() {
        // Test successful build
        let ship = ShipBuilder::new()
            .id(1)
            .name("Test Ship")
            .cargo_capacity(500)
            .fuel_capacity(1200.0)
            .crew_capacity(5)
            .build()
            .unwrap();
        
        assert_eq!(ship.id, 1);
        assert_eq!(ship.name, "Test Ship");
        assert_eq!(ship.cargo_capacity, 500);
        assert_eq!(ship.fuel_capacity, 1200.0);
        assert_eq!(ship.crew_capacity, 5);
        
        // Test with defaults
        let ship_with_defaults = ShipBuilder::new()
            .id(2)
            .name("Default Ship")
            .build()
            .unwrap();
        
        assert_eq!(ship_with_defaults.cargo_capacity, 100);
        assert_eq!(ship_with_defaults.fuel_capacity, 1000.0);
        assert_eq!(ship_with_defaults.crew_capacity, 4);
        
        // Test missing required field
        let result = ShipBuilder::new().id(3).build();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("name"));
    }

    #[test]
    fn test_ship_factory_complete() {
        let cargo_ship = ShipFactory::create_ship(ShipType::Cargo, 1, "Cargo1".to_string());
        assert_eq!(cargo_ship.cargo_capacity, 2000);
        assert_eq!(cargo_ship.fuel_capacity, 1500.0);
        assert_eq!(cargo_ship.crew_capacity, 3);
        
        let mining_ship = ShipFactory::create_ship(ShipType::Mining, 2, "Miner1".to_string());
        assert_eq!(mining_ship.cargo_capacity, 800);
        assert_eq!(mining_ship.fuel_capacity, 2000.0);
        assert_eq!(mining_ship.crew_capacity, 6);
        
        // Test default naming
        let scout = ShipFactory::create_ship_with_defaults(ShipType::Scout, 99);
        assert_eq!(scout.name, "Scout-99");
    }

    #[tokio::test]
    async fn test_event_bus_complete() {
        let event_bus = EventBus::new();
        let mut receiver1 = event_bus.subscribe();
        let mut receiver2 = event_bus.subscribe();
        
        assert_eq!(event_bus.subscriber_count(), 2);
        
        let event = SimulationEvent::ResourceMined {
            ship_id: 1,
            resource_type: "Iron".to_string(),
            amount: 100,
        };
        
        let sent_count = event_bus.publish(event.clone()).unwrap();
        assert_eq!(sent_count, 2);
        
        // Both receivers should get the event
        let received1 = timeout(Duration::from_millis(100), receiver1.recv()).await.unwrap().unwrap();
        let received2 = timeout(Duration::from_millis(100), receiver2.recv()).await.unwrap().unwrap();
        
        assert!(matches!(received1, SimulationEvent::ResourceMined { amount: 100, .. }));
        assert!(matches!(received2, SimulationEvent::ResourceMined { amount: 100, .. }));
    }

    #[tokio::test]
    async fn test_resource_tracker_complete() {
        let event_bus = EventBus::new();
        let mut tracker = ResourceTracker::new(&event_bus);
        
        // Spawn tracker task
        let tracker_handle = tokio::spawn(async move {
            // Run tracker for a short time
            timeout(Duration::from_millis(100), tracker.run()).await.ok();
            tracker
        });
        
        // Send events
        event_bus.publish(SimulationEvent::ResourceMined {
            ship_id: 1,
            resource_type: "Iron".to_string(),
            amount: 50,
        }).unwrap();
        
        event_bus.publish(SimulationEvent::ResourceMined {
            ship_id: 2,
            resource_type: "Gold".to_string(),
            amount: 25,
        }).unwrap();
        
        // Allow processing time
        tokio::time::sleep(Duration::from_millis(50)).await;
        
        let tracker = tracker_handle.await.unwrap();
        assert_eq!(tracker.get_total_mined(), 75);
        assert_eq!(tracker.get_resources_by_type().get("Iron"), Some(&50));
        assert_eq!(tracker.get_resources_by_type().get("Gold"), Some(&25));
    }

    #[test]
    fn test_navigation_strategies_complete() {
        let from = (0.0, 0.0, 0.0);
        let to = (10.0, 10.0, 10.0);
        
        // Test direct navigation
        let direct = DirectNavigation;
        let direct_route = direct.calculate_route(from, to);
        assert_eq!(direct_route.len(), 2);
        assert_eq!(direct_route[0], from);
        assert_eq!(direct_route[1], to);
        
        // Test efficient navigation
        let efficient = EfficientNavigation::new(0.8);
        let efficient_route = efficient.calculate_route(from, to);
        assert_eq!(efficient_route.len(), 3);
        
        // Test safe navigation
        let safe = SafeNavigation::new(5.0);
        let safe_route = safe.calculate_route(from, to);
        assert_eq!(safe_route.len(), 4);
        
        // Test fuel cost estimation
        let direct_cost = direct.estimate_fuel_cost(from, to);
        let efficient_cost = efficient.estimate_fuel_cost(from, to);
        assert!(efficient_cost < direct_cost); // Should be more efficient
    }

    #[test]
    fn test_navigator_complete() {
        let mut navigator = Navigator::new(Box::new(DirectNavigation));
        let from = (0.0, 0.0, 0.0);
        let to = (10.0, 10.0, 10.0);
        
        let route = navigator.navigate(from, to);
        assert_eq!(route.len(), 2);
        
        // Change strategy
        navigator.set_strategy(Box::new(SafeNavigation::new(2.0)));
        let new_route = navigator.navigate(from, to);
        assert_eq!(new_route.len(), 4);
        
        // Test fuel estimation
        let fuel_cost = navigator.estimate_fuel_cost(from, to);
        assert!(fuel_cost > 0.0);
    }
}