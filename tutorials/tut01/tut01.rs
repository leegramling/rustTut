// Tutorial 01: Rust Design Patterns
// Complete the following exercises to practice implementing design patterns in Rust

use tokio::sync::broadcast;

// Exercise 1: Implement the Builder Pattern
// TODO: Complete the Ship struct with the following fields:
// - id: u32
// - name: String  
// - cargo_capacity: u32
// - fuel_capacity: f32
// - crew_capacity: u8
pub struct Ship {
    // TODO: Add fields here
}

// TODO: Implement the ShipBuilder struct
// It should have Option<T> fields for each Ship field
pub struct ShipBuilder {
    // TODO: Add optional fields here
}

impl ShipBuilder {
    // TODO: Implement new() method that returns a ShipBuilder with all None values
    pub fn new() -> Self {
        todo!("Create new ShipBuilder with all None values")
    }
    
    // TODO: Implement builder methods (id, name, cargo_capacity, fuel_capacity, crew_capacity)
    // Each method should take `mut self`, set the corresponding field, and return `self`
    pub fn id(mut self, id: u32) -> Self {
        todo!("Set id and return self")
    }
    
    pub fn name(mut self, name: impl Into<String>) -> Self {
        todo!("Set name and return self")
    }
    
    // TODO: Implement remaining builder methods
    
    // TODO: Implement build() method that returns Result<Ship, String>
    // Should return error if required fields (id, name) are missing
    // Should use default values for optional fields
    pub fn build(self) -> Result<Ship, String> {
        todo!("Build Ship from builder, handling missing required fields")
    }
}

// Exercise 2: Implement the Factory Pattern
#[derive(Debug, Clone)]
pub enum ShipType {
    Cargo,
    Mining,
    Transport,
    Scout,
}

pub struct ShipFactory;

impl ShipFactory {
    // TODO: Implement create_ship method that takes ShipType, id, and name
    // Should return different Ship configurations based on type:
    // Cargo: cargo_capacity=2000, fuel_capacity=1500.0, crew_capacity=3
    // Mining: cargo_capacity=800, fuel_capacity=2000.0, crew_capacity=6  
    // Transport: cargo_capacity=50, fuel_capacity=800.0, crew_capacity=12
    // Scout: cargo_capacity=20, fuel_capacity=500.0, crew_capacity=2
    pub fn create_ship(ship_type: ShipType, id: u32, name: String) -> Ship {
        todo!("Implement factory method using match on ship_type")
    }
}

// Exercise 3: Implement Observer Pattern with Channels
#[derive(Debug, Clone)]
pub enum SimulationEvent {
    ShipDocked { ship_id: u32, station_id: u32 },
    ShipUndocked { ship_id: u32, station_id: u32 },
    ResourceMined { ship_id: u32, resource_type: String, amount: u32 },
    CargoTransferred { from_id: u32, to_id: u32, resource: String, amount: u32 },
}

// TODO: Implement EventBus struct with a broadcast sender
pub struct EventBus {
    // TODO: Add broadcast::Sender<SimulationEvent> field
}

impl EventBus {
    // TODO: Implement new() method that creates a broadcast channel with capacity 1000
    pub fn new() -> Self {
        todo!("Create broadcast channel and return EventBus")
    }
    
    // TODO: Implement subscribe() method that returns a receiver
    pub fn subscribe(&self) -> broadcast::Receiver<SimulationEvent> {
        todo!("Return a new receiver from the sender")
    }
    
    // TODO: Implement publish() method that sends an event
    pub fn publish(&self, event: SimulationEvent) -> Result<usize, broadcast::error::SendError<SimulationEvent>> {
        todo!("Send event through the broadcast channel")
    }
}

// TODO: Implement ResourceTracker that observes mining events
pub struct ResourceTracker {
    // TODO: Add receiver field and total_resources_mined counter
}

impl ResourceTracker {
    // TODO: Implement new() method that subscribes to event bus
    pub fn new(event_bus: &EventBus) -> Self {
        todo!("Create ResourceTracker with event subscription")
    }
    
    // TODO: Implement async run() method that processes events
    // Should increment total_resources_mined when ResourceMined events are received
    pub async fn run(&mut self) {
        todo!("Process events in a loop, tracking mined resources")
    }
}

// Exercise 4: Implement Strategy Pattern
// TODO: Define NavigationStrategy trait with calculate_route method
// Method signature: fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)>

// TODO: Implement DirectNavigation strategy (returns vec![from, to])
pub struct DirectNavigation;

// TODO: Implement EfficientNavigation strategy (for now, same as DirectNavigation)
pub struct EfficientNavigation;

// TODO: Implement SafeNavigation strategy (for now, same as DirectNavigation) 
pub struct SafeNavigation;

// TODO: Implement Navigator struct that uses a NavigationStrategy
pub struct Navigator {
    // TODO: Add strategy field using Box<dyn NavigationStrategy>
}

impl Navigator {
    // TODO: Implement new() method that takes a boxed strategy
    pub fn new(strategy: Box<dyn NavigationStrategy>) -> Self {
        todo!("Create Navigator with given strategy")
    }
    
    // TODO: Implement set_strategy() method to change strategy
    pub fn set_strategy(&mut self, strategy: Box<dyn NavigationStrategy>) {
        todo!("Replace current strategy")
    }
    
    // TODO: Implement navigate() method that delegates to strategy
    pub fn navigate(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        todo!("Call strategy's calculate_route method")
    }
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ship_builder() {
        let ship = ShipBuilder::new()
            .id(1)
            .name("Test Ship")
            .cargo_capacity(500)
            .build()
            .unwrap();
        
        assert_eq!(ship.id, 1);
        assert_eq!(ship.name, "Test Ship");
        assert_eq!(ship.cargo_capacity, 500);
    }

    #[test]
    fn test_ship_factory() {
        let cargo_ship = ShipFactory::create_ship(ShipType::Cargo, 1, "Cargo1".to_string());
        assert_eq!(cargo_ship.cargo_capacity, 2000);
    }

    #[tokio::test]
    async fn test_event_bus() {
        let event_bus = EventBus::new();
        let mut receiver = event_bus.subscribe();
        
        let event = SimulationEvent::ResourceMined {
            ship_id: 1,
            resource_type: "Iron".to_string(),
            amount: 100,
        };
        
        event_bus.publish(event.clone()).unwrap();
        let received = receiver.recv().await.unwrap();
        
        matches!(received, SimulationEvent::ResourceMined { amount: 100, .. });
    }

    #[test]
    fn test_navigation_strategy() {
        let navigator = Navigator::new(Box::new(DirectNavigation));
        let route = navigator.navigate((0.0, 0.0, 0.0), (10.0, 10.0, 10.0));
        assert_eq!(route.len(), 2);
    }
}