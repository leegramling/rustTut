# Tutorial 01: Rust Design Patterns

## Learning Objectives
- Understand and implement common Rust design patterns
- Learn Builder, Factory, and Observer patterns in Rust context
- Apply patterns to space simulation components
- Practice ownership, borrowing, and lifetimes through patterns

## Key Concepts

### 1. Builder Pattern
The Builder pattern is especially useful in Rust due to its ownership system. It allows constructing complex objects step by step while maintaining compile-time safety.

```rust
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

// Usage
let ship = ShipBuilder::new()
    .id(1)
    .name("Starfighter")
    .cargo_capacity(500)
    .build()?;
```

### 2. Factory Pattern
Factory pattern in Rust can leverage enums and match expressions for type-safe object creation.

```rust
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
}
```

### 3. Observer Pattern with Channels
Rust's ownership system makes traditional observer patterns challenging. Instead, we use channels for event-driven communication.

```rust
use tokio::sync::broadcast;

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
}

// Observer implementation
pub struct ResourceTracker {
    event_receiver: broadcast::Receiver<SimulationEvent>,
    total_resources_mined: u32,
}

impl ResourceTracker {
    pub fn new(event_bus: &EventBus) -> Self {
        Self {
            event_receiver: event_bus.subscribe(),
            total_resources_mined: 0,
        }
    }
    
    pub async fn run(&mut self) {
        while let Ok(event) = self.event_receiver.recv().await {
            match event {
                SimulationEvent::ResourceMined { amount, .. } => {
                    self.total_resources_mined += amount;
                    println!("Total resources mined: {}", self.total_resources_mined);
                }
                _ => {} // Ignore other events
            }
        }
    }
}
```

### 4. Strategy Pattern with Traits
Use traits to define interchangeable algorithms.

```rust
pub trait NavigationStrategy {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)>;
}

pub struct DirectNavigation;
pub struct EfficientNavigation;
pub struct SafeNavigation;

impl NavigationStrategy for DirectNavigation {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        vec![from, to] // Direct line
    }
}

impl NavigationStrategy for EfficientNavigation {
    fn calculate_route(&self, from: (f32, f32, f32), to: (f32, f32, f32)) -> Vec<(f32, f32, f32)> {
        // Implement fuel-efficient pathfinding
        vec![from, to] // Simplified
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
}
```

## Practical Applications in Space Simulation

1. **Builder Pattern**: Constructing ships, stations, and other complex entities with many optional parameters
2. **Factory Pattern**: Creating different types of entities based on configuration or game state
3. **Observer Pattern**: Event-driven communication between simulation components
4. **Strategy Pattern**: Interchangeable AI behaviors, navigation algorithms, and resource management strategies

## Key Takeaways

- Rust's ownership system influences how patterns are implemented
- Use `Box<dyn Trait>` for trait objects when you need runtime polymorphism
- Channels provide a safe alternative to traditional observer patterns
- Builder pattern works well with Rust's move semantics
- Always handle errors appropriately with `Result<T, E>`

## Next Steps

In the next tutorial, we'll explore generics and traits in depth, building upon these patterns to create flexible and reusable simulation components.