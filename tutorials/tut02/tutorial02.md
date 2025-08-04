# Tutorial 02: Generics and Traits

## Learning Objectives
- Master Rust's generic type system for code reuse and performance
- Implement and use traits for shared behavior and polymorphism
- Understand associated types, trait bounds, and where clauses
- Apply generic programming to space simulation components
- Learn advanced trait patterns: blanket implementations, trait objects, and GATs

## Key Concepts

### 1. Generic Types and Functions

Generics in Rust enable code reuse while maintaining zero-cost abstractions through monomorphization.

```rust
// Generic resource container that works with any resource type
#[derive(Debug, Clone)]
pub struct ResourceContainer<T> {
    contents: Vec<T>,
    capacity: usize,
}

impl<T> ResourceContainer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            contents: Vec::new(),
            capacity,
        }
    }
    
    pub fn add(&mut self, item: T) -> Result<(), &'static str> {
        if self.contents.len() >= self.capacity {
            return Err("Container at capacity");
        }
        self.contents.push(item);
        Ok(())
    }
    
    pub fn remove(&mut self) -> Option<T> {
        self.contents.pop()
    }
    
    pub fn count(&self) -> usize {
        self.contents.len()
    }
    
    pub fn is_full(&self) -> bool {
        self.contents.len() >= self.capacity
    }
}

// Generic function with multiple type parameters
pub fn transfer_resources<T, U>(
    from: &mut ResourceContainer<T>,
    to: &mut ResourceContainer<U>,
    converter: impl Fn(T) -> U,
) -> Result<(), &'static str> {
    if let Some(item) = from.remove() {
        let converted = converter(item);
        to.add(converted)?;
        Ok(())
    } else {
        Err("Source container is empty")
    }
}
```

### 2. Traits for Shared Behavior

Traits define shared behavior that types can implement, enabling polymorphism and code reuse.

```rust
// Core trait for all simulation entities
pub trait Entity {
    fn id(&self) -> u32;
    fn position(&self) -> Vector3<f32>;
    fn set_position(&mut self, pos: Vector3<f32>);
    
    // Default implementation
    fn distance_to(&self, other: &dyn Entity) -> f32 {
        let diff = self.position() - other.position();
        (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt()
    }
}

// Trait for entities that can move
pub trait Movable: Entity {
    fn velocity(&self) -> Vector3<f32>;
    fn set_velocity(&mut self, vel: Vector3<f32>);
    fn max_speed(&self) -> f32;
    
    fn update_position(&mut self, dt: f32) {
        let current_pos = self.position();
        let vel = self.velocity();
        self.set_position(current_pos + vel * dt);
    }
    
    fn accelerate(&mut self, acceleration: Vector3<f32>, dt: f32) {
        let current_vel = self.velocity();
        let new_vel = current_vel + acceleration * dt;
        
        // Clamp to max speed
        let speed = new_vel.magnitude();
        if speed > self.max_speed() {
            let normalized = new_vel.normalize();
            self.set_velocity(normalized * self.max_speed());
        } else {
            self.set_velocity(new_vel);
        }
    }
}

// Trait for entities that can carry cargo
pub trait CargoCarrier: Entity {
    type CargoType;
    
    fn cargo_capacity(&self) -> usize;
    fn current_cargo(&self) -> &[Self::CargoType];
    fn add_cargo(&mut self, cargo: Self::CargoType) -> Result<(), &'static str>;
    fn remove_cargo(&mut self) -> Option<Self::CargoType>;
    
    fn cargo_mass(&self) -> f32 
    where 
        Self::CargoType: HasMass,
    {
        self.current_cargo().iter().map(|item| item.mass()).sum()
    }
}

// Helper trait for items with mass
pub trait HasMass {
    fn mass(&self) -> f32;
}
```

### 3. Associated Types vs Generic Parameters

Associated types provide a cleaner API when there's a logical relationship between types.

```rust
// Iterator-like trait using associated types
pub trait ResourceProducer {
    type Resource;
    type Error;
    
    fn produce(&mut self) -> Result<Self::Resource, Self::Error>;
    fn production_rate(&self) -> f32;
    fn can_produce(&self) -> bool;
}

// Mining station implementation
#[derive(Debug)]
pub struct MiningStation {
    id: u32,
    position: Vector3<f32>,
    ore_reserves: u32,
    mining_rate: f32,
}

#[derive(Debug, Clone)]
pub struct Ore {
    pub ore_type: OreType,
    pub purity: f32,
    pub mass: f32,
}

#[derive(Debug, Clone)]
pub enum OreType {
    Iron,
    Gold,
    Platinum,
    RareEarth,
}

impl HasMass for Ore {
    fn mass(&self) -> f32 {
        self.mass
    }
}

impl Entity for MiningStation {
    fn id(&self) -> u32 { self.id }
    fn position(&self) -> Vector3<f32> { self.position }
    fn set_position(&mut self, pos: Vector3<f32>) { self.position = pos; }
}

impl ResourceProducer for MiningStation {
    type Resource = Ore;
    type Error = &'static str;
    
    fn produce(&mut self) -> Result<Self::Resource, Self::Error> {
        if !self.can_produce() {
            return Err("No ore reserves remaining");
        }
        
        self.ore_reserves -= 1;
        Ok(Ore {
            ore_type: OreType::Iron, // Simplified
            purity: 0.8,
            mass: 10.0,
        })
    }
    
    fn production_rate(&self) -> f32 {
        self.mining_rate
    }
    
    fn can_produce(&self) -> bool {
        self.ore_reserves > 0
    }
}
```

### 4. Trait Bounds and Where Clauses

Complex trait bounds help express precise requirements for generic functions.

```rust
// Complex generic function with multiple bounds
pub fn optimize_route<E, P>(
    entities: &[E],
    pathfinder: &P,
    start: Vector3<f32>,
    goals: &[Vector3<f32>],
) -> Vec<Vector3<f32>>
where
    E: Entity + Movable,
    P: PathFinder + Clone,
    E::CargoType: HasMass + Clone,
    E: CargoCarrier,
{
    // Find the best route considering entity speeds and cargo masses
    let mut best_route = Vec::new();
    let mut best_cost = f32::INFINITY;
    
    for goal in goals {
        let route = pathfinder.find_path(start, *goal);
        let cost = calculate_route_cost(&route, entities);
        
        if cost < best_cost {
            best_cost = cost;
            best_route = route;
        }
    }
    
    best_route
}

pub trait PathFinder {
    fn find_path(&self, start: Vector3<f32>, goal: Vector3<f32>) -> Vec<Vector3<f32>>;
    fn estimate_cost(&self, start: Vector3<f32>, goal: Vector3<f32>) -> f32;
}

fn calculate_route_cost<E>(route: &[Vector3<f32>], entities: &[E]) -> f32
where
    E: Entity + Movable + CargoCarrier,
    E::CargoType: HasMass,
{
    let mut total_cost = 0.0;
    
    for entity in entities {
        let cargo_mass = entity.cargo_mass();
        let max_speed = entity.max_speed();
        
        // Cost increases with cargo mass and decreases with speed
        let entity_cost = cargo_mass / max_speed;
        total_cost += entity_cost;
    }
    
    // Add distance cost
    for window in route.windows(2) {
        let distance = (window[1] - window[0]).magnitude();
        total_cost += distance;
    }
    
    total_cost
}
```

### 5. Blanket Implementations and Trait Objects

Advanced trait patterns for maximum code reuse and runtime polymorphism.

```rust
// Blanket implementation for all entities that are movable
impl<T> Entity for T 
where 
    T: Movable,
{
    // Provide default implementations based on Movable trait
    fn id(&self) -> u32 {
        // Default implementation
        0
    }
    
    fn position(&self) -> Vector3<f32> {
        self.position()
    }
    
    fn set_position(&mut self, pos: Vector3<f32>) {
        self.set_position(pos)
    }
}

// Trait for serializable entities
pub trait Serializable {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: &[u8]) -> Result<Self, String> where Self: Sized;
}

// Automatic serialization for any entity that implements specific traits
impl<T> Serializable for T 
where 
    T: Entity + serde::Serialize + serde::DeserializeOwned + Clone,
{
    fn serialize(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
    
    fn deserialize(data: &[u8]) -> Result<Self, String> {
        bincode::deserialize(data).map_err(|e| e.to_string())
    }
}

// Trait objects for runtime polymorphism
pub struct Simulation {
    entities: Vec<Box<dyn Entity>>,
    movable_entities: Vec<Box<dyn Movable>>,
}

impl Simulation {
    pub fn add_entity(&mut self, entity: Box<dyn Entity>) {
        self.entities.push(entity);
    }
    
    pub fn add_movable(&mut self, entity: Box<dyn Movable>) {
        self.movable_entities.push(entity);
    }
    
    pub fn update(&mut self, dt: f32) {
        // Update all movable entities
        for entity in &mut self.movable_entities {
            entity.update_position(dt);
        }
    }
    
    pub fn find_nearest_entity(&self, position: Vector3<f32>) -> Option<&dyn Entity> {
        let mut nearest: Option<&dyn Entity> = None;
        let mut min_distance = f32::INFINITY;
        
        for entity in &self.entities {
            let distance = (entity.position() - position).magnitude();
            if distance < min_distance {
                min_distance = distance;
                nearest = Some(entity.as_ref());
            }
        }
        
        nearest
    }
}
```

### 6. Generic Associated Types (GATs)

Advanced pattern for highly flexible trait definitions.

```rust
// Generic Associated Type for flexible resource processing
pub trait ResourceProcessor {
    type Input<T>;
    type Output<T>;
    type Error;
    
    fn process<T>(&mut self, input: Self::Input<T>) -> Result<Self::Output<T>, Self::Error>
    where
        T: Clone + Send;
}

// Refinery that processes any resource type
pub struct Refinery {
    efficiency: f32,
    capacity: usize,
}

impl ResourceProcessor for Refinery {
    type Input<T> = Vec<T>;
    type Output<T> = Vec<T>;
    type Error = String;
    
    fn process<T>(&mut self, mut input: Self::Input<T>) -> Result<Self::Output<T>, Self::Error> 
    where 
        T: Clone + Send,
    {
        if input.len() > self.capacity {
            return Err("Input exceeds refinery capacity".to_string());
        }
        
        // Simulate processing by duplicating based on efficiency
        let output_size = (input.len() as f32 * self.efficiency) as usize;
        let mut output = Vec::new();
        
        for _ in 0..output_size {
            if let Some(item) = input.pop() {
                output.push(item);
            }
        }
        
        Ok(output)
    }
}
```

### 7. Const Generics for Compile-Time Parameters

Use const generics for array sizes and other compile-time constants.

```rust
// Fixed-size cargo hold with compile-time capacity
#[derive(Debug)]
pub struct FixedCargoHold<T, const CAPACITY: usize> {
    contents: [Option<T>; CAPACITY],
    count: usize,
}

impl<T, const CAPACITY: usize> FixedCargoHold<T, CAPACITY> {
    pub fn new() -> Self {
        Self {
            contents: std::array::from_fn(|_| None),
            count: 0,
        }
    }
    
    pub fn add(&mut self, item: T) -> Result<(), T> {
        if self.count >= CAPACITY {
            return Err(item);
        }
        
        // Find first empty slot
        for slot in &mut self.contents {
            if slot.is_none() {
                *slot = Some(item);
                self.count += 1;
                return Ok(());
            }
        }
        
        unreachable!()
    }
    
    pub fn remove(&mut self) -> Option<T> {
        // Remove from last occupied slot
        for slot in self.contents.iter_mut().rev() {
            if slot.is_some() {
                self.count -= 1;
                return slot.take();
            }
        }
        None
    }
    
    pub fn capacity() -> usize {
        CAPACITY
    }
    
    pub fn len(&self) -> usize {
        self.count
    }
}

// Different ship types with different cargo capacities
type SmallCargoHold<T> = FixedCargoHold<T, 10>;
type MediumCargoHold<T> = FixedCargoHold<T, 50>;
type LargeCargoHold<T> = FixedCargoHold<T, 200>;
```

## Practical Application: Generic Fleet Management System

```rust
use std::collections::HashMap;
use std::marker::PhantomData;

// Generic fleet manager that can handle any type of vessel
pub struct FleetManager<V, C> 
where 
    V: Entity + Movable + CargoCarrier<CargoType = C>,
    C: HasMass + Clone,
{
    vessels: HashMap<u32, V>,
    assignments: HashMap<u32, Assignment<C>>,
    _phantom: PhantomData<C>,
}

#[derive(Debug, Clone)]
pub struct Assignment<C> {
    pub destination: Vector3<f32>,
    pub cargo_to_pickup: Vec<C>,
    pub cargo_to_deliver: Vec<C>,
    pub priority: Priority,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

impl<V, C> FleetManager<V, C>
where 
    V: Entity + Movable + CargoCarrier<CargoType = C>,
    C: HasMass + Clone,
{
    pub fn new() -> Self {
        Self {
            vessels: HashMap::new(),
            assignments: HashMap::new(),
            _phantom: PhantomData,
        }
    }
    
    pub fn add_vessel(&mut self, vessel: V) {
        let id = vessel.id();
        self.vessels.insert(id, vessel);
    }
    
    pub fn assign_mission(&mut self, vessel_id: u32, assignment: Assignment<C>) -> Result<(), String> {
        if !self.vessels.contains_key(&vessel_id) {
            return Err("Vessel not found".to_string());
        }
        
        self.assignments.insert(vessel_id, assignment);
        Ok(())
    }
    
    pub fn find_best_vessel_for_cargo(&self, cargo: &[C]) -> Option<u32> {
        let total_cargo_mass: f32 = cargo.iter().map(|c| c.mass()).sum();
        
        let mut best_vessel = None;
        let mut best_score = f32::NEG_INFINITY;
        
        for (id, vessel) in &self.vessels {
            // Skip if vessel is already assigned
            if self.assignments.contains_key(id) {
                continue;
            }
            
            // Calculate suitability score
            let current_cargo_mass = vessel.cargo_mass();
            let remaining_capacity = vessel.cargo_capacity() as f32 - current_cargo_mass;
            
            if remaining_capacity >= total_cargo_mass {
                let speed_factor = vessel.max_speed();
                let efficiency = speed_factor / (current_cargo_mass + 1.0);
                
                if efficiency > best_score {
                    best_score = efficiency;
                    best_vessel = Some(*id);
                }
            }
        }
        
        best_vessel
    }
    
    pub fn update_fleet(&mut self, dt: f32) {
        for vessel in self.vessels.values_mut() {
            vessel.update_position(dt);
        }
    }
}
```

## Key Takeaways

1. **Generics**: Enable code reuse while maintaining performance through monomorphization
2. **Traits**: Define shared behavior and enable polymorphism
3. **Associated Types**: Cleaner API when types have logical relationships
4. **Trait Bounds**: Express complex requirements for generic functions
5. **Blanket Implementations**: Provide functionality for entire categories of types
6. **Trait Objects**: Enable runtime polymorphism when compile-time isn't sufficient
7. **Const Generics**: Compile-time parameters for array sizes and other constants

## Best Practices

- Use associated types when there's a single logical type relationship
- Use generic parameters when multiple types might be valid
- Prefer trait bounds to trait objects for performance
- Use const generics for compile-time configuration
- Keep trait definitions focused and cohesive
- Leverage blanket implementations for common patterns

## Next Steps

In the next tutorial, we'll explore functional programming patterns in Rust, building on these generic and trait concepts to create more expressive and composable simulation systems.