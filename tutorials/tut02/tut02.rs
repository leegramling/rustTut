// Tutorial 02: Generics and Traits
// Complete the following exercises to practice generics and traits in Rust

use std::collections::HashMap;
use std::marker::PhantomData;

// Helper types for the exercises
#[derive(Debug, Clone, PartialEq)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vector3<T> 
where 
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<Output = T> + Copy
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    
    pub fn magnitude(&self) -> T 
    where 
        T: std::ops::Add<Output = T> + From<f32>,
    {
        // Simplified magnitude calculation
        self.x + self.y + self.z
    }
}

impl<T> std::ops::Add for Vector3<T> 
where 
    T: std::ops::Add<Output = T>
{
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T> std::ops::Sub for Vector3<T> 
where 
    T: std::ops::Sub<Output = T>
{
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T> std::ops::Mul<T> for Vector3<T> 
where 
    T: std::ops::Mul<Output = T> + Copy
{
    type Output = Self;
    
    fn mul(self, scalar: T) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// Exercise 1: Generic Resource Container
// TODO: Complete the ResourceContainer implementation
#[derive(Debug, Clone)]
pub struct ResourceContainer<T> {
    // TODO: Add fields for contents (Vec<T>) and capacity (usize)
}

impl<T> ResourceContainer<T> {
    // TODO: Implement new() method that takes capacity and returns empty container
    pub fn new(capacity: usize) -> Self {
        todo!("Create new ResourceContainer with given capacity")
    }
    
    // TODO: Implement add() method that adds item if there's space
    // Should return Result<(), &'static str> with error "Container at capacity"
    pub fn add(&mut self, item: T) -> Result<(), &'static str> {
        todo!("Add item to container if space available")
    }
    
    // TODO: Implement remove() method that removes and returns last item
    pub fn remove(&mut self) -> Option<T> {
        todo!("Remove and return last item")
    }
    
    // TODO: Implement count() method that returns current number of items
    pub fn count(&self) -> usize {
        todo!("Return number of items in container")
    }
    
    // TODO: Implement is_full() method
    pub fn is_full(&self) -> bool {
        todo!("Check if container is at capacity")
    }
}

// TODO: Implement a generic transfer_resources function
// Should take two ResourceContainer references and a converter function
// Should move one item from source to destination using the converter
pub fn transfer_resources<T, U>(
    from: &mut ResourceContainer<T>,
    to: &mut ResourceContainer<U>,
    converter: impl Fn(T) -> U,
) -> Result<(), &'static str> {
    todo!("Transfer and convert resource from one container to another")
}

// Exercise 2: Entity Trait System
// TODO: Define Entity trait with the following methods:
// - id(&self) -> u32
// - position(&self) -> Vector3<f32>
// - set_position(&mut self, pos: Vector3<f32>)
// - distance_to(&self, other: &dyn Entity) -> f32 (default implementation)

// TODO: Define Movable trait that extends Entity with:
// - velocity(&self) -> Vector3<f32>
// - set_velocity(&mut self, vel: Vector3<f32>)
// - max_speed(&self) -> f32
// - update_position(&mut self, dt: f32) (default implementation)
// - accelerate(&mut self, acceleration: Vector3<f32>, dt: f32) (default implementation)

// TODO: Define CargoCarrier trait that extends Entity with:
// - Associated type CargoType
// - cargo_capacity(&self) -> usize
// - current_cargo(&self) -> &[Self::CargoType]
// - add_cargo(&mut self, cargo: Self::CargoType) -> Result<(), &'static str>
// - remove_cargo(&mut self) -> Option<Self::CargoType>
// - cargo_mass(&self) -> f32 where Self::CargoType: HasMass (default implementation)

// TODO: Define HasMass trait with:
// - mass(&self) -> f32

// Exercise 3: Resource Producer with Associated Types
// TODO: Define ResourceProducer trait with associated types:
// - type Resource
// - type Error  
// - produce(&mut self) -> Result<Self::Resource, Self::Error>
// - production_rate(&self) -> f32
// - can_produce(&self) -> bool

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

// TODO: Implement HasMass for Ore

#[derive(Debug)]
pub struct MiningStation {
    pub id: u32,
    pub position: Vector3<f32>,
    pub ore_reserves: u32,
    pub mining_rate: f32,
}

// TODO: Implement Entity for MiningStation

// TODO: Implement ResourceProducer for MiningStation
// Resource = Ore, Error = &'static str
// produce() should decrease ore_reserves and return Iron ore with purity 0.8, mass 10.0
// Return error "No ore reserves remaining" if ore_reserves is 0

// Exercise 4: Path Finding with Trait Bounds
// TODO: Define PathFinder trait with:
// - find_path(&self, start: Vector3<f32>, goal: Vector3<f32>) -> Vec<Vector3<f32>>
// - estimate_cost(&self, start: Vector3<f32>, goal: Vector3<f32>) -> f32

// TODO: Implement a generic optimize_route function with these bounds:
// - E: Entity + Movable + CargoCarrier
// - P: PathFinder + Clone  
// - E::CargoType: HasMass + Clone
// Should find the best route from available goals considering entity capabilities
pub fn optimize_route<E, P>(
    entities: &[E],
    pathfinder: &P,
    start: Vector3<f32>,
    goals: &[Vector3<f32>],
) -> Vec<Vector3<f32>>
where
    // TODO: Add appropriate trait bounds
{
    todo!("Find optimal route considering entity capabilities")
}

// Helper function for route cost calculation
fn calculate_route_cost<E>(route: &[Vector3<f32>], entities: &[E]) -> f32
where
    E: Entity + Movable + CargoCarrier,
    E::CargoType: HasMass,
{
    let mut total_cost = 0.0;
    
    // Cost based on entity cargo and speed
    for entity in entities {
        let cargo_mass = entity.cargo_mass();
        let max_speed = entity.max_speed();
        total_cost += cargo_mass / max_speed;
    }
    
    // Distance cost
    for window in route.windows(2) {
        let diff = window[1] - window[0];
        total_cost += diff.magnitude();
    }
    
    total_cost
}

// Exercise 5: Const Generics - Fixed Size Cargo Hold
// TODO: Implement FixedCargoHold with const generic CAPACITY
// Should use [Option<T>; CAPACITY] for storage and track count
#[derive(Debug)]
pub struct FixedCargoHold<T, const CAPACITY: usize> {
    // TODO: Add contents array and count field
}

impl<T, const CAPACITY: usize> FixedCargoHold<T, CAPACITY> {
    // TODO: Implement new() that creates empty cargo hold
    pub fn new() -> Self {
        todo!("Create new FixedCargoHold with all slots empty")
    }
    
    // TODO: Implement add() that finds first empty slot and adds item
    // Return Err(item) if no space available
    pub fn add(&mut self, item: T) -> Result<(), T> {
        todo!("Add item to first available slot")
    }
    
    // TODO: Implement remove() that removes from last occupied slot
    pub fn remove(&mut self) -> Option<T> {
        todo!("Remove item from last occupied slot")
    }
    
    // TODO: Implement capacity() that returns CAPACITY as usize
    pub fn capacity() -> usize {
        todo!("Return the compile-time capacity")
    }
    
    // TODO: Implement len() that returns current count
    pub fn len(&self) -> usize {
        todo!("Return current number of items")
    }
    
    // TODO: Implement is_empty()
    pub fn is_empty(&self) -> bool {
        todo!("Check if cargo hold is empty")
    }
}

// TODO: Create type aliases for different cargo hold sizes:
// type SmallCargoHold<T> = FixedCargoHold<T, 10>;
// type MediumCargoHold<T> = FixedCargoHold<T, 50>;  
// type LargeCargoHold<T> = FixedCargoHold<T, 200>;

// Exercise 6: Generic Fleet Manager
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct Assignment<C> {
    pub destination: Vector3<f32>,
    pub cargo_to_pickup: Vec<C>,
    pub cargo_to_deliver: Vec<C>,
    pub priority: Priority,
}

// TODO: Implement FleetManager with generic parameters V (vessel) and C (cargo)
// Should have appropriate trait bounds: V: Entity + Movable + CargoCarrier<CargoType = C>, C: HasMass + Clone
pub struct FleetManager<V, C> 
where 
    // TODO: Add trait bounds
{
    // TODO: Add fields for vessels (HashMap<u32, V>) and assignments (HashMap<u32, Assignment<C>>)
    // Include PhantomData<C> if needed
}

impl<V, C> FleetManager<V, C>
where 
    // TODO: Add same trait bounds as struct
{
    // TODO: Implement new() that creates empty fleet manager
    pub fn new() -> Self {
        todo!("Create new FleetManager")
    }
    
    // TODO: Implement add_vessel() that adds vessel using its ID as key
    pub fn add_vessel(&mut self, vessel: V) {
        todo!("Add vessel to fleet")
    }
    
    // TODO: Implement assign_mission() that assigns mission to vessel by ID
    // Return error if vessel not found
    pub fn assign_mission(&mut self, vessel_id: u32, assignment: Assignment<C>) -> Result<(), String> {
        todo!("Assign mission to vessel")
    }
    
    // TODO: Implement find_best_vessel_for_cargo() that finds most suitable unassigned vessel
    // Consider cargo mass, vessel capacity, and vessel speed
    // Return vessel ID of best match
    pub fn find_best_vessel_for_cargo(&self, cargo: &[C]) -> Option<u32> {
        todo!("Find best vessel for given cargo")
    }
    
    // TODO: Implement update_fleet() that updates positions of all vessels
    pub fn update_fleet(&mut self, dt: f32) {
        todo!("Update all vessel positions")
    }
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_container() {
        let mut container = ResourceContainer::<i32>::new(3);
        assert_eq!(container.count(), 0);
        assert!(!container.is_full());
        
        assert!(container.add(1).is_ok());
        assert!(container.add(2).is_ok());
        assert!(container.add(3).is_ok());
        assert!(container.is_full());
        
        assert!(container.add(4).is_err());
        
        assert_eq!(container.remove(), Some(3));
        assert_eq!(container.count(), 2);
    }

    #[test]
    fn test_transfer_resources() {
        let mut from = ResourceContainer::new(5);
        let mut to = ResourceContainer::new(5);
        
        from.add(10).unwrap();
        
        let result = transfer_resources(&mut from, &mut to, |x| x * 2);
        assert!(result.is_ok());
        assert_eq!(from.count(), 0);
        assert_eq!(to.count(), 1);
    }

    #[test] 
    fn test_fixed_cargo_hold() {
        let mut hold = FixedCargoHold::<i32, 3>::new();
        assert_eq!(hold.len(), 0);
        assert!(hold.is_empty());
        assert_eq!(FixedCargoHold::<i32, 3>::capacity(), 3);
        
        assert!(hold.add(1).is_ok());
        assert!(hold.add(2).is_ok());
        assert!(hold.add(3).is_ok());
        assert!(hold.add(4).is_err());
        
        assert_eq!(hold.remove(), Some(3));
        assert_eq!(hold.len(), 2);
    }

    // Additional tests would go here for traits and fleet manager
}