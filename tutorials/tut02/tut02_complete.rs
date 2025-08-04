// Tutorial 02: Generics and Traits - Complete Solutions
// This file contains the complete implementations for all exercises

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
        T: std::ops::Add<Output = T> + From<f32> + std::ops::Mul<Output = T>,
    {
        // Simplified magnitude for demo (real implementation would use sqrt)
        let sum = self.x * self.x + self.y * self.y + self.z * self.z;
        sum
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

// Exercise 1: Generic Resource Container - Complete Implementation
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
    
    pub fn is_empty(&self) -> bool {
        self.contents.is_empty()
    }
    
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.contents.len()
    }
}

impl<T> Default for ResourceContainer<T> {
    fn default() -> Self {
        Self::new(100) // Default capacity
    }
}

// Generic transfer function - Complete Implementation
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

// Batch transfer function (bonus)
pub fn transfer_batch<T, U>(
    from: &mut ResourceContainer<T>,
    to: &mut ResourceContainer<U>,
    converter: impl Fn(T) -> U,
    count: usize,
) -> Result<usize, &'static str> {
    let mut transferred = 0;
    
    for _ in 0..count {
        match transfer_resources(from, to, &converter) {
            Ok(()) => transferred += 1,
            Err(_) => break,
        }
    }
    
    if transferred > 0 {
        Ok(transferred)
    } else {
        Err("No items could be transferred")
    }
}

// Exercise 2: Entity Trait System - Complete Implementation
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

pub trait Movable: Entity {
    fn velocity(&self) -> Vector3<f32>;
    fn set_velocity(&mut self, vel: Vector3<f32>);
    fn max_speed(&self) -> f32;
    
    // Default implementations
    fn update_position(&mut self, dt: f32) {
        let current_pos = self.position();
        let vel = self.velocity();
        self.set_position(current_pos + vel * dt);
    }
    
    fn accelerate(&mut self, acceleration: Vector3<f32>, dt: f32) {
        let current_vel = self.velocity();
        let new_vel = current_vel + acceleration * dt;
        
        // Clamp to max speed
        let speed_squared = new_vel.x * new_vel.x + new_vel.y * new_vel.y + new_vel.z * new_vel.z;
        let max_speed = self.max_speed();
        
        if speed_squared > max_speed * max_speed {
            let speed = speed_squared.sqrt();
            let normalized = new_vel * (max_speed / speed);
            self.set_velocity(normalized);
        } else {
            self.set_velocity(new_vel);
        }
    }
    
    fn current_speed(&self) -> f32 {
        let vel = self.velocity();
        (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z).sqrt()
    }
}

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
    
    fn remaining_cargo_capacity(&self) -> usize {
        self.cargo_capacity() - self.current_cargo().len()
    }
    
    fn is_cargo_full(&self) -> bool {
        self.current_cargo().len() >= self.cargo_capacity()
    }
}

pub trait HasMass {
    fn mass(&self) -> f32;
}

// Exercise 3: Resource Producer with Associated Types - Complete Implementation
pub trait ResourceProducer {
    type Resource;
    type Error;
    
    fn produce(&mut self) -> Result<Self::Resource, Self::Error>;
    fn production_rate(&self) -> f32;
    fn can_produce(&self) -> bool;
    
    // Default implementations
    fn estimated_production_time(&self) -> f32 {
        if self.can_produce() {
            1.0 / self.production_rate()
        } else {
            f32::INFINITY
        }
    }
    
    fn batch_produce(&mut self, count: usize) -> Vec<Self::Resource> {
        let mut results = Vec::new();
        for _ in 0..count {
            if let Ok(resource) = self.produce() {
                results.push(resource);
            } else {
                break;
            }
        }
        results
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ore {
    pub ore_type: OreType,
    pub purity: f32,
    pub mass: f32,
}

#[derive(Debug, Clone, PartialEq)]
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

impl Ore {
    pub fn value(&self) -> f32 {
        let base_value = match self.ore_type {
            OreType::Iron => 10.0,
            OreType::Gold => 100.0,
            OreType::Platinum => 500.0,
            OreType::RareEarth => 1000.0,
        };
        
        base_value * self.purity * self.mass
    }
}

#[derive(Debug)]
pub struct MiningStation {
    pub id: u32,
    pub position: Vector3<f32>,
    pub ore_reserves: u32,
    pub mining_rate: f32,
    pub ore_type: OreType,
}

impl Entity for MiningStation {
    fn id(&self) -> u32 { 
        self.id 
    }
    
    fn position(&self) -> Vector3<f32> { 
        self.position 
    }
    
    fn set_position(&mut self, pos: Vector3<f32>) { 
        self.position = pos; 
    }
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
            ore_type: self.ore_type.clone(),
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

// Exercise 4: Path Finding with Trait Bounds - Complete Implementation
pub trait PathFinder {
    fn find_path(&self, start: Vector3<f32>, goal: Vector3<f32>) -> Vec<Vector3<f32>>;
    fn estimate_cost(&self, start: Vector3<f32>, goal: Vector3<f32>) -> f32;
}

// Simple pathfinder implementation
#[derive(Debug, Clone)]
pub struct DirectPathFinder;

impl PathFinder for DirectPathFinder {
    fn find_path(&self, start: Vector3<f32>, goal: Vector3<f32>) -> Vec<Vector3<f32>> {
        vec![start, goal]
    }
    
    fn estimate_cost(&self, start: Vector3<f32>, goal: Vector3<f32>) -> f32 {
        let diff = goal - start;
        (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt()
    }
}

// Advanced pathfinder with waypoints
#[derive(Debug, Clone)]
pub struct SafePathFinder {
    pub safety_margin: f32,
}

impl PathFinder for SafePathFinder {
    fn find_path(&self, start: Vector3<f32>, goal: Vector3<f32>) -> Vec<Vector3<f32>> {
        // Add intermediate waypoints for safety
        let mid = Vector3::new(
            (start.x + goal.x) / 2.0,
            (start.y + goal.y) / 2.0 + self.safety_margin,
            (start.z + goal.z) / 2.0,
        );
        
        vec![start, mid, goal]
    }
    
    fn estimate_cost(&self, start: Vector3<f32>, goal: Vector3<f32>) -> f32 {
        let path = self.find_path(start, goal);
        let mut cost = 0.0;
        
        for window in path.windows(2) {
            let diff = window[1] - window[0];
            cost += (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt();
        }
        
        cost
    }
}

pub fn optimize_route<E, P>(
    entities: &[E],
    pathfinder: &P,
    start: Vector3<f32>,
    goals: &[Vector3<f32>],
) -> Vec<Vector3<f32>>
where
    E: Entity + Movable + CargoCarrier,
    E::CargoType: HasMass + Clone,
    P: PathFinder + Clone,
{
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
    
    if best_route.is_empty() {
        // Fallback to direct route to first goal
        if let Some(&first_goal) = goals.first() {
            best_route = pathfinder.find_path(start, first_goal);
        }
    }
    
    best_route
}

fn calculate_route_cost<E>(route: &[Vector3<f32>], entities: &[E]) -> f32
where
    E: Entity + Movable + CargoCarrier,
    E::CargoType: HasMass,
{
    let mut total_cost = 0.0;
    
    // Cost based on entity capabilities
    for entity in entities {
        let cargo_mass = entity.cargo_mass();
        let max_speed = entity.max_speed();
        
        // Higher cargo mass increases cost, higher speed decreases cost
        let entity_cost = (cargo_mass + 1.0) / (max_speed + 1.0);
        total_cost += entity_cost;
    }
    
    // Distance cost
    for window in route.windows(2) {
        let diff = window[1] - window[0];
        let distance = (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z).sqrt();
        total_cost += distance;
    }
    
    total_cost
}

// Exercise 5: Const Generics - Fixed Size Cargo Hold - Complete Implementation
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
        
        unreachable!("Should have found empty slot")
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
    
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    pub fn is_full(&self) -> bool {
        self.count >= CAPACITY
    }
    
    pub fn remaining_capacity(&self) -> usize {
        CAPACITY - self.count
    }
    
    // Bonus: Get item at specific index without removing
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < CAPACITY {
            self.contents[index].as_ref()
        } else {
            None
        }
    }
    
    // Bonus: Iterator over items
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.contents.iter().filter_map(|slot| slot.as_ref())
    }
}

impl<T, const CAPACITY: usize> Default for FixedCargoHold<T, CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

// Type aliases for different cargo hold sizes
pub type SmallCargoHold<T> = FixedCargoHold<T, 10>;
pub type MediumCargoHold<T> = FixedCargoHold<T, 50>;
pub type LargeCargoHold<T> = FixedCargoHold<T, 200>;

// Exercise 6: Generic Fleet Manager - Complete Implementation
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

impl<C> Assignment<C> {
    pub fn new(destination: Vector3<f32>, priority: Priority) -> Self {
        Self {
            destination,
            cargo_to_pickup: Vec::new(),
            cargo_to_deliver: Vec::new(),
            priority,
        }
    }
    
    pub fn with_pickup(mut self, cargo: Vec<C>) -> Self {
        self.cargo_to_pickup = cargo;
        self
    }
    
    pub fn with_delivery(mut self, cargo: Vec<C>) -> Self {
        self.cargo_to_deliver = cargo;
        self
    }
}

pub struct FleetManager<V, C> 
where 
    V: Entity + Movable + CargoCarrier<CargoType = C>,
    C: HasMass + Clone,
{
    vessels: HashMap<u32, V>,
    assignments: HashMap<u32, Assignment<C>>,
    _phantom: PhantomData<C>,
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
    
    pub fn remove_vessel(&mut self, vessel_id: u32) -> Option<V> {
        self.assignments.remove(&vessel_id);
        self.vessels.remove(&vessel_id)
    }
    
    pub fn assign_mission(&mut self, vessel_id: u32, assignment: Assignment<C>) -> Result<(), String> {
        if !self.vessels.contains_key(&vessel_id) {
            return Err("Vessel not found".to_string());
        }
        
        self.assignments.insert(vessel_id, assignment);
        Ok(())
    }
    
    pub fn clear_assignment(&mut self, vessel_id: u32) -> Option<Assignment<C>> {
        self.assignments.remove(&vessel_id)
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
            
            // Check if vessel can carry the cargo
            let current_cargo_mass = vessel.cargo_mass();
            let max_cargo_mass = vessel.cargo_capacity() as f32 * 10.0; // Assume average item mass of 10
            let remaining_capacity = max_cargo_mass - current_cargo_mass;
            
            if remaining_capacity >= total_cargo_mass {
                // Calculate suitability score based on speed and remaining capacity
                let speed_factor = vessel.max_speed();
                let capacity_efficiency = remaining_capacity / max_cargo_mass;
                let score = speed_factor * capacity_efficiency;
                
                if score > best_score {
                    best_score = score;
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
    
    pub fn get_vessel(&self, vessel_id: u32) -> Option<&V> {
        self.vessels.get(&vessel_id)
    }
    
    pub fn get_vessel_mut(&mut self, vessel_id: u32) -> Option<&mut V> {
        self.vessels.get_mut(&vessel_id)
    }
    
    pub fn get_assignment(&self, vessel_id: u32) -> Option<&Assignment<C>> {
        self.assignments.get(&vessel_id)
    }
    
    pub fn vessel_count(&self) -> usize {
        self.vessels.len()
    }
    
    pub fn assigned_vessel_count(&self) -> usize {
        self.assignments.len()
    }
    
    pub fn available_vessel_count(&self) -> usize {
        self.vessels.len() - self.assignments.len()
    }
    
    // Bonus: Find vessels by priority
    pub fn vessels_by_priority(&self, priority: Priority) -> Vec<u32> {
        self.assignments
            .iter()
            .filter(|(_, assignment)| assignment.priority == priority)
            .map(|(id, _)| *id)
            .collect()
    }
    
    // Bonus: Auto-assign cargo to best available vessel
    pub fn auto_assign_cargo(&mut self, cargo: Vec<C>, destination: Vector3<f32>, priority: Priority) -> Result<u32, String> {
        if let Some(vessel_id) = self.find_best_vessel_for_cargo(&cargo) {
            let assignment = Assignment::new(destination, priority).with_pickup(cargo);
            self.assign_mission(vessel_id, assignment)?;
            Ok(vessel_id)
        } else {
            Err("No suitable vessel available".to_string())
        }
    }
}

impl<V, C> Default for FleetManager<V, C>
where 
    V: Entity + Movable + CargoCarrier<CargoType = C>,
    C: HasMass + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// Example ship implementation for testing
#[derive(Debug, Clone)]
pub struct Ship {
    pub id: u32,
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub max_speed: f32,
    pub cargo_hold: ResourceContainer<Ore>,
}

impl Entity for Ship {
    fn id(&self) -> u32 { self.id }
    fn position(&self) -> Vector3<f32> { self.position }
    fn set_position(&mut self, pos: Vector3<f32>) { self.position = pos; }
}

impl Movable for Ship {
    fn velocity(&self) -> Vector3<f32> { self.velocity }
    fn set_velocity(&mut self, vel: Vector3<f32>) { self.velocity = vel; }
    fn max_speed(&self) -> f32 { self.max_speed }
}

impl CargoCarrier for Ship {
    type CargoType = Ore;
    
    fn cargo_capacity(&self) -> usize { 
        self.cargo_hold.capacity 
    }
    
    fn current_cargo(&self) -> &[Self::CargoType] { 
        &self.cargo_hold.contents 
    }
    
    fn add_cargo(&mut self, cargo: Self::CargoType) -> Result<(), &'static str> {
        self.cargo_hold.add(cargo)
    }
    
    fn remove_cargo(&mut self) -> Option<Self::CargoType> {
        self.cargo_hold.remove()
    }
}

// Comprehensive test suite
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_container_complete() {
        let mut container = ResourceContainer::<i32>::new(3);
        assert_eq!(container.count(), 0);
        assert!(!container.is_full());
        assert!(container.is_empty());
        assert_eq!(container.remaining_capacity(), 3);
        
        assert!(container.add(1).is_ok());
        assert!(container.add(2).is_ok());
        assert!(container.add(3).is_ok());
        assert!(container.is_full());
        assert_eq!(container.remaining_capacity(), 0);
        
        assert!(container.add(4).is_err()); // Should fail
        
        assert_eq!(container.remove(), Some(3));
        assert_eq!(container.count(), 2);
        assert!(!container.is_full());
    }

    #[test]
    fn test_transfer_resources_complete() {
        let mut from = ResourceContainer::new(5);
        let mut to = ResourceContainer::new(5);
        
        from.add(10).unwrap();
        from.add(20).unwrap();
        
        let result = transfer_resources(&mut from, &mut to, |x| x * 2);
        assert!(result.is_ok());
        assert_eq!(from.count(), 1);
        assert_eq!(to.count(), 1);
        
        // Test batch transfer
        from.add(30).unwrap();
        let transferred = transfer_batch(&mut from, &mut to, |x| x / 2, 2).unwrap();
        assert_eq!(transferred, 2);
    }

    #[test] 
    fn test_fixed_cargo_hold_complete() {
        let mut hold = FixedCargoHold::<i32, 3>::new();
        assert_eq!(hold.len(), 0);
        assert!(hold.is_empty());
        assert!(!hold.is_full());
        assert_eq!(FixedCargoHold::<i32, 3>::capacity(), 3);
        assert_eq!(hold.remaining_capacity(), 3);
        
        assert!(hold.add(1).is_ok());
        assert!(hold.add(2).is_ok());
        assert!(hold.add(3).is_ok());
        assert!(hold.is_full());
        assert_eq!(hold.remaining_capacity(), 0);
        
        let failed_add = hold.add(4);
        assert!(failed_add.is_err());
        assert_eq!(failed_add.unwrap_err(), 4); // Returns the item back
        
        assert_eq!(hold.remove(), Some(3));
        assert_eq!(hold.len(), 2);
        assert!(!hold.is_full());
        
        // Test iterator
        let items: Vec<&i32> = hold.iter().collect();
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_mining_station() {
        let mut station = MiningStation {
            id: 1,
            position: Vector3::new(0.0, 0.0, 0.0),
            ore_reserves: 5,
            mining_rate: 2.0,
            ore_type: OreType::Gold,
        };
        
        assert!(station.can_produce());
        assert_eq!(station.production_rate(), 2.0);
        
        let ore = station.produce().unwrap();
        assert_eq!(ore.ore_type, OreType::Gold);
        assert_eq!(ore.purity, 0.8);
        assert_eq!(ore.mass(), 10.0);
        assert_eq!(station.ore_reserves, 4);
        
        // Test batch production
        let batch = station.batch_produce(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(station.ore_reserves, 1);
        
        let last_batch = station.batch_produce(5);
        assert_eq!(last_batch.len(), 1); // Only 1 remaining
        assert!(!station.can_produce());
    }

    #[test]
    fn test_pathfinder() {
        let direct = DirectPathFinder;
        let start = Vector3::new(0.0, 0.0, 0.0);
        let goal = Vector3::new(10.0, 0.0, 0.0);
        
        let path = direct.find_path(start, goal);
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], start);
        assert_eq!(path[1], goal);
        
        let cost = direct.estimate_cost(start, goal);
        assert_eq!(cost, 10.0);
        
        let safe = SafePathFinder { safety_margin: 5.0 };
        let safe_path = safe.find_path(start, goal);
        assert_eq!(safe_path.len(), 3);
        assert!(safe.estimate_cost(start, goal) > cost); // Should be longer
    }

    #[test]
    fn test_fleet_manager() {
        let mut fleet = FleetManager::new();
        
        let ship = Ship {
            id: 1,
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            max_speed: 10.0,
            cargo_hold: ResourceContainer::new(100),
        };
        
        fleet.add_vessel(ship);
        assert_eq!(fleet.vessel_count(), 1);
        assert_eq!(fleet.available_vessel_count(), 1);
        
        let assignment = Assignment::new(
            Vector3::new(50.0, 0.0, 0.0),
            Priority::High,
        );
        
        assert!(fleet.assign_mission(1, assignment).is_ok());
        assert_eq!(fleet.assigned_vessel_count(), 1);
        assert_eq!(fleet.available_vessel_count(), 0);
        
        // Test auto-assignment
        let cargo = vec![
            Ore { ore_type: OreType::Iron, purity: 0.9, mass: 5.0 }
        ];
        
        // Clear assignment first
        fleet.clear_assignment(1);
        
        let assigned_vessel = fleet.auto_assign_cargo(
            cargo,
            Vector3::new(100.0, 0.0, 0.0),
            Priority::Medium,
        );
        
        assert!(assigned_vessel.is_ok());
        assert_eq!(assigned_vessel.unwrap(), 1);
    }

    #[test]
    fn test_ship_implementation() {
        let mut ship = Ship {
            id: 42,
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(1.0, 0.0, 0.0),
            max_speed: 5.0,
            cargo_hold: ResourceContainer::new(10),
        };
        
        // Test Entity trait
        assert_eq!(ship.id(), 42);
        
        // Test Movable trait
        ship.update_position(2.0);
        assert_eq!(ship.position(), Vector3::new(2.0, 0.0, 0.0));
        
        // Test acceleration clamping
        ship.accelerate(Vector3::new(10.0, 0.0, 0.0), 1.0);
        assert!(ship.current_speed() <= ship.max_speed() + 0.01); // Allow for floating point errors
        
        // Test CargoCarrier trait
        let ore = Ore {
            ore_type: OreType::Iron,
            purity: 0.8,
            mass: 15.0,
        };
        
        assert!(ship.add_cargo(ore.clone()).is_ok());
        assert_eq!(ship.current_cargo().len(), 1);
        assert_eq!(ship.cargo_mass(), 15.0);
        assert!(!ship.is_cargo_full());
        
        let removed = ship.remove_cargo();
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().mass, 15.0);
    }
}