// Tutorial 06: Data-Oriented Programming
// Complete the following exercises to practice data-oriented design patterns in Rust

use std::collections::HashMap;
use std::any::{Any, TypeId};
use nalgebra::{Vector3, Matrix4};
use rayon::prelude::*;

// Exercise 1: Entity-Component-System Foundation

pub type EntityId = u32;

// TODO: Define Component trait with type_id() method
// Should be: 'static + Send + Sync
pub trait Component {
    // TODO: Add type_id() method that returns TypeId
}

// Exercise 2: Component Definitions

// TODO: Define Position component with:
// - x, y, z: f32 fields
// - new(x, y, z) constructor
// - to_vector3() method returning Vector3<f32>
// - distance_to(other: &Position) -> f32 method
#[derive(Debug, Clone, Copy)]
pub struct Position {
    // TODO: Add fields
}

impl Component for Position {
    // TODO: Implement Component trait
}

impl Position {
    // TODO: Implement new() constructor
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        todo!("Create new Position")
    }
    
    // TODO: Implement to_vector3() method
    pub fn to_vector3(&self) -> Vector3<f32> {
        todo!("Convert to Vector3")
    }
    
    // TODO: Implement distance_to() method
    pub fn distance_to(&self, other: &Position) -> f32 {
        todo!("Calculate distance to other position")
    }
}

// TODO: Define Velocity component with:
// - dx, dy, dz: f32 fields
// - new(dx, dy, dz) constructor
// - magnitude() -> f32 method
// - normalize(&mut self) method
#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    // TODO: Add fields
}

impl Component for Velocity {
    // TODO: Implement Component trait
}

impl Velocity {
    // TODO: Implement new() constructor
    pub fn new(dx: f32, dy: f32, dz: f32) -> Self {
        todo!("Create new Velocity")
    }
    
    // TODO: Implement magnitude() method
    pub fn magnitude(&self) -> f32 {
        todo!("Calculate velocity magnitude")
    }
    
    // TODO: Implement normalize() method
    pub fn normalize(&mut self) {
        todo!("Normalize velocity vector")
    }
}

// TODO: Define Ship component with:
// - ship_class: ShipClass
// - hull_integrity: f32 (0.0 to 1.0)
// - fuel_level: f32 (0.0 to 1.0)
// - cargo_capacity: f32
// - current_cargo: f32
// - crew_count: u32
// - max_crew: u32
#[derive(Debug, Clone)]
pub struct Ship {
    // TODO: Add fields
}

impl Component for Ship {
    // TODO: Implement Component trait
}

#[derive(Debug, Clone, Copy)]
pub enum ShipClass {
    Fighter,
    Merchant,
    Transport,
    Capital,
    Station,
}

impl Ship {
    // TODO: Implement new() constructor that sets defaults based on ship_class
    // Fighter: (10.0 cargo, 1 crew), Merchant: (500.0, 5), Transport: (1000.0, 10), 
    // Capital: (2000.0, 100), Station: (10000.0, 1000)
    pub fn new(ship_class: ShipClass) -> Self {
        todo!("Create new Ship with class-specific defaults")
    }
    
    // TODO: Implement can_carry() method - check if ship can carry additional cargo
    pub fn can_carry(&self, cargo_amount: f32) -> bool {
        todo!("Check if ship can carry additional cargo")
    }
    
    // TODO: Implement is_operational() method
    // Ship is operational if hull_integrity > 0.1, fuel_level > 0.0, crew_count > 0
    pub fn is_operational(&self) -> bool {
        todo!("Check if ship is operational")
    }
}

// TODO: Define Resource component with:
// - resource_type: ResourceType
// - quantity: f32
// - quality: f32 (0.0 to 1.0)
// - extraction_rate: f32
#[derive(Debug, Clone)]
pub struct Resource {
    // TODO: Add fields
}

impl Component for Resource {
    // TODO: Implement Component trait
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Iron,
    Gold,
    Platinum,
    Hydrogen,
    Helium3,
    Crystals,
    Organics,
}

impl Resource {
    // TODO: Implement new() constructor (set quality to 0.5, extraction_rate to 1.0)
    pub fn new(resource_type: ResourceType, quantity: f32) -> Self {
        todo!("Create new Resource")
    }
    
    // TODO: Implement value() method
    // Base values: Iron=10, Gold=100, Platinum=500, Hydrogen=5, Helium3=1000, Crystals=2000, Organics=50
    // Total value = base_value * quantity * quality
    pub fn value(&self) -> f32 {
        todo!("Calculate resource value")
    }
}

// Exercise 3: Dense Component Storage

// TODO: Implement ComponentStorage<T> with:
// - components: Vec<T>
// - entity_to_index: HashMap<EntityId, usize>
// - index_to_entity: Vec<EntityId>
// - free_indices: Vec<usize>
pub struct ComponentStorage<T: Component> {
    // TODO: Add fields
}

impl<T: Component> ComponentStorage<T> {
    // TODO: Implement new() constructor
    pub fn new() -> Self {
        todo!("Create new ComponentStorage")
    }
    
    // TODO: Implement insert() method
    // Reuse free indices when available, otherwise append
    pub fn insert(&mut self, entity: EntityId, component: T) {
        todo!("Insert component for entity")
    }
    
    // TODO: Implement get() method
    pub fn get(&self, entity: EntityId) -> Option<&T> {
        todo!("Get component reference for entity")
    }
    
    // TODO: Implement get_mut() method
    pub fn get_mut(&mut self, entity: EntityId) -> Option<&mut T> {
        todo!("Get mutable component reference for entity")
    }
    
    // TODO: Implement remove() method
    // Move last element to fill gap to maintain dense storage
    pub fn remove(&mut self, entity: EntityId) -> Option<T> {
        todo!("Remove component for entity")
    }
    
    // TODO: Implement iter() method for cache-friendly iteration
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &T)> {
        todo!("Return iterator over (EntityId, &T)")
    }
    
    // TODO: Implement iter_mut() method
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        todo!("Return mutable iterator over (EntityId, &mut T)")
    }
    
    // TODO: Implement as_slice() for SIMD operations
    pub fn as_slice(&self) -> &[T] {
        todo!("Return component slice")
    }
    
    // TODO: Implement as_mut_slice() for SIMD operations
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        todo!("Return mutable component slice")
    }
    
    // TODO: Implement len() method
    pub fn len(&self) -> usize {
        todo!("Return number of components")
    }
    
    // TODO: Implement is_empty() method
    pub fn is_empty(&self) -> bool {
        todo!("Check if storage is empty")
    }
}

// Exercise 4: World ECS Container

// TODO: Implement World struct with:
// - next_entity_id: EntityId
// - storages: HashMap<TypeId, Box<dyn Any + Send + Sync>>
pub struct World {
    // TODO: Add fields
}

impl World {
    // TODO: Implement new() constructor
    pub fn new() -> Self {
        todo!("Create new World")
    }
    
    // TODO: Implement create_entity() method
    pub fn create_entity(&mut self) -> EntityId {
        todo!("Create new entity and return its ID")
    }
    
    // TODO: Implement add_component() method
    pub fn add_component<T: Component>(&mut self, entity: EntityId, component: T) {
        todo!("Add component to entity")
    }
    
    // TODO: Implement get_component() method
    pub fn get_component<T: Component>(&self, entity: EntityId) -> Option<&T> {
        todo!("Get component reference for entity")
    }
    
    // TODO: Implement get_component_mut() method
    pub fn get_component_mut<T: Component>(&mut self, entity: EntityId) -> Option<&mut T> {
        todo!("Get mutable component reference for entity")
    }
    
    // TODO: Implement remove_component() method
    pub fn remove_component<T: Component>(&mut self, entity: EntityId) -> Option<T> {
        todo!("Remove component from entity")
    }
    
    // TODO: Implement query() method for single component type
    pub fn query<T: Component>(&self) -> Option<impl Iterator<Item = (EntityId, &T)>> {
        todo!("Query all entities with component T")
    }
    
    // TODO: Implement query_mut() method
    pub fn query_mut<T: Component>(&mut self) -> Option<impl Iterator<Item = (EntityId, &mut T)>> {
        todo!("Query all entities with component T (mutable)")
    }
    
    // TODO: Implement get_components() for two component types
    pub fn get_components<T1, T2>(&self, entity: EntityId) -> Option<(&T1, &T2)>
    where
        T1: Component,
        T2: Component,
    {
        todo!("Get two components for entity")
    }
    
    // TODO: Implement query_components() for batch query of two component types
    pub fn query_components<T1, T2>(&self) -> Vec<(EntityId, &T1, &T2)>
    where
        T1: Component,
        T2: Component,
    {
        todo!("Query all entities with both T1 and T2 components")
    }
    
    // TODO: Implement helper methods get_storage(), get_storage_mut(), get_or_create_storage()
    fn get_storage<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        todo!("Get storage for component type T")
    }
    
    fn get_storage_mut<T: Component>(&mut self) -> Option<&mut ComponentStorage<T>> {
        todo!("Get mutable storage for component type T")
    }
    
    fn get_or_create_storage<T: Component>(&mut self) -> &mut ComponentStorage<T> {
        todo!("Get or create storage for component type T")
    }
}

// Exercise 5: System Implementation

// TODO: Define System trait with update() method
pub trait System {
    // TODO: Add update method that takes &mut World and dt: f32
}

// TODO: Implement MovementSystem that updates positions based on velocities
pub struct MovementSystem;

impl System for MovementSystem {
    // TODO: Implement update() method
    // Update all positions by adding velocity * dt
    fn update(&mut self, world: &mut World, dt: f32) {
        todo!("Update positions based on velocities")
    }
}

// TODO: Implement PhysicsSystem with gravity simulation
pub struct PhysicsSystem {
    // TODO: Add gravity_constant: f32 field
}

impl PhysicsSystem {
    // TODO: Implement new() constructor (gravity_constant = 9.81)
    pub fn new() -> Self {
        todo!("Create new PhysicsSystem")
    }
}

impl System for PhysicsSystem {
    // TODO: Implement update() method with parallel gravity calculations
    // Use rayon for parallel processing of gravity between entities
    fn update(&mut self, world: &mut World, dt: f32) {
        todo!("Apply gravity forces between entities")
    }
}

// TODO: Implement ResourceExtractionSystem
pub struct ResourceExtractionSystem;

impl System for ResourceExtractionSystem {
    // TODO: Implement update() method
    // Find ships near resources (distance < 50.0) and extract resources
    // Update both resource quantity and ship cargo
    fn update(&mut self, world: &mut World, dt: f32) {
        todo!("Process resource extraction")
    }
}

// Exercise 6: Spatial Data Structures

// TODO: Define AABB (Axis-Aligned Bounding Box) struct with:
// - min_x, min_y, min_z: f32
// - max_x, max_y, max_z: f32
#[derive(Debug, Clone, Copy)]
pub struct AABB {
    // TODO: Add fields
}

impl AABB {
    // TODO: Implement new() constructor
    pub fn new(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> Self {
        todo!("Create new AABB")
    }
    
    // TODO: Implement contains_point() method
    pub fn contains_point(&self, pos: &Position) -> bool {
        todo!("Check if AABB contains position")
    }
    
    // TODO: Implement intersects() method
    pub fn intersects(&self, other: &AABB) -> bool {
        todo!("Check if two AABBs intersect")
    }
    
    // TODO: Implement center() method
    pub fn center(&self) -> Position {
        todo!("Calculate AABB center")
    }
    
    // TODO: Implement subdivide() method returning [AABB; 8] for octree
    fn subdivide(&self) -> [AABB; 8] {
        todo!("Subdivide AABB into 8 octants")
    }
}

// TODO: Define OctreeNode struct with:
// - bounds: AABB
// - entities: Vec<EntityId>
// - children: Option<Box<[OctreeNode; 8]>>
// - depth: usize
#[derive(Debug)]
struct OctreeNode {
    // TODO: Add fields
}

impl OctreeNode {
    // TODO: Implement new() constructor
    fn new(bounds: AABB, depth: usize) -> Self {
        todo!("Create new OctreeNode")
    }
    
    // TODO: Implement insert() method
    // Insert entity at position, subdividing if necessary
    fn insert(&mut self, entity: EntityId, position: &Position, max_depth: usize, max_entities: usize) {
        todo!("Insert entity into octree node")
    }
    
    // TODO: Implement query_range() method
    // Find all entities within query bounds
    fn query_range(&self, query_bounds: &AABB, results: &mut Vec<EntityId>) {
        todo!("Query entities within bounds")
    }
}

// TODO: Define Octree struct with:
// - root: OctreeNode
// - max_depth: usize
// - max_entities_per_node: usize
pub struct Octree {
    // TODO: Add fields
}

impl Octree {
    // TODO: Implement new() constructor
    pub fn new(bounds: AABB, max_depth: usize, max_entities_per_node: usize) -> Self {
        todo!("Create new Octree")
    }
    
    // TODO: Implement insert() method
    pub fn insert(&mut self, entity: EntityId, position: &Position) {
        todo!("Insert entity into octree")
    }
    
    // TODO: Implement query_range() method
    pub fn query_range(&self, bounds: &AABB) -> Vec<EntityId> {
        todo!("Query entities within bounds")
    }
    
    // TODO: Implement query_sphere() method
    // Create AABB around sphere and query
    pub fn query_sphere(&self, center: &Position, radius: f32) -> Vec<EntityId> {
        todo!("Query entities within sphere")
    }
    
    // TODO: Implement clear() method
    pub fn clear(&mut self) {
        todo!("Clear octree")
    }
}

// Exercise 7: SIMD-Optimized Operations (Optional - requires target features)

// TODO: Implement SIMDMovementSystem for vectorized position updates
pub struct SIMDMovementSystem;

impl SIMDMovementSystem {
    // TODO: Implement update_positions_scalar() fallback method
    fn update_positions_scalar(
        positions: &mut [Position],
        velocities: &[Velocity], 
        dt: f32,
    ) {
        todo!("Update positions using scalar operations")
    }
    
    // TODO: Implement SIMD version for x86_64 with AVX (optional)
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn update_positions_avx(
        positions: &mut [Position],
        velocities: &[Velocity],
        dt: f32,
    ) {
        // This is advanced - implement scalar fallback first
        Self::update_positions_scalar(positions, velocities, dt);
    }
}

impl System for SIMDMovementSystem {
    // TODO: Implement update() method that chooses between SIMD and scalar
    fn update(&mut self, world: &mut World, dt: f32) {
        todo!("Update positions using SIMD or scalar operations")
    }
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_distance() {
        let pos1 = Position::new(0.0, 0.0, 0.0);
        let pos2 = Position::new(3.0, 4.0, 0.0);
        
        assert_eq!(pos1.distance_to(&pos2), 5.0);
    }

    #[test]
    fn test_velocity_magnitude() {
        let mut vel = Velocity::new(3.0, 4.0, 0.0);
        assert_eq!(vel.magnitude(), 5.0);
        
        vel.normalize();
        assert!((vel.magnitude() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_ship_creation() {
        let ship = Ship::new(ShipClass::Merchant);
        assert!(ship.is_operational());
        assert!(ship.can_carry(100.0));
        assert!(!ship.can_carry(600.0)); // Over capacity
    }

    #[test]
    fn test_component_storage() {
        let mut storage = ComponentStorage::<Position>::new();
        let pos = Position::new(1.0, 2.0, 3.0);
        
        storage.insert(1, pos);
        assert_eq!(storage.len(), 1);
        
        let retrieved = storage.get(1).unwrap();
        assert_eq!(retrieved.x, 1.0);
        
        let removed = storage.remove(1).unwrap();
        assert_eq!(removed.y, 2.0);
        assert!(storage.is_empty());
    }

    #[test]
    fn test_world_ecs() {
        let mut world = World::new();
        
        let entity = world.create_entity();
        world.add_component(entity, Position::new(1.0, 2.0, 3.0));
        world.add_component(entity, Velocity::new(0.1, 0.2, 0.3));
        
        let (pos, vel) = world.get_components::<Position, Velocity>(entity).unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(vel.dx, 0.1);
    }

    #[test]
    fn test_aabb() {
        let aabb = AABB::new(-10.0, -10.0, -10.0, 10.0, 10.0, 10.0);
        let pos = Position::new(5.0, 5.0, 5.0);
        
        assert!(aabb.contains_point(&pos));
        
        let center = aabb.center();
        assert_eq!(center.x, 0.0);
    }

    #[test]
    fn test_movement_system() {
        let mut world = World::new();
        let mut system = MovementSystem;
        
        let entity = world.create_entity();
        world.add_component(entity, Position::new(0.0, 0.0, 0.0));
        world.add_component(entity, Velocity::new(1.0, 1.0, 1.0));
        
        system.update(&mut world, 1.0);
        
        let pos = world.get_component::<Position>(entity).unwrap();
        assert_eq!(pos.x, 1.0);
        assert_eq!(pos.y, 1.0);
        assert_eq!(pos.z, 1.0);
    }

    #[test]
    fn test_resource_value() {
        let resource = Resource::new(ResourceType::Gold, 10.0);
        assert_eq!(resource.value(), 500.0); // 100 * 10 * 0.5
    }

    #[test] 
    fn test_octree_basic() {
        let bounds = AABB::new(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0);
        let mut octree = Octree::new(bounds, 4, 10);
        
        let pos = Position::new(10.0, 10.0, 10.0);
        octree.insert(1, &pos);
        
        let results = octree.query_sphere(&pos, 20.0);
        assert!(results.contains(&1));
    }
}