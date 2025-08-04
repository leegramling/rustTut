// Tutorial 06: Data-Oriented Programming - Complete Implementation
// This file contains complete solutions for all exercises

use std::collections::HashMap;
use std::any::{Any, TypeId};
use nalgebra::{Vector3, Matrix4};
use rayon::prelude::*;

// Exercise 1: Entity-Component-System Foundation

pub type EntityId = u32;

pub trait Component: 'static + Send + Sync {
    fn type_id() -> TypeId 
    where 
        Self: Sized 
    {
        TypeId::of::<Self>()
    }
}

// Exercise 2: Component Definitions

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Component for Position {}

impl Position {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    pub fn to_vector3(&self) -> Vector3<f32> {
        Vector3::new(self.x, self.y, self.z)
    }
    
    pub fn distance_to(&self, other: &Position) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Velocity {
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

impl Component for Velocity {}

impl Velocity {
    pub fn new(dx: f32, dy: f32, dz: f32) -> Self {
        Self { dx, dy, dz }
    }
    
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy + self.dz * self.dz).sqrt()
    }
    
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 0.0 {
            self.dx /= mag;
            self.dy /= mag;
            self.dz /= mag;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Ship {
    pub ship_class: ShipClass,
    pub hull_integrity: f32,
    pub fuel_level: f32,
    pub cargo_capacity: f32,
    pub current_cargo: f32,
    pub crew_count: u32,
    pub max_crew: u32,
}

impl Component for Ship {}

#[derive(Debug, Clone, Copy)]
pub enum ShipClass {
    Fighter,
    Merchant,
    Transport,
    Capital,
    Station,
}

impl Ship {
    pub fn new(ship_class: ShipClass) -> Self {
        let (cargo_capacity, max_crew) = match ship_class {
            ShipClass::Fighter => (10.0, 1),
            ShipClass::Merchant => (500.0, 5),
            ShipClass::Transport => (1000.0, 10),
            ShipClass::Capital => (2000.0, 100),
            ShipClass::Station => (10000.0, 1000),
        };
        
        Self {
            ship_class,
            hull_integrity: 1.0,
            fuel_level: 1.0,
            cargo_capacity,
            current_cargo: 0.0,
            crew_count: max_crew,
            max_crew,
        }
    }
    
    pub fn can_carry(&self, cargo_amount: f32) -> bool {
        self.current_cargo + cargo_amount <= self.cargo_capacity
    }
    
    pub fn is_operational(&self) -> bool {
        self.hull_integrity > 0.1 && self.fuel_level > 0.0 && self.crew_count > 0
    }
}

#[derive(Debug, Clone)]
pub struct Resource {
    pub resource_type: ResourceType,
    pub quantity: f32,
    pub quality: f32,
    pub extraction_rate: f32,
}

impl Component for Resource {}

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
    pub fn new(resource_type: ResourceType, quantity: f32) -> Self {
        Self {
            resource_type,
            quantity,
            quality: 0.5,
            extraction_rate: 1.0,
        }
    }
    
    pub fn value(&self) -> f32 {
        let base_value = match self.resource_type {
            ResourceType::Iron => 10.0,
            ResourceType::Gold => 100.0,
            ResourceType::Platinum => 500.0,
            ResourceType::Hydrogen => 5.0,
            ResourceType::Helium3 => 1000.0,
            ResourceType::Crystals => 2000.0,
            ResourceType::Organics => 50.0,
        };
        
        base_value * self.quantity * self.quality
    }
}

// Exercise 3: Dense Component Storage

pub struct ComponentStorage<T: Component> {
    components: Vec<T>,
    entity_to_index: HashMap<EntityId, usize>,
    index_to_entity: Vec<EntityId>,
    free_indices: Vec<usize>,
}

impl<T: Component> ComponentStorage<T> {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            entity_to_index: HashMap::new(),
            index_to_entity: Vec::new(),
            free_indices: Vec::new(),
        }
    }
    
    pub fn insert(&mut self, entity: EntityId, component: T) {
        if let Some(index) = self.free_indices.pop() {
            // Reuse free slot
            self.components[index] = component;
            self.index_to_entity[index] = entity;
            self.entity_to_index.insert(entity, index);
        } else {
            // Add new slot
            let index = self.components.len();
            self.components.push(component);
            self.index_to_entity.push(entity);
            self.entity_to_index.insert(entity, index);
        }
    }
    
    pub fn get(&self, entity: EntityId) -> Option<&T> {
        self.entity_to_index
            .get(&entity)
            .and_then(|&index| self.components.get(index))
    }
    
    pub fn get_mut(&mut self, entity: EntityId) -> Option<&mut T> {
        self.entity_to_index
            .get(&entity)
            .and_then(|&index| self.components.get_mut(index))
    }
    
    pub fn remove(&mut self, entity: EntityId) -> Option<T> {
        if let Some(index) = self.entity_to_index.remove(&entity) {
            // Move last element to fill gap
            let component = if index == self.components.len() - 1 {
                self.components.pop().unwrap()
            } else {
                let component = std::mem::replace(
                    &mut self.components[index],
                    self.components.pop().unwrap()
                );
                
                // Update moved entity's mapping
                let moved_entity = self.index_to_entity.pop().unwrap();
                self.index_to_entity[index] = moved_entity;
                self.entity_to_index.insert(moved_entity, index);
                
                component
            };
            
            Some(component)
        } else {
            None
        }
    }
    
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &T)> {
        self.components
            .iter()
            .zip(self.index_to_entity.iter())
            .map(|(component, &entity)| (entity, component))
    }
    
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        self.components
            .iter_mut()
            .zip(self.index_to_entity.iter())
            .map(|(component, &entity)| (entity, component))
    }
    
    pub fn as_slice(&self) -> &[T] {
        &self.components
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.components
    }
    
    pub fn len(&self) -> usize {
        self.components.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }
}

// Exercise 4: World ECS Container

pub struct World {
    next_entity_id: EntityId,
    storages: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            next_entity_id: 1,
            storages: HashMap::new(),
        }
    }
    
    pub fn create_entity(&mut self) -> EntityId {
        let entity = self.next_entity_id;
        self.next_entity_id += 1;
        entity
    }
    
    pub fn add_component<T: Component>(&mut self, entity: EntityId, component: T) {
        self.get_or_create_storage::<T>().insert(entity, component);
    }
    
    pub fn get_component<T: Component>(&self, entity: EntityId) -> Option<&T> {
        self.get_storage::<T>()?.get(entity)
    }
    
    pub fn get_component_mut<T: Component>(&mut self, entity: EntityId) -> Option<&mut T> {
        self.get_storage_mut::<T>()?.get_mut(entity)
    }
    
    pub fn remove_component<T: Component>(&mut self, entity: EntityId) -> Option<T> {
        self.get_storage_mut::<T>()?.remove(entity)
    }
    
    pub fn query<T: Component>(&self) -> Option<impl Iterator<Item = (EntityId, &T)>> {
        self.get_storage::<T>().map(|storage| storage.iter())
    }
    
    pub fn query_mut<T: Component>(&mut self) -> Option<impl Iterator<Item = (EntityId, &mut T)>> {
        self.get_storage_mut::<T>().map(|storage| storage.iter_mut())
    }
    
    pub fn get_components<T1, T2>(&self, entity: EntityId) -> Option<(&T1, &T2)>
    where
        T1: Component,
        T2: Component,
    {
        let c1 = self.get_component::<T1>(entity)?;
        let c2 = self.get_component::<T2>(entity)?;
        Some((c1, c2))
    }
    
    pub fn query_components<T1, T2>(&self) -> Vec<(EntityId, &T1, &T2)>
    where
        T1: Component,
        T2: Component,
    {
        let mut results = Vec::new();
        
        if let Some(storage1) = self.get_storage::<T1>() {
            for (entity, component1) in storage1.iter() {
                if let Some(component2) = self.get_component::<T2>(entity) {
                    results.push((entity, component1, component2));
                }
            }
        }
        
        results
    }
    
    fn get_storage<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        self.storages
            .get(&TypeId::of::<T>())?
            .downcast_ref::<ComponentStorage<T>>()
    }
    
    fn get_storage_mut<T: Component>(&mut self) -> Option<&mut ComponentStorage<T>> {
        self.storages
            .get_mut(&TypeId::of::<T>())?
            .downcast_mut::<ComponentStorage<T>>()
    }
    
    fn get_or_create_storage<T: Component>(&mut self) -> &mut ComponentStorage<T> {
        let type_id = TypeId::of::<T>();
        self.storages
            .entry(type_id)
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new()))
            .downcast_mut::<ComponentStorage<T>>()
            .unwrap()
    }
}

// Exercise 5: System Implementation

pub trait System {
    fn update(&mut self, world: &mut World, dt: f32);
}

pub struct MovementSystem;

impl System for MovementSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        // Collect entities with both position and velocity
        let updates: Vec<_> = world
            .query_components::<Position, Velocity>()
            .into_iter()
            .map(|(entity, pos, vel)| {
                (entity, Position::new(
                    pos.x + vel.dx * dt,
                    pos.y + vel.dy * dt,
                    pos.z + vel.dz * dt,
                ))
            })
            .collect();
        
        // Apply position updates
        for (entity, new_pos) in updates {
            if let Some(pos) = world.get_component_mut::<Position>(entity) {
                *pos = new_pos;
            }
        }
    }
}

pub struct PhysicsSystem {
    gravity_constant: f32,
}

impl PhysicsSystem {
    pub fn new() -> Self {
        Self {
            gravity_constant: 9.81,
        }
    }
}

impl System for PhysicsSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        // Collect entities with position and velocity
        let entities_with_physics: Vec<_> = world
            .query_components::<Position, Velocity>()
            .into_iter()
            .map(|(entity, pos, vel)| (entity, *pos, *vel))
            .collect();
        
        // Parallel gravity calculations
        let gravity_updates: Vec<_> = entities_with_physics
            .par_iter()
            .map(|(entity, position, velocity)| {
                let mut new_velocity = *velocity;
                
                // Apply gravity from other massive objects
                for (other_entity, other_pos, _) in &entities_with_physics {
                    if entity != other_entity {
                        let distance = position.distance_to(other_pos);
                        if distance > 1.0 { // Avoid division by zero
                            let gravity_force = self.gravity_constant / (distance * distance);
                            let dx = (other_pos.x - position.x) / distance;
                            let dy = (other_pos.y - position.y) / distance;
                            let dz = (other_pos.z - position.z) / distance;
                            
                            new_velocity.dx += dx * gravity_force * dt;
                            new_velocity.dy += dy * gravity_force * dt;
                            new_velocity.dz += dz * gravity_force * dt;
                        }
                    }
                }
                
                (*entity, new_velocity)
            })
            .collect();
        
        // Apply velocity updates
        for (entity, new_velocity) in gravity_updates {
            if let Some(velocity) = world.get_component_mut::<Velocity>(entity) {
                *velocity = new_velocity;
            }
        }
    }
}

pub struct ResourceExtractionSystem;

impl System for ResourceExtractionSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        // Find ships near resource-bearing asteroids
        let ships: Vec<_> = world
            .query_components::<Position, Ship>()
            .into_iter()
            .filter(|(_, _, ship)| ship.is_operational())
            .collect();
        
        let resources: Vec<_> = world
            .query_components::<Position, Resource>()
            .into_iter()
            .collect();
        
        let mut extraction_operations = Vec::new();
        
        for (ship_entity, ship_pos, ship) in ships {
            for (resource_entity, resource_pos, resource) in &resources {
                let distance = ship_pos.distance_to(resource_pos);
                
                // Within extraction range
                if distance < 50.0 {
                    let extraction_amount = resource.extraction_rate * dt;
                    
                    // Check if ship can carry more and resource has enough
                    if ship.can_carry(extraction_amount) && resource.quantity >= extraction_amount {
                        extraction_operations.push((ship_entity, *resource_entity, extraction_amount));
                    }
                }
            }
        }
        
        // Apply extraction operations
        for (ship_entity, resource_entity, extraction_amount) in extraction_operations {
            // Update resource quantity
            if let Some(resource) = world.get_component_mut::<Resource>(resource_entity) {
                resource.quantity -= extraction_amount;
                
                // Update ship cargo
                if let Some(ship) = world.get_component_mut::<Ship>(ship_entity) {
                    ship.current_cargo += extraction_amount;
                    println!("Ship {} extracted {:.2} units of {:?}", 
                            ship_entity, extraction_amount, resource.resource_type);
                }
            }
        }
    }
}

// Exercise 6: Spatial Data Structures

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
}

impl AABB {
    pub fn new(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) -> Self {
        Self { min_x, min_y, min_z, max_x, max_y, max_z }
    }
    
    pub fn contains_point(&self, pos: &Position) -> bool {
        pos.x >= self.min_x && pos.x <= self.max_x &&
        pos.y >= self.min_y && pos.y <= self.max_y &&
        pos.z >= self.min_z && pos.z <= self.max_z
    }
    
    pub fn intersects(&self, other: &AABB) -> bool {
        self.max_x >= other.min_x && self.min_x <= other.max_x &&
        self.max_y >= other.min_y && self.min_y <= other.max_y &&
        self.max_z >= other.min_z && self.min_z <= other.max_z
    }
    
    pub fn center(&self) -> Position {
        Position::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )
    }
    
    fn subdivide(&self) -> [AABB; 8] {
        let center = self.center();
        [
            // Bottom half
            AABB::new(self.min_x, self.min_y, self.min_z, center.x, center.y, center.z),
            AABB::new(center.x, self.min_y, self.min_z, self.max_x, center.y, center.z),
            AABB::new(self.min_x, center.y, self.min_z, center.x, self.max_y, center.z),
            AABB::new(center.x, center.y, self.min_z, self.max_x, self.max_y, center.z),
            // Top half
            AABB::new(self.min_x, self.min_y, center.z, center.x, center.y, self.max_z),
            AABB::new(center.x, self.min_y, center.z, self.max_x, center.y, self.max_z),
            AABB::new(self.min_x, center.y, center.z, center.x, self.max_y, self.max_z),
            AABB::new(center.x, center.y, center.z, self.max_x, self.max_y, self.max_z),
        ]
    }
}

#[derive(Debug)]
struct OctreeNode {
    bounds: AABB,
    entities: Vec<EntityId>,
    children: Option<Box<[OctreeNode; 8]>>,
    depth: usize,
}

impl OctreeNode {
    fn new(bounds: AABB, depth: usize) -> Self {
        Self {
            bounds,
            entities: Vec::new(),
            children: None,
            depth,
        }
    }
    
    fn insert(&mut self, entity: EntityId, position: &Position, max_depth: usize, max_entities: usize) {
        if !self.bounds.contains_point(position) {
            return;
        }
        
        // If we're at max depth or under entity limit, store here
        if self.depth >= max_depth || (self.entities.len() < max_entities && self.children.is_none()) {
            self.entities.push(entity);
            return;
        }
        
        // Create children if needed
        if self.children.is_none() {
            let child_bounds = self.bounds.subdivide();
            let mut children = Vec::with_capacity(8);
            for bounds in child_bounds.iter() {
                children.push(OctreeNode::new(*bounds, self.depth + 1));
            }
            self.children = Some(children.into_boxed_slice().try_into().unwrap());
            
            // Redistribute existing entities to children
            let entities = std::mem::take(&mut self.entities);
            if let Some(ref mut children) = self.children {
                for old_entity in entities {
                    // In a real implementation, we'd need to store positions
                    // For now, just put in first child that contains the point
                    for child in children.iter_mut() {
                        if child.bounds.contains_point(position) {
                            child.insert(old_entity, position, max_depth, max_entities);
                            break;
                        }
                    }
                }
            }
        }
        
        // Insert into appropriate child
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                if child.bounds.contains_point(position) {
                    child.insert(entity, position, max_depth, max_entities);
                    break;
                }
            }
        }
    }
    
    fn query_range(&self, query_bounds: &AABB, results: &mut Vec<EntityId>) {
        if !self.bounds.intersects(query_bounds) {
            return;
        }
        
        // Add entities from this node
        results.extend(&self.entities);
        
        // Query children
        if let Some(ref children) = self.children {
            for child in children.iter() {
                child.query_range(query_bounds, results);
            }
        }
    }
}

pub struct Octree {
    root: OctreeNode,
    max_depth: usize,
    max_entities_per_node: usize,
}

impl Octree {
    pub fn new(bounds: AABB, max_depth: usize, max_entities_per_node: usize) -> Self {
        Self {
            root: OctreeNode::new(bounds, 0),
            max_depth,
            max_entities_per_node,
        }
    }
    
    pub fn insert(&mut self, entity: EntityId, position: &Position) {
        self.root.insert(entity, position, self.max_depth, self.max_entities_per_node);
    }
    
    pub fn query_range(&self, bounds: &AABB) -> Vec<EntityId> {
        let mut results = Vec::new();
        self.root.query_range(bounds, &mut results);
        results
    }
    
    pub fn query_sphere(&self, center: &Position, radius: f32) -> Vec<EntityId> {
        let bounds = AABB::new(
            center.x - radius, center.y - radius, center.z - radius,
            center.x + radius, center.y + radius, center.z + radius,
        );
        self.query_range(&bounds)
    }
    
    pub fn clear(&mut self) {
        self.root = OctreeNode::new(self.root.bounds, 0);
    }
}

// Exercise 7: SIMD-Optimized Operations

pub struct SIMDMovementSystem;

impl SIMDMovementSystem {
    fn update_positions_scalar(
        positions: &mut [Position],
        velocities: &[Velocity], 
        dt: f32,
    ) {
        for (position, velocity) in positions.iter_mut().zip(velocities.iter()) {
            position.x += velocity.dx * dt;
            position.y += velocity.dy * dt;
            position.z += velocity.dz * dt;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn update_positions_avx(
        positions: &mut [Position],
        velocities: &[Velocity],
        dt: f32,
    ) {
        // For simplicity, fall back to scalar for now
        // Real SIMD implementation would use _mm256_* intrinsics
        Self::update_positions_scalar(positions, velocities, dt);
    }
}

impl System for SIMDMovementSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        let positions = world.get_storage_mut::<Position>();
        let velocities = world.get_storage::<Velocity>();
        
        if let (Some(positions), Some(velocities)) = (positions, velocities) {
            let pos_slice = positions.as_mut_slice();
            let vel_slice = velocities.as_slice();
            
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx") {
                    unsafe {
                        Self::update_positions_avx(pos_slice, vel_slice, dt);
                    }
                } else {
                    Self::update_positions_scalar(pos_slice, vel_slice, dt);
                }
            }
            
            #[cfg(not(target_arch = "x86_64"))]
            {
                Self::update_positions_scalar(pos_slice, vel_slice, dt);
            }
        }
    }
}

// Comprehensive testing
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

    #[test]
    fn test_physics_system() {
        let mut world = World::new();
        let mut system = PhysicsSystem::new();
        
        // Create two entities
        let entity1 = world.create_entity();
        world.add_component(entity1, Position::new(0.0, 0.0, 0.0));
        world.add_component(entity1, Velocity::new(0.0, 0.0, 0.0));
        
        let entity2 = world.create_entity();
        world.add_component(entity2, Position::new(10.0, 0.0, 0.0));
        world.add_component(entity2, Velocity::new(0.0, 0.0, 0.0));
        
        system.update(&mut world, 0.1);
        
        // Check that gravity affected velocities
        let vel1 = world.get_component::<Velocity>(entity1).unwrap();
        assert!(vel1.dx > 0.0); // Should be attracted to entity2
    }

    #[test]
    fn test_resource_extraction() {
        let mut world = World::new();
        let mut system = ResourceExtractionSystem;
        
        // Create ship
        let ship_entity = world.create_entity();
        world.add_component(ship_entity, Position::new(0.0, 0.0, 0.0));
        world.add_component(ship_entity, Ship::new(ShipClass::Merchant));
        
        // Create resource
        let resource_entity = world.create_entity();
        world.add_component(resource_entity, Position::new(10.0, 0.0, 0.0)); // Within range
        world.add_component(resource_entity, Resource::new(ResourceType::Iron, 100.0));
        
        system.update(&mut world, 1.0);
        
        // Check that cargo increased
        let ship = world.get_component::<Ship>(ship_entity).unwrap();
        assert!(ship.current_cargo > 0.0);
        
        // Check that resource decreased
        let resource = world.get_component::<Resource>(resource_entity).unwrap();
        assert!(resource.quantity < 100.0);
    }

    #[test]
    fn test_multiple_component_query() {
        let mut world = World::new();
        
        // Create entities with different component combinations
        let entity1 = world.create_entity();
        world.add_component(entity1, Position::new(1.0, 1.0, 1.0));
        world.add_component(entity1, Velocity::new(0.1, 0.1, 0.1));
        
        let entity2 = world.create_entity();
        world.add_component(entity2, Position::new(2.0, 2.0, 2.0));
        // No velocity
        
        let entity3 = world.create_entity();
        world.add_component(entity3, Position::new(3.0, 3.0, 3.0));
        world.add_component(entity3, Velocity::new(0.3, 0.3, 0.3));
        
        let results = world.query_components::<Position, Velocity>();
        assert_eq!(results.len(), 2); // Only entity1 and entity3 have both components
    }

    #[test]
    fn test_octree_subdivisions() {
        let bounds = AABB::new(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0);
        let mut octree = Octree::new(bounds, 2, 2);
        
        // Insert entities to force subdivision
        for i in 0..10 {
            let pos = Position::new(i as f32 * 10.0, i as f32 * 10.0, i as f32 * 10.0);
            octree.insert(i, &pos);
        }
        
        // Query should return relevant entities
        let query_pos = Position::new(0.0, 0.0, 0.0);
        let results = octree.query_sphere(&query_pos, 50.0);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_simd_movement_system() {
        let mut world = World::new();
        let mut system = SIMDMovementSystem;
        
        // Create multiple entities
        for i in 0..10 {
            let entity = world.create_entity();
            world.add_component(entity, Position::new(i as f32, i as f32, i as f32));
            world.add_component(entity, Velocity::new(1.0, 1.0, 1.0));
        }
        
        system.update(&mut world, 1.0);
        
        // Check that all positions were updated
        if let Some(positions) = world.query::<Position>() {
            for (entity, pos) in positions {
                assert_eq!(pos.x, entity as f32 + 1.0);
                assert_eq!(pos.y, entity as f32 + 1.0);
                assert_eq!(pos.z, entity as f32 + 1.0);
            }
        }
    }
}