# Tutorial 06: Data-Oriented Programming

## Learning Objectives
- Master data-oriented design principles in systems programming
- Implement Entity-Component-System (ECS) patterns for high performance
- Design cache-friendly data structures for optimal memory access
- Apply SIMD operations for parallel data processing
- Build efficient spatial data structures for game engines
- Understand memory layout optimization techniques
- Create data structures that leverage CPU architecture

## Lesson: Data-Oriented Programming

### What is Data-Oriented Programming?

Data-Oriented Programming is a programming paradigm that focuses on organizing code around the data and how it flows through the system, rather than around objects and their methods. The key principle is:

**"Transform data efficiently, not hide data behind abstractions"**

### Core Principles

#### 1. Data Layout Matters
- **Cache locality**: Keep related data close in memory
- **Structure of Arrays (SoA)**: Separate different data types
- **Array of Structures (AoS)**: Group related data together
- **Memory alignment**: Align data to processor word boundaries

#### 2. Batch Processing
- **SIMD (Single Instruction, Multiple Data)**: Process multiple data points simultaneously
- **Vectorization**: Use CPU vector instructions
- **Loop optimization**: Minimize branches and memory access patterns

#### 3. Separate Data from Behavior
- **Data is data**: Pure data structures without methods
- **Systems process data**: Functions operate on data collections
- **Composition over inheritance**: Combine simple data types

### Why Data-Oriented Programming?

#### Performance Benefits:
1. **Cache efficiency**: Better memory access patterns
2. **Parallelization**: Easier to parallelize data processing
3. **SIMD utilization**: Vectorized operations on homogeneous data
4. **Predictable performance**: Less pointer chasing and virtual calls

#### Design Benefits:
1. **Separation of concerns**: Data and behavior are separate
2. **Testability**: Pure functions are easier to test
3. **Scalability**: Easier to distribute data processing
4. **Flexibility**: Recombine data in different ways

### Memory Hierarchy and Performance

#### CPU Cache Hierarchy:
```
CPU Registers  ~1 cycle    32-64 bytes
L1 Cache       ~4 cycles   32-64 KB
L2 Cache       ~10 cycles  256 KB - 1 MB
L3 Cache       ~40 cycles  8-32 MB
Main Memory    ~200 cycles Several GB
Disk Storage   ~10M cycles Terabytes
```

#### Cache Line Impact:
- **Cache line size**: Typically 64 bytes
- **False sharing**: Multiple threads accessing different data in same cache line
- **Spatial locality**: Accessing nearby memory addresses
- **Temporal locality**: Accessing same memory repeatedly

### Entity-Component-System (ECS) Pattern

ECS is a specific application of data-oriented principles:

#### Traditional OOP Approach:
```rust
// Object-oriented approach
struct Ship {
    position: Vector3,
    velocity: Vector3,
    health: f32,
    // ... many other fields
}

impl Ship {
    fn update(&mut self, dt: f32) {
        self.position += self.velocity * dt;
        // Process all aspects of the ship
    }
}
```

#### ECS Approach:
```rust
// Data-oriented ECS approach
struct Position(Vector3);
struct Velocity(Vector3);
struct Health(f32);

// System processes all entities with specific components
fn movement_system(positions: &mut [Position], velocities: &[Velocity], dt: f32) {
    for (pos, vel) in positions.iter_mut().zip(velocities.iter()) {
        pos.0 += vel.0 * dt;
    }
}
```

### Structure of Arrays vs Array of Structures

#### Array of Structures (AoS):
```rust
struct Particle {
    x: f32, y: f32, z: f32,  // Position
    dx: f32, dy: f32, dz: f32, // Velocity
}

let particles: Vec<Particle> = vec![/* ... */];
// Memory layout: [x,y,z,dx,dy,dz][x,y,z,dx,dy,dz][x,y,z,dx,dy,dz]...
```

#### Structure of Arrays (SoA):
```rust
struct ParticleSystem {
    positions_x: Vec<f32>,
    positions_y: Vec<f32>,
    positions_z: Vec<f32>,
    velocities_x: Vec<f32>,
    velocities_y: Vec<f32>,
    velocities_z: Vec<f32>,
}
// Memory layout: [x,x,x,x...][y,y,y,y...][z,z,z,z...][dx,dx,dx...]
```

### SIMD and Vectorization

#### What is SIMD?
- **Single Instruction, Multiple Data**
- **Vector processing**: Apply same operation to multiple data points
- **Hardware support**: Modern CPUs have SIMD instruction sets (SSE, AVX)
- **Parallelism**: Process 4, 8, or 16 values simultaneously

#### When SIMD Helps:
- **Homogeneous data**: Same operations on similar data types
- **Mathematical operations**: Vector math, transformations
- **Data parallel algorithms**: Sorting, searching, filtering
- **Regular access patterns**: Sequential or strided memory access

### Spatial Data Structures

For games and simulations, spatial organization is crucial:

#### Common Structures:
- **Octree/Quadtree**: Hierarchical space partitioning
- **Grid/Hash grid**: Regular space subdivision
- **BSP trees**: Binary space partitioning
- **R-trees**: Rectangle trees for arbitrary shapes

#### Trade-offs:
- **Memory usage** vs **query speed**
- **Update cost** vs **query cost**
- **Static** vs **dynamic** scenarios

### Rust's Advantages for Data-Oriented Programming

#### Memory Safety:
- **No garbage collection**: Predictable memory usage
- **Zero-cost abstractions**: High-level code compiles to efficient machine code
- **Ownership system**: Clear data ownership and lifetimes

#### Performance Features:
- **SIMD support**: Standard library and external crates
- **Memory layout control**: `#[repr(C)]`, `#[repr(packed)]`
- **Efficient iterators**: Zero-cost iterator chains
- **Inline assembly**: Direct hardware access when needed

### When to Use Data-Oriented Programming

#### Good for:
- **Performance-critical systems**
- **Data processing pipelines**
- **Game engines and simulations**
- **Scientific computing**
- **Real-time systems**

#### Consider alternatives for:
- **Complex business logic**
- **User interfaces**
- **I/O heavy applications**
- **Prototype development**

### Space Simulation Applications

In our space simulation, data-oriented programming excels at:
- **Physics updates**: Process thousands of entities efficiently
- **Collision detection**: Spatial queries on large datasets
- **Rendering**: Batch similar objects for GPU efficiency
- **AI processing**: Decision making for multiple agents
- **Resource management**: Track resources across many containers

## Key Concepts

### 1. Entity-Component-System (ECS) Architecture

ECS separates data (Components) from behavior (Systems), enabling better cache locality and parallelization.

```rust
use std::collections::HashMap;
use std::any::{Any, TypeId};
use nalgebra::{Vector3, Matrix4};

// Entity is just an ID
pub type EntityId = u32;

// Component trait for type-safe storage
pub trait Component: 'static + Send + Sync {
    fn type_id() -> TypeId where Self: Sized {
        TypeId::of::<Self>()
    }
}

// Position component for spatial entities
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

// Velocity component for moving entities
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

// Ship component with space-specific data
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

// Resource component for mining and trading
#[derive(Debug, Clone)]
pub struct Resource {
    pub resource_type: ResourceType,
    pub quantity: f32,
    pub quality: f32, // 0.0 to 1.0
    pub extraction_rate: f32, // for asteroids
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

// Weapons component for combat entities
#[derive(Debug, Clone)]
pub struct Weapons {
    pub weapon_systems: Vec<WeaponSystem>,
    pub targeting_range: f32,
    pub fire_rate: f32,
    pub last_fired: f32,
}

impl Component for Weapons {}

#[derive(Debug, Clone)]
pub struct WeaponSystem {
    pub weapon_type: WeaponType,
    pub damage: f32,
    pub range: f32,
    pub energy_cost: f32,
    pub ammo_count: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub enum WeaponType {
    Laser,
    Plasma,
    Kinetic,
    Missile,
    EMP,
}
```

### 2. Component Storage Systems

Efficient storage patterns that optimize for cache locality and batch processing.

```rust
use std::collections::HashMap;
use std::any::TypeId;

// Dense component storage for cache-friendly iteration
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
    
    // Batch iteration for cache-friendly processing
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
    
    // Get raw component data for SIMD operations
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

// World manages all entities and component storages
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
    
    // Get multiple component types for entity
    pub fn get_components<T1, T2>(&self, entity: EntityId) -> Option<(&T1, &T2)>
    where
        T1: Component,
        T2: Component,
    {
        let c1 = self.get_component::<T1>(entity)?;
        let c2 = self.get_component::<T2>(entity)?;
        Some((c1, c2))
    }
    
    // Batch query for multiple component types
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
```

### 3. System Implementation for Batch Processing

Systems process components in batches for optimal performance.

```rust
use rayon::prelude::*;

// System trait for processing components
pub trait System {
    fn update(&mut self, world: &mut World, dt: f32);
}

// Movement system updates positions based on velocities
pub struct MovementSystem;

impl System for MovementSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        // Get both position and velocity storages
        let positions = world.get_storage_mut::<Position>();
        let velocities = world.get_storage::<Velocity>();
        
        if let (Some(positions), Some(velocities)) = (positions, velocities) {
            // Process entities that have both components
            for (entity, velocity) in velocities.iter() {
                if let Some(position) = positions.get_mut(entity) {
                    position.x += velocity.dx * dt;
                    position.y += velocity.dy * dt;
                    position.z += velocity.dz * dt;
                }
            }
        }
    }
}

// Physics system with parallel processing
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

// Resource extraction system
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
        
        for (ship_entity, ship_pos, ship) in ships {
            for (resource_entity, resource_pos, resource) in &resources {
                let distance = ship_pos.distance_to(resource_pos);
                
                // Within extraction range
                if distance < 50.0 {
                    let extraction_amount = resource.extraction_rate * dt;
                    
                    // Check if ship can carry more
                    if ship.can_carry(extraction_amount) {
                        // Update resource quantity
                        if let Some(resource_mut) = world.get_component_mut::<Resource>(*resource_entity) {
                            if resource_mut.quantity > extraction_amount {
                                resource_mut.quantity -= extraction_amount;
                                
                                // Update ship cargo
                                if let Some(ship_mut) = world.get_component_mut::<Ship>(ship_entity) {
                                    ship_mut.current_cargo += extraction_amount;
                                    println!("Ship {} extracted {:.2} units of {:?}", 
                                            ship_entity, extraction_amount, resource.resource_type);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// Combat system with spatial optimization
pub struct CombatSystem {
    max_combat_range: f32,
}

impl CombatSystem {
    pub fn new() -> Self {
        Self {
            max_combat_range: 1000.0,
        }
    }
}

impl System for CombatSystem {
    fn update(&mut self, world: &mut World, dt: f32) {
        // Find entities with weapons
        let combatants: Vec<_> = world
            .query_components::<Position, Weapons>()
            .into_iter()
            .collect();
        
        let targets: Vec<_> = world
            .query_components::<Position, Ship>()
            .into_iter()
            .filter(|(_, _, ship)| ship.is_operational())
            .collect();
        
        for (attacker_entity, attacker_pos, weapons) in &combatants {
            // Check if weapons can fire
            if weapons.last_fired + (1.0 / weapons.fire_rate) > dt {
                continue;
            }
            
            // Find targets in range
            for (target_entity, target_pos, _) in &targets {
                if attacker_entity == target_entity {
                    continue;
                }
                
                let distance = attacker_pos.distance_to(target_pos);
                if distance <= weapons.targeting_range {
                    // Calculate damage
                    let total_damage = weapons.weapon_systems
                        .iter()
                        .filter(|weapon| distance <= weapon.range)
                        .map(|weapon| weapon.damage)
                        .sum::<f32>();
                    
                    if total_damage > 0.0 {
                        // Apply damage to target
                        if let Some(target_ship) = world.get_component_mut::<Ship>(*target_entity) {
                            target_ship.hull_integrity -= total_damage * dt;
                            target_ship.hull_integrity = target_ship.hull_integrity.max(0.0);
                            
                            println!("Ship {} attacked ship {} for {:.2} damage (hull: {:.2})", 
                                    attacker_entity, target_entity, total_damage, target_ship.hull_integrity);
                        }
                        
                        // Update weapon cooldown
                        if let Some(weapons_mut) = world.get_component_mut::<Weapons>(*attacker_entity) {
                            weapons_mut.last_fired = 0.0;
                        }
                        
                        break; // One target per frame
                    }
                }
            }
        }
    }
}
```

### 4. SIMD-Optimized Operations

Using SIMD instructions for parallel processing of component data.

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-optimized position updates
pub struct SIMDMovementSystem;

impl SIMDMovementSystem {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn update_positions_avx(
        positions: &mut [Position],
        velocities: &[Velocity],
        dt: f32,
    ) {
        let dt_vec = _mm256_set1_ps(dt);
        
        // Process 8 positions at a time (256-bit AVX)
        for i in (0..positions.len().min(velocities.len())).step_by(8) {
            if i + 8 <= positions.len() && i + 8 <= velocities.len() {
                // Load positions
                let pos_x = _mm256_loadu_ps(&positions[i].x as *const f32);
                let pos_y = _mm256_loadu_ps(&positions[i].y as *const f32);
                let pos_z = _mm256_loadu_ps(&positions[i].z as *const f32);
                
                // Load velocities
                let vel_x = _mm256_loadu_ps(&velocities[i].dx as *const f32);
                let vel_y = _mm256_loadu_ps(&velocities[i].dy as *const f32);
                let vel_z = _mm256_loadu_ps(&velocities[i].dz as *const f32);
                
                // Calculate new positions: pos = pos + vel * dt
                let new_pos_x = _mm256_fmadd_ps(vel_x, dt_vec, pos_x);
                let new_pos_y = _mm256_fmadd_ps(vel_y, dt_vec, pos_y);
                let new_pos_z = _mm256_fmadd_ps(vel_z, dt_vec, pos_z);
                
                // Store results
                _mm256_storeu_ps(&mut positions[i].x as *mut f32, new_pos_x);
                _mm256_storeu_ps(&mut positions[i].y as *mut f32, new_pos_y);
                _mm256_storeu_ps(&mut positions[i].z as *mut f32, new_pos_z);
            } else {
                // Handle remaining elements scalar
                for j in i..positions.len().min(velocities.len()) {
                    positions[j].x += velocities[j].dx * dt;
                    positions[j].y += velocities[j].dy * dt;
                    positions[j].z += velocities[j].dz * dt;
                }
                break;
            }
        }
    }
    
    // Fallback scalar implementation
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
```

### 5. Spatial Data Structures

Efficient spatial indexing for collision detection and proximity queries.

```rust
// Octree for 3D spatial partitioning
pub struct Octree {
    root: OctreeNode,
    max_depth: usize,
    max_entities_per_node: usize,
}

#[derive(Debug)]
struct OctreeNode {
    bounds: AABB,
    entities: Vec<EntityId>,
    children: Option<Box<[OctreeNode; 8]>>,
    depth: usize,
}

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
            
            // Redistribute existing entities
            let entities = std::mem::take(&mut self.entities);
            if let Some(ref mut children) = self.children {
                for entity in entities {
                    // Note: We'd need position lookup here in real implementation
                    // For now, just add to first child
                    children[0].entities.push(entity);
                }
            }
        }
        
        // Insert into appropriate child
        if let Some(ref mut children) = self.children {
            for child in children.iter_mut() {
                child.insert(entity, position, max_depth, max_entities);
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
```

## Key Takeaways

1. **Data Locality**: Organize data for cache-friendly access patterns
2. **ECS Architecture**: Separate data from behavior for better performance
3. **SIMD Operations**: Use parallel instructions for batch processing
4. **Spatial Structures**: Optimize collision detection with spatial indexing
5. **Memory Layout**: Design structures that align with CPU cache lines
6. **Batch Processing**: Process similar data together for efficiency
7. **Parallel Execution**: Use multiple CPU cores effectively

## Best Practices

- Design components for minimal size and optimal alignment
- Use structure-of-arrays (SoA) instead of array-of-structures (AoS)
- Process components in batches for cache efficiency
- Leverage SIMD instructions for parallel operations
- Use spatial data structures for range queries
- Profile memory access patterns and optimize hot paths
- Consider CPU cache hierarchy in data structure design

## Performance Considerations

- Cache line size affects data layout optimization
- Memory bandwidth limits throughput more than CPU speed
- Branch prediction impacts conditional processing
- SIMD requires proper data alignment
- Spatial structures trade memory for query performance
- Lock-free algorithms reduce contention in parallel systems

## Next Steps

In the next tutorial, we'll explore high-performance collections, learning how to implement custom data structures optimized for specific use cases in our space simulation engine.