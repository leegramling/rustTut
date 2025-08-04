# Tutorial 07: High-Performance Collections

## Learning Objectives
- Master specialized collection types for high-performance applications
- Implement custom allocators and memory-efficient data structures
- Design cache-friendly collections with optimal memory layouts
- Apply SIMD operations to collection processing
- Build arena allocators for bulk memory management
- Create lock-free concurrent collections
- Optimize collection access patterns for modern CPU architectures
- Implement spatial collections for game engine performance

## Key Concepts

### 1. Vec Optimization and Custom Collections

Understanding when and how to optimize Vec operations for performance-critical code.

```rust
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::mem::{self, MaybeUninit};

// High-performance vector with custom allocation strategies
pub struct FastVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
}

impl<T> FastVec<T> {
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        
        let layout = Layout::array::<T>(capacity).unwrap();
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            panic!("Allocation failed");
        }
        
        Self {
            ptr: unsafe { NonNull::new_unchecked(ptr as *mut T) },
            len: 0,
            capacity,
        }
    }
    
    pub fn push(&mut self, item: T) {
        if self.len == self.capacity {
            self.grow();
        }
        
        unsafe {
            let end = self.ptr.as_ptr().add(self.len);
            std::ptr::write(end, item);
        }
        
        self.len += 1;
    }
    
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        
        self.len -= 1;
        
        unsafe {
            let item_ptr = self.ptr.as_ptr().add(self.len);
            Some(std::ptr::read(item_ptr))
        }
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            Some(&*self.ptr.as_ptr().add(index))
        }
    }
    
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            Some(&mut *self.ptr.as_ptr().add(index))
        }
    }
    
    // Unsafe but fast access for performance-critical code
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        debug_assert!(index < self.len);
        &*self.ptr.as_ptr().add(index)
    }
    
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < self.len);
        &mut *self.ptr.as_ptr().add(index)
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    // Bulk operations for better performance
    pub fn extend_from_slice(&mut self, slice: &[T]) 
    where 
        T: Copy 
    {
        let required_capacity = self.len + slice.len();
        if required_capacity > self.capacity {
            self.reserve(required_capacity - self.capacity);
        }
        
        unsafe {
            let dst = self.ptr.as_ptr().add(self.len);
            std::ptr::copy_nonoverlapping(slice.as_ptr(), dst, slice.len());
        }
        
        self.len += slice.len();
    }
    
    pub fn reserve(&mut self, additional: usize) {
        let required_capacity = self.len + additional;
        if required_capacity <= self.capacity {
            return;
        }
        
        let new_capacity = required_capacity.next_power_of_two().max(4);
        self.grow_to(new_capacity);
    }
    
    fn grow(&mut self) {
        let new_capacity = if self.capacity == 0 { 4 } else { self.capacity * 2 };
        self.grow_to(new_capacity);
    }
    
    fn grow_to(&mut self, new_capacity: usize) {
        let old_layout = if self.capacity == 0 {
            Layout::new::<T>()
        } else {
            Layout::array::<T>(self.capacity).unwrap()
        };
        
        let new_layout = Layout::array::<T>(new_capacity).unwrap();
        
        let new_ptr = if self.capacity == 0 {
            unsafe { alloc(new_layout) }
        } else {
            unsafe {
                std::alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size())
            }
        };
        
        if new_ptr.is_null() {
            panic!("Allocation failed");
        }
        
        self.ptr = unsafe { NonNull::new_unchecked(new_ptr as *mut T) };
        self.capacity = new_capacity;
    }
    
    // Iterator support
    pub fn iter(&self) -> FastVecIter<T> {
        FastVecIter {
            ptr: self.ptr.as_ptr(),
            end: unsafe { self.ptr.as_ptr().add(self.len) },
            _phantom: std::marker::PhantomData,
        }
    }
    
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.len)
        }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
        }
    }
}

impl<T> Drop for FastVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        
        // Deallocate memory
        if self.capacity > 0 {
            let layout = Layout::array::<T>(self.capacity).unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

pub struct FastVecIter<T> {
    ptr: *const T,
    end: *const T,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Iterator for FastVecIter<T> {
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                let item = std::ptr::read(self.ptr);
                self.ptr = self.ptr.add(1);
                Some(item)
            }
        }
    }
}

// Space simulation example: Fast particle system
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub life: f32,
    pub mass: f32,
}

impl Particle {
    pub fn new(position: [f32; 3], velocity: [f32; 3]) -> Self {
        Self {
            position,
            velocity,
            life: 1.0,
            mass: 1.0,
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        for i in 0..3 {
            self.position[i] += self.velocity[i] * dt;
        }
        self.life -= dt;
    }
    
    pub fn is_alive(&self) -> bool {
        self.life > 0.0
    }
}

pub struct ParticleSystem {
    particles: FastVec<Particle>,
    max_particles: usize,
}

impl ParticleSystem {
    pub fn new(max_particles: usize) -> Self {
        Self {
            particles: FastVec::with_capacity(max_particles),
            max_particles,
        }
    }
    
    pub fn emit(&mut self, position: [f32; 3], velocity: [f32; 3]) {
        if self.particles.len() < self.max_particles {
            self.particles.push(Particle::new(position, velocity));
        }
    }
    
    pub fn update(&mut self, dt: f32) {
        // Update all particles
        for i in 0..self.particles.len() {
            unsafe {
                self.particles.get_unchecked_mut(i).update(dt);
            }
        }
        
        // Remove dead particles (retain_mut equivalent)
        let mut write_idx = 0;
        for read_idx in 0..self.particles.len() {
            let particle = unsafe { self.particles.get_unchecked(read_idx) };
            if particle.is_alive() {
                if write_idx != read_idx {
                    unsafe {
                        *self.particles.get_unchecked_mut(write_idx) = *particle;
                    }
                }
                write_idx += 1;
            }
        }
        
        // Update length to remove dead particles
        unsafe {
            self.particles.len = write_idx;
        }
    }
    
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
    
    pub fn particles(&self) -> &[Particle] {
        self.particles.as_slice()
    }
}
```

### 2. Arena Allocators for Bulk Memory Management

Arena allocators provide fast allocation and automatic cleanup for temporary objects.

```rust
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::cell::Cell;

// Simple arena allocator for temporary allocations
pub struct Arena {
    chunks: Vec<Chunk>,
    current_chunk: usize,
    chunk_size: usize,
}

struct Chunk {
    data: NonNull<u8>,
    size: usize,
    offset: Cell<usize>,
}

impl Arena {
    pub fn new() -> Self {
        Self::with_chunk_size(64 * 1024) // 64KB chunks
    }
    
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            current_chunk: 0,
            chunk_size,
        }
    }
    
    pub fn alloc<T>(&self, value: T) -> &mut T {
        let layout = Layout::new::<T>();
        let ptr = self.alloc_layout(layout).as_ptr() as *mut T;
        
        unsafe {
            std::ptr::write(ptr, value);
            &mut *ptr
        }
    }
    
    pub fn alloc_slice<T>(&self, slice: &[T]) -> &mut [T] 
    where 
        T: Copy 
    {
        let layout = Layout::array::<T>(slice.len()).unwrap();
        let ptr = self.alloc_layout(layout).as_ptr() as *mut T;
        
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            std::slice::from_raw_parts_mut(ptr, slice.len())
        }
    }
    
    pub fn alloc_array<T, const N: usize>(&self, array: [T; N]) -> &mut [T; N] {
        let layout = Layout::new::<[T; N]>();
        let ptr = self.alloc_layout(layout).as_ptr() as *mut [T; N];
        
        unsafe {
            std::ptr::write(ptr, array);
            &mut *ptr
        }
    }
    
    fn alloc_layout(&self, layout: Layout) -> NonNull<u8> {
        let size = layout.size();
        let align = layout.align();
        
        // Try current chunk first
        if let Some(chunk) = self.chunks.get(self.current_chunk) {
            if let Some(ptr) = chunk.try_alloc(size, align) {
                return ptr;
            }
        }
        
        // Try other chunks
        for (i, chunk) in self.chunks.iter().enumerate() {
            if i != self.current_chunk {
                if let Some(ptr) = chunk.try_alloc(size, align) {
                    return ptr;
                }
            }
        }
        
        // Need new chunk
        self.add_chunk();
        let chunk = &self.chunks[self.chunks.len() - 1];
        chunk.try_alloc(size, align).expect("New chunk should have space")
    }
    
    fn add_chunk(&self) {
        let chunk_size = self.chunk_size.max(1024); // Minimum chunk size
        let layout = Layout::array::<u8>(chunk_size).unwrap();
        let data = unsafe { alloc(layout) };
        
        if data.is_null() {
            panic!("Arena allocation failed");
        }
        
        let chunk = Chunk {
            data: unsafe { NonNull::new_unchecked(data) },
            size: chunk_size,
            offset: Cell::new(0),
        };
        
        // This is a bit of a hack - we're modifying through shared reference
        // In real code, you'd want RefCell or UnsafeCell
        let chunks_ptr = &self.chunks as *const Vec<Chunk> as *mut Vec<Chunk>;
        unsafe {
            (*chunks_ptr).push(chunk);
        }
    }
    
    pub fn reset(&mut self) {
        for chunk in &self.chunks {
            chunk.offset.set(0);
        }
        self.current_chunk = 0;
    }
    
    pub fn total_allocated(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.offset.get()).sum()
    }
    
    pub fn total_capacity(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.size).sum()
    }
}

impl Chunk {
    fn try_alloc(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        let current_offset = self.offset.get();
        let aligned_offset = (current_offset + align - 1) & !(align - 1);
        let end_offset = aligned_offset + size;
        
        if end_offset <= self.size {
            self.offset.set(end_offset);
            unsafe {
                Some(NonNull::new_unchecked(
                    self.data.as_ptr().add(aligned_offset)
                ))
            }
        } else {
            None
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            let layout = Layout::array::<u8>(chunk.size).unwrap();
            unsafe {
                dealloc(chunk.data.as_ptr(), layout);
            }
        }
    }
}

// Space simulation: Temporary collision detection structures
pub struct CollisionArena {
    arena: Arena,
}

impl CollisionArena {
    pub fn new() -> Self {
        Self {
            arena: Arena::with_chunk_size(128 * 1024), // 128KB for collision data
        }
    }
    
    pub fn create_collision_pairs<'a>(&'a self, objects: &[CollisionObject]) -> &'a mut [CollisionPair] {
        let mut pairs = Vec::new();
        
        for i in 0..objects.len() {
            for j in (i + 1)..objects.len() {
                let distance = objects[i].position.distance_to(&objects[j].position);
                if distance < objects[i].radius + objects[j].radius {
                    pairs.push(CollisionPair {
                        object_a: i,
                        object_b: j,
                        distance,
                        normal: calculate_collision_normal(&objects[i], &objects[j]),
                    });
                }
            }
        }
        
        self.arena.alloc_slice(&pairs)
    }
    
    pub fn create_spatial_grid<'a>(&'a self, bounds: AABB, cell_size: f32) -> &'a mut SpatialGrid {
        let grid_width = ((bounds.max_x - bounds.min_x) / cell_size).ceil() as usize;
        let grid_height = ((bounds.max_y - bounds.min_y) / cell_size).ceil() as usize;
        let grid_depth = ((bounds.max_z - bounds.min_z) / cell_size).ceil() as usize;
        
        let total_cells = grid_width * grid_height * grid_depth;
        let cells = vec![SpatialCell::new(); total_cells];
        
        self.arena.alloc(SpatialGrid {
            bounds,
            cell_size,
            width: grid_width,
            height: grid_height,
            depth: grid_depth,
            cells: self.arena.alloc_slice(&cells),
        })
    }
    
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CollisionObject {
    pub position: Position,
    pub radius: f32,
    pub mass: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct CollisionPair {
    pub object_a: usize,
    pub object_b: usize,
    pub distance: f32,
    pub normal: [f32; 3],
}

#[derive(Debug)]
pub struct SpatialGrid<'a> {
    pub bounds: AABB,
    pub cell_size: f32,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub cells: &'a mut [SpatialCell],
}

#[derive(Debug, Clone)]
pub struct SpatialCell {
    pub objects: Vec<usize>,
}

impl SpatialCell {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }
}

fn calculate_collision_normal(a: &CollisionObject, b: &CollisionObject) -> [f32; 3] {
    let dx = b.position.x - a.position.x;
    let dy = b.position.y - a.position.y;
    let dz = b.position.z - a.position.z;
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    
    if length > 0.0 {
        [dx / length, dy / length, dz / length]
    } else {
        [1.0, 0.0, 0.0]
    }
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

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    pub fn distance_to(&self, other: &Position) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}
```

### 3. SIMD-Optimized Collection Operations

Using SIMD instructions for parallel processing of collection data.

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// SIMD-optimized operations on collections
pub struct SimdVec<T> {
    data: Vec<T>,
}

impl SimdVec<f32> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }
    
    pub fn push(&mut self, value: f32) {
        self.data.push(value);
    }
    
    pub fn extend_from_slice(&mut self, slice: &[f32]) {
        self.data.extend_from_slice(slice);
    }
    
    // SIMD-optimized addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    pub unsafe fn add_simd(&mut self, other: &[f32]) {
        assert_eq!(self.data.len(), other.len());
        
        let len = self.data.len();
        let mut i = 0;
        
        // Process 8 elements at a time with AVX
        while i + 8 <= len {
            let a = _mm256_loadu_ps(self.data.as_ptr().add(i));
            let b = _mm256_loadu_ps(other.as_ptr().add(i));
            let result = _mm256_add_ps(a, b);
            _mm256_storeu_ps(self.data.as_mut_ptr().add(i), result);
            i += 8;
        }
        
        // Handle remaining elements
        for j in i..len {
            self.data[j] += other[j];
        }
    }
    
    // Fallback scalar addition
    pub fn add_scalar(&mut self, other: &[f32]) {
        assert_eq!(self.data.len(), other.len());
        
        for (a, b) in self.data.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }
    
    // Public interface that chooses best implementation
    pub fn add(&mut self, other: &[f32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { self.add_simd(other); }
            } else {
                self.add_scalar(other);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.add_scalar(other);
        }
    }
    
    // SIMD-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    pub unsafe fn dot_product_simd(&self, other: &[f32]) -> f32 {
        assert_eq!(self.data.len(), other.len());
        
        let len = self.data.len();
        let mut i = 0;
        let mut sum_vec = _mm256_setzero_ps();
        
        // Process 8 elements at a time
        while i + 8 <= len {
            let a = _mm256_loadu_ps(self.data.as_ptr().add(i));
            let b = _mm256_loadu_ps(other.as_ptr().add(i));
            let product = _mm256_mul_ps(a, b);
            sum_vec = _mm256_add_ps(sum_vec, product);
            i += 8;
        }
        
        // Horizontal sum of the vector
        let mut result: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum_vec);
        let mut total = result.iter().sum::<f32>();
        
        // Handle remaining elements
        for j in i..len {
            total += self.data[j] * other[j];
        }
        
        total
    }
    
    pub fn dot_product_scalar(&self, other: &[f32]) -> f32 {
        assert_eq!(self.data.len(), other.len());
        
        self.data.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    pub fn dot_product(&self, other: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { self.dot_product_simd(other) }
            } else {
                self.dot_product_scalar(other)
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.dot_product_scalar(other)
        }
    }
    
    // SIMD-optimized normalization
    pub fn normalize(&mut self) {
        let magnitude = self.dot_product(&self.data).sqrt();
        if magnitude > 0.0 {
            self.scale(1.0 / magnitude);
        }
    }
    
    pub fn scale(&mut self, factor: f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                unsafe { self.scale_simd(factor); }
            } else {
                self.scale_scalar(factor);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.scale_scalar(factor);
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx")]
    unsafe fn scale_simd(&mut self, factor: f32) {
        let factor_vec = _mm256_set1_ps(factor);
        let len = self.data.len();
        let mut i = 0;
        
        while i + 8 <= len {
            let data_vec = _mm256_loadu_ps(self.data.as_ptr().add(i));
            let result = _mm256_mul_ps(data_vec, factor_vec);
            _mm256_storeu_ps(self.data.as_mut_ptr().add(i), result);
            i += 8;
        }
        
        // Handle remaining elements
        for j in i..len {
            self.data[j] *= factor;
        }
    }
    
    fn scale_scalar(&mut self, factor: f32) {
        for value in &mut self.data {
            *value *= factor;
        }
    }
    
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// Space simulation: SIMD-optimized physics calculations
pub struct PhysicsSystem {
    positions_x: SimdVec<f32>,
    positions_y: SimdVec<f32>,
    positions_z: SimdVec<f32>,
    velocities_x: SimdVec<f32>,
    velocities_y: SimdVec<f32>,
    velocities_z: SimdVec<f32>,
    masses: SimdVec<f32>,
}

impl PhysicsSystem {
    pub fn new() -> Self {
        Self {
            positions_x: SimdVec::new(),
            positions_y: SimdVec::new(),
            positions_z: SimdVec::new(),
            velocities_x: SimdVec::new(),
            velocities_y: SimdVec::new(),
            velocities_z: SimdVec::new(),
            masses: SimdVec::new(),
        }
    }
    
    pub fn add_object(&mut self, position: [f32; 3], velocity: [f32; 3], mass: f32) {
        self.positions_x.push(position[0]);
        self.positions_y.push(position[1]);
        self.positions_z.push(position[2]);
        self.velocities_x.push(velocity[0]);
        self.velocities_y.push(velocity[1]);
        self.velocities_z.push(velocity[2]);
        self.masses.push(mass);
    }
    
    // SIMD-optimized position update
    pub fn update_positions(&mut self, dt: f32) {
        let dt_vec = vec![dt; self.positions_x.len()];
        
        // Create temporary velocity * dt vectors
        let mut vel_x_dt = self.velocities_x.as_slice().to_vec();
        let mut vel_y_dt = self.velocities_y.as_slice().to_vec();
        let mut vel_z_dt = self.velocities_z.as_slice().to_vec();
        
        // Scale velocities by dt
        SimdVec { data: vel_x_dt }.scale(dt);
        SimdVec { data: vel_y_dt }.scale(dt);
        SimdVec { data: vel_z_dt }.scale(dt);
        
        // Add to positions
        self.positions_x.add(&vel_x_dt);
        self.positions_y.add(&vel_y_dt);
        self.positions_z.add(&vel_z_dt);
    }
    
    pub fn object_count(&self) -> usize {
        self.positions_x.len()
    }
    
    pub fn get_position(&self, index: usize) -> Option<[f32; 3]> {
        if index < self.object_count() {
            Some([
                self.positions_x.as_slice()[index],
                self.positions_y.as_slice()[index],
                self.positions_z.as_slice()[index],
            ])
        } else {
            None
        }
    }
}
```

### 4. Lock-Free Concurrent Collections

High-performance concurrent data structures without traditional locking.

```rust
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;

// Lock-free stack using atomic operations
pub struct LockFreeStack<T> {
    head: AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: T,
    next: *mut Node<T>,
}

impl<T> LockFreeStack<T> {
    pub fn new() -> Self {
        Self {
            head: AtomicPtr::new(ptr::null_mut()),
        }
    }
    
    pub fn push(&self, data: T) {
        let new_node = Box::into_raw(Box::new(Node {
            data,
            next: ptr::null_mut(),
        }));
        
        loop {
            let head = self.head.load(Ordering::Acquire);
            unsafe {
                (*new_node).next = head;
            }
            
            match self.head.compare_exchange_weak(
                head,
                new_node,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue, // Retry
            }
        }
    }
    
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }
            
            let next = unsafe { (*head).next };
            
            match self.head.compare_exchange_weak(
                head,
                next,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    let node = unsafe { Box::from_raw(head) };
                    return Some(node.data);
                }
                Err(_) => continue, // Retry
            }
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire).is_null()
    }
}

impl<T> Drop for LockFreeStack<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

// Lock-free queue using ring buffer
pub struct LockFreeQueue<T> {
    buffer: Vec<AtomicPtr<T>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> LockFreeQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.next_power_of_two(); // Ensure power of 2 for fast modulo
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(AtomicPtr::new(ptr::null_mut()));
        }
        
        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }
    
    pub fn push(&self, data: T) -> Result<(), T> {
        let data_ptr = Box::into_raw(Box::new(data));
        
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let head = self.head.load(Ordering::Acquire);
            
            // Check if queue is full
            if (tail + 1) % self.capacity == head {
                unsafe {
                    let data = Box::from_raw(data_ptr);
                    return Err(*data);
                }
            }
            
            let slot = &self.buffer[tail % self.capacity];
            
            // Try to store the data
            match slot.compare_exchange_weak(
                ptr::null_mut(),
                data_ptr,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully stored, now try to advance tail
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        tail + 1,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    return Ok(());
                }
                Err(_) => {
                    // Slot was not empty, try next position
                    continue;
                }
            }
        }
    }
    
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            
            // Check if queue is empty
            if head == tail {
                return None;
            }
            
            let slot = &self.buffer[head % self.capacity];
            let data_ptr = slot.load(Ordering::Acquire);
            
            if data_ptr.is_null() {
                continue; // Slot is being modified
            }
            
            // Try to take the data
            match slot.compare_exchange_weak(
                data_ptr,
                ptr::null_mut(),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully took data, advance head
                    let _ = self.head.compare_exchange_weak(
                        head,
                        head + 1,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                    
                    let data = unsafe { Box::from_raw(data_ptr) };
                    return Some(*data);
                }
                Err(_) => continue, // Retry
            }
        }
    }
    
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail + self.capacity - head) % self.capacity
    }
    
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Clean up any remaining items
        while self.pop().is_some() {}
    }
}

// Space simulation: Lock-free message passing between threads
#[derive(Debug, Clone)]
pub enum SpaceMessage {
    ShipMovement { ship_id: u32, position: [f32; 3] },
    WeaponFired { ship_id: u32, target_id: u32, damage: f32 },
    ResourceMined { ship_id: u32, resource_type: String, amount: f32 },
    SystemAlert { message: String, priority: u8 },
}

pub struct MessageBus {
    queues: Vec<LockFreeQueue<SpaceMessage>>,
    broadcast_stack: LockFreeStack<SpaceMessage>,
}

impl MessageBus {
    pub fn new(worker_count: usize) -> Self {
        let mut queues = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            queues.push(LockFreeQueue::new(1024));
        }
        
        Self {
            queues,
            broadcast_stack: LockFreeStack::new(),
        }
    }
    
    pub fn send_to_worker(&self, worker_id: usize, message: SpaceMessage) -> Result<(), SpaceMessage> {
        if worker_id < self.queues.len() {
            self.queues[worker_id].push(message)
        } else {
            Err(message)
        }
    }
    
    pub fn broadcast(&self, message: SpaceMessage) {
        self.broadcast_stack.push(message);
    }
    
    pub fn receive_from_worker(&self, worker_id: usize) -> Option<SpaceMessage> {
        if worker_id < self.queues.len() {
            self.queues[worker_id].pop()
        } else {
            None
        }
    }
    
    pub fn receive_broadcast(&self) -> Option<SpaceMessage> {
        self.broadcast_stack.pop()
    }
    
    pub fn worker_count(&self) -> usize {
        self.queues.len()
    }
}
```

## Key Takeaways

1. **Custom Collections**: Build specialized data structures for specific performance needs
2. **Arena Allocation**: Use bulk allocation for temporary objects and automatic cleanup
3. **SIMD Operations**: Leverage parallel instructions for mathematical computations
4. **Lock-Free Algorithms**: Achieve high concurrency without traditional locking
5. **Memory Layout**: Design data structures for optimal cache performance
6. **Batch Processing**: Process collections in chunks for better performance
7. **Zero-Copy Operations**: Minimize memory allocations and copies in hot paths

## Best Practices

- Profile before optimizing - measure actual performance bottlenecks
- Use arena allocators for temporary data that shares lifetimes
- Implement SIMD with runtime feature detection for portability
- Design lock-free structures carefully - they're complex but powerful
- Consider cache line alignment for high-performance structures
- Use unsafe code judiciously and document safety invariants
- Provide both safe and unsafe APIs when performance is critical

## Performance Considerations

- Cache locality affects performance more than algorithmic complexity
- SIMD requires proper data alignment and vectorizable operations
- Lock-free algorithms trade complexity for reduced contention
- Arena allocators reduce fragmentation and allocation overhead
- Bulk operations amortize per-element costs
- Memory bandwidth often limits performance more than CPU speed

## Next Steps

In the next tutorial, we'll explore test-driven development patterns, learning how to build robust and reliable systems through comprehensive testing strategies in our space simulation engine.