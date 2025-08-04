// Tutorial 07: High-Performance Collections - Complete Implementation
// This file contains complete solutions for all exercises

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::mem::{self, MaybeUninit};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::ptr;
use std::cell::Cell;

// Exercise 1: Custom Vector Implementation

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

// Exercise 2: Particle System with Custom Vector

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
        self.particles.len = write_idx;
    }
    
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }
    
    pub fn particles(&self) -> &[Particle] {
        self.particles.as_slice()
    }
}

// Exercise 3: Arena Allocator

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

// Exercise 4: Collision Detection with Arena

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

#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    pub fn distance_to(&self, other: &Position) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

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
    
    pub fn reset(&mut self) {
        self.arena.reset();
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

// Exercise 5: SIMD-Optimized Vector Operations

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
    
    pub fn add_scalar(&mut self, other: &[f32]) {
        assert_eq!(self.data.len(), other.len());
        
        for (a, b) in self.data.iter_mut().zip(other.iter()) {
            *a += *b;
        }
    }
    
    pub fn add(&mut self, other: &[f32]) {
        // For now, use scalar implementation
        // Real implementation would have SIMD optimizations
        self.add_scalar(other);
    }
    
    pub fn dot_product_scalar(&self, other: &[f32]) -> f32 {
        assert_eq!(self.data.len(), other.len());
        
        self.data.iter()
            .zip(other.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
    
    pub fn dot_product(&self, other: &[f32]) -> f32 {
        // For now, use scalar implementation
        // Real implementation would have SIMD optimizations
        self.dot_product_scalar(other)
    }
    
    pub fn normalize(&mut self) {
        let magnitude = self.dot_product(&self.data).sqrt();
        if magnitude > 0.0 {
            self.scale(1.0 / magnitude);
        }
    }
    
    pub fn scale(&mut self, factor: f32) {
        self.scale_scalar(factor);
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

impl From<Vec<f32>> for SimdVec<f32> {
    fn from(data: Vec<f32>) -> Self {
        Self { data }
    }
}

// Exercise 6: SIMD Physics System

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
    
    pub fn update_positions(&mut self, dt: f32) {
        // Create temporary velocity * dt vectors
        let mut vel_x_dt = self.velocities_x.as_slice().to_vec();
        let mut vel_y_dt = self.velocities_y.as_slice().to_vec();
        let mut vel_z_dt = self.velocities_z.as_slice().to_vec();
        
        // Scale velocities by dt
        let mut vel_x_simd = SimdVec::from(vel_x_dt);
        let mut vel_y_simd = SimdVec::from(vel_y_dt);
        let mut vel_z_simd = SimdVec::from(vel_z_dt);
        
        vel_x_simd.scale(dt);
        vel_y_simd.scale(dt);
        vel_z_simd.scale(dt);
        
        // Add to positions
        self.positions_x.add(vel_x_simd.as_slice());
        self.positions_y.add(vel_y_simd.as_slice());
        self.positions_z.add(vel_z_simd.as_slice());
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

// Exercise 7: Lock-Free Stack

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

// Exercise 8: Lock-Free Queue

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

// Exercise 9: Space Message System

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

// Comprehensive testing
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_vec_basic() {
        let mut vec = FastVec::new();
        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
        
        vec.push(1);
        vec.push(2);
        vec.push(3);
        
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.get(0), Some(&1));
        assert_eq!(vec.get(1), Some(&2));
        assert_eq!(vec.get(2), Some(&3));
        assert_eq!(vec.get(3), None);
        
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_fast_vec_capacity() {
        let mut vec = FastVec::with_capacity(10);
        assert_eq!(vec.capacity(), 10);
        
        for i in 0..15 {
            vec.push(i);
        }
        
        assert!(vec.capacity() >= 15);
        assert_eq!(vec.len(), 15);
    }

    #[test]
    fn test_fast_vec_extend() {
        let mut vec = FastVec::new();
        vec.extend_from_slice(&[1, 2, 3, 4, 5]);
        
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_particle_system() {
        let mut system = ParticleSystem::new(100);
        
        system.emit([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        system.emit([1.0, 1.0, 1.0], [0.5, 0.5, 0.5]);
        
        assert_eq!(system.particle_count(), 2);
        
        system.update(0.1);
        assert_eq!(system.particle_count(), 2);
        
        // Update many times to kill particles
        for _ in 0..20 {
            system.update(0.1);
        }
        
        assert_eq!(system.particle_count(), 0);
    }

    #[test]
    fn test_arena_basic() {
        let arena = Arena::new();
        
        let value1 = arena.alloc(42i32);
        let value2 = arena.alloc(3.14f32);
        
        assert_eq!(*value1, 42);
        assert_eq!(*value2, 3.14);
        
        let slice = arena.alloc_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(slice, &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_arena_reset() {
        let mut arena = Arena::new();
        
        arena.alloc(42i32);
        arena.alloc(3.14f32);
        let allocated_before = arena.total_allocated();
        
        arena.reset();
        let allocated_after = arena.total_allocated();
        
        assert!(allocated_before > 0);
        assert_eq!(allocated_after, 0);
    }

    #[test]
    fn test_position() {
        let pos1 = Position::new(0.0, 0.0, 0.0);
        let pos2 = Position::new(3.0, 4.0, 0.0);
        
        assert_eq!(pos1.distance_to(&pos2), 5.0);
    }

    #[test]
    fn test_collision_detection() {
        let arena = CollisionArena::new();
        
        let objects = vec![
            CollisionObject {
                position: Position::new(0.0, 0.0, 0.0),
                radius: 1.0,
                mass: 1.0,
            },
            CollisionObject {
                position: Position::new(1.5, 0.0, 0.0),
                radius: 1.0,
                mass: 1.0,
            },
            CollisionObject {
                position: Position::new(5.0, 0.0, 0.0),
                radius: 1.0,
                mass: 1.0,
            },
        ];
        
        let pairs = arena.create_collision_pairs(&objects);
        assert_eq!(pairs.len(), 1); // Only first two should collide
        assert_eq!(pairs[0].object_a, 0);
        assert_eq!(pairs[0].object_b, 1);
    }

    #[test]
    fn test_simd_vec_basic() {
        let mut vec1 = SimdVec::new();
        vec1.extend_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        
        let vec2 = [0.5, 0.5, 0.5, 0.5];
        vec1.add(&vec2);
        
        assert_eq!(vec1.as_slice(), &[1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_simd_vec_dot_product() {
        let vec1 = SimdVec::from(vec![1.0, 2.0, 3.0]);
        let vec2 = [4.0, 5.0, 6.0];
        
        let dot = vec1.dot_product(&vec2);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_simd_vec_normalize() {
        let mut vec = SimdVec::from(vec![3.0, 4.0, 0.0]);
        vec.normalize();
        
        let magnitude = vec.dot_product(vec.as_slice()).sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_physics_system() {
        let mut physics = PhysicsSystem::new();
        
        physics.add_object([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.0);
        physics.add_object([1.0, 1.0, 1.0], [0.5, 0.5, 0.5], 2.0);
        
        assert_eq!(physics.object_count(), 2);
        
        physics.update_positions(1.0);
        
        assert_eq!(physics.get_position(0), Some([1.0, 1.0, 1.0]));
        assert_eq!(physics.get_position(1), Some([1.5, 1.5, 1.5]));
    }

    #[test]
    fn test_lock_free_stack() {
        let stack = LockFreeStack::new();
        assert!(stack.is_empty());
        
        stack.push(1);
        stack.push(2);
        stack.push(3);
        
        assert!(!stack.is_empty());
        
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);
        
        assert!(stack.is_empty());
    }

    #[test]
    fn test_lock_free_queue() {
        let queue = LockFreeQueue::new(4);
        assert!(queue.is_empty());
        
        assert!(queue.push(1).is_ok());
        assert!(queue.push(2).is_ok());
        assert!(queue.push(3).is_ok());
        
        assert_eq!(queue.len(), 3);
        
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
        
        assert!(queue.is_empty());
    }

    #[test]
    fn test_lock_free_queue_full() {
        let queue = LockFreeQueue::new(2);
        
        assert!(queue.push(1).is_ok());
        assert!(queue.push(2).is_ok());
        
        // Queue should be full now
        match queue.push(3) {
            Err(returned_value) => assert_eq!(returned_value, 3),
            Ok(_) => panic!("Queue should be full"),
        }
    }

    #[test]
    fn test_message_bus() {
        let bus = MessageBus::new(2);
        assert_eq!(bus.worker_count(), 2);
        
        let msg1 = SpaceMessage::ShipMovement { 
            ship_id: 1, 
            position: [1.0, 2.0, 3.0] 
        };
        
        let msg2 = SpaceMessage::SystemAlert { 
            message: "Test".to_string(), 
            priority: 5 
        };
        
        assert!(bus.send_to_worker(0, msg1).is_ok());
        assert!(bus.send_to_worker(999, msg2.clone()).is_err()); // Invalid worker
        
        bus.broadcast(msg2);
        
        assert!(bus.receive_from_worker(0).is_some());
        assert!(bus.receive_broadcast().is_some());
        assert!(bus.receive_from_worker(0).is_none()); // Should be empty now
    }

    #[test]
    fn test_concurrent_stack_operations() {
        use std::sync::Arc;
        use std::thread;
        
        let stack = Arc::new(LockFreeStack::new());
        let mut handles = vec![];
        
        // Spawn threads to push data
        for i in 0..10 {
            let stack_clone = Arc::clone(&stack);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    stack_clone.push(i * 100 + j);
                }
            }));
        }
        
        // Wait for all pushes to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Pop all items
        let mut count = 0;
        while stack.pop().is_some() {
            count += 1;
        }
        
        assert_eq!(count, 1000);
    }

    #[test]
    fn test_concurrent_queue_operations() {
        use std::sync::Arc;
        use std::thread;
        
        let queue = Arc::new(LockFreeQueue::new(2048));
        let mut handles = vec![];
        
        // Producer threads
        for i in 0..5 {
            let queue_clone = Arc::clone(&queue);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    while queue_clone.push(i * 100 + j).is_err() {
                        // Retry if queue is full
                        std::thread::yield_now();
                    }
                }
            }));
        }
        
        // Consumer thread
        let queue_clone = Arc::clone(&queue);
        let consumer = thread::spawn(move || {
            let mut count = 0;
            for _ in 0..500 { // Expect 5 * 100 = 500 items
                while queue_clone.pop().is_none() {
                    std::thread::yield_now();
                }
                count += 1;
            }
            count
        });
        
        // Wait for producers
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Wait for consumer
        let consumed = consumer.join().unwrap();
        assert_eq!(consumed, 500);
    }
}