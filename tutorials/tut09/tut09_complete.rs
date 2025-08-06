// Tutorial 09: Unsafe Rust and FFI - Complete Implementation
// 
// This file contains complete implementations of unsafe Rust patterns and FFI integrations
// for high-performance space simulation systems.

use std::ffi::{CStr, CString, c_char, c_int, c_float, c_void};
use std::ptr;
use std::marker::PhantomData;
use std::alloc::{GlobalAlloc, Layout, alloc, dealloc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use std::cell::UnsafeCell;

// ================================
// Exercise 1: Safe FFI Wrappers
// ================================

// Mock external C function bindings (in real code, these would come from a C library)
extern "C" {
    // Note: These are mock declarations - in real code you'd link against actual C library
    fn physics_create_world(gravity: *const c_float) -> *mut c_void;
    fn physics_destroy_world(world: *mut c_void);
    fn physics_create_ship(
        world: *mut c_void,
        mass: c_float,
        position: *const c_float,
        dimensions: *const c_float,
    ) -> c_int;
    fn physics_step_simulation(world: *mut c_void, dt: c_float) -> c_int;
    fn physics_get_transform(
        world: *mut c_void,
        object_id: c_int,
        position: *mut c_float,
        rotation: *mut c_float,
    ) -> c_int;
    fn physics_apply_force(
        world: *mut c_void,
        object_id: c_int,
        force: *const c_float,
        local_pos: *const c_float,
    ) -> c_int;
}

// Mock implementations for testing (normally these would be in a separate C library)
#[no_mangle]
pub extern "C" fn physics_create_world(_gravity: *const c_float) -> *mut c_void {
    // Mock implementation - returns a dummy pointer
    Box::into_raw(Box::new(42i32)) as *mut c_void
}

#[no_mangle]
pub extern "C" fn physics_destroy_world(world: *mut c_void) {
    if !world.is_null() {
        unsafe {
            let _ = Box::from_raw(world as *mut i32);
        }
    }
}

#[no_mangle]
pub extern "C" fn physics_create_ship(
    _world: *mut c_void,
    _mass: c_float,
    _position: *const c_float,
    _dimensions: *const c_float,
) -> c_int {
    // Mock implementation - returns incrementing IDs
    static mut NEXT_ID: c_int = 1;
    unsafe {
        let id = NEXT_ID;
        NEXT_ID += 1;
        id
    }
}

#[no_mangle]
pub extern "C" fn physics_step_simulation(_world: *mut c_void, _dt: c_float) -> c_int {
    0 // Success
}

#[no_mangle]
pub extern "C" fn physics_get_transform(
    _world: *mut c_void,
    _object_id: c_int,
    position: *mut c_float,
    rotation: *mut c_float,
) -> c_int {
    if !position.is_null() && !rotation.is_null() {
        unsafe {
            // Mock transform data
            *position.add(0) = 0.0;
            *position.add(1) = 0.0;
            *position.add(2) = 0.0;
            *rotation.add(0) = 0.0;
            *rotation.add(1) = 0.0;
            *rotation.add(2) = 0.0;
            *rotation.add(3) = 1.0;
        }
    }
    0 // Success
}

#[no_mangle]
pub extern "C" fn physics_apply_force(
    _world: *mut c_void,
    _object_id: c_int,
    _force: *const c_float,
    _local_pos: *const c_float,
) -> c_int {
    0 // Success
}

// Safe wrapper for the physics world
pub struct PhysicsWorld {
    world_ptr: *mut c_void,
    next_object_id: i32,
    objects: HashMap<i32, ObjectInfo>,
}

#[derive(Debug, Clone)]
struct ObjectInfo {
    object_type: String,
    mass: f32,
}

#[derive(Debug)]
pub enum PhysicsError {
    WorldCreationFailed,
    ObjectCreationFailed,
    InvalidObjectId,
    SimulationStepFailed,
    TransformQueryFailed,
    ForceApplicationFailed,
}

impl PhysicsWorld {
    pub fn new(gravity: [f32; 3]) -> Result<Self, PhysicsError> {
        let world_ptr = unsafe {
            physics_create_world(gravity.as_ptr())
        };
        
        if world_ptr.is_null() {
            return Err(PhysicsError::WorldCreationFailed);
        }
        
        Ok(Self {
            world_ptr,
            next_object_id: 1,
            objects: HashMap::new(),
        })
    }
    
    pub fn create_ship(
        &mut self,
        mass: f32,
        position: [f32; 3],
        dimensions: [f32; 3],
    ) -> Result<i32, PhysicsError> {
        let object_id = unsafe {
            physics_create_ship(
                self.world_ptr,
                mass,
                position.as_ptr(),
                dimensions.as_ptr(),
            )
        };
        
        if object_id < 0 {
            return Err(PhysicsError::ObjectCreationFailed);
        }
        
        self.objects.insert(object_id, ObjectInfo {
            object_type: "ship".to_string(),
            mass,
        });
        
        Ok(object_id)
    }
    
    pub fn step_simulation(&mut self, dt: f32) -> Result<(), PhysicsError> {
        let result = unsafe {
            physics_step_simulation(self.world_ptr, dt)
        };
        
        if result != 0 {
            Err(PhysicsError::SimulationStepFailed)
        } else {
            Ok(())
        }
    }
    
    pub fn get_transform(&self, object_id: i32) -> Result<([f32; 3], [f32; 4]), PhysicsError> {
        if !self.objects.contains_key(&object_id) {
            return Err(PhysicsError::InvalidObjectId);
        }
        
        let mut position = [0.0f32; 3];
        let mut rotation = [0.0f32; 4];
        
        let result = unsafe {
            physics_get_transform(
                self.world_ptr,
                object_id,
                position.as_mut_ptr(),
                rotation.as_mut_ptr(),
            )
        };
        
        if result != 0 {
            Err(PhysicsError::TransformQueryFailed)
        } else {
            Ok((position, rotation))
        }
    }
    
    pub fn apply_force(&mut self, object_id: i32, force: [f32; 3], local_pos: [f32; 3]) -> Result<(), PhysicsError> {
        if !self.objects.contains_key(&object_id) {
            return Err(PhysicsError::InvalidObjectId);
        }
        
        let result = unsafe {
            physics_apply_force(
                self.world_ptr,
                object_id,
                force.as_ptr(),
                local_pos.as_ptr(),
            )
        };
        
        if result != 0 {
            Err(PhysicsError::ForceApplicationFailed)
        } else {
            Ok(())
        }
    }
}

impl Drop for PhysicsWorld {
    fn drop(&mut self) {
        unsafe {
            physics_destroy_world(self.world_ptr);
        }
    }
}

// PhysicsWorld is not thread-safe due to C library limitations
unsafe impl Send for PhysicsWorld {}

// ================================
// Exercise 2: Custom Allocators
// ================================

pub struct ArenaAllocator {
    memory: UnsafeCell<Vec<u8>>,
    position: AtomicUsize,
    capacity: usize,
}

impl ArenaAllocator {
    pub fn new(capacity: usize) -> Self {
        let mut memory = Vec::with_capacity(capacity);
        unsafe {
            memory.set_len(capacity);
        }
        
        Self {
            memory: UnsafeCell::new(memory),
            position: AtomicUsize::new(0),
            capacity,
        }
    }
    
    pub fn allocate<T>(&self) -> Option<*mut T> {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        
        self.allocate_layout(Layout::from_size_align(size, align).ok()?)
            .map(|ptr| ptr as *mut T)
    }
    
    pub fn allocate_layout(&self, layout: Layout) -> Option<*mut u8> {
        let size = layout.size();
        let align = layout.align();
        
        loop {
            let current_pos = self.position.load(Ordering::Acquire);
            
            // Align the current position
            let aligned_pos = align_up(current_pos, align);
            let new_pos = aligned_pos + size;
            
            if new_pos > self.capacity {
                return None; // Out of memory
            }
            
            // Try to atomically update position
            match self.position.compare_exchange_weak(
                current_pos,
                new_pos,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully allocated
                    let memory = unsafe { &*self.memory.get() };
                    let ptr = unsafe { memory.as_ptr().add(aligned_pos) as *mut u8 };
                    return Some(ptr);
                }
                Err(_) => {
                    // Another thread updated position, retry
                    continue;
                }
            }
        }
    }
    
    pub fn reset(&self) {
        self.position.store(0, Ordering::Release);
    }
    
    pub fn used_bytes(&self) -> usize {
        self.position.load(Ordering::Acquire)
    }
    
    pub fn available_bytes(&self) -> usize {
        self.capacity - self.used_bytes()
    }
}

unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

pub struct ArenaBox<'a, T> {
    ptr: *mut T,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> ArenaBox<'a, T> {
    pub fn new_in(arena: &'a ArenaAllocator, value: T) -> Option<Self> {
        let ptr = arena.allocate::<T>()?;
        unsafe {
            // SAFETY: ptr is valid and properly aligned, allocated from arena
            ptr.write(value);
        }
        Some(Self {
            ptr,
            _phantom: PhantomData,
        })
    }
}

impl<T> std::ops::Deref for ArenaBox<'_, T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        unsafe {
            // SAFETY: ptr is valid for the lifetime of the arena
            &*self.ptr
        }
    }
}

impl<T> std::ops::DerefMut for ArenaBox<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            // SAFETY: ptr is valid and we have exclusive access
            &mut *self.ptr
        }
    }
}

impl<T> Drop for ArenaBox<'_, T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: ptr points to a valid T that we own
            // Run destructor but don't deallocate (arena handles bulk deallocation)
            std::ptr::drop_in_place(self.ptr);
        }
    }
}

// ================================
// Exercise 3: Pool Allocator
// ================================

pub struct PoolAllocator<T> {
    memory: UnsafeCell<Vec<u8>>,
    free_list: AtomicUsize, // Index into free list (usize::MAX = end)
    capacity: usize,
    _phantom: PhantomData<T>,
}

impl<T> PoolAllocator<T> {
    pub fn new(capacity: usize) -> Self {
        let object_size = std::mem::size_of::<T>().max(std::mem::size_of::<usize>());
        let total_size = object_size * capacity;
        
        let mut memory = vec![0u8; total_size];
        
        // Initialize free list
        for i in 0..capacity {
            let offset = i * object_size;
            let next_index = if i + 1 < capacity { i + 1 } else { usize::MAX };
            
            unsafe {
                // SAFETY: offset is within bounds and properly aligned for usize
                let ptr = memory.as_mut_ptr().add(offset) as *mut usize;
                *ptr = next_index;
            }
        }
        
        Self {
            memory: UnsafeCell::new(memory),
            free_list: AtomicUsize::new(0),
            capacity,
            _phantom: PhantomData,
        }
    }
    
    pub fn allocate(&self) -> Option<*mut T> {
        let object_size = std::mem::size_of::<T>().max(std::mem::size_of::<usize>());
        
        loop {
            let current_index = self.free_list.load(Ordering::Acquire);
            
            if current_index == usize::MAX {
                return None; // Pool exhausted
            }
            
            let memory = unsafe { &*self.memory.get() };
            let current_ptr = unsafe {
                // SAFETY: current_index is valid and within bounds
                memory.as_ptr().add(current_index * object_size) as *const usize
            };
            
            let next_index = unsafe {
                // SAFETY: current_ptr points to valid memory within our allocation
                *current_ptr
            };
            
            // Try to atomically update free list head
            match self.free_list.compare_exchange_weak(
                current_index,
                next_index,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully allocated
                    return Some(current_ptr as *mut T);
                }
                Err(_) => {
                    // Another thread updated free list, retry
                    continue;
                }
            }
        }
    }
    
    pub fn deallocate(&self, ptr: *mut T) {
        if ptr.is_null() {
            return;
        }
        
        let object_size = std::mem::size_of::<T>().max(std::mem::size_of::<usize>());
        let memory = unsafe { &*self.memory.get() };
        let memory_start = memory.as_ptr() as usize;
        let ptr_addr = ptr as usize;
        
        // Verify pointer is within our memory range
        assert!(ptr_addr >= memory_start);
        assert!(ptr_addr < memory_start + memory.len());
        assert!((ptr_addr - memory_start) % object_size == 0);
        
        let index = (ptr_addr - memory_start) / object_size;
        let list_ptr = ptr as *mut usize;
        
        loop {
            let current_head = self.free_list.load(Ordering::Acquire);
            
            unsafe {
                // SAFETY: ptr points to memory we own and is properly aligned
                *list_ptr = current_head;
            }
            
            match self.free_list.compare_exchange_weak(
                current_head,
                index,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(_) => continue,
            }
        }
    }
}

unsafe impl<T: Send> Send for PoolAllocator<T> {}
unsafe impl<T: Send> Sync for PoolAllocator<T> {}

pub struct PoolBox<T> {
    ptr: *mut T,
    pool: *const PoolAllocator<T>,
}

impl<T> PoolBox<T> {
    pub fn new_in(pool: &PoolAllocator<T>, value: T) -> Option<Self> {
        let ptr = pool.allocate()?;
        unsafe {
            // SAFETY: ptr is valid and properly aligned
            ptr.write(value);
        }
        Some(Self {
            ptr,
            pool: pool as *const _,
        })
    }
}

impl<T> std::ops::Deref for PoolBox<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        unsafe {
            // SAFETY: ptr is valid for lifetime of pool
            &*self.ptr
        }
    }
}

impl<T> std::ops::DerefMut for PoolBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            // SAFETY: ptr is valid and we have exclusive access
            &mut *self.ptr
        }
    }
}

impl<T> Drop for PoolBox<T> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: ptr points to valid T that we own
            std::ptr::drop_in_place(self.ptr);
            
            // SAFETY: pool pointer is valid for program duration
            let pool = &*self.pool;
            pool.deallocate(self.ptr);
        }
    }
}

// ================================
// Exercise 4: Raw Pointer Manipulation
// ================================

pub struct RawVec<T> {
    ptr: *mut T,
    cap: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> RawVec<T> {
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            cap: 0,
            len: 0,
            _phantom: PhantomData,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        
        let layout = Layout::array::<T>(capacity).expect("Capacity overflow");
        let ptr = unsafe {
            alloc(layout) as *mut T
        };
        
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        
        Self {
            ptr,
            cap: capacity,
            len: 0,
            _phantom: PhantomData,
        }
    }
    
    pub fn push(&mut self, value: T) -> Result<(), CollectionError> {
        if self.len == self.cap {
            self.grow()?;
        }
        
        unsafe {
            // SAFETY: len < cap, so this is within bounds
            self.ptr.add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }
    
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                // SAFETY: len was > 0, so this index contains a valid T
                Some(self.ptr.add(self.len).read())
            }
        }
    }
    
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe {
                // SAFETY: index is within bounds
                Some(&*self.ptr.add(index))
            }
        } else {
            None
        }
    }
    
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            unsafe {
                // SAFETY: index is within bounds
                Some(&mut *self.ptr.add(index))
            }
        } else {
            None
        }
    }
    
    pub fn len(&self) -> usize {
        self.len
    }
    
    pub fn capacity(&self) -> usize {
        self.cap
    }
    
    fn grow(&mut self) -> Result<(), CollectionError> {
        let new_cap = if self.cap == 0 { 1 } else { self.cap * 2 };
        
        let new_layout = Layout::array::<T>(new_cap)
            .map_err(|_| CollectionError::AllocationFailed)?;
        
        let new_ptr = if self.cap == 0 {
            unsafe { alloc(new_layout) as *mut T }
        } else {
            let old_layout = Layout::array::<T>(self.cap).unwrap();
            unsafe {
                std::alloc::realloc(
                    self.ptr as *mut u8,
                    old_layout,
                    new_layout.size(),
                ) as *mut T
            }
        };
        
        if new_ptr.is_null() {
            return Err(CollectionError::OutOfMemory);
        }
        
        self.ptr = new_ptr;
        self.cap = new_cap;
        Ok(())
    }
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                // SAFETY: i is within bounds and elements are valid
                std::ptr::drop_in_place(self.ptr.add(i));
            }
        }
        
        // Deallocate memory
        let layout = Layout::array::<T>(self.cap).unwrap();
        unsafe {
            // SAFETY: ptr was allocated with this layout
            dealloc(self.ptr as *mut u8, layout);
        }
    }
}

#[derive(Debug)]
pub enum CollectionError {
    OutOfMemory,
    AllocationFailed,
}

// ================================
// Exercise 5: C String Handling
// ================================

pub struct CStringManager;

impl CStringManager {
    pub fn to_c_string(rust_str: &str) -> Result<CString, StringError> {
        CString::new(rust_str).map_err(|_| StringError::ContainsNull)
    }
    
    pub fn from_c_string(c_str: *const c_char) -> Result<String, StringError> {
        if c_str.is_null() {
            return Err(StringError::NullPointer);
        }
        
        let c_str = unsafe {
            // SAFETY: We checked for null above
            CStr::from_ptr(c_str)
        };
        
        c_str.to_str()
            .map_err(|_| StringError::InvalidUtf8)
            .map(|s| s.to_string())
    }
    
    pub fn c_strlen(c_str: *const c_char) -> Result<usize, StringError> {
        if c_str.is_null() {
            return Err(StringError::NullPointer);
        }
        
        let c_str = unsafe {
            // SAFETY: We checked for null above
            CStr::from_ptr(c_str)
        };
        
        Ok(c_str.to_bytes().len())
    }
}

#[derive(Debug)]
pub enum StringError {
    NullPointer,
    InvalidUtf8,
    ContainsNull,
}

// ================================
// Exercise 6: Memory Layout and Alignment
// ================================

#[repr(C)]
pub struct CCompatibleTransform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // Quaternion [x, y, z, w]
    pub scale: [f32; 3],
}

// Static assertions to verify size and alignment
const _: () = assert!(std::mem::size_of::<CCompatibleTransform>() == 40);
const _: () = assert!(std::mem::align_of::<CCompatibleTransform>() == 4);

#[repr(C, packed)]
pub struct PackedNetworkMessage {
    pub message_type: u8,
    pub entity_id: u32,
    pub position: [f32; 3],
    pub timestamp: u64,
}

impl PackedNetworkMessage {
    pub fn new(message_type: u8, entity_id: u32, position: [f32; 3]) -> Self {
        Self {
            message_type,
            entity_id,
            position,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
    
    pub fn serialize(&self) -> &[u8] {
        unsafe {
            // SAFETY: PackedNetworkMessage has a defined layout
            std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }
    
    pub fn deserialize(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() != std::mem::size_of::<Self>() {
            return None;
        }
        
        unsafe {
            // SAFETY: We verified the size matches
            Some(&*(bytes.as_ptr() as *const Self))
        }
    }
}

// ================================
// Exercise 7: Callback Functions
// ================================

pub type CollisionCallback = extern "C" fn(
    object1_id: c_int,
    object2_id: c_int,
    user_data: *mut c_void,
) -> c_int;

pub struct CallbackManager {
    collision_callback: Option<Box<dyn FnMut(i32, i32) -> bool + 'static>>,
}

impl CallbackManager {
    pub fn new() -> Self {
        Self {
            collision_callback: None,
        }
    }
    
    pub fn register_collision_callback<F>(&mut self, callback: F) -> Result<(), CallbackError>
    where
        F: FnMut(i32, i32) -> bool + 'static,
    {
        self.collision_callback = Some(Box::new(callback));
        Ok(())
    }
}

extern "C" fn collision_callback_wrapper(
    object1_id: c_int,
    object2_id: c_int,
    user_data: *mut c_void,
) -> c_int {
    if user_data.is_null() {
        return 0;
    }
    
    unsafe {
        // SAFETY: user_data points to a valid CallbackManager
        let manager = &mut *(user_data as *mut CallbackManager);
        if let Some(ref mut callback) = manager.collision_callback {
            if callback(object1_id, object2_id) {
                1
            } else {
                0
            }
        } else {
            0
        }
    }
}

#[derive(Debug)]
pub enum CallbackError {
    InvalidCallback,
    RegistrationFailed,
}

// ================================
// Exercise 8: Global Allocator
// ================================

pub struct TrackingAllocator {
    allocation_count: AtomicUsize,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            allocation_count: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
        }
    }
    
    pub fn stats(&self) -> AllocationStats {
        AllocationStats {
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            peak_allocated: self.peak_allocated.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub allocation_count: usize,
    pub total_allocated: usize,
    pub peak_allocated: usize,
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = std::alloc::System.alloc(layout);
        if !ptr.is_null() {
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
            let new_total = self.total_allocated.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            
            // Update peak if necessary
            let mut current_peak = self.peak_allocated.load(Ordering::Relaxed);
            while new_total > current_peak {
                match self.peak_allocated.compare_exchange_weak(
                    current_peak,
                    new_total,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current_peak = actual,
                }
            }
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        std::alloc::System.dealloc(ptr, layout);
        self.allocation_count.fetch_sub(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

// ================================
// Exercise 9: SIMD Operations
// ================================

pub struct SimdVectorOps;

impl SimdVectorOps {
    pub fn add_arrays(a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), SimdError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(SimdError::DimensionMismatch);
        }
        
        // Simple scalar implementation (SIMD would require target-specific code)
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
        
        Ok(())
    }
    
    pub fn multiply_scalar(array: &mut [f32], scalar: f32) -> Result<(), SimdError> {
        if array.is_empty() {
            return Err(SimdError::InvalidInput);
        }
        
        for element in array.iter_mut() {
            *element *= scalar;
        }
        
        Ok(())
    }
    
    pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch);
        }
        
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        
        Ok(sum)
    }
}

#[derive(Debug)]
pub enum SimdError {
    DimensionMismatch,
    InvalidInput,
}

// ================================
// Exercise 10: Integration Example
// ================================

pub struct HighPerformanceSimulation {
    physics_world: PhysicsWorld,
    arena: ArenaAllocator,
    projectile_pool: PoolAllocator<Projectile>,
    callback_manager: CallbackManager,
}

#[derive(Debug)]
pub struct Projectile {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub damage: f32,
    pub lifetime: f32,
}

impl HighPerformanceSimulation {
    pub fn new() -> Result<Self, SimulationError> {
        let physics_world = PhysicsWorld::new([0.0, 0.0, 0.0])
            .map_err(SimulationError::PhysicsError)?;
        
        let arena = ArenaAllocator::new(1024 * 1024); // 1MB arena
        let projectile_pool = PoolAllocator::new(1000); // 1000 projectiles max
        let callback_manager = CallbackManager::new();
        
        Ok(Self {
            physics_world,
            arena,
            projectile_pool,
            callback_manager,
        })
    }
    
    pub fn add_ship(&mut self, position: [f32; 3], mass: f32) -> Result<i32, SimulationError> {
        self.physics_world
            .create_ship(mass, position, [10.0, 5.0, 20.0])
            .map_err(SimulationError::PhysicsError)
    }
    
    pub fn fire_projectile(&mut self, position: [f32; 3], velocity: [f32; 3]) -> Result<(), SimulationError> {
        let projectile = Projectile {
            position,
            velocity,
            damage: 10.0,
            lifetime: 5.0,
        };
        
        let _pool_box = PoolBox::new_in(&self.projectile_pool, projectile)
            .ok_or(SimulationError::AllocationError)?;
        
        // In a real implementation, we'd store the PoolBox somewhere
        Ok(())
    }
    
    pub fn step_simulation(&mut self, dt: f32) -> Result<(), SimulationError> {
        self.physics_world
            .step_simulation(dt)
            .map_err(SimulationError::PhysicsError)?;
        
        // Reset arena for temporary objects created this frame
        self.arena.reset();
        
        Ok(())
    }
}

#[derive(Debug)]
pub enum SimulationError {
    PhysicsError(PhysicsError),
    AllocationError,
    CallbackError(CallbackError),
}

// ================================
// Helper Functions
// ================================

pub fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

pub fn is_aligned(addr: usize, align: usize) -> bool {
    addr & (align - 1) == 0
}

pub fn null_terminate_string(s: &str) -> Vec<u8> {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

// ================================
// Testing
// ================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_allocator() {
        let arena = ArenaAllocator::new(1024);
        
        // Test allocation
        let ptr1 = arena.allocate::<i32>();
        assert!(ptr1.is_some());
        
        // Test used bytes tracking
        assert!(arena.used_bytes() >= std::mem::size_of::<i32>());
        
        // Test reset
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
    }
    
    #[test]
    fn test_pool_allocator() {
        let pool = PoolAllocator::<i32>::new(10);
        
        // Test allocation
        let ptr1 = pool.allocate();
        assert!(ptr1.is_some());
        
        let ptr2 = pool.allocate();
        assert!(ptr2.is_some());
        
        // Test deallocation
        pool.deallocate(ptr1.unwrap());
        
        // Should be able to allocate again
        let ptr3 = pool.allocate();
        assert!(ptr3.is_some());
        
        pool.deallocate(ptr2.unwrap());
        pool.deallocate(ptr3.unwrap());
    }
    
    #[test]
    fn test_raw_vec() {
        let mut vec = RawVec::new();
        
        // Test push/pop
        vec.push(42).unwrap();
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.get(0), Some(&42));
        assert_eq!(vec.pop(), Some(42));
        assert_eq!(vec.len(), 0);
        
        // Test growth
        for i in 0..10 {
            vec.push(i).unwrap();
        }
        assert_eq!(vec.len(), 10);
        assert!(vec.capacity() >= 10);
    }
    
    #[test]
    fn test_c_string_handling() {
        let rust_str = "Hello, World!";
        let c_string = CStringManager::to_c_string(rust_str).unwrap();
        let back_to_rust = CStringManager::from_c_string(c_string.as_ptr()).unwrap();
        assert_eq!(rust_str, back_to_rust);
        
        let len = CStringManager::c_strlen(c_string.as_ptr()).unwrap();
        assert_eq!(len, rust_str.len());
    }
    
    #[test]
    fn test_memory_layout() {
        let transform = CCompatibleTransform {
            position: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        };
        
        // Verify we can serialize and access fields
        assert_eq!(transform.position[0], 1.0);
        assert_eq!(transform.rotation[3], 1.0);
        
        // Test packed message
        let msg = PackedNetworkMessage::new(1, 42, [1.0, 2.0, 3.0]);
        let serialized = msg.serialize();
        let deserialized = PackedNetworkMessage::deserialize(serialized).unwrap();
        assert_eq!(deserialized.message_type, 1);
        assert_eq!(deserialized.entity_id, 42);
    }
    
    #[test]
    fn test_simd_operations() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];
        
        SimdVectorOps::add_arrays(&a, &b, &mut result).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        
        let dot = SimdVectorOps::dot_product(&a, &b).unwrap();
        assert_eq!(dot, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    }
    
    #[test]
    fn test_physics_integration() {
        let mut simulation = HighPerformanceSimulation::new().unwrap();
        
        let ship_id = simulation.add_ship([0.0, 0.0, 0.0], 1000.0).unwrap();
        assert!(ship_id > 0);
        
        simulation.fire_projectile([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        
        simulation.step_simulation(0.016).unwrap();
    }
    
    #[test]
    fn test_helper_functions() {
        assert_eq!(align_up(7, 4), 8);
        assert_eq!(align_up(8, 4), 8);
        assert_eq!(align_up(9, 4), 12);
        
        assert!(is_aligned(8, 4));
        assert!(!is_aligned(9, 4));
        
        let null_term = null_terminate_string("hello");
        assert_eq!(null_term, b"hello\0");
    }
}