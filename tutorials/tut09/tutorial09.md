# Tutorial 09: Unsafe Rust and FFI

## Learning Objectives
- Master the principles and use cases for unsafe Rust programming
- Understand when and why to use unsafe code in system programming contexts
- Learn Foreign Function Interface (FFI) patterns for C/C++ interoperability
- Implement safe abstractions around unsafe operations
- Apply memory management techniques for external library integration
- Create bindings for C libraries used in space simulation systems
- Design safe APIs that encapsulate unsafe implementations
- Handle raw pointers, memory layout, and ABI compatibility safely

## Lesson: Unsafe Rust and Foreign Function Interfaces

### What is Unsafe Rust?

Unsafe Rust is a subset of Rust that allows you to bypass some of the language's safety guarantees. While Rust's type system prevents most memory safety issues at compile time, there are scenarios where manual memory management, raw pointer manipulation, or interaction with foreign code requires "unsafe" operations.

**Important**: Unsafe doesn't mean "broken" or "dangerous" by default—it means the programmer takes responsibility for upholding Rust's safety invariants manually.

### The Philosophy of Unsafe Rust

#### Core Principles:
1. **Encapsulation**: Unsafe code should be wrapped in safe APIs
2. **Minimalism**: Use unsafe only when necessary and in the smallest possible scope
3. **Documentation**: Every unsafe block should explain why it's safe
4. **Testing**: Unsafe code requires more rigorous testing
5. **Audit Trail**: Make unsafe code easy to find and review

#### Rust's Safety Guarantees:
- **Memory safety**: No use-after-free, double-free, or buffer overflows
- **Thread safety**: No data races between threads
- **Type safety**: No invalid type conversions or uninitialized values

### When to Use Unsafe Rust

#### Legitimate Use Cases:
1. **FFI (Foreign Function Interface)**: Calling C/C++ libraries
2. **Performance optimization**: When Rust's safety checks are provably unnecessary
3. **Low-level system programming**: Direct hardware access, custom allocators
4. **Implementing safe abstractions**: Building safe APIs on unsafe foundations
5. **Interfacing with legacy systems**: Working with existing C codebases

#### What Unsafe Allows:
- **Raw pointer dereferencing**: `*raw_ptr`
- **Calling unsafe functions**: Including FFI functions
- **Accessing/modifying static mut variables**
- **Implementing unsafe traits**: Like `Send` and `Sync`
- **Accessing union fields**

### FFI (Foreign Function Interface)

FFI enables Rust to interact with code written in other languages, primarily C and C++. This is crucial for:
- **System integration**: Using existing C libraries
- **Performance-critical code**: Leveraging optimized C/C++ implementations
- **Legacy system compatibility**: Working with established codebases
- **Platform APIs**: Accessing OS-specific functionality

#### ABI (Application Binary Interface) Compatibility:
- **C ABI**: Most portable, used by default for FFI
- **Calling conventions**: How function parameters are passed
- **Memory layout**: Struct alignment and padding
- **Name mangling**: How function names are encoded

### Memory Management in FFI

#### Key Challenges:
1. **Ownership transfer**: Who owns allocated memory?
2. **Lifetime management**: When is it safe to free memory?
3. **Error handling**: How do foreign functions report errors?
4. **Resource cleanup**: Ensuring proper cleanup on panic or error

#### RAII (Resource Acquisition Is Initialization):
Rust's RAII patterns ensure resources are cleaned up automatically, even when interfacing with C code that requires manual cleanup.

### Safe Abstractions Around Unsafe Code

The key to using unsafe Rust effectively is building **safe abstractions** that encapsulate unsafe operations:

```rust
// Unsafe implementation hidden behind safe API
pub struct SafeBuffer {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

impl SafeBuffer {
    pub fn new(capacity: usize) -> Self {
        let ptr = unsafe {
            // Unsafe allocation, but we maintain invariants
            std::alloc::alloc_zeroed(std::alloc::Layout::array::<u8>(capacity).unwrap())
        };
        Self { ptr, len: 0, capacity }
    }
    
    // Safe API - bounds checking prevents buffer overflows
    pub fn push(&mut self, byte: u8) -> Result<(), BufferFullError> {
        if self.len >= self.capacity {
            return Err(BufferFullError);
        }
        
        unsafe {
            // Safe because we checked bounds above
            *self.ptr.add(self.len) = byte;
        }
        self.len += 1;
        Ok(())
    }
}

impl Drop for SafeBuffer {
    fn drop(&mut self) {
        unsafe {
            // Safe cleanup - ptr came from alloc, layout matches
            std::alloc::dealloc(
                self.ptr,
                std::alloc::Layout::array::<u8>(self.capacity).unwrap()
            );
        }
    }
}
```

### FFI Patterns and Best Practices

#### 1. C Function Bindings:
```rust
use std::ffi::{CStr, CString, c_char, c_int};

// External C function declaration
extern "C" {
    fn c_function(input: *const c_char) -> c_int;
}

// Safe Rust wrapper
pub fn safe_c_function(input: &str) -> Result<i32, StringConversionError> {
    let c_string = CString::new(input)?;
    let result = unsafe {
        c_function(c_string.as_ptr())
    };
    Ok(result)
}
```

#### 2. Struct Layout Compatibility:
```rust
#[repr(C)]
pub struct CCompatibleStruct {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// Ensures same memory layout as C struct
static_assertions::assert_eq_size!(CCompatibleStruct, [u8; 12]);
```

#### 3. Callback Functions:
```rust
type CallbackFunction = extern "C" fn(data: *mut c_void) -> c_int;

extern "C" {
    fn register_callback(callback: CallbackFunction, user_data: *mut c_void);
}

// Safe callback wrapper
extern "C" fn safe_callback_wrapper(data: *mut c_void) -> c_int {
    let callback: &mut dyn FnMut() -> i32 = unsafe {
        &mut *(data as *mut &mut dyn FnMut() -> i32)
    };
    callback()
}
```

### Space Simulation Context

In our space simulation engine, unsafe Rust and FFI are particularly useful for:

#### 1. **High-Performance Physics Libraries**:
- Interfacing with optimized C++ physics engines (Bullet Physics, PhysX)
- GPU compute integration (CUDA, OpenCL)
- SIMD-optimized mathematical operations

#### 2. **Platform Integration**:
- Operating system APIs for threading, memory management
- Graphics APIs (Vulkan, DirectX, OpenGL)
- Network stack optimization

#### 3. **Legacy System Integration**:
- Existing C/C++ simulation codebases
- Scientific computing libraries (BLAS, LAPACK)
- Database connectors and file format parsers

#### 4. **Performance-Critical Operations**:
- Custom memory allocators for specific use cases
- Lock-free data structures
- Optimized serialization/deserialization

### Common Pitfalls and How to Avoid Them

#### 1. **Use-After-Free**:
```rust
// WRONG: Dangling pointer
let mut vec = vec![1, 2, 3];
let ptr = vec.as_mut_ptr();
drop(vec);
// unsafe { *ptr = 42; } // Use-after-free!

// CORRECT: Ensure lifetime overlap
let mut vec = vec![1, 2, 3];
let ptr = vec.as_mut_ptr();
unsafe { *ptr = 42; } // Safe within vec's lifetime
```

#### 2. **Buffer Overflows**:
```rust
// WRONG: No bounds checking
unsafe fn write_unchecked(buffer: *mut u8, index: usize, value: u8) {
    *buffer.add(index) = value; // Could overflow!
}

// CORRECT: Bounds checking in safe wrapper
pub fn write_checked(buffer: &mut [u8], index: usize, value: u8) -> Result<(), IndexError> {
    if index >= buffer.len() {
        return Err(IndexError::OutOfBounds);
    }
    unsafe {
        *buffer.as_mut_ptr().add(index) = value; // Safe due to bounds check
    }
    Ok(())
}
```

#### 3. **Null Pointer Dereferencing**:
```rust
// WRONG: No null check
unsafe fn dereference_unchecked(ptr: *const i32) -> i32 {
    *ptr // Could be null!
}

// CORRECT: Null checking
pub fn dereference_safe(ptr: *const i32) -> Option<i32> {
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { *ptr })
    }
}
```

### Testing Unsafe Code

Unsafe code requires additional testing strategies:

#### 1. **Property-Based Testing**:
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_safe_buffer_operations(
        operations in prop::collection::vec(
            prop::oneof![
                Just(BufferOp::Push(prop::any::<u8>())),
                Just(BufferOp::Pop),
            ],
            0..100
        )
    ) {
        let mut buffer = SafeBuffer::new(50);
        for op in operations {
            match op {
                BufferOp::Push(byte) => {
                    let _ = buffer.push(byte);
                }
                BufferOp::Pop => {
                    let _ = buffer.pop();
                }
            }
            // Invariants should hold after every operation
            assert!(buffer.len() <= buffer.capacity());
        }
    }
}
```

#### 2. **Memory Sanitization**:
- Use **AddressSanitizer** (ASan) to detect memory errors
- **Miri** - Rust's interpreter for detecting undefined behavior
- **Valgrind** for additional memory error detection

#### 3. **Fuzzing**:
```rust
#[cfg(fuzzing)]
pub fn fuzz_target(data: &[u8]) {
    if let Ok(input) = std::str::from_utf8(data) {
        let _ = our_unsafe_parser(input);
    }
}
```

### Performance Considerations

#### When Unsafe Provides Benefits:
1. **Eliminating bounds checks**: When you can prove safety
2. **Custom memory layouts**: For cache optimization
3. **Zero-copy operations**: Direct memory manipulation
4. **SIMD optimizations**: Hand-tuned vectorized code

#### When to Avoid Unsafe:
1. **Premature optimization**: Profile first
2. **Unclear safety**: If you can't prove it's safe
3. **Available safe alternatives**: When safe code is fast enough
4. **Complex unsafe patterns**: High maintenance cost

### Documentation and Safety Comments

Every unsafe block should include a safety comment explaining why it's safe:

```rust
unsafe {
    // SAFETY: We know `ptr` is valid because:
    // 1. It was allocated with the same layout we're using
    // 2. The index is bounds-checked above
    // 3. The lifetime of the allocation extends beyond this use
    *ptr.add(index) = value;
}
```

### Why This Matters for Space Simulation

In our space simulation engine, unsafe Rust enables:

#### 1. **Performance**: 
- Zero-cost abstractions where Rust's safety checks are provably unnecessary
- Direct memory manipulation for high-frequency operations
- Custom allocators optimized for specific access patterns

#### 2. **Interoperability**: 
- Integration with existing C/C++ physics engines
- Platform-specific optimizations and system calls
- GPU computing frameworks (CUDA, OpenCL)

#### 3. **System Programming**: 
- Custom networking protocols with precise memory layout control
- Real-time systems with deterministic memory allocation
- Hardware interface programming for specialized devices

#### 4. **Legacy Integration**: 
- Wrapping existing simulation libraries
- Interfacing with scientific computing frameworks
- Database and file format compatibility

The key is using unsafe Rust **judiciously** - only when necessary, always with safe abstractions, and with thorough testing and documentation.

## Key Concepts

### 1. Safe FFI Wrappers for C Libraries

Creating safe Rust interfaces for C libraries commonly used in simulation systems.

```rust
use std::ffi::{CStr, CString, c_char, c_int, c_float, c_void};
use std::ptr;
use std::slice;

// ===================================
// Physics Engine FFI (e.g., Bullet Physics)
// ===================================

// External C library functions
extern "C" {
    // World management
    fn bt_create_world() -> *mut c_void;
    fn bt_destroy_world(world: *mut c_void);
    fn bt_step_simulation(world: *mut c_void, time_step: c_float) -> c_int;
    
    // Rigid body management
    fn bt_create_rigid_body(
        world: *mut c_void,
        mass: c_float,
        position: *const c_float, // [x, y, z]
        rotation: *const c_float, // [x, y, z, w] quaternion
    ) -> *mut c_void;
    
    fn bt_destroy_rigid_body(world: *mut c_void, body: *mut c_void);
    fn bt_get_body_transform(body: *mut c_void, position: *mut c_float, rotation: *mut c_float);
    fn bt_set_body_velocity(body: *mut c_void, velocity: *const c_float);
    
    // Collision detection
    fn bt_get_collision_pairs(
        world: *mut c_void,
        pairs_buffer: *mut c_int,
        max_pairs: c_int,
    ) -> c_int;
}

// Safe Rust wrapper for physics world
pub struct PhysicsWorld {
    world_ptr: *mut c_void,
    bodies: Vec<PhysicsBody>,
}

pub struct PhysicsBody {
    body_ptr: *mut c_void,
    world_ptr: *mut c_void, // Keep reference to world for cleanup
    id: usize,
}

#[derive(Debug, Clone)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // Quaternion [x, y, z, w]
}

#[derive(Debug)]
pub struct CollisionPair {
    pub body1_id: usize,
    pub body2_id: usize,
}

#[derive(Debug)]
pub enum PhysicsError {
    WorldCreationFailed,
    BodyCreationFailed,
    InvalidBodyId,
    SimulationError,
}

impl PhysicsWorld {
    pub fn new() -> Result<Self, PhysicsError> {
        let world_ptr = unsafe { bt_create_world() };
        
        if world_ptr.is_null() {
            return Err(PhysicsError::WorldCreationFailed);
        }
        
        Ok(Self {
            world_ptr,
            bodies: Vec::new(),
        })
    }
    
    pub fn create_rigid_body(
        &mut self,
        mass: f32,
        transform: Transform,
    ) -> Result<usize, PhysicsError> {
        let body_ptr = unsafe {
            bt_create_rigid_body(
                self.world_ptr,
                mass,
                transform.position.as_ptr(),
                transform.rotation.as_ptr(),
            )
        };
        
        if body_ptr.is_null() {
            return Err(PhysicsError::BodyCreationFailed);
        }
        
        let body_id = self.bodies.len();
        self.bodies.push(PhysicsBody {
            body_ptr,
            world_ptr: self.world_ptr,
            id: body_id,
        });
        
        Ok(body_id)
    }
    
    pub fn step_simulation(&mut self, time_step: f32) -> Result<(), PhysicsError> {
        let result = unsafe {
            bt_step_simulation(self.world_ptr, time_step)
        };
        
        if result != 0 {
            Err(PhysicsError::SimulationError)
        } else {
            Ok(())
        }
    }
    
    pub fn get_body_transform(&self, body_id: usize) -> Result<Transform, PhysicsError> {
        let body = self.bodies.get(body_id)
            .ok_or(PhysicsError::InvalidBodyId)?;
        
        let mut position = [0.0f32; 3];
        let mut rotation = [0.0f32; 4];
        
        unsafe {
            bt_get_body_transform(
                body.body_ptr,
                position.as_mut_ptr(),
                rotation.as_mut_ptr(),
            );
        }
        
        Ok(Transform { position, rotation })
    }
    
    pub fn set_body_velocity(&mut self, body_id: usize, velocity: [f32; 3]) -> Result<(), PhysicsError> {
        let body = self.bodies.get(body_id)
            .ok_or(PhysicsError::InvalidBodyId)?;
        
        unsafe {
            bt_set_body_velocity(body.body_ptr, velocity.as_ptr());
        }
        
        Ok(())
    }
    
    pub fn get_collision_pairs(&self) -> Result<Vec<CollisionPair>, PhysicsError> {
        const MAX_PAIRS: usize = 1000;
        let mut pairs_buffer = vec![0i32; MAX_PAIRS * 2]; // Each pair has 2 IDs
        
        let pair_count = unsafe {
            bt_get_collision_pairs(
                self.world_ptr,
                pairs_buffer.as_mut_ptr(),
                MAX_PAIRS as c_int,
            )
        };
        
        if pair_count < 0 {
            return Err(PhysicsError::SimulationError);
        }
        
        let mut collision_pairs = Vec::new();
        for i in 0..(pair_count as usize) {
            collision_pairs.push(CollisionPair {
                body1_id: pairs_buffer[i * 2] as usize,
                body2_id: pairs_buffer[i * 2 + 1] as usize,
            });
        }
        
        Ok(collision_pairs)
    }
}

impl Drop for PhysicsWorld {
    fn drop(&mut self) {
        // Clean up all bodies first
        for body in &self.bodies {
            unsafe {
                bt_destroy_rigid_body(self.world_ptr, body.body_ptr);
            }
        }
        
        // Then clean up the world
        unsafe {
            bt_destroy_world(self.world_ptr);
        }
    }
}

// Thread safety: PhysicsWorld is not thread-safe due to C library limitations
// Users must handle synchronization externally
unsafe impl Send for PhysicsWorld {}
// Note: Not implementing Sync as the C library is not thread-safe

// ===================================
// Math Library FFI (e.g., Intel MKL)
// ===================================

extern "C" {
    // BLAS Level 1: Vector operations
    fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        incx: c_int,
        y: *mut c_float,
        incy: c_int,
    );
    
    fn cblas_sdot(
        n: c_int,
        x: *const c_float,
        incx: c_int,
        y: *const c_float,
        incy: c_int,
    ) -> c_float;
    
    // BLAS Level 2: Matrix-vector operations
    fn cblas_sgemv(
        layout: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        x: *const c_float,
        incx: c_int,
        beta: c_float,
        y: *mut c_float,
        incy: c_int,
    );
}

// Constants for BLAS operations
const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_TRANS: c_int = 112;

pub struct HighPerformanceMath;

impl HighPerformanceMath {
    /// Compute y = alpha * x + y (SAXPY operation)
    pub fn vector_add_scaled(alpha: f32, x: &[f32], y: &mut [f32]) -> Result<(), MathError> {
        if x.len() != y.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        if x.is_empty() {
            return Ok(());
        }
        
        unsafe {
            cblas_saxpy(
                x.len() as c_int,
                alpha,
                x.as_ptr(),
                1, // increment = 1 (contiguous)
                y.as_mut_ptr(),
                1,
            );
        }
        
        Ok(())
    }
    
    /// Compute dot product of two vectors
    pub fn dot_product(x: &[f32], y: &[f32]) -> Result<f32, MathError> {
        if x.len() != y.len() {
            return Err(MathError::DimensionMismatch);
        }
        
        if x.is_empty() {
            return Ok(0.0);
        }
        
        let result = unsafe {
            cblas_sdot(
                x.len() as c_int,
                x.as_ptr(),
                1,
                y.as_ptr(),
                1,
            )
        };
        
        Ok(result)
    }
    
    /// Matrix-vector multiplication: y = alpha * A * x + beta * y
    pub fn matrix_vector_multiply(
        alpha: f32,
        matrix: &[f32], // Row-major order
        rows: usize,
        cols: usize,
        x: &[f32],
        beta: f32,
        y: &mut [f32],
    ) -> Result<(), MathError> {
        if matrix.len() != rows * cols {
            return Err(MathError::InvalidMatrixDimensions);
        }
        if x.len() != cols {
            return Err(MathError::DimensionMismatch);
        }
        if y.len() != rows {
            return Err(MathError::DimensionMismatch);
        }
        
        unsafe {
            cblas_sgemv(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                rows as c_int,
                cols as c_int,
                alpha,
                matrix.as_ptr(),
                cols as c_int, // leading dimension
                x.as_ptr(),
                1,
                beta,
                y.as_mut_ptr(),
                1,
            );
        }
        
        Ok(())
    }
}

#[derive(Debug)]
pub enum MathError {
    DimensionMismatch,
    InvalidMatrixDimensions,
}
```

### 2. Custom Memory Allocators for Simulation Performance

Implementing specialized allocators for different simulation subsystems.

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::marker::PhantomData;

// ===================================
// Arena Allocator for Temporary Objects
// ===================================

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
        
        // Atomic allocation to support concurrent access
        loop {
            let current_pos = self.position.load(Ordering::Acquire);
            
            // Align the current position
            let aligned_pos = (current_pos + align - 1) & !(align - 1);
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

// RAII wrapper for arena-allocated objects
pub struct ArenaBox<'a, T> {
    ptr: *mut T,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T> ArenaBox<'a, T> {
    pub fn new_in(arena: &'a ArenaAllocator, value: T) -> Option<Self> {
        let ptr = arena.allocate::<T>()?;
        unsafe {
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
        unsafe { &*self.ptr }
    }
}

impl<T> std::ops::DerefMut for ArenaBox<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

impl<T> Drop for ArenaBox<'_, T> {
    fn drop(&mut self) {
        unsafe {
            // Run destructor but don't deallocate
            // (arena handles bulk deallocation)
            std::ptr::drop_in_place(self.ptr);
        }
    }
}

// ===================================
// Pool Allocator for Fixed-Size Objects
// ===================================

pub struct PoolAllocator<T> {
    memory: UnsafeCell<Vec<u8>>,
    free_list: AtomicUsize, // Index into free list
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
                memory.as_ptr().add(current_index * object_size) as *const usize
            };
            
            let next_index = unsafe { *current_ptr };
            
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

// RAII wrapper for pool-allocated objects
pub struct PoolBox<T> {
    ptr: *mut T,
    pool: *const PoolAllocator<T>,
}

impl<T> PoolBox<T> {
    pub fn new_in(pool: &PoolAllocator<T>, value: T) -> Option<Self> {
        let ptr = pool.allocate()?;
        unsafe {
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
        unsafe { &*self.ptr }
    }
}

impl<T> std::ops::DerefMut for PoolBox<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

impl<T> Drop for PoolBox<T> {
    fn drop(&mut self) {
        unsafe {
            // Run destructor
            std::ptr::drop_in_place(self.ptr);
            
            // Return to pool
            let pool = &*self.pool;
            pool.deallocate(self.ptr);
        }
    }
}

// ===================================
// Global Allocator Integration
// ===================================

pub struct SimulationAllocator {
    system_allocator: System,
    allocation_count: AtomicUsize,
    total_allocated: AtomicUsize,
}

impl SimulationAllocator {
    pub const fn new() -> Self {
        Self {
            system_allocator: System,
            allocation_count: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
        }
    }
    
    pub fn allocation_stats(&self) -> AllocationStats {
        AllocationStats {
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AllocationStats {
    pub allocation_count: usize,
    pub total_allocated: usize,
}

unsafe impl GlobalAlloc for SimulationAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.system_allocator.alloc(layout);
        if !ptr.is_null() {
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
            self.total_allocated.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.system_allocator.dealloc(ptr, layout);
        self.allocation_count.fetch_sub(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

// Example usage in simulation systems
pub struct SimulationFrame<'arena> {
    arena: &'arena ArenaAllocator,
    temp_entities: Vec<ArenaBox<'arena, TempEntity>>,
    projectile_pool: &'arena PoolAllocator<Projectile>,
}

#[derive(Debug)]
pub struct TempEntity {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub lifetime: f32,
}

#[derive(Debug)]
pub struct Projectile {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub damage: f32,
    pub active: bool,
}

impl<'arena> SimulationFrame<'arena> {
    pub fn new(
        arena: &'arena ArenaAllocator,
        projectile_pool: &'arena PoolAllocator<Projectile>,
    ) -> Self {
        Self {
            arena,
            temp_entities: Vec::new(),
            projectile_pool,
        }
    }
    
    pub fn spawn_explosion(&mut self, position: [f32; 3]) -> Option<()> {
        for i in 0..10 {
            let entity = TempEntity {
                position,
                velocity: [
                    (i as f32 - 5.0) * 2.0,
                    (i as f32 - 5.0) * 2.0,
                    (i as f32 - 5.0) * 2.0,
                ],
                lifetime: 2.0,
            };
            
            let arena_box = ArenaBox::new_in(self.arena, entity)?;
            self.temp_entities.push(arena_box);
        }
        Some(())
    }
    
    pub fn fire_projectile(&self, position: [f32; 3], velocity: [f32; 3]) -> Option<PoolBox<Projectile>> {
        let projectile = Projectile {
            position,
            velocity,
            damage: 10.0,
            active: true,
        };
        
        PoolBox::new_in(self.projectile_pool, projectile)
    }
}
```

### 3. GPU Computing Integration with Unsafe Rust

Integrating CUDA and OpenCL for GPU-accelerated physics and simulation.

```rust
use std::ffi::{c_void, c_int, c_float};
use std::ptr;
use std::slice;

// ===================================
// CUDA Integration
// ===================================

// CUDA Runtime API bindings
extern "C" {
    fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(devPtr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaGetLastError() -> c_int;
    fn cudaGetErrorString(error: c_int) -> *const std::ffi::c_char;
}

// CUDA kernel launch (simplified)
extern "C" {
    fn launch_physics_kernel(
        positions: *mut c_float,
        velocities: *mut c_float,
        forces: *const c_float,
        count: c_int,
        dt: c_float,
    );
    
    fn launch_collision_kernel(
        positions: *const c_float,
        radii: *const c_float,
        collision_pairs: *mut c_int,
        max_pairs: c_int,
        count: c_int,
    ) -> c_int;
}

// CUDA memory copy kinds
const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const CUDA_SUCCESS: c_int = 0;

#[derive(Debug)]
pub enum CudaError {
    AllocationFailed,
    MemoryCopyFailed,
    KernelLaunchFailed(String),
    SynchronizationFailed,
}

pub struct CudaBuffer<T> {
    device_ptr: *mut c_void,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T> CudaBuffer<T> {
    pub fn new(count: usize) -> Result<Self, CudaError> {
        let size = count * std::mem::size_of::<T>();
        let mut device_ptr = ptr::null_mut();
        
        let result = unsafe {
            cudaMalloc(&mut device_ptr, size)
        };
        
        if result != CUDA_SUCCESS {
            return Err(CudaError::AllocationFailed);
        }
        
        Ok(Self {
            device_ptr,
            size,
            _phantom: PhantomData,
        })
    }
    
    pub fn copy_from_host(&self, host_data: &[T]) -> Result<(), CudaError> {
        let byte_size = host_data.len() * std::mem::size_of::<T>();
        if byte_size > self.size {
            return Err(CudaError::MemoryCopyFailed);
        }
        
        let result = unsafe {
            cudaMemcpy(
                self.device_ptr,
                host_data.as_ptr() as *const c_void,
                byte_size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        
        if result != CUDA_SUCCESS {
            Err(CudaError::MemoryCopyFailed)
        } else {
            Ok(())
        }
    }
    
    pub fn copy_to_host(&self, host_data: &mut [T]) -> Result<(), CudaError> {
        let byte_size = host_data.len() * std::mem::size_of::<T>();
        if byte_size > self.size {
            return Err(CudaError::MemoryCopyFailed);
        }
        
        let result = unsafe {
            cudaMemcpy(
                host_data.as_mut_ptr() as *mut c_void,
                self.device_ptr,
                byte_size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        
        if result != CUDA_SUCCESS {
            Err(CudaError::MemoryCopyFailed)
        } else {
            Ok(())
        }
    }
    
    pub fn as_device_ptr(&self) -> *mut c_void {
        self.device_ptr
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.device_ptr);
        }
    }
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Send> Sync for CudaBuffer<T> {}

pub struct CudaPhysicsEngine {
    d_positions: CudaBuffer<f32>,
    d_velocities: CudaBuffer<f32>,
    d_forces: CudaBuffer<f32>,
    d_radii: CudaBuffer<f32>,
    d_collision_pairs: CudaBuffer<i32>,
    max_entities: usize,
}

impl CudaPhysicsEngine {
    pub fn new(max_entities: usize) -> Result<Self, CudaError> {
        // Each position/velocity has 3 components (x, y, z)
        let d_positions = CudaBuffer::new(max_entities * 3)?;
        let d_velocities = CudaBuffer::new(max_entities * 3)?;
        let d_forces = CudaBuffer::new(max_entities * 3)?;
        let d_radii = CudaBuffer::new(max_entities)?;
        
        // Collision pairs buffer: [count, pair1_a, pair1_b, pair2_a, pair2_b, ...]
        let d_collision_pairs = CudaBuffer::new(max_entities * 2 + 1)?;
        
        Ok(Self {
            d_positions,
            d_velocities,
            d_forces,
            d_radii,
            d_collision_pairs,
            max_entities,
        })
    }
    
    pub fn update_physics(
        &self,
        positions: &mut [[f32; 3]],
        velocities: &[[f32; 3]],
        forces: &[[f32; 3]],
        dt: f32,
    ) -> Result<(), CudaError> {
        let entity_count = positions.len().min(self.max_entities);
        
        // Flatten 3D arrays to 1D for GPU transfer
        let flat_positions: Vec<f32> = positions[..entity_count]
            .iter()
            .flat_map(|pos| pos.iter().copied())
            .collect();
        
        let flat_velocities: Vec<f32> = velocities[..entity_count]
            .iter()
            .flat_map(|vel| vel.iter().copied())
            .collect();
            
        let flat_forces: Vec<f32> = forces[..entity_count]
            .iter()
            .flat_map(|force| force.iter().copied())
            .collect();
        
        // Upload data to GPU
        self.d_positions.copy_from_host(&flat_positions)?;
        self.d_velocities.copy_from_host(&flat_velocities)?;
        self.d_forces.copy_from_host(&flat_forces)?;
        
        // Launch physics kernel
        unsafe {
            launch_physics_kernel(
                self.d_positions.as_device_ptr() as *mut c_float,
                self.d_velocities.as_device_ptr() as *mut c_float,
                self.d_forces.as_device_ptr() as *const c_float,
                entity_count as c_int,
                dt,
            );
        }
        
        // Check for kernel launch errors
        let error = unsafe { cudaGetLastError() };
        if error != CUDA_SUCCESS {
            let error_str = unsafe {
                let c_str = cudaGetErrorString(error);
                std::ffi::CStr::from_ptr(c_str).to_string_lossy().into_owned()
            };
            return Err(CudaError::KernelLaunchFailed(error_str));
        }
        
        // Synchronize to ensure kernel completion
        let sync_result = unsafe { cudaDeviceSynchronize() };
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::SynchronizationFailed);
        }
        
        // Download updated positions
        let mut updated_positions = vec![0.0f32; entity_count * 3];
        self.d_positions.copy_to_host(&mut updated_positions)?;
        
        // Convert back to 3D arrays
        for (i, pos) in positions[..entity_count].iter_mut().enumerate() {
            pos[0] = updated_positions[i * 3];
            pos[1] = updated_positions[i * 3 + 1];
            pos[2] = updated_positions[i * 3 + 2];
        }
        
        Ok(())
    }
    
    pub fn detect_collisions(
        &self,
        positions: &[[f32; 3]],
        radii: &[f32],
    ) -> Result<Vec<(usize, usize)>, CudaError> {
        let entity_count = positions.len().min(radii.len()).min(self.max_entities);
        
        // Prepare data for GPU
        let flat_positions: Vec<f32> = positions[..entity_count]
            .iter()
            .flat_map(|pos| pos.iter().copied())
            .collect();
        
        // Upload data
        self.d_positions.copy_from_host(&flat_positions)?;
        self.d_radii.copy_from_host(&radii[..entity_count])?;
        
        // Launch collision detection kernel
        let max_pairs = unsafe {
            launch_collision_kernel(
                self.d_positions.as_device_ptr() as *const c_float,
                self.d_radii.as_device_ptr() as *const c_float,
                self.d_collision_pairs.as_device_ptr() as *mut c_int,
                (self.max_entities * 2) as c_int,
                entity_count as c_int,
            )
        };
        
        if max_pairs < 0 {
            return Err(CudaError::KernelLaunchFailed("Collision detection failed".to_string()));
        }
        
        // Synchronize
        let sync_result = unsafe { cudaDeviceSynchronize() };
        if sync_result != CUDA_SUCCESS {
            return Err(CudaError::SynchronizationFailed);
        }
        
        // Download collision pairs
        let mut collision_data = vec![0i32; max_pairs as usize * 2 + 1];
        self.d_collision_pairs.copy_to_host(&mut collision_data)?;
        
        let pair_count = collision_data[0] as usize;
        let mut collision_pairs = Vec::new();
        
        for i in 0..pair_count {
            let entity1 = collision_data[i * 2 + 1] as usize;
            let entity2 = collision_data[i * 2 + 2] as usize;
            collision_pairs.push((entity1, entity2));
        }
        
        Ok(collision_pairs)
    }
}

// ===================================
// High-Level Safe Interface
// ===================================

pub struct GpuAcceleratedSimulation {
    cuda_engine: Option<CudaPhysicsEngine>,
    fallback_cpu: bool,
}

impl GpuAcceleratedSimulation {
    pub fn new(max_entities: usize) -> Self {
        let cuda_engine = match CudaPhysicsEngine::new(max_entities) {
            Ok(engine) => {
                println!("CUDA physics engine initialized successfully");
                Some(engine)
            }
            Err(e) => {
                println!("Failed to initialize CUDA: {:?}, falling back to CPU", e);
                None
            }
        };
        
        Self {
            cuda_engine,
            fallback_cpu: cuda_engine.is_none(),
        }
    }
    
    pub fn update_physics(
        &self,
        positions: &mut [[f32; 3]],
        velocities: &[[f32; 3]],
        forces: &[[f32; 3]],
        dt: f32,
    ) -> Result<(), String> {
        if let Some(ref cuda_engine) = self.cuda_engine {
            match cuda_engine.update_physics(positions, velocities, forces, dt) {
                Ok(()) => Ok(()),
                Err(e) => {
                    // Fallback to CPU if GPU fails
                    println!("GPU physics failed: {:?}, using CPU fallback", e);
                    self.update_physics_cpu(positions, velocities, forces, dt);
                    Ok(())
                }
            }
        } else {
            self.update_physics_cpu(positions, velocities, forces, dt);
            Ok(())
        }
    }
    
    fn update_physics_cpu(
        &self,
        positions: &mut [[f32; 3]],
        velocities: &[[f32; 3]],
        forces: &[[f32; 3]],
        dt: f32,
    ) {
        // CPU fallback implementation
        for i in 0..positions.len() {
            for j in 0..3 {
                // Simple Euler integration: pos += vel * dt + 0.5 * force * dt^2
                positions[i][j] += velocities[i][j] * dt + 0.5 * forces[i][j] * dt * dt;
            }
        }
    }
    
    pub fn is_gpu_accelerated(&self) -> bool {
        self.cuda_engine.is_some()
    }
}
```

## Practical Application: Space Simulation C++ Physics Integration

```rust
use std::ffi::{CStr, CString, c_char, c_int, c_float, c_void};
use std::ptr;
use std::collections::HashMap;

// Complete example integrating with a C++ physics library for space simulation
extern "C" {
    // C++ physics library interface
    fn physics_create_world(gravity: *const c_float) -> *mut c_void;
    fn physics_destroy_world(world: *mut c_void);
    fn physics_step(world: *mut c_void, dt: c_float) -> c_int;
    
    fn physics_create_ship(
        world: *mut c_void,
        mass: c_float,
        position: *const c_float,
        dimensions: *const c_float,
    ) -> c_int;
    
    fn physics_create_asteroid(
        world: *mut c_void,
        mass: c_float,
        position: *const c_float,
        radius: c_float,
    ) -> c_int;
    
    fn physics_get_object_transform(
        world: *mut c_void,
        object_id: c_int,
        position: *mut c_float,
        rotation: *mut c_float,
    ) -> c_int;
    
    fn physics_apply_force(
        world: *mut c_void,
        object_id: c_int,
        force: *const c_float,
        position: *const c_float,
    ) -> c_int;
    
    fn physics_get_collisions(
        world: *mut c_void,
        collision_buffer: *mut CollisionInfo,
        max_collisions: c_int,
    ) -> c_int;
}

#[repr(C)]
pub struct CollisionInfo {
    pub object1_id: c_int,
    pub object2_id: c_int,
    pub contact_point: [c_float; 3],
    pub normal: [c_float; 3],
    pub impulse: c_float,
}

pub struct SpacePhysicsWorld {
    world_ptr: *mut c_void,
    next_object_id: i32,
    objects: HashMap<i32, PhysicsObjectInfo>,
}

#[derive(Debug, Clone)]
pub struct PhysicsObjectInfo {
    pub object_type: ObjectType,
    pub mass: f32,
}

#[derive(Debug, Clone)]
pub enum ObjectType {
    Ship { dimensions: [f32; 3] },
    Asteroid { radius: f32 },
    Station { dimensions: [f32; 3] },
}

#[derive(Debug, Clone)]
pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 4], // Quaternion
}

#[derive(Debug)]
pub struct CollisionEvent {
    pub object1_id: i32,
    pub object2_id: i32,
    pub contact_point: [f32; 3],
    pub normal: [f32; 3],
    pub impulse: f32,
}

impl SpacePhysicsWorld {
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
        
        self.objects.insert(object_id, PhysicsObjectInfo {
            object_type: ObjectType::Ship { dimensions },
            mass,
        });
        
        Ok(object_id)
    }
    
    pub fn create_asteroid(
        &mut self,
        mass: f32,
        position: [f32; 3],
        radius: f32,
    ) -> Result<i32, PhysicsError> {
        let object_id = unsafe {
            physics_create_asteroid(
                self.world_ptr,
                mass,
                position.as_ptr(),
                radius,
            )
        };
        
        if object_id < 0 {
            return Err(PhysicsError::ObjectCreationFailed);
        }
        
        self.objects.insert(object_id, PhysicsObjectInfo {
            object_type: ObjectType::Asteroid { radius },
            mass,
        });
        
        Ok(object_id)
    }
    
    pub fn get_transform(&self, object_id: i32) -> Result<Transform, PhysicsError> {
        if !self.objects.contains_key(&object_id) {
            return Err(PhysicsError::InvalidObjectId);
        }
        
        let mut position = [0.0f32; 3];
        let mut rotation = [0.0f32; 4];
        
        let result = unsafe {
            physics_get_object_transform(
                self.world_ptr,
                object_id,
                position.as_mut_ptr(),
                rotation.as_mut_ptr(),
            )
        };
        
        if result != 0 {
            return Err(PhysicsError::TransformQueryFailed);
        }
        
        Ok(Transform { position, rotation })
    }
    
    pub fn apply_force(
        &mut self,
        object_id: i32,
        force: [f32; 3],
        local_position: [f32; 3],
    ) -> Result<(), PhysicsError> {
        if !self.objects.contains_key(&object_id) {
            return Err(PhysicsError::InvalidObjectId);
        }
        
        let result = unsafe {
            physics_apply_force(
                self.world_ptr,
                object_id,
                force.as_ptr(),
                local_position.as_ptr(),
            )
        };
        
        if result != 0 {
            Err(PhysicsError::ForceApplicationFailed)
        } else {
            Ok(())
        }
    }
    
    pub fn step_simulation(&mut self, dt: f32) -> Result<Vec<CollisionEvent>, PhysicsError> {
        let step_result = unsafe {
            physics_step(self.world_ptr, dt)
        };
        
        if step_result != 0 {
            return Err(PhysicsError::SimulationStepFailed);
        }
        
        // Get collision events
        const MAX_COLLISIONS: usize = 100;
        let mut collision_buffer = vec![
            CollisionInfo {
                object1_id: 0,
                object2_id: 0,
                contact_point: [0.0; 3],
                normal: [0.0; 3],
                impulse: 0.0,
            };
            MAX_COLLISIONS
        ];
        
        let collision_count = unsafe {
            physics_get_collisions(
                self.world_ptr,
                collision_buffer.as_mut_ptr(),
                MAX_COLLISIONS as c_int,
            )
        };
        
        if collision_count < 0 {
            return Err(PhysicsError::CollisionQueryFailed);
        }
        
        let mut collision_events = Vec::new();
        for i in 0..(collision_count as usize) {
            let collision = &collision_buffer[i];
            collision_events.push(CollisionEvent {
                object1_id: collision.object1_id,
                object2_id: collision.object2_id,
                contact_point: collision.contact_point,
                normal: collision.normal,
                impulse: collision.impulse,
            });
        }
        
        Ok(collision_events)
    }
    
    pub fn get_object_info(&self, object_id: i32) -> Option<&PhysicsObjectInfo> {
        self.objects.get(&object_id)
    }
}

impl Drop for SpacePhysicsWorld {
    fn drop(&mut self) {
        unsafe {
            physics_destroy_world(self.world_ptr);
        }
    }
}

#[derive(Debug)]
pub enum PhysicsError {
    WorldCreationFailed,
    ObjectCreationFailed,
    InvalidObjectId,
    TransformQueryFailed,
    ForceApplicationFailed,
    SimulationStepFailed,
    CollisionQueryFailed,
}

// High-level simulation integration
pub struct SpaceSimulation {
    physics_world: SpacePhysicsWorld,
    ships: Vec<Ship>,
    asteroids: Vec<Asteroid>,
}

#[derive(Debug)]
pub struct Ship {
    pub physics_id: i32,
    pub name: String,
    pub fuel: f32,
    pub cargo: f32,
    pub max_thrust: f32,
}

#[derive(Debug)]
pub struct Asteroid {
    pub physics_id: i32,
    pub mineral_content: f32,
    pub hardness: f32,
}

impl SpaceSimulation {
    pub fn new() -> Result<Self, PhysicsError> {
        let physics_world = SpacePhysicsWorld::new([0.0, 0.0, 0.0])?; // No gravity in space
        
        Ok(Self {
            physics_world,
            ships: Vec::new(),
            asteroids: Vec::new(),
        })
    }
    
    pub fn add_ship(
        &mut self,
        name: String,
        position: [f32; 3],
        mass: f32,
        dimensions: [f32; 3],
    ) -> Result<usize, PhysicsError> {
        let physics_id = self.physics_world.create_ship(mass, position, dimensions)?;
        
        let ship = Ship {
            physics_id,
            name,
            fuel: 100.0,
            cargo: 0.0,
            max_thrust: mass * 10.0, // Simple thrust calculation
        };
        
        self.ships.push(ship);
        Ok(self.ships.len() - 1)
    }
    
    pub fn add_asteroid(
        &mut self,
        position: [f32; 3],
        radius: f32,
        density: f32,
    ) -> Result<usize, PhysicsError> {
        let mass = density * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
        let physics_id = self.physics_world.create_asteroid(mass, position, radius)?;
        
        let asteroid = Asteroid {
            physics_id,
            mineral_content: radius * density * 0.1, // Simplified mineral calculation
            hardness: density,
        };
        
        self.asteroids.push(asteroid);
        Ok(self.asteroids.len() - 1)
    }
    
    pub fn apply_ship_thrust(
        &mut self,
        ship_index: usize,
        thrust_direction: [f32; 3],
        thrust_magnitude: f32,
    ) -> Result<(), PhysicsError> {
        let ship = self.ships.get_mut(ship_index)
            .ok_or(PhysicsError::InvalidObjectId)?;
        
        if ship.fuel <= 0.0 {
            return Ok((); // No fuel, no thrust
        }
        
        let actual_thrust = thrust_magnitude.min(ship.max_thrust);
        let force = [
            thrust_direction[0] * actual_thrust,
            thrust_direction[1] * actual_thrust,
            thrust_direction[2] * actual_thrust,
        ];
        
        // Apply thrust at center of mass
        self.physics_world.apply_force(ship.physics_id, force, [0.0, 0.0, 0.0])?;
        
        // Consume fuel
        ship.fuel -= actual_thrust * 0.01; // Simplified fuel consumption
        ship.fuel = ship.fuel.max(0.0);
        
        Ok(())
    }
    
    pub fn step_simulation(&mut self, dt: f32) -> Result<Vec<SimulationEvent>, PhysicsError> {
        let collisions = self.physics_world.step_simulation(dt)?;
        let mut events = Vec::new();
        
        for collision in collisions {
            // Process collision between objects
            let obj1_info = self.physics_world.get_object_info(collision.object1_id);
            let obj2_info = self.physics_world.get_object_info(collision.object2_id);
            
            match (obj1_info, obj2_info) {
                (Some(PhysicsObjectInfo { object_type: ObjectType::Ship { .. }, .. }),
                 Some(PhysicsObjectInfo { object_type: ObjectType::Asteroid { .. }, .. })) => {
                    events.push(SimulationEvent::ShipAsteroidCollision {
                        ship_physics_id: collision.object1_id,
                        asteroid_physics_id: collision.object2_id,
                        impact_force: collision.impulse,
                    });
                }
                _ => {
                    events.push(SimulationEvent::GenericCollision {
                        object1_id: collision.object1_id,
                        object2_id: collision.object2_id,
                        impulse: collision.impulse,
                    });
                }
            }
        }
        
        Ok(events)
    }
    
    pub fn get_ship_transform(&self, ship_index: usize) -> Result<Transform, PhysicsError> {
        let ship = self.ships.get(ship_index)
            .ok_or(PhysicsError::InvalidObjectId)?;
        
        self.physics_world.get_transform(ship.physics_id)
    }
}

#[derive(Debug)]
pub enum SimulationEvent {
    ShipAsteroidCollision {
        ship_physics_id: i32,
        asteroid_physics_id: i32,
        impact_force: f32,
    },
    GenericCollision {
        object1_id: i32,
        object2_id: i32,
        impulse: f32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_integration() {
        let mut simulation = SpaceSimulation::new().expect("Failed to create simulation");
        
        // Add a ship
        let ship_index = simulation.add_ship(
            "Test Ship".to_string(),
            [0.0, 0.0, 0.0],
            1000.0,
            [10.0, 5.0, 20.0],
        ).expect("Failed to add ship");
        
        // Add an asteroid
        let _asteroid_index = simulation.add_asteroid(
            [100.0, 0.0, 0.0],
            5.0,
            2000.0,
        ).expect("Failed to add asteroid");
        
        // Apply thrust to ship
        simulation.apply_ship_thrust(
            ship_index,
            [1.0, 0.0, 0.0], // Forward thrust
            5000.0,
        ).expect("Failed to apply thrust");
        
        // Step simulation
        for _ in 0..100 {
            let events = simulation.step_simulation(0.016).expect("Simulation step failed");
            
            for event in events {
                println!("Simulation event: {:?}", event);
            }
            
            // Check ship position
            if let Ok(transform) = simulation.get_ship_transform(ship_index) {
                println!("Ship position: {:?}", transform.position);
            }
        }
    }
}
```

## Key Takeaways

1. **Unsafe Encapsulation**: Always wrap unsafe code in safe APIs that maintain Rust's safety invariants
2. **FFI Safety**: Use proper C string conversion, null pointer checks, and RAII for resource management
3. **Memory Management**: Implement custom allocators when standard allocation patterns don't fit your needs
4. **Error Handling**: Foreign functions require careful error handling and validation
5. **Performance**: Unsafe code enables zero-cost abstractions and direct hardware access when needed
6. **Testing**: Unsafe code requires more rigorous testing, including property-based tests and sanitizers

## Best Practices

- Document every unsafe block with safety comments explaining why it's safe
- Use `#[repr(C)]` for structs that cross FFI boundaries
- Implement proper Drop traits for RAII resource management
- Validate all inputs from foreign functions
- Use tools like Miri, AddressSanitizer, and fuzzing for testing
- Keep unsafe blocks as small as possible
- Prefer safe abstractions over exposing unsafe interfaces

## Performance Considerations

- Unsafe code can eliminate bounds checking when you can prove safety
- FFI calls have overhead - batch operations when possible
- Custom allocators can provide significant performance improvements
- GPU integration requires careful memory management and synchronization
- Profile unsafe code to ensure it provides the expected benefits

## Next Steps

In the next tutorial, we'll explore Procedural Macros, learning how to extend Rust's syntax and generate code at compile time to create powerful domain-specific languages and code generation tools for our simulation engine.