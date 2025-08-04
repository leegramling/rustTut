# Tutorial 00: From C++20 to Rust - A Developer's Guide

## Learning Objectives
- Understand fundamental differences between C++20 and Rust paradigms
- Learn Rust's ownership system as an alternative to manual memory management
- Master enums, structs, traits, and methods in Rust context
- Build class-like objects using Rust's composition over inheritance model
- Apply Option/Result types instead of nullable pointers and exceptions
- Use generics and trait objects to replace C++ templates and virtual inheritance
- Implement space simulation examples that demonstrate Rust's unique strengths

## Core Paradigm Shifts

### Memory Management: RAII vs Ownership

**C++20 Approach:**
```cpp
// C++20: Manual lifetime management with smart pointers
class SpaceShip {
private:
    std::unique_ptr<Engine> engine_;
    std::shared_ptr<CrewManifest> crew_;
    std::vector<std::unique_ptr<Weapon>> weapons_;
    
public:
    SpaceShip(std::unique_ptr<Engine> engine) 
        : engine_(std::move(engine)) {}
    
    void addWeapon(std::unique_ptr<Weapon> weapon) {
        weapons_.push_back(std::move(weapon));
    }
    
    // Need explicit copy/move constructors for complex ownership
    SpaceShip(const SpaceShip&) = delete; // Often deleted
    SpaceShip& operator=(const SpaceShip&) = delete;
    SpaceShip(SpaceShip&&) = default;
    SpaceShip& operator=(SpaceShip&&) = default;
};
```

**Rust Approach:**
```rust
// Rust: Ownership system handles memory automatically
pub struct SpaceShip {
    engine: Engine,                    // Owned directly
    crew: Arc<CrewManifest>,          // Shared ownership when needed
    weapons: Vec<Weapon>,             // Owned collection
}

impl SpaceShip {
    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
            crew: Arc::new(CrewManifest::new()),
            weapons: Vec::new(),
        }
    }
    
    pub fn add_weapon(&mut self, weapon: Weapon) {
        self.weapons.push(weapon);
    }
    
    // Move semantics are default, Clone is explicit when needed
}

// Key differences:
// - No need for explicit smart pointer management
// - Move semantics by default prevent accidental copies
// - Borrowing system prevents use-after-free at compile time
// - No overhead of reference counting unless explicitly used (Arc/Rc)
```

### Error Handling: Exceptions vs Result Types

**C++20 Approach:**
```cpp
// C++20: Exception-based error handling
class NavigationSystem {
public:
    Position calculateJump(const Position& from, const Position& to) {
        if (from.distance(to) > MAX_JUMP_DISTANCE) {
            throw std::runtime_error("Jump distance exceeds maximum range");
        }
        
        if (!hasEnoughFuel()) {
            throw InsufficientFuelException("Not enough fuel for jump");
        }
        
        return performJumpCalculation(from, to);
    }
    
    // Caller must handle exceptions
    void executeJump() {
        try {
            auto destination = calculateJump(currentPos_, targetPos_);
            jumpTo(destination);
        } catch (const std::exception& e) {
            std::cerr << "Jump failed: " << e.what() << std::endl;
            handleJumpFailure();
        }
    }
};
```

**Rust Approach:**
```rust
// Rust: Result types make errors explicit and recoverable
pub struct NavigationSystem {
    max_jump_distance: f32,
    fuel_level: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum NavigationError {
    #[error("Jump distance {distance:.1} exceeds maximum range {max_range:.1}")]
    DistanceTooFar { distance: f32, max_range: f32 },
    #[error("Insufficient fuel: need {required:.1}, have {available:.1}")]
    InsufficientFuel { required: f32, available: f32 },
    #[error("Navigation computer malfunction: {reason}")]
    ComputerError { reason: String },
}

impl NavigationSystem {
    pub fn calculate_jump(&self, from: &Position, to: &Position) -> Result<Position, NavigationError> {
        let distance = from.distance_to(to);
        
        if distance > self.max_jump_distance {
            return Err(NavigationError::DistanceTooFar {
                distance,
                max_range: self.max_jump_distance,
            });
        }
        
        let fuel_required = distance * 0.1;
        if fuel_required > self.fuel_level {
            return Err(NavigationError::InsufficientFuel {
                required: fuel_required,
                available: self.fuel_level,
            });
        }
        
        Ok(self.perform_jump_calculation(from, to))
    }
    
    // Error handling is explicit and composable
    pub async fn execute_jump(&mut self, target: Position) -> Result<(), NavigationError> {
        let current = self.get_current_position();
        let destination = self.calculate_jump(&current, &target)?; // ? operator propagates errors
        self.jump_to(destination).await?;
        Ok(())
    }
}

// Usage shows explicit error handling
async fn navigate_ship() {
    let mut nav = NavigationSystem::new();
    let target = Position::new(100.0, 200.0, 300.0);
    
    match nav.execute_jump(target).await {
        Ok(()) => println!("Jump successful!"),
        Err(NavigationError::DistanceTooFar { distance, max_range }) => {
            println!("Cannot jump {:.1} units, maximum is {:.1}", distance, max_range);
        }
        Err(NavigationError::InsufficientFuel { required, available }) => {
            println!("Need {:.1} fuel units, but only have {:.1}", required, available);
        }
        Err(e) => println!("Navigation failed: {}", e),
    }
}
```

## Structs and Methods: Classes Reimagined

### Basic Structure Definition

**C++20 Class:**
```cpp
class SpaceStation {
private:
    std::string name_;
    Position coordinates_;
    double docking_fee_;
    std::vector<DockingBay> bays_;
    bool operational_;

public:
    // Constructor
    SpaceStation(std::string name, Position pos, double fee) 
        : name_(std::move(name)), coordinates_(pos), docking_fee_(fee), operational_(true) {}
    
    // Getters/Setters
    const std::string& name() const { return name_; }
    void setName(const std::string& name) { name_ = name; }
    
    Position coordinates() const { return coordinates_; }
    void setCoordinates(const Position& pos) { coordinates_ = pos; }
    
    // Methods
    bool canDock(const SpaceShip& ship) const;
    DockingPermit requestDocking(const SpaceShip& ship);
    
    // Static factory method
    static SpaceStation createTradingPost(const std::string& name, Position pos) {
        auto station = SpaceStation(name, pos, 100.0);
        station.addTradingBay();
        return station;
    }
};
```

**Rust Structure:**
```rust
// Rust struct with associated methods
#[derive(Debug, Clone)]
pub struct SpaceStation {
    pub name: String,                    // Public field (no need for getter/setter)
    pub coordinates: Position,
    pub docking_fee: f64,
    bays: Vec<DockingBay>,              // Private field
    operational: bool,
}

impl SpaceStation {
    // Constructor (associated function)
    pub fn new(name: String, coordinates: Position, docking_fee: f64) -> Self {
        Self {
            name,
            coordinates,
            docking_fee,
            bays: Vec::new(),
            operational: true,
        }
    }
    
    // Methods take &self (immutable reference) or &mut self (mutable reference)
    pub fn can_dock(&self, ship: &SpaceShip) -> bool {
        self.operational && 
        self.bays.iter().any(|bay| bay.is_available() && bay.can_accommodate(ship))
    }
    
    pub fn request_docking(&mut self, ship: &SpaceShip) -> Result<DockingPermit, DockingError> {
        if !self.can_dock(ship) {
            return Err(DockingError::NoAvailableBays);
        }
        
        let bay = self.bays.iter_mut()
            .find(|bay| bay.is_available() && bay.can_accommodate(ship))
            .ok_or(DockingError::NoAvailableBays)?;
            
        bay.reserve_for(ship);
        Ok(DockingPermit::new(self.name.clone(), bay.id(), self.docking_fee))
    }
    
    // Associated function (like static method in C++)
    pub fn create_trading_post(name: String, coordinates: Position) -> Self {
        let mut station = Self::new(name, coordinates, 100.0);
        station.add_trading_bay();
        station
    }
    
    // Getters for private fields (when needed)
    pub fn is_operational(&self) -> bool {
        self.operational
    }
    
    pub fn bay_count(&self) -> usize {
        self.bays.len()
    }
    
    // Mutable access when needed
    pub fn set_operational(&mut self, operational: bool) {
        self.operational = operational;
    }
    
    fn add_trading_bay(&mut self) {
        self.bays.push(DockingBay::new_trading_bay());
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct DockingBay {
    id: u32,
    bay_type: BayType,
    occupied: bool,
    size_limit: ShipSize,
}

#[derive(Debug, Clone)]
pub enum BayType {
    Standard,
    Trading,
    Repair,
    Refueling,
}

#[derive(Debug, Clone)]
pub struct DockingPermit {
    station_name: String,
    bay_id: u32,
    fee: f64,
    expires_at: std::time::Instant,
}

#[derive(Debug, thiserror::Error)]
pub enum DockingError {
    #[error("No available docking bays")]
    NoAvailableBays,
    #[error("Ship too large for available bays")]
    ShipTooLarge,
    #[error("Station is not operational")]
    StationOffline,
}
```

## Enums: Powerful Pattern Matching

### C++20 Enums and Variants

**C++20 Approach:**
```cpp
// C++20: enum class + std::variant for complex enums
enum class MessageType {
    Navigation,
    Trading,
    Emergency,
    System
};

// Complex data requires separate structs + variant
struct NavigationMessage {
    Position destination;
    double speed;
    bool emergency_stop;
};

struct TradingMessage {
    std::string commodity;
    double quantity;
    double price_per_unit;
};

struct EmergencyMessage {
    std::string emergency_type;
    Position location;
    int severity_level;
};

using ShipMessage = std::variant<NavigationMessage, TradingMessage, EmergencyMessage>;

// Pattern matching requires visitor pattern
class MessageHandler {
public:
    void processMessage(const ShipMessage& message) {
        std::visit([this](const auto& msg) {
            using T = std::decay_t<decltype(msg)>;
            if constexpr (std::is_same_v<T, NavigationMessage>) {
                this->handleNavigation(msg);
            } else if constexpr (std::is_same_v<T, TradingMessage>) {
                this->handleTrading(msg);
            } else if constexpr (std::is_same_v<T, EmergencyMessage>) {
                this->handleEmergency(msg);
            }
        }, message);
    }
    
private:
    void handleNavigation(const NavigationMessage& msg) { /* ... */ }
    void handleTrading(const TradingMessage& msg) { /* ... */ }
    void handleEmergency(const EmergencyMessage& msg) { /* ... */ }
};
```

**Rust Approach:**
```rust
// Rust: Enums can contain data directly and have methods
#[derive(Debug, Clone)]
pub enum ShipMessage {
    Navigation {
        destination: Position,
        speed: f64,
        emergency_stop: bool,
    },
    Trading {
        commodity: String,
        quantity: f64,
        price_per_unit: f64,
    },
    Emergency {
        emergency_type: String,
        location: Position,
        severity_level: u8,
    },
    System {
        command: SystemCommand,
        parameters: HashMap<String, String>,
    },
}

#[derive(Debug, Clone)]
pub enum SystemCommand {
    Shutdown,
    Restart,
    DiagnosticScan,
    UpdateFirmware { version: String },
}

impl ShipMessage {
    // Enums can have methods
    pub fn is_urgent(&self) -> bool {
        match self {
            ShipMessage::Emergency { severity_level, .. } => *severity_level > 7,
            ShipMessage::Navigation { emergency_stop: true, .. } => true,
            ShipMessage::System { command: SystemCommand::Shutdown, .. } => true,
            _ => false,
        }
    }
    
    pub fn priority(&self) -> MessagePriority {
        match self {
            ShipMessage::Emergency { severity_level, .. } => {
                if *severity_level > 8 {
                    MessagePriority::Critical
                } else {
                    MessagePriority::High
                }
            }
            ShipMessage::Navigation { emergency_stop: true, .. } => MessagePriority::High,
            ShipMessage::System { .. } => MessagePriority::Medium,
            ShipMessage::Trading { .. } => MessagePriority::Low,
            _ => MessagePriority::Normal,
        }
    }
    
    // Constructor methods
    pub fn navigation_to(destination: Position, speed: f64) -> Self {
        ShipMessage::Navigation {
            destination,
            speed,
            emergency_stop: false,
        }
    }
    
    pub fn emergency_stop() -> Self {
        ShipMessage::Navigation {
            destination: Position::zero(),
            speed: 0.0,
            emergency_stop: true,
        }
    }
    
    pub fn trade_offer(commodity: String, quantity: f64, price: f64) -> Self {
        ShipMessage::Trading {
            commodity,
            quantity,
            price_per_unit: price,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    Low,
    Normal,
    Medium,
    High,
    Critical,
}

// Pattern matching is built-in and exhaustive
pub struct MessageHandler;

impl MessageHandler {
    pub fn process_message(&mut self, message: ShipMessage) -> Result<(), ProcessingError> {
        match message {
            ShipMessage::Navigation { destination, speed, emergency_stop } => {
                if emergency_stop {
                    self.handle_emergency_stop()?;
                } else {
                    self.handle_navigation(destination, speed)?;
                }
            }
            
            ShipMessage::Trading { commodity, quantity, price_per_unit } => {
                self.handle_trading(&commodity, quantity, price_per_unit)?;
            }
            
            ShipMessage::Emergency { emergency_type, location, severity_level } => {
                self.handle_emergency(&emergency_type, location, severity_level)?;
            }
            
            ShipMessage::System { command, parameters } => {
                match command {
                    SystemCommand::Shutdown => self.initiate_shutdown(),
                    SystemCommand::Restart => self.restart_systems(),
                    SystemCommand::DiagnosticScan => self.run_diagnostics(),
                    SystemCommand::UpdateFirmware { version } => {
                        self.update_firmware(&version)?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Helper methods...
    fn handle_navigation(&mut self, dest: Position, speed: f64) -> Result<(), ProcessingError> {
        println!("Navigating to {:?} at speed {:.1}", dest, speed);
        Ok(())
    }
    
    fn handle_emergency_stop(&mut self) -> Result<(), ProcessingError> {
        println!("EMERGENCY STOP INITIATED");
        Ok(())
    }
    
    fn handle_trading(&mut self, commodity: &str, qty: f64, price: f64) -> Result<(), ProcessingError> {
        println!("Trading {:.1} units of {} at {:.2} credits per unit", qty, commodity, price);
        Ok(())
    }
    
    fn handle_emergency(&mut self, emergency_type: &str, location: Position, severity: u8) -> Result<(), ProcessingError> {
        println!("EMERGENCY: {} at {:?} (severity {})", emergency_type, location, severity);
        Ok(())
    }
    
    fn initiate_shutdown(&mut self) { println!("Shutting down systems..."); }
    fn restart_systems(&mut self) { println!("Restarting systems..."); }
    fn run_diagnostics(&mut self) { println!("Running diagnostic scan..."); }
    fn update_firmware(&mut self, version: &str) -> Result<(), ProcessingError> {
        println!("Updating firmware to version {}", version);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Navigation system offline")]
    NavigationOffline,
    #[error("Trading system error: {reason}")]
    TradingError { reason: String },
    #[error("Emergency protocol failed")]
    EmergencyProtocolFailed,
}
```

## Option and Result: Null Safety

### Replacing Nullable Pointers

**C++20 Approach:**
```cpp
// C++20: Raw pointers, nullptr, and optional
class FleetManager {
private:
    std::map<int, std::unique_ptr<SpaceShip>> ships_;
    
public:
    // Returns nullptr if ship not found
    SpaceShip* findShip(int id) {
        auto it = ships_.find(id);
        return it != ships_.end() ? it->second.get() : nullptr;
    }
    
    // std::optional for modern C++
    std::optional<Position> getShipPosition(int id) {
        if (auto ship = findShip(id)) {
            return ship->getPosition();
        }
        return std::nullopt;
    }
    
    // Exception-based error handling
    void moveShip(int id, const Position& destination) {
        auto ship = findShip(id);
        if (!ship) {
            throw std::runtime_error("Ship not found: " + std::to_string(id));
        }
        
        if (!ship->isOperational()) {
            throw std::runtime_error("Ship is not operational");
        }
        
        ship->moveTo(destination);
    }
};
```

**Rust Approach:**
```rust
// Rust: Option<T> and Result<T, E> eliminate null pointer errors
use std::collections::HashMap;

pub struct FleetManager {
    ships: HashMap<u32, SpaceShip>,
    next_ship_id: u32,
}

impl FleetManager {
    pub fn new() -> Self {
        Self {
            ships: HashMap::new(),
            next_ship_id: 1,
        }
    }
    
    // Option<T> makes absence explicit
    pub fn find_ship(&self, id: u32) -> Option<&SpaceShip> {
        self.ships.get(&id)
    }
    
    pub fn find_ship_mut(&mut self, id: u32) -> Option<&mut SpaceShip> {
        self.ships.get_mut(&id)
    }
    
    // Chaining operations with Option
    pub fn get_ship_position(&self, id: u32) -> Option<Position> {
        self.find_ship(id)?.get_position()  // ? operator handles None case
    }
    
    pub fn get_ship_fuel_level(&self, id: u32) -> Option<f64> {
        self.find_ship(id)?.fuel_level()
    }
    
    // Result<T, E> for operations that can fail
    pub fn move_ship(&mut self, id: u32, destination: Position) -> Result<(), FleetError> {
        let ship = self.find_ship_mut(id)
            .ok_or(FleetError::ShipNotFound { id })?;
        
        if !ship.is_operational() {
            return Err(FleetError::ShipNotOperational { id });
        }
        
        if ship.fuel_level().unwrap_or(0.0) < destination.distance_to(&ship.get_position().unwrap()) * 0.1 {
            return Err(FleetError::InsufficientFuel { 
                id, 
                required: destination.distance_to(&ship.get_position().unwrap()) * 0.1,
                available: ship.fuel_level().unwrap_or(0.0),
            });
        }
        
        ship.move_to(destination);
        Ok(())
    }
    
    // Combining Option and Result
    pub fn transfer_cargo(&mut self, from_id: u32, to_id: u32, cargo_type: &str, amount: f64) -> Result<(), FleetError> {
        // Check both ships exist
        let from_ship_cargo = self.find_ship(from_id)
            .ok_or(FleetError::ShipNotFound { id: from_id })?
            .get_cargo_amount(cargo_type);
        
        if from_ship_cargo < amount {
            return Err(FleetError::InsufficientCargo {
                ship_id: from_id,
                cargo_type: cargo_type.to_string(),
                requested: amount,
                available: from_ship_cargo,
            });
        }
        
        // This requires two mutable references, so we need to be careful
        let to_ship_capacity = self.find_ship(to_id)
            .ok_or(FleetError::ShipNotFound { id: to_id })?
            .remaining_cargo_capacity();
            
        if to_ship_capacity < amount {
            return Err(FleetError::InsufficientCapacity {
                ship_id: to_id,
                requested: amount,
                available: to_ship_capacity,
            });
        }
        
        // Perform transfer - we need to split the borrow
        if let Some(from_ship) = self.ships.get_mut(&from_id) {
            from_ship.remove_cargo(cargo_type, amount);
        }
        
        if let Some(to_ship) = self.ships.get_mut(&to_id) {
            to_ship.add_cargo(cargo_type, amount);
        }
        
        Ok(())
    }
    
    // Collecting results with error propagation
    pub fn get_fleet_status(&self) -> Result<FleetStatus, FleetError> {
        let mut total_ships = 0;
        let mut operational_ships = 0;
        let mut total_cargo = 0.0;
        let mut positions = Vec::new();
        
        for (id, ship) in &self.ships {
            total_ships += 1;
            
            if ship.is_operational() {
                operational_ships += 1;
            }
            
            total_cargo += ship.total_cargo();
            
            // Handle potential None values gracefully
            if let Some(position) = ship.get_position() {
                positions.push((*id, position));
            }
        }
        
        Ok(FleetStatus {
            total_ships,
            operational_ships,
            total_cargo,
            ship_positions: positions,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FleetError {
    #[error("Ship {id} not found")]
    ShipNotFound { id: u32 },
    
    #[error("Ship {id} is not operational")]
    ShipNotOperational { id: u32 },
    
    #[error("Ship {id} has insufficient fuel: need {required:.1}, have {available:.1}")]
    InsufficientFuel { id: u32, required: f64, available: f64 },
    
    #[error("Ship {ship_id} has insufficient {cargo_type}: need {requested:.1}, have {available:.1}")]
    InsufficientCargo { ship_id: u32, cargo_type: String, requested: f64, available: f64 },
    
    #[error("Ship {ship_id} has insufficient capacity: need {requested:.1}, have {available:.1}")]
    InsufficientCapacity { ship_id: u32, requested: f64, available: f64 },
}

#[derive(Debug)]
pub struct FleetStatus {
    pub total_ships: usize,
    pub operational_ships: usize,
    pub total_cargo: f64,
    pub ship_positions: Vec<(u32, Position)>,
}
```

## Traits: Interface and Behavior

### From Virtual Inheritance to Traits

**C++20 Virtual Inheritance:**
```cpp
// C++20: Virtual inheritance and abstract base classes
class Trackable {
public:
    virtual ~Trackable() = default;
    virtual Position getPosition() const = 0;
    virtual void updatePosition(const Position& pos) = 0;
};

class Commandable {
public:
    virtual ~Commandable() = default;
    virtual void executeCommand(const Command& cmd) = 0;
    virtual bool canExecuteCommand(const Command& cmd) const = 0;
};

class Damageable {
public:
    virtual ~Damageable() = default;
    virtual void takeDamage(double amount) = 0;
    virtual double getHealth() const = 0;
    virtual bool isDestroyed() const = 0;
};

// Multiple inheritance
class SpaceShip : public Trackable, public Commandable, public Damageable {
private:
    Position position_;
    double health_;
    std::string ship_class_;
    
public:
    // Must implement all virtual methods
    Position getPosition() const override { return position_; }
    void updatePosition(const Position& pos) override { position_ = pos; }
    
    void executeCommand(const Command& cmd) override {
        // Implementation...
    }
    
    bool canExecuteCommand(const Command& cmd) const override {
        return health_ > 0 && /* other conditions */;
    }
    
    void takeDamage(double amount) override {
        health_ -= amount;
        if (health_ < 0) health_ = 0;
    }
    
    double getHealth() const override { return health_; }
    bool isDestroyed() const override { return health_ <= 0; }
};

// Using virtual inheritance requires runtime polymorphism
void processTrackables(std::vector<std::unique_ptr<Trackable>>& objects) {
    for (auto& obj : objects) {
        auto pos = obj->getPosition();
        // Process position...
    }
}
```

**Rust Traits:**
```rust
// Rust: Traits provide behavior without inheritance
use std::fmt::Debug;

// Traits define shared behavior
pub trait Trackable {
    fn position(&self) -> Position;
    fn set_position(&mut self, position: Position);
    
    // Default implementation
    fn distance_to<T: Trackable>(&self, other: &T) -> f64 {
        self.position().distance_to(&other.position())
    }
}

pub trait Commandable {
    type Command;  // Associated type
    type Error;
    
    fn execute_command(&mut self, command: Self::Command) -> Result<(), Self::Error>;
    fn can_execute_command(&self, command: &Self::Command) -> bool;
    
    // Default implementation with multiple trait bounds
    fn try_execute_command(&mut self, command: Self::Command) -> Result<(), Self::Error> 
    where 
        Self::Command: Clone + Debug,
        Self::Error: Debug,
    {
        if self.can_execute_command(&command) {
            println!("Executing command: {:?}", command);
            self.execute_command(command)
        } else {
            println!("Cannot execute command: {:?}", command);
            // Would need a specific error variant
            self.execute_command(command) // This will fail appropriately
        }
    }
}

pub trait Damageable {
    fn take_damage(&mut self, amount: f64);
    fn health(&self) -> f64;
    fn max_health(&self) -> f64;
    
    // Default implementations
    fn is_destroyed(&self) -> bool {
        self.health() <= 0.0
    }
    
    fn health_percentage(&self) -> f64 {
        self.health() / self.max_health()
    }
    
    fn heal(&mut self, amount: f64) {
        let current = self.health();
        let new_health = (current + amount).min(self.max_health());
        let heal_amount = new_health - current;
        self.take_damage(-heal_amount); // Negative damage = healing
    }
}

// SpaceShip implements multiple traits
#[derive(Debug, Clone)]
pub struct SpaceShip {
    position: Position,
    health: f64,
    max_health: f64,
    ship_class: ShipClass,
    fuel: f64,
    cargo: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum ShipCommand {
    MoveTo(Position),
    Dock(u32), // Station ID
    Trade { commodity: String, amount: f64 },
    Attack(u32), // Target ID
    Repair,
}

#[derive(Debug, thiserror::Error)]
pub enum ShipCommandError {
    #[error("Insufficient fuel")]
    InsufficientFuel,
    #[error("Ship is destroyed")]
    ShipDestroyed,
    #[error("Invalid target")]
    InvalidTarget,
    #[error("Cannot dock: {reason}")]
    DockingFailed { reason: String },
}

// Implementing traits for SpaceShip
impl Trackable for SpaceShip {
    fn position(&self) -> Position {
        self.position
    }
    
    fn set_position(&mut self, position: Position) {
        self.position = position;
    }
}

impl Commandable for SpaceShip {
    type Command = ShipCommand;
    type Error = ShipCommandError;
    
    fn execute_command(&mut self, command: Self::Command) -> Result<(), Self::Error> {
        if self.is_destroyed() {
            return Err(ShipCommandError::ShipDestroyed);
        }
        
        match command {
            ShipCommand::MoveTo(destination) => {
                let distance = self.position().distance_to(&destination);
                let fuel_needed = distance * 0.1;
                
                if self.fuel < fuel_needed {
                    return Err(ShipCommandError::InsufficientFuel);
                }
                
                self.fuel -= fuel_needed;
                self.set_position(destination);
                println!("Ship moved to {:?}", destination);
                Ok(())
            }
            
            ShipCommand::Dock(station_id) => {
                // Implementation would check if station exists and is in range
                println!("Attempting to dock at station {}", station_id);
                Ok(())
            }
            
            ShipCommand::Trade { commodity, amount } => {
                // Implementation would handle trading logic
                println!("Trading {:.1} units of {}", amount, commodity);
                Ok(())
            }
            
            ShipCommand::Attack(target_id) => {
                println!("Attacking target {}", target_id);
                Ok(())
            }
            
            ShipCommand::Repair => {
                self.heal(20.0);
                println!("Ship repaired, health now {:.1}", self.health());
                Ok(())
            }
        }
    }
    
    fn can_execute_command(&self, command: &Self::Command) -> bool {
        if self.is_destroyed() {
            return false;
        }
        
        match command {
            ShipCommand::MoveTo(destination) => {
                let distance = self.position().distance_to(destination);
                let fuel_needed = distance * 0.1;
                self.fuel >= fuel_needed
            }
            ShipCommand::Dock(_) => self.fuel > 0.0,
            ShipCommand::Trade { .. } => true,
            ShipCommand::Attack(_) => self.fuel > 0.0,
            ShipCommand::Repair => self.health() < self.max_health(),
        }
    }
}

impl Damageable for SpaceShip {
    fn take_damage(&mut self, amount: f64) {
        self.health = (self.health - amount).max(0.0);
        if self.is_destroyed() {
            println!("Ship {} has been destroyed!", self.ship_class.name());
        }
    }
    
    fn health(&self) -> f64 {
        self.health
    }
    
    fn max_health(&self) -> f64 {
        self.max_health
    }
}

// Trait objects for dynamic dispatch (similar to virtual inheritance)
pub fn process_trackables(objects: &[Box<dyn Trackable>]) {
    for obj in objects {
        let pos = obj.position();
        println!("Object at position: {:?}", pos);
    }
}

// Generic functions for static dispatch (zero-cost)
pub fn move_trackable_object<T: Trackable>(object: &mut T, destination: Position) {
    let old_pos = object.position();
    object.set_position(destination);
    println!("Moved object from {:?} to {:?}", old_pos, destination);
}

// Trait bounds allow flexible generic programming
pub fn simulate_combat<T, U>(attacker: &mut T, defender: &mut U) 
where 
    T: Trackable + Commandable<Command = ShipCommand, Error = ShipCommandError>,
    U: Trackable + Damageable + Debug,
{
    let distance = attacker.distance_to(defender);
    if distance < 100.0 {
        println!("Attacker at {:?} engaging target at {:?}", 
                 attacker.position(), defender.position());
        
        if let Ok(()) = attacker.execute_command(ShipCommand::Attack(0)) {
            defender.take_damage(25.0);
            println!("Target hit! Target health: {:.1}", defender.health());
        }
    } else {
        println!("Target out of range (distance: {:.1})", distance);
    }
}
```

## Generics and Trait Objects

### Templates vs Generics

**C++20 Templates:**
```cpp
// C++20: Templates with concepts (modern C++)
template<typename T>
concept Spacecraft = requires(T t) {
    { t.getPosition() } -> std::convertible_to<Position>;
    { t.getFuelLevel() } -> std::convertible_to<double>;
    { t.isOperational() } -> std::convertible_to<bool>;
};

template<Spacecraft T>
class FlightController {
private:
    std::vector<T> controlled_craft_;
    
public:
    void addCraft(T craft) {
        controlled_craft_.push_back(std::move(craft));
    }
    
    void updateAll(double dt) {
        for (auto& craft : controlled_craft_) {
            if (craft.isOperational()) {
                auto pos = craft.getPosition();
                // Update logic...
            }
        }
    }
    
    // Template method
    template<typename Predicate>
    std::vector<T> findCraft(Predicate pred) {
        std::vector<T> result;
        std::copy_if(controlled_craft_.begin(), controlled_craft_.end(),
                     std::back_inserter(result), pred);
        return result;
    }
};

// Usage requires template instantiation
auto fighter_controller = FlightController<Fighter>();
auto transport_controller = FlightController<Transport>();
```

**Rust Generics:**
```rust
// Rust: Trait bounds provide similar functionality with better error messages
pub trait Spacecraft {
    fn position(&self) -> Position;
    fn fuel_level(&self) -> f64;
    fn is_operational(&self) -> bool;
    
    // Associated types
    type FuelType: Clone + Debug;
    type NavigationSystem: Navigation;
    
    // Default implementation
    fn can_reach(&self, destination: &Position) -> bool {
        let distance = self.position().distance_to(destination);
        let fuel_needed = distance * 0.1;
        self.fuel_level() >= fuel_needed && self.is_operational()
    }
}

// Generic struct with trait bounds
pub struct FlightController<T: Spacecraft> {
    controlled_craft: Vec<T>,
    sector_bounds: AABB,
}

impl<T: Spacecraft> FlightController<T> {
    pub fn new(sector_bounds: AABB) -> Self {
        Self {
            controlled_craft: Vec::new(),
            sector_bounds,
        }
    }
    
    pub fn add_craft(&mut self, craft: T) {
        self.controlled_craft.push(craft);
    }
    
    pub fn update_all(&mut self, dt: f64) {
        for craft in &mut self.controlled_craft {
            if craft.is_operational() {
                let pos = craft.position();
                // Update logic...
                println!("Updating craft at {:?}", pos);
            }
        }
    }
    
    // Generic method with closure parameter
    pub fn find_craft<F>(&self, predicate: F) -> Vec<&T> 
    where 
        F: Fn(&T) -> bool 
    {
        self.controlled_craft
            .iter()
            .filter(|craft| predicate(craft))
            .collect()
    }
    
    // Method with multiple trait bounds
    pub fn find_and_command<F, C>(&mut self, finder: F, command: C) -> Result<usize, String>
    where
        F: Fn(&T) -> bool,
        C: Fn(&mut T) -> Result<(), String>,
    {
        let mut count = 0;
        for craft in &mut self.controlled_craft {
            if finder(craft) {
                command(craft)?;
                count += 1;
            }
        }
        Ok(count)
    }
}

// Multiple implementations for different types
impl Spacecraft for Fighter {
    fn position(&self) -> Position { self.coords }
    fn fuel_level(&self) -> f64 { self.fuel }
    fn is_operational(&self) -> bool { self.systems_online && self.fuel > 0.0 }
    
    type FuelType = MilitaryFuel;
    type NavigationSystem = TacticalNav;
}

impl Spacecraft for Transport {
    fn position(&self) -> Position { self.current_position }
    fn fuel_level(&self) -> f64 { self.fuel_tanks.total_fuel() }
    fn is_operational(&self) -> bool { 
        !self.critical_systems_failed && self.fuel_level() > 0.0 
    }
    
    type FuelType = CommercialFuel;
    type NavigationSystem = CommercialNav;
}

// Trait objects for heterogeneous collections
pub struct MixedFleet {
    spacecraft: Vec<Box<dyn Spacecraft<FuelType = UniversalFuel, NavigationSystem = StandardNav>>>,
}

impl MixedFleet {
    pub fn new() -> Self {
        Self {
            spacecraft: Vec::new(),
        }
    }
    
    pub fn add_craft<T>(&mut self, craft: T) 
    where 
        T: Spacecraft<FuelType = UniversalFuel, NavigationSystem = StandardNav> + 'static 
    {
        self.spacecraft.push(Box::new(craft));
    }
    
    pub fn total_operational(&self) -> usize {
        self.spacecraft
            .iter()
            .filter(|craft| craft.is_operational())
            .count()
    }
    
    pub fn positions(&self) -> Vec<Position> {
        self.spacecraft
            .iter()
            .map(|craft| craft.position())
            .collect()
    }
}

// Advanced generic patterns
pub struct Formation<T: Spacecraft + Clone> {
    leader: T,
    followers: Vec<T>,
    formation_type: FormationType,
}

impl<T: Spacecraft + Clone> Formation<T> {
    pub fn new(leader: T, formation_type: FormationType) -> Self {
        Self {
            leader,
            followers: Vec::new(),
            formation_type,
        }
    }
    
    pub fn add_follower(&mut self, follower: T) {
        self.followers.push(follower);
    }
    
    // Method using associated types
    pub fn coordinate_navigation(&mut self) -> Result<(), NavigationError>
    where
        T::NavigationSystem: FormationCapable,
    {
        let leader_pos = self.leader.position();
        let formation_positions = self.formation_type.calculate_positions(leader_pos, self.followers.len());
        
        for (follower, target_pos) in self.followers.iter_mut().zip(formation_positions.iter()) {
            follower.navigate_to_formation_position(*target_pos)?;
        }
        
        Ok(())
    }
}

// Trait with generic parameters
pub trait Navigation {
    type Error;
    
    fn navigate_to(&mut self, destination: Position) -> Result<(), Self::Error>;
    fn calculate_route(&self, from: Position, to: Position) -> Vec<Position>;
    
    // Generic method within trait
    fn navigate_formation<I>(&mut self, waypoints: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = Position>;
}

// Supporting types and traits
pub trait FormationCapable: Navigation {
    fn navigate_to_formation_position(&mut self, position: Position) -> Result<(), Self::Error>;
    fn maintain_formation_distance(&mut self, distance: f64) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone)]
pub enum FormationType {
    Line,
    V,
    Delta,
    Box,
}

impl FormationType {
    pub fn calculate_positions(&self, leader_pos: Position, follower_count: usize) -> Vec<Position> {
        // Implementation would calculate formation positions
        vec![leader_pos; follower_count] // Simplified
    }
}

// Fuel and navigation system types
#[derive(Debug, Clone)]
pub struct MilitaryFuel;
#[derive(Debug, Clone)]
pub struct CommercialFuel;
#[derive(Debug, Clone)]
pub struct UniversalFuel;

pub struct TacticalNav;
pub struct CommercialNav;
pub struct StandardNav;

impl Navigation for TacticalNav {
    type Error = NavigationError;
    
    fn navigate_to(&mut self, _destination: Position) -> Result<(), Self::Error> {
        Ok(())
    }
    
    fn calculate_route(&self, _from: Position, _to: Position) -> Vec<Position> {
        vec![]
    }
    
    fn navigate_formation<I>(&mut self, _waypoints: I) -> Result<(), Self::Error>
    where
        I: Iterator<Item = Position>
    {
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum NavigationError {
    #[error("Navigation system offline")]
    SystemOffline,
    #[error("Invalid destination")]
    InvalidDestination,
}
```

## Key Takeaways

1. **Memory Safety**: Rust's ownership system eliminates entire classes of bugs
2. **Error Handling**: Result types make errors explicit and composable
3. **Pattern Matching**: Enums with data enable powerful, safe code patterns
4. **Null Safety**: Option types eliminate null pointer dereferencing
5. **Zero-Cost Abstractions**: Traits provide polymorphism without runtime overhead
6. **Explicit Design**: Rust makes behavior and lifetime explicit rather than implicit
7. **Composition**: Traits favor composition over inheritance hierarchies

## Best Practices for C++ Developers

- Think in terms of ownership and borrowing rather than shared_ptr/unique_ptr
- Use Result for recoverable errors instead of exceptions
- Leverage pattern matching instead of complex if/else chains
- Use Option instead of nullable pointers
- Design with traits instead of virtual inheritance
- Prefer explicit error handling over try/catch blocks
- Use the type system to prevent bugs at compile time

## Next Steps

Now that you understand Rust's fundamental concepts compared to C++20, you can proceed with the specialized tutorials that build a complete space simulation engine while demonstrating advanced Rust patterns and performance techniques.