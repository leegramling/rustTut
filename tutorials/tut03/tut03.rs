// Tutorial 03: Functional Programming
// Complete the following exercises to practice functional programming in Rust

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

// Helper types for exercises
#[derive(Debug, Clone, PartialEq)]
pub struct Ship {
    pub id: u32,
    pub cargo_capacity: f32,
    pub current_cargo: f32,
    pub fuel_level: f32,
    pub position: (f32, f32, f32),
}

impl Ship {
    pub fn cargo_utilization(&self) -> f32 {
        self.current_cargo / self.cargo_capacity
    }
    
    pub fn needs_refuel(&self) -> bool {
        self.fuel_level < 0.3
    }
    
    pub fn distance_from_origin(&self) -> f32 {
        let (x, y, z) = self.position;
        (x * x + y * y + z * z).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct RawOre {
    pub mass: f32,
    pub purity: f32,
}

#[derive(Debug, Clone)]
pub struct RefinedMetal {
    pub mass: f32,
    pub quality: f32,
}

// Exercise 1: Closures and Higher-Order Functions
// TODO: Implement process_resources function that takes a Vec<T> and a closure F
// Should apply the closure to each item and return Vec<R>
pub fn process_resources<T, F, R>(
    resources: Vec<T>,
    processor: F,
) -> Vec<R>
where
    F: Fn(T) -> R,
{
    todo!("Apply processor function to each resource")
}

// TODO: Implement a closure-based ore refining function
// Should take Vec<RawOre> and return Vec<RefinedMetal>
// Use a closure that:
// - If purity > 0.8, use efficiency 1.1, otherwise 0.9
// - mass = original_mass * purity * efficiency  
// - quality = purity * efficiency
pub fn process_mining_output(raw_ores: Vec<RawOre>) -> Vec<RefinedMetal> {
    todo!("Process ores using closure with efficiency calculation")
}

// Exercise 2: Iterator Patterns and Lazy Evaluation
#[derive(Debug, Default)]
pub struct CargoStats {
    pub total_capacity: f32,
    pub total_cargo: f32,
}

impl CargoStats {
    pub fn utilization_rate(&self) -> f32 {
        if self.total_capacity > 0.0 {
            self.total_cargo / self.total_capacity
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct FleetAnalysis {
    pub total_ships: usize,
    pub efficient_ships: Vec<u32>,      // >80% cargo utilization
    pub fuel_alerts: Vec<u32>,          // need refuel
    pub distant_ships: Vec<u32>,        // >100 units from origin
    pub cargo_stats: CargoStats,
}

// TODO: Implement analyze_fleet using functional iterator patterns
// Should use a single fold operation to collect:
// - efficient_ships: ships with cargo_utilization() > 0.8
// - fuel_alerts: ships that need_refuel()
// - cargo_stats: sum of cargo_capacity and current_cargo
// Then use additional iterator chains for distant_ships
pub fn analyze_fleet(ships: &[Ship]) -> FleetAnalysis {
    todo!("Analyze fleet using iterator patterns and fold")
}

// Exercise 3: Functional Error Handling
#[derive(Debug, Clone)]
pub enum ResourceError {
    InsufficientQuantity,
    InvalidResource,
    TransferFailed,
    StorageFull,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceType {
    Iron,
    Gold,
    Water,
    Fuel,
    Food,
}

#[derive(Debug, Clone)]
pub struct ResourceTransaction {
    pub resource_type: ResourceType,
    pub quantity: f32,
    pub source_id: u32,
    pub destination_id: u32,
}

pub struct ResourceManager {
    storage: HashMap<(u32, ResourceType), f32>, // (entity_id, resource_type) -> quantity
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }
    
    // TODO: Implement transfer_resource using functional error handling
    // Chain these operations using and_then:
    // 1. check_availability - verify source has enough resources
    // 2. validate_destination - ensure destination_id != 0  
    // 3. execute_transfer - perform the actual transfer
    pub fn transfer_resource(
        &mut self,
        transaction: ResourceTransaction,
    ) -> Result<(), ResourceError> {
        todo!("Chain validation and transfer operations using and_then")
    }
    
    // TODO: Implement helper methods for transfer_resource
    fn check_availability(&self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        todo!("Check if source has sufficient resources")
    }
    
    fn validate_destination(&self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        todo!("Validate destination (return error if destination_id == 0)")
    }
    
    fn execute_transfer(&mut self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        todo!("Execute the transfer by updating storage HashMap")
    }
    
    pub fn get_resource_quantity(&self, entity_id: u32, resource_type: &ResourceType) -> f32 {
        *self.storage.get(&(entity_id, resource_type.clone())).unwrap_or(&0.0)
    }
    
    pub fn set_resource_quantity(&mut self, entity_id: u32, resource_type: ResourceType, quantity: f32) {
        self.storage.insert((entity_id, resource_type), quantity);
    }
    
    // TODO: Implement process_transactions using partition_map pattern
    // Should separate successful and failed transactions
    // Return (successful_transactions, failed_transactions_with_errors)
    pub fn process_transactions(
        &mut self,
        transactions: Vec<ResourceTransaction>,
    ) -> (Vec<ResourceTransaction>, Vec<(ResourceTransaction, ResourceError)>) {
        todo!("Process transactions and partition results using functional approach")
    }
    
    // TODO: Implement generate_report using functional patterns
    // Use fold to calculate total resources by type
    // Count unique entities
    pub fn generate_report(&self) -> ResourceReport {
        todo!("Generate report using fold and functional patterns")
    }
}

#[derive(Debug)]
pub struct ResourceReport {
    pub total_entities: usize,
    pub resource_totals: HashMap<ResourceType, f32>,
}

// Helper enum for partition_map pattern
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

// TODO: Implement IteratorExt trait with partition_map method
// Should separate iterator items into two collections based on Either result
pub trait IteratorExt: Iterator {
    fn partition_map<A, B, F>(self, f: F) -> (Vec<A>, Vec<B>)
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Either<A, B>;
}

impl<T: Iterator> IteratorExt for T {
    fn partition_map<A, B, F>(self, f: F) -> (Vec<A>, Vec<B>)
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Either<A, B>,
    {
        todo!("Implement partition_map using fold or manual iteration")
    }
}

// Exercise 4: Custom Iterators
// TODO: Implement RouteIterator that generates waypoints from start to destination
pub struct RouteIterator {
    // TODO: Add fields for current_position, destination, step_size, steps_remaining
}

impl RouteIterator {
    // TODO: Implement new() that calculates steps needed and sets up iterator
    pub fn new(start: (f32, f32, f32), end: (f32, f32, f32), step_size: f32) -> Self {
        todo!("Calculate distance and steps, initialize iterator")
    }
}

impl Iterator for RouteIterator {
    type Item = (f32, f32, f32);
    
    // TODO: Implement next() that returns waypoints along the route
    // Should return current position and advance to next waypoint
    // Return None when destination is reached
    fn next(&mut self) -> Option<Self::Item> {
        todo!("Generate next waypoint along route")
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        // TODO: Return accurate size hint based on steps_remaining
        todo!("Return size hint")
    }
}

// TODO: Implement ExactSizeIterator for RouteIterator

// Exercise 5: Resource Simulation Iterator
// TODO: Implement ResourceSimulation iterator that simulates resource changes over time
pub struct ResourceSimulation {
    // TODO: Add fields for initial_resources, consumption_rates, production_rates
    // current_time, time_step, max_time
}

impl ResourceSimulation {
    // TODO: Implement new() constructor
    pub fn new(
        initial: HashMap<ResourceType, f32>,
        consumption: HashMap<ResourceType, f32>,
        production: HashMap<ResourceType, f32>,
        time_step: f32,
        max_time: f32,
    ) -> Self {
        todo!("Initialize resource simulation")
    }
}

impl Iterator for ResourceSimulation {
    type Item = (f32, HashMap<ResourceType, f32>); // (time, resources)
    
    // TODO: Implement next() that advances simulation and returns (time, resource_state)
    // Calculate resource levels: initial + (production - consumption) * current_time
    // Ensure resources don't go below 0
    fn next(&mut self) -> Option<Self::Item> {
        todo!("Advance simulation time and calculate resource levels")
    }
}

// Exercise 6: Functional Data Processing Pipeline
#[derive(Debug, Clone)]
pub struct SensorReading {
    pub sensor_id: u32,
    pub sensor_type: SensorType,
    pub value: f32,
    pub timestamp: f32,
}

#[derive(Debug, Clone)]
pub enum SensorType {
    Temperature,
    Pressure,
    Radiation,
}

#[derive(Debug, Clone)]
pub struct ProcessedReading {
    pub sensor_id: u32,
    pub processed_value: f32,
    pub confidence: f32,
    pub timestamp: f32,
}

pub struct DataPipeline;

impl DataPipeline {
    // TODO: Implement process_sensor_data using a functional pipeline
    // Chain these operations:
    // 1. filter(is_valid_reading) - value is finite and timestamp > 0
    // 2. map(calibrate_reading) - apply calibration multipliers
    // 3. map(apply_noise_reduction) - reduce extreme values
    // 4. filter(is_within_normal_range) - check against sensor limits
    // 5. map(convert_to_processed) - convert to ProcessedReading
    pub fn process_sensor_data(
        raw_data: Vec<SensorReading>,
    ) -> Vec<ProcessedReading> {
        todo!("Implement functional processing pipeline")
    }
    
    // TODO: Implement helper functions for the pipeline
    fn is_valid_reading(reading: &SensorReading) -> bool {
        todo!("Check if reading is valid")
    }
    
    fn calibrate_reading(mut reading: SensorReading) -> SensorReading {
        todo!("Apply calibration: Temperature *1.02, Pressure *0.98, Radiation *1.05")
    }
    
    fn apply_noise_reduction(mut reading: SensorReading) -> SensorReading {
        todo!("Reduce noise: if |value| > 1000, multiply by 0.9")
    }
    
    fn is_within_normal_range(reading: &SensorReading) -> bool {
        todo!("Check ranges: Temp -100..200, Pressure 0..10000, Radiation 0..1000")
    }
    
    fn convert_to_processed(reading: SensorReading) -> ProcessedReading {
        todo!("Convert to ProcessedReading with confidence calculation")
    }
    
    fn calculate_confidence(reading: &SensorReading) -> f32 {
        todo!("Calculate confidence: 1.0 - (|value|/100).min(1.0), minimum 0.1")
    }
}

// Exercise 7: Fleet Route Optimization
#[derive(Debug, Clone)]
pub struct OptimizedRoute {
    pub route_id: usize,
    pub ship_id: u32,
    pub destination: (f32, f32, f32),
    pub estimated_time: f32,
    pub fuel_required: f32,
}

pub struct FleetOptimizer;

impl FleetOptimizer {
    // TODO: Implement optimize_routes using functional patterns
    // 1. enumerate() ships with indices
    // 2. filter_map() to find best destination for each ship
    // 3. map() to create OptimizedRoute
    // 4. sorted_by() estimated_time
    pub fn optimize_routes(
        ships: Vec<Ship>,
        destinations: Vec<(f32, f32, f32)>,
    ) -> Vec<OptimizedRoute> {
        todo!("Optimize routes using functional pipeline")
    }
    
    // TODO: Implement helper functions
    fn find_best_destination(
        ship: &Ship,
        destinations: &[(f32, f32, f32)],
    ) -> Option<(f32, f32, f32)> {
        todo!("Find closest destination based on travel time")
    }
    
    fn create_optimized_route(
        route_id: usize,
        ship: Ship,
        destination: (f32, f32, f32),
    ) -> OptimizedRoute {
        todo!("Create OptimizedRoute with time and fuel estimates")
    }
    
    fn calculate_distance(pos1: (f32, f32, f32), pos2: (f32, f32, f32)) -> f32 {
        let (dx, dy, dz) = (pos2.0 - pos1.0, pos2.1 - pos1.1, pos2.2 - pos1.2);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

// TODO: Implement SortedIterator extension trait
pub trait SortedIterator: Iterator {
    fn sorted_by<F>(self, compare: F) -> Vec<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering;
}

impl<T: Iterator> SortedIterator for T {
    fn sorted_by<F>(self, compare: F) -> Vec<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        todo!("Collect iterator into Vec and sort")
    }
}

// Test your implementations
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_resources() {
        let numbers = vec![1, 2, 3, 4, 5];
        let doubled = process_resources(numbers, |x| x * 2);
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_ore_processing() {
        let ores = vec![
            RawOre { mass: 100.0, purity: 0.9 },
            RawOre { mass: 50.0, purity: 0.7 },
        ];
        
        let refined = process_mining_output(ores);
        assert_eq!(refined.len(), 2);
        
        // High purity ore (0.9 > 0.8) should use efficiency 1.1
        assert!((refined[0].mass - 99.0).abs() < 0.1); // 100 * 0.9 * 1.1
        assert!((refined[0].quality - 0.99).abs() < 0.01); // 0.9 * 1.1
        
        // Low purity ore (0.7 < 0.8) should use efficiency 0.9  
        assert!((refined[1].mass - 31.5).abs() < 0.1); // 50 * 0.7 * 0.9
        assert!((refined[1].quality - 0.63).abs() < 0.01); // 0.7 * 0.9
    }

    #[test]
    fn test_fleet_analysis() {
        let ships = vec![
            Ship {
                id: 1,
                cargo_capacity: 100.0,
                current_cargo: 90.0, // 90% utilization - efficient
                fuel_level: 0.2,     // Needs refuel
                position: (50.0, 0.0, 0.0), // Distance 50 < 100
            },
            Ship {
                id: 2, 
                cargo_capacity: 200.0,
                current_cargo: 100.0, // 50% utilization - not efficient
                fuel_level: 0.8,      // No refuel needed
                position: (150.0, 0.0, 0.0), // Distance 150 > 100 - distant
            },
        ];
        
        let analysis = analyze_fleet(&ships);
        assert_eq!(analysis.total_ships, 2);
        assert_eq!(analysis.efficient_ships, vec![1]);
        assert_eq!(analysis.fuel_alerts, vec![1]);
        assert_eq!(analysis.distant_ships, vec![2]);
        assert_eq!(analysis.cargo_stats.total_capacity, 300.0);
        assert_eq!(analysis.cargo_stats.total_cargo, 190.0);
    }

    #[test]
    fn test_route_iterator() {
        let mut route = RouteIterator::new((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 5.0);
        
        assert_eq!(route.next(), Some((0.0, 0.0, 0.0)));
        assert_eq!(route.next(), Some((5.0, 0.0, 0.0)));
        assert_eq!(route.next(), Some((10.0, 0.0, 0.0)));
        assert_eq!(route.next(), None);
    }

    #[test]
    fn test_resource_simulation() {
        let mut initial = HashMap::new();
        initial.insert(ResourceType::Fuel, 100.0);
        
        let mut consumption = HashMap::new();
        consumption.insert(ResourceType::Fuel, 10.0);
        
        let mut production = HashMap::new();
        production.insert(ResourceType::Fuel, 5.0);
        
        let mut sim = ResourceSimulation::new(initial, consumption, production, 1.0, 3.0);
        
        let (time, resources) = sim.next().unwrap();
        assert_eq!(time, 0.0);
        assert_eq!(resources[&ResourceType::Fuel], 100.0);
        
        let (time, resources) = sim.next().unwrap();
        assert_eq!(time, 1.0);
        assert_eq!(resources[&ResourceType::Fuel], 95.0); // 100 + (5-10)*1
        
        let (time, resources) = sim.next().unwrap();
        assert_eq!(time, 2.0);
        assert_eq!(resources[&ResourceType::Fuel], 90.0); // 100 + (5-10)*2
    }
}