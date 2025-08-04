// Tutorial 03: Functional Programming - Complete Solutions
// This file contains the complete implementations for all exercises

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
        if self.cargo_capacity > 0.0 {
            self.current_cargo / self.cargo_capacity
        } else {
            0.0
        }
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

// Exercise 1: Closures and Higher-Order Functions - Complete Implementation
pub fn process_resources<T, F, R>(
    resources: Vec<T>,
    processor: F,
) -> Vec<R>
where
    F: Fn(T) -> R,
{
    resources.into_iter().map(processor).collect()
}

// Alternative implementation using fold
pub fn process_resources_fold<T, F, R>(
    resources: Vec<T>,
    processor: F,
) -> Vec<R>
where
    F: Fn(T) -> R,
{
    resources.into_iter().fold(Vec::new(), |mut acc, item| {
        acc.push(processor(item));
        acc
    })
}

pub fn process_mining_output(raw_ores: Vec<RawOre>) -> Vec<RefinedMetal> {
    process_resources(raw_ores, |ore| {
        let efficiency = if ore.purity > 0.8 { 1.1 } else { 0.9 };
        RefinedMetal {
            mass: ore.mass * ore.purity * efficiency,
            quality: ore.purity * efficiency,
        }
    })
}

// Bonus: More complex processing with error handling
pub fn process_mining_output_with_validation(
    raw_ores: Vec<RawOre>,
) -> Result<Vec<RefinedMetal>, String> {
    raw_ores
        .into_iter()
        .map(|ore| {
            if ore.mass <= 0.0 {
                Err("Invalid ore mass".to_string())
            } else if ore.purity <= 0.0 || ore.purity > 1.0 {
                Err("Invalid ore purity".to_string())
            } else {
                let efficiency = if ore.purity > 0.8 { 1.1 } else { 0.9 };
                Ok(RefinedMetal {
                    mass: ore.mass * ore.purity * efficiency,
                    quality: ore.purity * efficiency,
                })
            }
        })
        .collect()
}

// Exercise 2: Iterator Patterns and Lazy Evaluation - Complete Implementation
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
    pub efficient_ships: Vec<u32>,
    pub fuel_alerts: Vec<u32>,
    pub distant_ships: Vec<u32>,
    pub cargo_stats: CargoStats,
}

pub fn analyze_fleet(ships: &[Ship]) -> FleetAnalysis {
    let total_ships = ships.len();
    
    // Single fold operation to collect multiple statistics
    let (efficient_ships, fuel_alerts, cargo_stats) = ships
        .iter()
        .filter(|ship| ship.fuel_level > 0.0) // Only consider active ships
        .fold(
            (Vec::new(), Vec::new(), CargoStats::default()),
            |(mut efficient, mut alerts, mut stats), ship| {
                // Update cargo statistics
                stats.total_capacity += ship.cargo_capacity;
                stats.total_cargo += ship.current_cargo;
                
                // Check for efficient ships
                if ship.cargo_utilization() > 0.8 {
                    efficient.push(ship.id);
                }
                
                // Check for fuel alerts
                if ship.needs_refuel() {
                    alerts.push(ship.id);
                }
                
                (efficient, alerts, stats)
            },
        );
    
    // Separate iterator chain for distant ships
    let distant_ships: Vec<u32> = ships
        .iter()
        .filter(|ship| ship.distance_from_origin() > 100.0)
        .map(|ship| ship.id)
        .collect();
    
    FleetAnalysis {
        total_ships,
        efficient_ships,
        fuel_alerts,
        distant_ships,
        cargo_stats,
    }
}

// Bonus: More detailed fleet analysis
pub fn detailed_fleet_analysis(ships: &[Ship]) -> DetailedFleetAnalysis {
    let ship_stats = ships
        .iter()
        .map(|ship| ShipStats {
            id: ship.id,
            efficiency_score: ship.cargo_utilization() * ship.fuel_level,
            risk_level: if ship.needs_refuel() && ship.distance_from_origin() > 50.0 {
                RiskLevel::High
            } else if ship.needs_refuel() || ship.distance_from_origin() > 100.0 {
                RiskLevel::Medium
            } else {
                RiskLevel::Low
            },
        })
        .collect();
    
    let average_efficiency = ships
        .iter()
        .map(|ship| ship.cargo_utilization())
        .fold(0.0, |acc, util| acc + util) / ships.len() as f32;
    
    DetailedFleetAnalysis {
        ship_stats,
        average_efficiency,
        total_ships: ships.len(),
    }
}

#[derive(Debug)]
pub struct DetailedFleetAnalysis {
    pub ship_stats: Vec<ShipStats>,
    pub average_efficiency: f32,
    pub total_ships: usize,
}

#[derive(Debug)]
pub struct ShipStats {
    pub id: u32,
    pub efficiency_score: f32,
    pub risk_level: RiskLevel,
}

#[derive(Debug, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

// Exercise 3: Functional Error Handling - Complete Implementation
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
    storage: HashMap<(u32, ResourceType), f32>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }
    
    pub fn transfer_resource(
        &mut self,
        transaction: ResourceTransaction,
    ) -> Result<(), ResourceError> {
        self.check_availability(&transaction)
            .and_then(|_| self.validate_destination(&transaction))
            .and_then(|_| self.execute_transfer(&transaction))
    }
    
    fn check_availability(&self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        let available = self.get_resource_quantity(tx.source_id, &tx.resource_type);
        
        if available >= tx.quantity {
            Ok(())
        } else {
            Err(ResourceError::InsufficientQuantity)
        }
    }
    
    fn validate_destination(&self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        if tx.destination_id == 0 {
            Err(ResourceError::InvalidResource)
        } else {
            Ok(())
        }
    }
    
    fn execute_transfer(&mut self, tx: &ResourceTransaction) -> Result<(), ResourceError> {
        // Remove from source
        let source_key = (tx.source_id, tx.resource_type.clone());
        let current_source = self.storage.get(&source_key).unwrap_or(&0.0);
        self.storage.insert(source_key, current_source - tx.quantity);
        
        // Add to destination
        let dest_key = (tx.destination_id, tx.resource_type.clone());
        let current_dest = self.storage.get(&dest_key).unwrap_or(&0.0);
        self.storage.insert(dest_key, current_dest + tx.quantity);
        
        Ok(())
    }
    
    pub fn get_resource_quantity(&self, entity_id: u32, resource_type: &ResourceType) -> f32 {
        *self.storage.get(&(entity_id, resource_type.clone())).unwrap_or(&0.0)
    }
    
    pub fn set_resource_quantity(&mut self, entity_id: u32, resource_type: ResourceType, quantity: f32) {
        self.storage.insert((entity_id, resource_type), quantity);
    }
    
    pub fn process_transactions(
        &mut self,
        transactions: Vec<ResourceTransaction>,
    ) -> (Vec<ResourceTransaction>, Vec<(ResourceTransaction, ResourceError)>) {
        transactions
            .into_iter()
            .partition_map(|tx| {
                match self.transfer_resource(tx.clone()) {
                    Ok(()) => Either::Left(tx),
                    Err(e) => Either::Right((tx, e)),
                }
            })
    }
    
    pub fn generate_report(&self) -> ResourceReport {
        let resource_totals = self.storage
            .iter()
            .fold(HashMap::new(), |mut acc, ((_, resource_type), &quantity)| {
                *acc.entry(resource_type.clone()).or_insert(0.0) += quantity;
                acc
            });
        
        let entity_count = self.storage
            .keys()
            .map(|(entity_id, _)| *entity_id)
            .collect::<HashSet<_>>()
            .len();
        
        ResourceReport {
            total_entities: entity_count,
            resource_totals,
        }
    }
    
    // Bonus: Batch operations with validation
    pub fn validate_and_execute_batch(
        &mut self,
        transactions: Vec<ResourceTransaction>,
    ) -> Result<(), Vec<(ResourceTransaction, ResourceError)>> {
        // First, validate all transactions
        let validation_results: Vec<_> = transactions
            .iter()
            .map(|tx| (tx.clone(), self.check_availability(tx).and_then(|_| self.validate_destination(tx))))
            .collect();
        
        let failures: Vec<_> = validation_results
            .iter()
            .filter_map(|(tx, result)| match result {
                Err(e) => Some((tx.clone(), e.clone())),
                Ok(()) => None,
            })
            .collect();
        
        if !failures.is_empty() {
            return Err(failures);
        }
        
        // Execute all transactions if validation passed
        for tx in transactions {
            self.execute_transfer(&tx)?;
        }
        
        Ok(())
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
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

pub trait IteratorExt: Iterator {
    fn partition_map<A, B, F>(self, f: F) -> (Vec<A>, Vec<B>)
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Either<A, B>;
}

impl<T: Iterator> IteratorExt for T {
    fn partition_map<A, B, F>(self, mut f: F) -> (Vec<A>, Vec<B>)
    where
        Self: Sized,
        F: FnMut(Self::Item) -> Either<A, B>,
    {
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for item in self {
            match f(item) {
                Either::Left(a) => left.push(a),
                Either::Right(b) => right.push(b),
            }
        }
        
        (left, right)
    }
}

// Exercise 4: Custom Iterators - Complete Implementation
pub struct RouteIterator {
    current_position: (f32, f32, f32),
    destination: (f32, f32, f32),
    step_size: f32,
    steps_remaining: usize,
    direction: (f32, f32, f32),
}

impl RouteIterator {
    pub fn new(start: (f32, f32, f32), end: (f32, f32, f32), step_size: f32) -> Self {
        let distance = {
            let (dx, dy, dz) = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        
        let steps_remaining = if distance <= step_size {
            2 // Start and end
        } else {
            (distance / step_size).ceil() as usize + 1
        };
        
        let direction = if distance > 0.0 {
            let (dx, dy, dz) = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
            (dx / distance, dy / distance, dz / distance)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        Self {
            current_position: start,
            destination: end,
            step_size,
            steps_remaining,
            direction,
        }
    }
}

impl Iterator for RouteIterator {
    type Item = (f32, f32, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.steps_remaining == 0 {
            return None;
        }
        
        let current = self.current_position;
        
        if self.steps_remaining == 1 {
            // Last step - return destination
            self.current_position = self.destination;
        } else {
            // Move towards destination
            self.current_position = (
                self.current_position.0 + self.direction.0 * self.step_size,
                self.current_position.1 + self.direction.1 * self.step_size,
                self.current_position.2 + self.direction.2 * self.step_size,
            );
        }
        
        self.steps_remaining -= 1;
        Some(current)
    }
    
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.steps_remaining, Some(self.steps_remaining))
    }
}

impl ExactSizeIterator for RouteIterator {}

// Bonus: Advanced route iterator with waypoints
pub struct WaypointRouteIterator {
    waypoints: Vec<(f32, f32, f32)>,
    current_segment: Option<RouteIterator>,
    current_waypoint_index: usize,
    step_size: f32,
}

impl WaypointRouteIterator {
    pub fn new(waypoints: Vec<(f32, f32, f32)>, step_size: f32) -> Self {
        let current_segment = if waypoints.len() >= 2 {
            Some(RouteIterator::new(waypoints[0], waypoints[1], step_size))
        } else {
            None
        };
        
        Self {
            waypoints,
            current_segment,
            current_waypoint_index: 0,
            step_size,
        }
    }
}

impl Iterator for WaypointRouteIterator {
    type Item = (f32, f32, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut segment) = self.current_segment {
                if let Some(position) = segment.next() {
                    return Some(position);
                } else {
                    // Current segment finished, move to next
                    self.current_waypoint_index += 1;
                    
                    if self.current_waypoint_index + 1 < self.waypoints.len() {
                        // Create next segment
                        self.current_segment = Some(RouteIterator::new(
                            self.waypoints[self.current_waypoint_index],
                            self.waypoints[self.current_waypoint_index + 1],
                            self.step_size,
                        ));
                    } else {
                        // No more segments
                        self.current_segment = None;
                        return None;
                    }
                }
            } else {
                return None;
            }
        }
    }
}

// Exercise 5: Resource Simulation Iterator - Complete Implementation
pub struct ResourceSimulation {
    initial_resources: HashMap<ResourceType, f32>,
    consumption_rates: HashMap<ResourceType, f32>,
    production_rates: HashMap<ResourceType, f32>,
    current_time: f32,
    time_step: f32,
    max_time: f32,
}

impl ResourceSimulation {
    pub fn new(
        initial: HashMap<ResourceType, f32>,
        consumption: HashMap<ResourceType, f32>,
        production: HashMap<ResourceType, f32>,
        time_step: f32,
        max_time: f32,
    ) -> Self {
        Self {
            initial_resources: initial,
            consumption_rates: consumption,
            production_rates: production,
            current_time: 0.0,
            time_step,
            max_time,
        }
    }
}

impl Iterator for ResourceSimulation {
    type Item = (f32, HashMap<ResourceType, f32>);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_time > self.max_time {
            return None;
        }
        
        let current_resources = self.initial_resources
            .keys()
            .chain(self.consumption_rates.keys())
            .chain(self.production_rates.keys())
            .collect::<HashSet<_>>()
            .into_iter()
            .map(|resource_type| {
                let initial = self.initial_resources.get(resource_type).unwrap_or(&0.0);
                let consumption = self.consumption_rates.get(resource_type).unwrap_or(&0.0);
                let production = self.production_rates.get(resource_type).unwrap_or(&0.0);
                
                let net_change = (production - consumption) * self.current_time;
                let current_amount = (initial + net_change).max(0.0);
                
                (resource_type.clone(), current_amount)
            })
            .collect();
        
        let result = (self.current_time, current_resources);
        self.current_time += self.time_step;
        
        Some(result)
    }
}

// Bonus: Resource simulation with events
pub struct AdvancedResourceSimulation {
    base_simulation: ResourceSimulation,
    events: Vec<ResourceEvent>,
}

#[derive(Debug, Clone)]
pub struct ResourceEvent {
    pub time: f32,
    pub resource_type: ResourceType,
    pub change: f32,
    pub description: String,
}

impl AdvancedResourceSimulation {
    pub fn new(
        initial: HashMap<ResourceType, f32>,
        consumption: HashMap<ResourceType, f32>,
        production: HashMap<ResourceType, f32>,
        time_step: f32,
        max_time: f32,
        events: Vec<ResourceEvent>,
    ) -> Self {
        Self {
            base_simulation: ResourceSimulation::new(initial, consumption, production, time_step, max_time),
            events,
        }
    }
}

impl Iterator for AdvancedResourceSimulation {
    type Item = (f32, HashMap<ResourceType, f32>, Vec<ResourceEvent>);
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some((time, mut resources)) = self.base_simulation.next() {
            // Apply events that occur at this time
            let current_events: Vec<_> = self.events
                .iter()
                .filter(|event| (event.time - time).abs() < self.base_simulation.time_step / 2.0)
                .cloned()
                .collect();
            
            for event in &current_events {
                if let Some(current) = resources.get_mut(&event.resource_type) {
                    *current = (*current + event.change).max(0.0);
                }
            }
            
            Some((time, resources, current_events))
        } else {
            None
        }
    }
}

// Exercise 6: Functional Data Processing Pipeline - Complete Implementation
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
    pub fn process_sensor_data(
        raw_data: Vec<SensorReading>,
    ) -> Vec<ProcessedReading> {
        raw_data
            .into_iter()
            .filter(Self::is_valid_reading)
            .map(Self::calibrate_reading)
            .map(Self::apply_noise_reduction)
            .filter(Self::is_within_normal_range)
            .map(Self::convert_to_processed)
            .collect()
    }
    
    fn is_valid_reading(reading: &SensorReading) -> bool {
        reading.value.is_finite() && reading.timestamp > 0.0
    }
    
    fn calibrate_reading(mut reading: SensorReading) -> SensorReading {
        reading.value *= match reading.sensor_type {
            SensorType::Temperature => 1.02,
            SensorType::Pressure => 0.98,
            SensorType::Radiation => 1.05,
        };
        reading
    }
    
    fn apply_noise_reduction(mut reading: SensorReading) -> SensorReading {
        if reading.value.abs() > 1000.0 {
            reading.value *= 0.9;
        }
        reading
    }
    
    fn is_within_normal_range(reading: &SensorReading) -> bool {
        match reading.sensor_type {
            SensorType::Temperature => (-100.0..=200.0).contains(&reading.value),
            SensorType::Pressure => (0.0..=10000.0).contains(&reading.value),
            SensorType::Radiation => (0.0..=1000.0).contains(&reading.value),
        }
    }
    
    fn convert_to_processed(reading: SensorReading) -> ProcessedReading {
        ProcessedReading {
            sensor_id: reading.sensor_id,
            processed_value: reading.value,
            confidence: Self::calculate_confidence(&reading),
            timestamp: reading.timestamp,
        }
    }
    
    fn calculate_confidence(reading: &SensorReading) -> f32 {
        let normalized_value = reading.value.abs() / 100.0;
        (1.0 - normalized_value.min(1.0)).max(0.1)
    }
    
    // Bonus: Parallel processing
    pub fn process_sensor_data_parallel(
        raw_data: Vec<SensorReading>,
    ) -> Vec<ProcessedReading> {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let chunk_size = (raw_data.len() / 4).max(1);
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        
        for chunk in raw_data.chunks(chunk_size) {
            let chunk = chunk.to_vec();
            let results = Arc::clone(&results);
            
            let handle = thread::spawn(move || {
                let processed = Self::process_sensor_data(chunk);
                results.lock().unwrap().extend(processed);
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        Arc::try_unwrap(results).unwrap().into_inner().unwrap()
    }
}

// Exercise 7: Fleet Route Optimization - Complete Implementation
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
    pub fn optimize_routes(
        ships: Vec<Ship>,
        destinations: Vec<(f32, f32, f32)>,
    ) -> Vec<OptimizedRoute> {
        ships
            .into_iter()
            .enumerate()
            .filter_map(|(index, ship)| {
                Self::find_best_destination(&ship, &destinations)
                    .map(|dest| (index, ship, dest))
            })
            .map(|(index, ship, destination)| {
                Self::create_optimized_route(index, ship, destination)
            })
            .sorted_by(|a, b| {
                a.estimated_time
                    .partial_cmp(&b.estimated_time)
                    .unwrap_or(Ordering::Equal)
            })
    }
    
    fn find_best_destination(
        ship: &Ship,
        destinations: &[(f32, f32, f32)],
    ) -> Option<(f32, f32, f32)> {
        destinations
            .iter()
            .map(|&dest| {
                let distance = Self::calculate_distance(ship.position, dest);
                let travel_time = distance / ship.fuel_level.max(1.0);
                (dest, travel_time)
            })
            .min_by(|(_, time_a), (_, time_b)| {
                time_a.partial_cmp(time_b).unwrap_or(Ordering::Equal)
            })
            .map(|(dest, _)| dest)
    }
    
    fn create_optimized_route(
        route_id: usize,
        ship: Ship,
        destination: (f32, f32, f32),
    ) -> OptimizedRoute {
        let distance = Self::calculate_distance(ship.position, destination);
        let estimated_time = distance / ship.fuel_level.max(1.0);
        
        OptimizedRoute {
            route_id,
            ship_id: ship.id,
            destination,
            estimated_time,
            fuel_required: distance * 0.1,
        }
    }
    
    fn calculate_distance(pos1: (f32, f32, f32), pos2: (f32, f32, f32)) -> f32 {
        let (dx, dy, dz) = (pos2.0 - pos1.0, pos2.1 - pos1.1, pos2.2 - pos1.2);
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    // Bonus: Advanced optimization with multiple criteria
    pub fn multi_criteria_optimization(
        ships: Vec<Ship>,
        destinations: Vec<(f32, f32, f32)>,
        weights: OptimizationWeights,
    ) -> Vec<OptimizedRoute> {
        ships
            .into_iter()
            .enumerate()
            .filter_map(|(index, ship)| {
                Self::find_best_destination_weighted(&ship, &destinations, &weights)
                    .map(|dest| (index, ship, dest))
            })
            .map(|(index, ship, destination)| {
                Self::create_optimized_route(index, ship, destination)
            })
            .sorted_by(|a, b| {
                let score_a = Self::calculate_route_score(a, &weights);
                let score_b = Self::calculate_route_score(b, &weights);
                score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
            })
    }
    
    fn find_best_destination_weighted(
        ship: &Ship,
        destinations: &[(f32, f32, f32)],
        weights: &OptimizationWeights,
    ) -> Option<(f32, f32, f32)> {
        destinations
            .iter()
            .map(|&dest| {
                let distance = Self::calculate_distance(ship.position, dest);
                let travel_time = distance / ship.fuel_level.max(1.0);
                let fuel_cost = distance * 0.1;
                
                let score = weights.time_weight * travel_time + weights.fuel_weight * fuel_cost;
                (dest, score)
            })
            .min_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap_or(Ordering::Equal)
            })
            .map(|(dest, _)| dest)
    }
    
    fn calculate_route_score(route: &OptimizedRoute, weights: &OptimizationWeights) -> f32 {
        weights.time_weight * route.estimated_time + weights.fuel_weight * route.fuel_required
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationWeights {
    pub time_weight: f32,
    pub fuel_weight: f32,
}

impl Default for OptimizationWeights {
    fn default() -> Self {
        Self {
            time_weight: 1.0,
            fuel_weight: 1.0,
        }
    }
}

pub trait SortedIterator: Iterator {
    fn sorted_by<F>(self, compare: F) -> Vec<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering;
}

impl<T: Iterator> SortedIterator for T {
    fn sorted_by<F>(self, mut compare: F) -> Vec<Self::Item>
    where
        Self: Sized,
        F: FnMut(&Self::Item, &Self::Item) -> Ordering,
    {
        let mut vec: Vec<Self::Item> = self.collect();
        vec.sort_by(|a, b| compare(a, b));
        vec
    }
}

// Comprehensive test suite
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_resources_complete() {
        let numbers = vec![1, 2, 3, 4, 5];
        let doubled = process_resources(numbers.clone(), |x| x * 2);
        assert_eq!(doubled, vec![2, 4, 6, 8, 10]);
        
        // Test with fold version
        let tripled = process_resources_fold(numbers, |x| x * 3);
        assert_eq!(tripled, vec![3, 6, 9, 12, 15]);
    }

    #[test]
    fn test_ore_processing_complete() {
        let ores = vec![
            RawOre { mass: 100.0, purity: 0.9 },
            RawOre { mass: 50.0, purity: 0.7 },
        ];
        
        let refined = process_mining_output(ores);
        assert_eq!(refined.len(), 2);
        
        // High purity ore (0.9 > 0.8) should use efficiency 1.1
        assert!((refined[0].mass - 99.0).abs() < 0.1);
        assert!((refined[0].quality - 0.99).abs() < 0.01);
        
        // Low purity ore (0.7 < 0.8) should use efficiency 0.9
        assert!((refined[1].mass - 31.5).abs() < 0.1);
        assert!((refined[1].quality - 0.63).abs() < 0.01);
    }

    #[test]
    fn test_ore_processing_with_validation() {
        let ores = vec![
            RawOre { mass: 100.0, purity: 0.9 },
            RawOre { mass: -50.0, purity: 0.7 }, // Invalid mass
        ];
        
        let result = process_mining_output_with_validation(ores);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid ore mass"));
    }

    #[test]
    fn test_fleet_analysis_complete() {
        let ships = vec![
            Ship {
                id: 1,
                cargo_capacity: 100.0,
                current_cargo: 90.0,
                fuel_level: 0.2,
                position: (50.0, 0.0, 0.0),
            },
            Ship {
                id: 2,
                cargo_capacity: 200.0,
                current_cargo: 100.0,
                fuel_level: 0.8,
                position: (150.0, 0.0, 0.0),
            },
        ];
        
        let analysis = analyze_fleet(&ships);
        assert_eq!(analysis.total_ships, 2);
        assert_eq!(analysis.efficient_ships, vec![1]);
        assert_eq!(analysis.fuel_alerts, vec![1]);
        assert_eq!(analysis.distant_ships, vec![2]);
        assert_eq!(analysis.cargo_stats.total_capacity, 300.0);
        assert_eq!(analysis.cargo_stats.total_cargo, 190.0);
        
        // Test detailed analysis
        let detailed = detailed_fleet_analysis(&ships);
        assert_eq!(detailed.total_ships, 2);
        assert!(detailed.average_efficiency > 0.0);
        assert_eq!(detailed.ship_stats[0].risk_level, RiskLevel::High); // Low fuel + distant
    }

    #[test]
    fn test_resource_manager_complete() {
        let mut manager = ResourceManager::new();
        
        // Set up initial resources
        manager.set_resource_quantity(1, ResourceType::Iron, 100.0);
        manager.set_resource_quantity(2, ResourceType::Iron, 50.0);
        
        // Test successful transfer
        let tx = ResourceTransaction {
            resource_type: ResourceType::Iron,
            quantity: 30.0,
            source_id: 1,
            destination_id: 2,
        };
        
        assert!(manager.transfer_resource(tx).is_ok());
        assert_eq!(manager.get_resource_quantity(1, &ResourceType::Iron), 70.0);
        assert_eq!(manager.get_resource_quantity(2, &ResourceType::Iron), 80.0);
        
        // Test insufficient quantity
        let tx2 = ResourceTransaction {
            resource_type: ResourceType::Iron,
            quantity: 100.0,
            source_id: 1,
            destination_id: 2,
        };
        
        assert!(matches!(manager.transfer_resource(tx2), Err(ResourceError::InsufficientQuantity)));
        
        // Test report generation
        let report = manager.generate_report();
        assert_eq!(report.total_entities, 2);
        assert_eq!(report.resource_totals[&ResourceType::Iron], 150.0);
    }

    #[test]
    fn test_resource_transactions_batch() {
        let mut manager = ResourceManager::new();
        manager.set_resource_quantity(1, ResourceType::Fuel, 100.0);
        manager.set_resource_quantity(1, ResourceType::Water, 50.0);
        
        let transactions = vec![
            ResourceTransaction {
                resource_type: ResourceType::Fuel,
                quantity: 20.0,
                source_id: 1,
                destination_id: 2,
            },
            ResourceTransaction {
                resource_type: ResourceType::Water,
                quantity: 200.0, // Should fail - insufficient
                source_id: 1,
                destination_id: 2,
            },
        ];
        
        let (successful, failed) = manager.process_transactions(transactions);
        assert_eq!(successful.len(), 1);
        assert_eq!(failed.len(), 1);
        assert!(matches!(failed[0].1, ResourceError::InsufficientQuantity));
    }

    #[test]
    fn test_route_iterator_complete() {
        let mut route = RouteIterator::new((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 5.0);
        
        assert_eq!(route.len(), 3);
        assert_eq!(route.next(), Some((0.0, 0.0, 0.0)));
        assert_eq!(route.len(), 2);
        assert_eq!(route.next(), Some((5.0, 0.0, 0.0)));
        assert_eq!(route.next(), Some((10.0, 0.0, 0.0)));
        assert_eq!(route.next(), None);
        
        // Test single step route
        let mut short_route = RouteIterator::new((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5.0);
        assert_eq!(short_route.len(), 2);
        assert_eq!(short_route.next(), Some((0.0, 0.0, 0.0)));
        assert_eq!(short_route.next(), Some((1.0, 0.0, 0.0)));
        assert_eq!(short_route.next(), None);
    }

    #[test]
    fn test_waypoint_route_iterator() {
        let waypoints = vec![(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)];
        let mut route = WaypointRouteIterator::new(waypoints, 2.5);
        
        // Should visit points along both segments
        let points: Vec<_> = route.collect();
        assert!(points.len() > 4); // Multiple segments with intermediate points
    }

    #[test]
    fn test_resource_simulation_complete() {
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
        assert_eq!(resources[&ResourceType::Fuel], 95.0);
        
        let (time, resources) = sim.next().unwrap();
        assert_eq!(time, 2.0);
        assert_eq!(resources[&ResourceType::Fuel], 90.0);
        
        let (time, resources) = sim.next().unwrap();
        assert_eq!(time, 3.0);
        assert_eq!(resources[&ResourceType::Fuel], 85.0);
        
        assert!(sim.next().is_none());
    }

    #[test]
    fn test_sensor_data_pipeline() {
        let raw_data = vec![
            SensorReading {
                sensor_id: 1,
                sensor_type: SensorType::Temperature,
                value: 50.0,
                timestamp: 1.0,
            },
            SensorReading {
                sensor_id: 2,
                sensor_type: SensorType::Temperature,
                value: f32::INFINITY, // Invalid
                timestamp: 2.0,
            },
            SensorReading {
                sensor_id: 3,
                sensor_type: SensorType::Pressure,
                value: 5000.0,
                timestamp: 3.0,
            },
        ];
        
        let processed = DataPipeline::process_sensor_data(raw_data);
        assert_eq!(processed.len(), 2); // One invalid reading filtered out
        
        // Check calibration was applied
        assert!((processed[0].processed_value - 51.0).abs() < 0.1); // 50 * 1.02
        assert!((processed[1].processed_value - 4900.0).abs() < 0.1); // 5000 * 0.98
    }

    #[test]
    fn test_fleet_optimization() {
        let ships = vec![
            Ship {
                id: 1,
                cargo_capacity: 100.0,
                current_cargo: 50.0,
                fuel_level: 80.0,
                position: (0.0, 0.0, 0.0),
            },
            Ship {
                id: 2,
                cargo_capacity: 200.0,
                current_cargo: 100.0,
                fuel_level: 60.0,
                position: (10.0, 0.0, 0.0),
            },
        ];
        
        let destinations = vec![(5.0, 0.0, 0.0), (20.0, 0.0, 0.0)];
        
        let routes = FleetOptimizer::optimize_routes(ships, destinations);
        assert_eq!(routes.len(), 2);
        
        // Routes should be sorted by estimated time
        assert!(routes[0].estimated_time <= routes[1].estimated_time);
    }

    #[test]
    fn test_multi_criteria_optimization() {
        let ships = vec![
            Ship {
                id: 1,
                cargo_capacity: 100.0,
                current_cargo: 50.0,
                fuel_level: 80.0,
                position: (0.0, 0.0, 0.0),
            },
        ];
        
        let destinations = vec![(5.0, 0.0, 0.0), (50.0, 0.0, 0.0)];
        
        let weights = OptimizationWeights {
            time_weight: 2.0,
            fuel_weight: 1.0,
        };
        
        let routes = FleetOptimizer::multi_criteria_optimization(ships, destinations, weights);
        assert_eq!(routes.len(), 1);
        
        // Should prefer closer destination due to time weight
        let distance = FleetOptimizer::calculate_distance((0.0, 0.0, 0.0), routes[0].destination);
        assert!(distance < 30.0); // Should pick closer destination
    }

    #[test] 
    fn test_partition_map() {
        let numbers = vec![1, 2, 3, 4, 5, 6];
        let (evens, odds): (Vec<i32>, Vec<i32>) = numbers
            .into_iter()
            .partition_map(|n| {
                if n % 2 == 0 {
                    Either::Left(n)
                } else {
                    Either::Right(n)
                }
            });
        
        assert_eq!(evens, vec![2, 4, 6]);
        assert_eq!(odds, vec![1, 3, 5]);
    }

    #[test]
    fn test_sorted_iterator() {
        let numbers = vec![5, 2, 8, 1, 9, 3];
        let sorted = numbers.into_iter().sorted_by(|a, b| a.cmp(b));
        assert_eq!(sorted, vec![1, 2, 3, 5, 8, 9]);
    }
}