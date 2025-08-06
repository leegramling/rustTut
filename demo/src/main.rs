// Space Simulation Demo - Rust Simulator
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ship {
    pub id: u32,
    pub name: String,
    pub position: Position,
    pub cargo: CargoHold,
    pub crew: CrewManifest,
    pub credits: f64,
    pub fuel: f64,
    pub max_fuel: f64,
    pub status: ShipStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sector: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoHold {
    pub materials: HashMap<String, f64>,
    pub parts: HashMap<String, u32>,
    pub robots: u32,
    pub capacity: f64,
    pub used: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrewManifest {
    pub engineers: u32,
    pub pilots: u32,
    pub miners: u32,
    pub capacity: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShipStatus {
    Idle,
    Traveling { destination: Position, eta: f64 },
    Docked { port: String },
    Mining,
    Loading,
    Unloading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub position: Position,
    pub services: PortServices,
    pub market: Market,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortServices {
    pub refuel: bool,
    pub repair: bool,
    pub crew_transfer: bool,
    pub cargo_handling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Market {
    pub buy_prices: HashMap<String, f64>,
    pub sell_prices: HashMap<String, f64>,
    pub demand: HashMap<String, f64>,
    pub supply: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationEvent {
    pub timestamp: f64,
    pub event_type: EventType,
    pub ship_id: u32,
    pub description: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    Travel,
    Dock,
    Undock,
    LoadCargo,
    UnloadCargo,
    CrewTransfer,
    Transaction,
    FuelUpdate,
    StatusChange,
}

pub struct SpaceSimulator {
    ships: HashMap<u32, Ship>,
    ports: HashMap<String, Port>,
    events: Vec<SimulationEvent>,
    current_time: f64,
    next_ship_id: u32,
}

impl SpaceSimulator {
    pub fn new() -> Self {
        let mut simulator = Self {
            ships: HashMap::new(),
            ports: HashMap::new(),
            events: Vec::new(),
            current_time: 0.0,
            next_ship_id: 1,
        };
        
        simulator.initialize_ports();
        simulator.create_demo_ship();
        simulator
    }
    
    fn initialize_ports(&mut self) {
        // Mining Station Alpha
        let mut mining_market = HashMap::new();
        mining_market.insert("iron_ore".to_string(), 10.0);
        mining_market.insert("copper_ore".to_string(), 15.0);
        mining_market.insert("rare_earth".to_string(), 100.0);
        
        let mut mining_sell = HashMap::new();
        mining_sell.insert("fuel".to_string(), 2.0);
        mining_sell.insert("food".to_string(), 5.0);
        mining_sell.insert("mining_equipment".to_string(), 500.0);
        
        let mining_station = Port {
            name: "Mining Station Alpha".to_string(),
            position: Position {
                x: 100.0,
                y: 50.0,
                z: 25.0,
                sector: "Asteroid Belt".to_string(),
            },
            services: PortServices {
                refuel: true,
                repair: true,
                crew_transfer: true,
                cargo_handling: true,
            },
            market: Market {
                buy_prices: mining_market,
                sell_prices: mining_sell,
                demand: HashMap::new(),
                supply: HashMap::new(),
            },
        };
        
        // Trade Hub Beta
        let mut trade_buy = HashMap::new();
        trade_buy.insert("iron_ore".to_string(), 12.0);
        trade_buy.insert("copper_ore".to_string(), 18.0);
        trade_buy.insert("rare_earth".to_string(), 120.0);
        
        let mut trade_sell = HashMap::new();
        trade_sell.insert("fuel".to_string(), 1.8);
        trade_sell.insert("food".to_string(), 4.0);
        trade_sell.insert("electronics".to_string(), 200.0);
        trade_sell.insert("ship_parts".to_string(), 300.0);
        
        let trade_hub = Port {
            name: "Trade Hub Beta".to_string(),
            position: Position {
                x: -75.0,
                y: 100.0,
                z: -30.0,
                sector: "Trade Sector".to_string(),
            },
            services: PortServices {
                refuel: true,
                repair: true,
                crew_transfer: true,
                cargo_handling: true,
            },
            market: Market {
                buy_prices: trade_buy,
                sell_prices: trade_sell,
                demand: HashMap::new(),
                supply: HashMap::new(),
            },
        };
        
        self.ports.insert("Mining Station Alpha".to_string(), mining_station);
        self.ports.insert("Trade Hub Beta".to_string(), trade_hub);
    }
    
    fn create_demo_ship(&mut self) {
        let mut materials = HashMap::new();
        materials.insert("iron_ore".to_string(), 0.0);
        materials.insert("copper_ore".to_string(), 0.0);
        materials.insert("rare_earth".to_string(), 0.0);
        
        let mut parts = HashMap::new();
        parts.insert("mining_equipment".to_string(), 2);
        parts.insert("ship_parts".to_string(), 1);
        
        let ship = Ship {
            id: self.next_ship_id,
            name: "SS Prospector".to_string(),
            position: Position {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                sector: "Home Base".to_string(),
            },
            cargo: CargoHold {
                materials,
                parts,
                robots: 3,
                capacity: 1000.0,
                used: 150.0, // Initial cargo weight
            },
            crew: CrewManifest {
                engineers: 2,
                pilots: 1,
                miners: 4,
                capacity: 10,
            },
            credits: 50000.0,
            fuel: 800.0,
            max_fuel: 1000.0,
            status: ShipStatus::Idle,
        };
        
        self.ships.insert(self.next_ship_id, ship);
        self.next_ship_id += 1;
        
        self.log_event(1, EventType::StatusChange, "Ship created and ready for operations", 
                      serde_json::json!({"initial_credits": 50000.0, "fuel": 800.0}));
    }
    
    pub async fn run_simulation(&mut self) {
        println!("🚀 Starting Space Resource Management Simulation");
        println!("================================================");
        
        // Mission plan: Travel to mining station, load materials, travel to trade hub, sell materials
        let ship_id = 1;
        
        // Phase 1: Travel to Mining Station Alpha
        self.travel_to_port(ship_id, "Mining Station Alpha").await;
        
        // Phase 2: Mine and load raw materials
        self.dock_at_port(ship_id, "Mining Station Alpha").await;
        self.load_materials(ship_id, "Mining Station Alpha").await;
        self.refuel(ship_id, "Mining Station Alpha").await;
        self.undock_from_port(ship_id, "Mining Station Alpha").await;
        
        // Phase 3: Travel to Trade Hub Beta
        self.travel_to_port(ship_id, "Trade Hub Beta").await;
        
        // Phase 4: Sell materials and resupply
        self.dock_at_port(ship_id, "Trade Hub Beta").await;
        self.sell_materials(ship_id, "Trade Hub Beta").await;
        self.load_crew_and_parts(ship_id, "Trade Hub Beta").await;
        self.undock_from_port(ship_id, "Trade Hub Beta").await;
        
        // Phase 5: Return home
        self.travel_to_home(ship_id).await;
        
        println!("\n🎯 Mission Complete! Final Status:");
        self.print_ship_status(ship_id);
        self.print_mission_summary();
    }
    
    async fn travel_to_port(&mut self, ship_id: u32, port_name: &str) {
        // Calculate distance first
        let (distance, port_position) = {
            if let (Some(ship), Some(port)) = (self.ships.get(&ship_id), self.ports.get(port_name)) {
                let distance = self.calculate_distance(&ship.position, &port.position);
                (distance, port.position.clone())
            } else {
                return;
            }
        };
        
        let travel_time = distance / 50.0; // 50 units per hour
        let fuel_cost = distance * 0.5;
        
        // Update ship state
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Traveling {
                destination: port_position.clone(),
                eta: self.current_time + travel_time,
            };
            ship.fuel -= fuel_cost;
        }
        
        self.log_event(ship_id, EventType::Travel, 
                      &format!("Traveling to {} (Distance: {:.1} AU, ETA: {:.1}h, Fuel cost: {:.1})", 
                              port_name, distance, travel_time, fuel_cost),
                      serde_json::json!({
                          "destination": port_name,
                          "distance": distance,
                          "travel_time": travel_time,
                          "fuel_cost": fuel_cost
                      }));
        
        // Simulate travel time
        let steps = 10;
        for i in 1..=steps {
            sleep(Duration::from_millis(500)).await;
            let progress = (i as f64) / (steps as f64);
            self.current_time += travel_time / (steps as f64);
            
            println!("🚀 Travel Progress: {:.0}% - ETA: {:.1}h", 
                    progress * 100.0, 
                    travel_time - (progress * travel_time));
            
            self.output_simulation_state();
        }
        
        // Arrive at destination
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.position = port_position.clone();
            ship.status = ShipStatus::Idle;
        }
        
        let fuel_remaining = self.ships.get(&ship_id).map(|s| s.fuel).unwrap_or(0.0);
        self.log_event(ship_id, EventType::StatusChange, 
                      &format!("Arrived at {}", port_name),
                      serde_json::json!({"port": port_name, "fuel_remaining": fuel_remaining}));
    }
    
    async fn dock_at_port(&mut self, ship_id: u32, port_name: &str) {
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Docked { port: port_name.to_string() };
            
            let docking_fee = 100.0;
            ship.credits -= docking_fee;
            
            self.log_event(ship_id, EventType::Dock,
                          &format!("Docked at {} (Fee: {} credits)", port_name, docking_fee),
                          serde_json::json!({"port": port_name, "fee": docking_fee}));
            
            sleep(Duration::from_millis(1000)).await;
            self.current_time += 0.5; // 30 minutes to dock
            self.output_simulation_state();
        }
    }
    
    async fn load_materials(&mut self, ship_id: u32, port_name: &str) {
        // Check if ship and port exist
        if self.ships.get(&ship_id).is_none() || self.ports.get(port_name).is_none() {
            return;
        }
        
        // Set loading status
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Loading;
        }
        
        // Simulate mining/loading time
        let materials_to_load = vec![
            ("iron_ore", 200.0, 8.0),
            ("copper_ore", 150.0, 12.0),
            ("rare_earth", 50.0, 80.0),
        ];
        
        for (material, amount, unit_cost) in materials_to_load {
            let loading_time = amount / 50.0; // 50 units per hour loading rate
            let cost = amount * unit_cost;
            
            // Update ship cargo and credits
            if let Some(ship) = self.ships.get_mut(&ship_id) {
                ship.credits -= cost;
                ship.cargo.materials.insert(material.to_string(), amount);
                ship.cargo.used += amount;
            }
            
            self.log_event(ship_id, EventType::LoadCargo,
                          &format!("Loading {} units of {} (Cost: {} credits, Time: {:.1}h)", 
                                  amount, material, cost, loading_time),
                          serde_json::json!({
                              "material": material,
                              "amount": amount,
                              "cost": cost,
                              "loading_time": loading_time
                          }));
            
            sleep(Duration::from_millis(1500)).await;
            self.current_time += loading_time;
            self.output_simulation_state();
        }
        
        // Set docked status
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Docked { port: port_name.to_string() };
        }
    }
    
    async fn sell_materials(&mut self, ship_id: u32, port_name: &str) {
        // Get port buy prices
        let buy_prices = if let Some(port) = self.ports.get(port_name) {
            port.market.buy_prices.clone()
        } else {
            return;
        };
        
        // Get ship materials to sell
        let cargo_materials = if let Some(ship) = self.ships.get(&ship_id) {
            ship.cargo.materials.clone()
        } else {
            return;
        };
        
        // Set unloading status
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Unloading;
        }
        
        let mut total_revenue = 0.0;
        
        for (material, amount) in cargo_materials {
            if let Some(&sell_price) = buy_prices.get(&material) {
                let revenue = amount * sell_price;
                let unloading_time = amount / 75.0; // 75 units per hour unloading rate
                
                // Update ship cargo and credits
                if let Some(ship) = self.ships.get_mut(&ship_id) {
                    ship.credits += revenue;
                    ship.cargo.materials.insert(material.clone(), 0.0);
                    ship.cargo.used -= amount;
                }
                
                total_revenue += revenue;
                
                self.log_event(ship_id, EventType::UnloadCargo,
                              &format!("Sold {} units of {} for {} credits (Time: {:.1}h)", 
                                      amount, material, revenue, unloading_time),
                              serde_json::json!({
                                  "material": material,
                                  "amount": amount,
                                  "revenue": revenue,
                                  "unloading_time": unloading_time
                              }));
                
                sleep(Duration::from_millis(1000)).await;
                self.current_time += unloading_time;
                self.output_simulation_state();
            }
        }
        
        self.log_event(ship_id, EventType::Transaction,
                      &format!("Total sales revenue: {} credits", total_revenue),
                      serde_json::json!({"total_revenue": total_revenue}));
        
        // Set docked status
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Docked { port: port_name.to_string() };
        }
    }
    
    async fn refuel(&mut self, ship_id: u32, port_name: &str) {
        if let (Some(ship), Some(_port)) = (self.ships.get_mut(&ship_id), self.ports.get(port_name)) {
            let fuel_needed = ship.max_fuel - ship.fuel;
            let fuel_cost = fuel_needed * 2.0; // 2 credits per fuel unit
            
            ship.fuel = ship.max_fuel;
            ship.credits -= fuel_cost;
            
            self.log_event(ship_id, EventType::FuelUpdate,
                          &format!("Refueled {} units for {} credits", fuel_needed, fuel_cost),
                          serde_json::json!({"fuel_added": fuel_needed, "cost": fuel_cost}));
            
            sleep(Duration::from_millis(800)).await;
            self.current_time += 0.3;
            self.output_simulation_state();
        }
    }
    
    async fn load_crew_and_parts(&mut self, ship_id: u32, port_name: &str) {
        // Check if ship and port exist
        if self.ships.get(&ship_id).is_none() || self.ports.get(port_name).is_none() {
            return;
        }
        
        // Hire additional crew
        let engineers_hired = 1;
        let crew_cost = engineers_hired as f64 * 5000.0; // 5000 credits per engineer
        
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.crew.engineers += engineers_hired;
            ship.credits -= crew_cost;
        }
        
        self.log_event(ship_id, EventType::CrewTransfer,
                      &format!("Hired {} engineers for {} credits", engineers_hired, crew_cost),
                      serde_json::json!({"engineers_hired": engineers_hired, "cost": crew_cost}));
        
        // Purchase ship parts
        let parts_purchased = 2;
        let parts_cost = parts_purchased as f64 * 300.0;
        
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            *ship.cargo.parts.get_mut("ship_parts").unwrap() += parts_purchased;
            ship.credits -= parts_cost;
            ship.cargo.used += parts_purchased as f64 * 10.0; // 10 units weight per part
        }
        
        self.log_event(ship_id, EventType::LoadCargo,
                      &format!("Purchased {} ship parts for {} credits", parts_purchased, parts_cost),
                      serde_json::json!({"parts_purchased": parts_purchased, "cost": parts_cost}));
        
        sleep(Duration::from_millis(1200)).await;
        self.current_time += 1.0;
        self.output_simulation_state();
    }
    
    async fn undock_from_port(&mut self, ship_id: u32, port_name: &str) {
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.status = ShipStatus::Idle;
            
            self.log_event(ship_id, EventType::Undock,
                          &format!("Undocked from {}", port_name),
                          serde_json::json!({"port": port_name}));
            
            sleep(Duration::from_millis(500)).await;
            self.current_time += 0.25;
            self.output_simulation_state();
        }
    }
    
    async fn travel_to_home(&mut self, ship_id: u32) {
        let home_position = Position {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            sector: "Home Base".to_string(),
        };
        
        // Calculate distance
        let (distance, travel_time, fuel_cost) = {
            if let Some(ship) = self.ships.get(&ship_id) {
                let distance = self.calculate_distance(&ship.position, &home_position);
                let travel_time = distance / 50.0;
                let fuel_cost = distance * 0.5;
                (distance, travel_time, fuel_cost)
            } else {
                return;
            }
        };
        
        // Update ship state
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.fuel -= fuel_cost;
            ship.status = ShipStatus::Traveling {
                destination: home_position.clone(),
                eta: self.current_time + travel_time,
            };
        }
        
        self.log_event(ship_id, EventType::Travel,
                      &format!("Returning home (Distance: {:.1} AU, Time: {:.1}h)", distance, travel_time),
                      serde_json::json!({"distance": distance, "travel_time": travel_time}));
        
        // Simulate travel
        for i in 1..=5 {
            sleep(Duration::from_millis(800)).await;
            let progress = (i as f64) / 5.0;
            self.current_time += travel_time / 5.0;
            
            println!("🏠 Homeward Progress: {:.0}%", progress * 100.0);
            self.output_simulation_state();
        }
        
        // Arrive at home
        if let Some(ship) = self.ships.get_mut(&ship_id) {
            ship.position = home_position;
            ship.status = ShipStatus::Idle;
        }
        
        let fuel_remaining = self.ships.get(&ship_id).map(|s| s.fuel).unwrap_or(0.0);
        self.log_event(ship_id, EventType::StatusChange, "Arrived at Home Base", 
                      serde_json::json!({"fuel_remaining": fuel_remaining}));
    }
    
    fn calculate_distance(&self, pos1: &Position, pos2: &Position) -> f64 {
        let dx = pos2.x - pos1.x;
        let dy = pos2.y - pos1.y;
        let dz = pos2.z - pos1.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    fn log_event(&mut self, ship_id: u32, event_type: EventType, description: &str, data: serde_json::Value) {
        let event = SimulationEvent {
            timestamp: self.current_time,
            event_type,
            ship_id,
            description: description.to_string(),
            data,
        };
        
        println!("📊 [T+{:.1}h] {}", self.current_time, description);
        self.events.push(event);
    }
    
    fn output_simulation_state(&self) {
        if let Some(ship) = self.ships.get(&1) {
            let state = serde_json::json!({
                "timestamp": self.current_time,
                "ship": ship,
                "latest_events": self.events.iter().rev().take(3).collect::<Vec<_>>()
            });
            
            // Output JSON for Python client to consume
            println!("SIM_DATA:{}", serde_json::to_string(&state).unwrap());
        }
    }
    
    fn print_ship_status(&self, ship_id: u32) {
        if let Some(ship) = self.ships.get(&ship_id) {
            println!("\n🚢 {} Status Report:", ship.name);
            println!("   Position: {:.1}, {:.1}, {:.1} ({})", 
                    ship.position.x, ship.position.y, ship.position.z, ship.position.sector);
            println!("   Credits: {:.2}", ship.credits);
            println!("   Fuel: {:.1}/{:.1}", ship.fuel, ship.max_fuel);
            println!("   Cargo Used: {:.1}/{:.1}", ship.cargo.used, ship.cargo.capacity);
            println!("   Status: {:?}", ship.status);
            
            println!("\n📦 Cargo Manifest:");
            for (material, amount) in &ship.cargo.materials {
                if *amount > 0.0 {
                    println!("   {}: {:.1} units", material, amount);
                }
            }
            
            for (part, count) in &ship.cargo.parts {
                if *count > 0 {
                    println!("   {}: {} units", part, count);
                }
            }
            
            if ship.cargo.robots > 0 {
                println!("   robots: {} units", ship.cargo.robots);
            }
            
            println!("\n👥 Crew: {} Engineers, {} Pilots, {} Miners", 
                    ship.crew.engineers, ship.crew.pilots, ship.crew.miners);
        }
    }
    
    fn print_mission_summary(&self) {
        println!("\n📈 Mission Summary:");
        println!("   Total mission time: {:.1} hours", self.current_time);
        println!("   Total events logged: {}", self.events.len());
        
        let mut total_revenue = 0.0;
        let mut total_expenses = 0.0;
        
        for event in &self.events {
            match event.event_type {
                EventType::UnloadCargo => {
                    if let Some(revenue) = event.data.get("revenue") {
                        total_revenue += revenue.as_f64().unwrap_or(0.0);
                    }
                }
                EventType::LoadCargo | EventType::FuelUpdate | EventType::Dock | EventType::CrewTransfer => {
                    if let Some(cost) = event.data.get("cost") {
                        total_expenses += cost.as_f64().unwrap_or(0.0);
                    }
                    if let Some(fee) = event.data.get("fee") {
                        total_expenses += fee.as_f64().unwrap_or(0.0);
                    }
                }
                _ => {}
            }
        }
        
        println!("   Total Revenue: {:.2} credits", total_revenue);
        println!("   Total Expenses: {:.2} credits", total_expenses);
        println!("   Net Profit: {:.2} credits", total_revenue - total_expenses);
    }
}

#[tokio::main]
async fn main() {
    let mut simulator = SpaceSimulator::new();
    simulator.run_simulation().await;
}