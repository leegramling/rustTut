use space_sim::SimulationEngine;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🚀 Space Simulation Engine Starting...");
    
    // Initialize the simulation engine
    let mut engine = SimulationEngine::new();
    
    // Main simulation loop
    loop {
        engine.tick().await?;
        sleep(Duration::from_millis(16)).await; // ~60 FPS
    }
}