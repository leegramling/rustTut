//! Space Simulation Engine
//! 
//! A high-performance simulation engine for managing space resources,
//! ship behavior, and inter-planetary logistics.

pub mod simulation;
pub mod resources;
pub mod entities;
pub mod behavior;
pub mod spatial;
pub mod network;
pub mod utils;

pub use simulation::SimulationEngine;
pub use resources::ResourceManager;
pub use entities::{Entity, Ship, Station, Asteroid};