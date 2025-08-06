# Space Resource Management Simulation Demo

A demonstration of our space simulation engine featuring a Rust-based simulator with a Python monitoring client.

## Overview

This demo simulates a single spaceship (SS Prospector) performing a complete resource management mission:

1. **Travel** to Mining Station Alpha
2. **Load** raw materials (iron ore, copper ore, rare earth elements)
3. **Travel** to Trade Hub Beta  
4. **Sell** materials for profit
5. **Load** crew and ship parts
6. **Return** to home base

The simulation tracks:
- ⏱️ Time for all operations
- 💰 Credits earned and spent
- ⛽ Fuel consumption
- 📦 Cargo loading/unloading
- 👥 Crew management
- 🔧 Parts and robot inventory

## Architecture

```
┌─────────────────┐    SIM_DATA JSON    ┌──────────────────┐
│   Rust Simulator│ ──────────────────► │  Python Client   │
│                 │                     │                  │
│ • Ship Logic    │    stdout/stdin     │ • Data Display   │
│ • Physics       │ ◄──────────────────  │ • Monitoring     │
│ • Economics     │                     │ • Analytics      │
└─────────────────┘                     └──────────────────┘
```

## Running the Demo

### Prerequisites

- Rust (2021 edition)
- Python 3.7+
- Cargo package manager

### Quick Start

1. **Run with Python client (recommended):**
   ```bash
   cd demo
   python3 client.py
   ```

2. **Run Rust simulator directly:**
   ```bash
   cd demo
   cargo run
   ```

## Features Demonstrated

### Ship Operations
- **Navigation**: 3D coordinate system with realistic travel times
- **Docking**: Port fees and docking procedures
- **Cargo Management**: Weight limits and capacity tracking
- **Fuel System**: Consumption based on distance and refueling costs
- **Economic System**: Dynamic pricing and profit/loss tracking

### Real-time Monitoring
- **Live Status Updates**: Ship position, fuel, cargo, credits
- **Event Logging**: All operations with timestamps and costs
- **Performance Metrics**: Profit analysis and mission efficiency
- **Visual Indicators**: Emoji-coded event types and progress bars

### Communication Protocol
The Rust simulator outputs structured JSON data prefixed with `SIM_DATA:`:

```json
{
  "timestamp": 15.2,
  "ship": {
    "id": 1,
    "name": "SS Prospector",
    "position": {"x": 100.0, "y": 50.0, "z": 25.0, "sector": "Asteroid Belt"},
    "credits": 45000.0,
    "fuel": 750.0,
    "status": {"Docked": {"port": "Mining Station Alpha"}},
    "cargo": {...},
    "crew": {...}
  },
  "latest_events": [...]
}
```

## Sample Output

```
🚀 Starting Space Resource Management Simulation
================================================

📊 [T+0.0h] Ship created and ready for operations
🚀 Travel Progress: 50% - ETA: 2.1h
📡 Real-time Ship Data [T+2.1h]
--------------------------------------------------
🚢 Ship: SS Prospector (ID: 1)
📍 Position: (100.0, 50.0, 25.0) - Asteroid Belt
🛸 Status: Traveling to (100.0, 50.0, 25.0)
💰 Credits: 48,900.00
⛽ Fuel: 748.5/1000.0 (74.9%)
📦 Cargo: 150.0/1000.0 (15.0%)
```

## Mission Economics

The demo simulates realistic space commerce:

### Costs
- **Docking Fees**: 100 credits per port
- **Fuel**: 2 credits per unit
- **Materials**: 8-80 credits per unit (depending on rarity)
- **Crew**: 5,000 credits per engineer
- **Parts**: 300 credits per ship part

### Revenue
- **Iron Ore**: 12 credits per unit
- **Copper Ore**: 18 credits per unit  
- **Rare Earth**: 120 credits per unit

### Typical Mission Results
- **Starting Credits**: 50,000
- **Total Revenue**: ~9,000-12,000
- **Total Expenses**: ~6,000-8,000
- **Net Profit**: ~3,000-5,000 (6-10% return)

## Extensions

This demo can be extended with:

- **Multiple Ships**: Fleet management
- **Dynamic Markets**: Supply/demand pricing
- **Random Events**: Pirates, equipment failures, discoveries
- **Multiplayer**: Multiple clients controlling different ships
- **Web Interface**: Browser-based monitoring
- **Database Persistence**: Mission history and analytics
- **Real-time Strategy**: Player decisions affecting outcomes

## Technical Details

### Rust Simulator
- **Async Runtime**: Tokio for non-blocking operations
- **Serialization**: Serde for JSON data exchange
- **Error Handling**: Comprehensive Result types
- **Modular Design**: Separated concerns for ships, ports, economics

### Python Client  
- **Real-time Processing**: Subprocess monitoring
- **Data Visualization**: Formatted console output
- **Event Analysis**: Automatic profit/loss calculation
- **Error Handling**: Graceful degradation and recovery

This demo showcases the practical application of Rust's performance and safety features in a game simulation context, while demonstrating effective inter-language communication patterns.