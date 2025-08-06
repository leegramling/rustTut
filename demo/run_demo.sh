#!/bin/bash
# Space Simulation Demo Runner

echo "🚀 Starting Space Resource Management Simulation Demo"
echo "======================================================"
echo ""
echo "This demo shows a complete resource management mission:"
echo "  1. Travel to Mining Station Alpha"
echo "  2. Load raw materials (iron ore, copper ore, rare earth)"
echo "  3. Travel to Trade Hub Beta"
echo "  4. Sell materials for profit"
echo "  5. Load crew and ship parts"
echo "  6. Return to home base"
echo ""
echo "The Python client will display real-time simulation data..."
echo ""
echo "Press Ctrl+C to stop the simulation"
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo not found. Please install Rust first."
    echo "   Visit: https://rustup.rs/"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Run the demo
python3 client.py