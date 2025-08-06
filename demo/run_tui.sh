#!/bin/bash
# Space Simulation Demo Runner with TUI Dashboard

echo "🚀 Starting Space Resource Management Simulation"
echo "================================================"
echo ""
echo "🖥️  DASHBOARD MODE - Non-scrolling real-time display"
echo ""
echo "This interactive dashboard shows:"
echo "  • Real-time ship status and location"
echo "  • Live resource tracking (fuel, cargo, credits)"
echo "  • Recent mission events with financial data"
echo "  • Mission statistics and profitability"
echo ""
echo "Controls:"
echo "  • Press 'q' to quit"
echo "  • Ctrl+C to stop simulation"
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

# Check if curses is available
python3 -c "import curses" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Warning: curses module not available"
    echo "   On Windows, try: pip install windows-curses"
    echo "   Falling back to scrolling client..."
    echo ""
    sleep 2
    python3 client.py
    exit 0
fi

echo "Starting dashboard in 3 seconds..."
sleep 3

# Clear screen and run TUI
clear
python3 client_curses.py