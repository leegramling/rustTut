#!/usr/bin/env python3
"""
Space Simulation Python Client with Curses TUI
Real-time dashboard view without scrolling text
"""

import json
import subprocess
import sys
import time
import threading
import curses
from datetime import datetime
from typing import Dict, Any, List, Optional

class SimulationTUI:
    def __init__(self):
        self.simulation_data = []
        self.latest_ship_data = None
        self.latest_events = []
        self.start_time = datetime.now()
        self.stdscr = None
        self.process = None
        self.running = True
        
    def init_curses(self):
        """Initialize curses display"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)  # Non-blocking input
        curses.curs_set(0)  # Hide cursor
        
        # Initialize colors if available
        if curses.has_colors():
            curses.start_color()
            curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Success/positive
            curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # Error/negative
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning/info
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Headers
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Special
            curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)    # Values
        
    def cleanup_curses(self):
        """Cleanup curses display"""
        if self.stdscr:
            try:
                curses.nocbreak()
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()
            except curses.error:
                pass  # Ignore cleanup errors
    
    def draw_header(self):
        """Draw the header section"""
        height, width = self.stdscr.getmaxyx()
        
        # Title
        title = "🚀 SPACE RESOURCE MANAGEMENT SIMULATION"
        subtitle = f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.stdscr.addstr(0, (width - len(title)) // 2, title, 
                          curses.color_pair(4) | curses.A_BOLD)
        self.stdscr.addstr(1, (width - len(subtitle)) // 2, subtitle, 
                          curses.color_pair(3))
        
        # Separator line
        self.stdscr.addstr(2, 0, "=" * width, curses.color_pair(4))
        
        return 3  # Return next available row
    
    def draw_ship_status(self, start_row: int) -> int:
        """Draw ship status panel"""
        if not self.latest_ship_data:
            self.stdscr.addstr(start_row, 2, "⏳ Waiting for simulation data...", 
                             curses.color_pair(3))
            return start_row + 2
        
        ship = self.latest_ship_data
        current_row = start_row
        
        # Ship info header
        self.stdscr.addstr(current_row, 2, f"🚢 {ship['name']} (ID: {ship['id']})", 
                          curses.color_pair(4) | curses.A_BOLD)
        current_row += 1
        
        # Position
        pos = ship['position']
        self.stdscr.addstr(current_row, 4, 
                          f"📍 Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) - {pos['sector']}")
        current_row += 1
        
        # Status
        status = ship['status']
        status_text = "📊 Status: "
        if isinstance(status, dict):
            if 'Traveling' in status:
                dest = status['Traveling']['destination']
                eta = status['Traveling']['eta']
                status_text += f"🛸 Traveling to ({dest['x']:.1f}, {dest['y']:.1f}, {dest['z']:.1f}) ETA: T+{eta:.1f}h"
            elif 'Docked' in status:
                status_text += f"🏭 Docked at {status['Docked']['port']}"
            else:
                status_text += f"{list(status.keys())[0] if status else 'Unknown'}"
        else:
            status_text += str(status)
        
        self.stdscr.addstr(current_row, 4, status_text)
        current_row += 1
        
        # Resources
        credits_color = curses.color_pair(1) if ship['credits'] >= 50000 else curses.color_pair(2)
        self.stdscr.addstr(current_row, 4, f"💰 Credits: {ship['credits']:,.2f}", credits_color)
        current_row += 1
        
        fuel_percent = ship['fuel'] / ship['max_fuel'] * 100
        fuel_color = curses.color_pair(1) if fuel_percent > 50 else curses.color_pair(3) if fuel_percent > 25 else curses.color_pair(2)
        self.stdscr.addstr(current_row, 4, 
                          f"⛽ Fuel: {ship['fuel']:.1f}/{ship['max_fuel']:.1f} ({fuel_percent:.1f}%)", 
                          fuel_color)
        current_row += 1
        
        # Cargo
        cargo = ship['cargo']
        cargo_percent = (cargo['used'] / cargo['capacity']) * 100
        cargo_color = curses.color_pair(2) if cargo_percent > 90 else curses.color_pair(3) if cargo_percent > 70 else curses.color_pair(1)
        self.stdscr.addstr(current_row, 4, 
                          f"📦 Cargo: {cargo['used']:.1f}/{cargo['capacity']:.1f} ({cargo_percent:.1f}%)", 
                          cargo_color)
        current_row += 1
        
        # Materials
        materials = cargo['materials']
        active_materials = {k: v for k, v in materials.items() if v > 0}
        if active_materials:
            self.stdscr.addstr(current_row, 6, "Materials:", curses.color_pair(3))
            current_row += 1
            for material, amount in list(active_materials.items())[:3]:  # Limit to 3 lines
                self.stdscr.addstr(current_row, 8, f"• {material}: {amount:.1f} units")
                current_row += 1
        
        # Parts and robots
        parts = cargo['parts']
        active_parts = {k: v for k, v in parts.items() if v > 0}
        if active_parts:
            self.stdscr.addstr(current_row, 6, "Parts:", curses.color_pair(3))
            current_row += 1
            for part, count in list(active_parts.items())[:2]:  # Limit to 2 lines
                self.stdscr.addstr(current_row, 8, f"• {part}: {count} units")
                current_row += 1
        
        if cargo['robots'] > 0:
            self.stdscr.addstr(current_row, 8, f"• robots: {cargo['robots']} units")
            current_row += 1
        
        # Crew
        crew = ship['crew']
        total_crew = crew['engineers'] + crew['pilots'] + crew['miners']
        self.stdscr.addstr(current_row, 4, 
                          f"👥 Crew: {total_crew}/{crew['capacity']} (E:{crew['engineers']}, P:{crew['pilots']}, M:{crew['miners']})")
        current_row += 2
        
        return current_row
    
    def draw_events(self, start_row: int) -> int:
        """Draw recent events panel"""
        height, width = self.stdscr.getmaxyx()
        
        self.stdscr.addstr(start_row, 2, "📋 Recent Events:", 
                          curses.color_pair(4) | curses.A_BOLD)
        current_row = start_row + 1
        
        if not self.latest_events:
            self.stdscr.addstr(current_row, 4, "No events yet...", curses.color_pair(3))
            return current_row + 2
        
        # Show last 8 events (or fewer if screen is small)
        max_events = min(8, height - current_row - 3)
        recent_events = self.latest_events[-max_events:]
        
        for event in reversed(recent_events):
            if current_row >= height - 2:
                break
                
            timestamp = event['timestamp']
            event_type = event['event_type']
            description = event['description']
            
            # Truncate description if too long
            max_desc_length = width - 25
            if len(description) > max_desc_length:
                description = description[:max_desc_length-3] + "..."
            
            # Color code by event type
            emoji = self.get_event_emoji(event_type)
            color = self.get_event_color(event_type)
            
            event_text = f"{emoji} [T+{timestamp:.1f}h] {description}"
            self.stdscr.addstr(current_row, 4, event_text, color)
            current_row += 1
            
            # Show financial data on next line if available
            if current_row < height - 2:
                data = event.get('data', {})
                financial_info = ""
                if 'cost' in data:
                    financial_info += f"💸 Cost: {data['cost']:,.2f} credits  "
                if 'revenue' in data:
                    financial_info += f"💰 Revenue: {data['revenue']:,.2f} credits  "
                if 'fuel_cost' in data:
                    financial_info += f"⛽ Fuel: {data['fuel_cost']:.1f} units"
                
                if financial_info and len(financial_info) < width - 10:
                    self.stdscr.addstr(current_row, 8, financial_info, curses.color_pair(6))
                    current_row += 1
        
        return current_row + 1
    
    def draw_mission_stats(self, start_row: int) -> int:
        """Draw mission statistics"""
        height, width = self.stdscr.getmaxyx()
        
        if start_row >= height - 3:
            return start_row
        
        self.stdscr.addstr(start_row, 2, "📈 Mission Statistics:", 
                          curses.color_pair(4) | curses.A_BOLD)
        current_row = start_row + 1
        
        runtime = datetime.now() - self.start_time
        self.stdscr.addstr(current_row, 4, f"⏱️  Runtime: {runtime.total_seconds():.1f}s")
        current_row += 1
        
        self.stdscr.addstr(current_row, 4, f"📊 Data Points: {len(self.simulation_data)}")
        current_row += 1
        
        if self.latest_ship_data:
            ship = self.latest_ship_data
            initial_credits = 50000.0
            profit = ship['credits'] - initial_credits
            profit_color = curses.color_pair(1) if profit >= 0 else curses.color_pair(2)
            self.stdscr.addstr(current_row, 4, 
                              f"💵 Mission Profit: {profit:,.2f} credits ({profit/initial_credits*100:+.1f}%)", 
                              profit_color)
            current_row += 1
        
        return current_row
    
    def draw_footer(self):
        """Draw footer with controls"""
        height, width = self.stdscr.getmaxyx()
        footer_text = "Press 'q' to quit | Ctrl+C to stop simulation"
        
        try:
            self.stdscr.addstr(height - 1, (width - len(footer_text)) // 2, footer_text, 
                              curses.color_pair(3))
        except curses.error:
            pass  # Ignore if we can't draw at the bottom
    
    def get_event_emoji(self, event_type: str) -> str:
        """Get emoji for event type"""
        emoji_map = {
            'Travel': '🚀',
            'Dock': '🏭',
            'Undock': '🚁',
            'LoadCargo': '📥',
            'UnloadCargo': '📤',
            'CrewTransfer': '👥',
            'Transaction': '💰',
            'FuelUpdate': '⛽',
            'StatusChange': '📊',
        }
        return emoji_map.get(event_type, '📋')
    
    def get_event_color(self, event_type: str) -> int:
        """Get color for event type"""
        color_map = {
            'Travel': curses.color_pair(4),
            'Dock': curses.color_pair(1),
            'Undock': curses.color_pair(1),
            'LoadCargo': curses.color_pair(3),
            'UnloadCargo': curses.color_pair(1),
            'CrewTransfer': curses.color_pair(5),
            'Transaction': curses.color_pair(1),
            'FuelUpdate': curses.color_pair(6),
            'StatusChange': curses.color_pair(4),
        }
        return color_map.get(event_type, curses.color_pair(0))
    
    def update_display(self):
        """Update the entire display"""
        try:
            self.stdscr.clear()
            
            current_row = self.draw_header()
            current_row = self.draw_ship_status(current_row)
            current_row = self.draw_events(current_row)
            current_row = self.draw_mission_stats(current_row)
            
            self.draw_footer()
            
            self.stdscr.refresh()
        except curses.error:
            # Handle display errors gracefully
            pass
    
    def process_simulator_output(self, line: str):
        """Process a line of output from the simulator"""
        if line.startswith("SIM_DATA:"):
            # Extract JSON data
            json_data = line[9:]  # Remove "SIM_DATA:" prefix
            try:
                data = json.loads(json_data)
                self.simulation_data.append(data)
                self.latest_ship_data = data.get('ship')
                
                # Update events
                events = data.get('latest_events', [])
                if events:
                    # Add new events to our list, avoiding duplicates
                    for event in events:
                        if not any(e['timestamp'] == event['timestamp'] and 
                                 e['description'] == event['description'] 
                                 for e in self.latest_events):
                            self.latest_events.append(event)
                    
                    # Keep only last 20 events
                    self.latest_events = self.latest_events[-20:]
                
                # Update display
                self.update_display()
                
            except json.JSONDecodeError:
                pass  # Ignore JSON errors silently in TUI mode
    
    def run_simulator_thread(self):
        """Run the Rust simulator in a separate thread"""
        try:
            self.process = subprocess.Popen(
                ['cargo', 'run'],
                cwd='.',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                line = line.strip()
                if line:
                    self.process_simulator_output(line)
            
            # Wait for process to complete
            self.process.wait()
            
        except Exception as e:
            # In TUI mode, we can't easily show errors, so just exit
            self.running = False
    
    def run(self):
        """Main run loop"""
        try:
            self.init_curses()
            
            # Start simulator in background thread
            simulator_thread = threading.Thread(target=self.run_simulator_thread)
            simulator_thread.daemon = True
            simulator_thread.start()
            
            # Initial display
            self.update_display()
            
            # Main event loop
            while self.running:
                # Check for user input
                try:
                    key = self.stdscr.getch()
                    if key == ord('q') or key == ord('Q'):
                        self.running = False
                        break
                except curses.error:
                    pass  # No input available
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
                # Check if simulator thread is still running
                if not simulator_thread.is_alive():
                    # Add a final message
                    height, width = self.stdscr.getmaxyx()
                    try:
                        self.stdscr.addstr(height // 2, (width - 30) // 2, 
                                         "🎯 SIMULATION COMPLETE!", 
                                         curses.color_pair(1) | curses.A_BOLD)
                        self.stdscr.addstr(height // 2 + 1, (width - 25) // 2, 
                                         "Press 'q' to exit", 
                                         curses.color_pair(3))
                        self.stdscr.refresh()
                    except curses.error:
                        pass
                    
                    # Wait for user to press 'q'
                    while True:
                        try:
                            key = self.stdscr.getch()
                            if key == ord('q') or key == ord('Q'):
                                break
                        except curses.error:
                            pass
                        time.sleep(0.1)
                    break
            
        except KeyboardInterrupt:
            self.running = False
        finally:
            # Cleanup
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.cleanup_curses()
            
            # Show final summary
            print("\n🎯 Space Simulation Complete!")
            if self.latest_ship_data:
                ship = self.latest_ship_data
                print(f"Final Credits: {ship['credits']:,.2f}")
                initial_credits = 50000.0
                profit = ship['credits'] - initial_credits
                print(f"Mission Profit: {profit:,.2f} credits ({profit/initial_credits*100:+.1f}%)")
            print(f"Data Points Collected: {len(self.simulation_data)}")
            print("Thank you for watching the simulation!")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Space Simulation Python Client with Curses TUI")
        print("Usage: python3 client_curses.py")
        print("Controls:")
        print("  q - Quit the application")
        print("  Ctrl+C - Stop simulation")
        return
    
    # Check for required dependencies
    try:
        import curses
    except ImportError:
        print("❌ Error: curses module not available")
        print("On Windows, try: pip install windows-curses")
        print("Falling back to regular client...")
        import subprocess
        subprocess.run([sys.executable, "client.py"])
        return
    
    # Check if we have a proper terminal
    if not sys.stdout.isatty():
        print("❌ Error: TUI mode requires a proper terminal")
        print("Try running in a terminal or use: python3 client.py")
        sys.exit(1)
    
    client = SimulationTUI()
    
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Client error: {e}")
        print("Try using the regular client: python3 client.py")
        sys.exit(1)

if __name__ == "__main__":
    main()