#!/usr/bin/env python3
"""
Space Simulation Python Client
Monitors simulation data from Rust simulator and displays formatted output
"""

import json
import subprocess
import sys
import time
import threading
from datetime import datetime
from typing import Dict, Any, List

class SimulationClient:
    def __init__(self):
        self.simulation_data = []
        self.latest_ship_data = None
        self.event_count = 0
        self.start_time = datetime.now()
        
    def run_simulator(self):
        """Run the Rust simulator and capture its output"""
        print("🚀 Starting Space Simulation Client")
        print("===================================")
        print(f"📅 Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # Run the Rust simulator
            process = subprocess.Popen(
                ['cargo', 'run'],
                cwd='.',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor output in real-time
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    self.process_simulator_output(line)
            
            # Wait for process to complete
            process.wait()
            
            if process.returncode != 0:
                stderr_output = process.stderr.read()
                print(f"❌ Simulator error: {stderr_output}")
            else:
                print("\n✅ Simulation completed successfully!")
                self.print_final_summary()
                
        except FileNotFoundError:
            print("❌ Error: Could not find cargo. Make sure Rust is installed.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error running simulator: {e}")
            sys.exit(1)
    
    def process_simulator_output(self, line: str):
        """Process a line of output from the simulator"""
        if line.startswith("SIM_DATA:"):
            # Extract JSON data
            json_data = line[9:]  # Remove "SIM_DATA:" prefix
            try:
                data = json.loads(json_data)
                self.process_simulation_data(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error: {e}")
        else:
            # Regular simulator output - just print it
            print(line)
    
    def process_simulation_data(self, data: Dict[str, Any]):
        """Process simulation data and display formatted information"""
        self.simulation_data.append(data)
        self.latest_ship_data = data.get('ship')
        
        # Display real-time ship status
        if self.latest_ship_data:
            self.display_ship_status(data['timestamp'], self.latest_ship_data)
            
        # Display recent events
        events = data.get('latest_events', [])
        if events:
            self.display_recent_events(events)
    
    def display_ship_status(self, timestamp: float, ship: Dict[str, Any]):
        """Display current ship status in a formatted way"""
        print(f"\n📡 Real-time Ship Data [T+{timestamp:.1f}h]")
        print("-" * 50)
        
        # Basic info
        print(f"🚢 Ship: {ship['name']} (ID: {ship['id']})")
        
        # Position
        pos = ship['position']
        print(f"📍 Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) - {pos['sector']}")
        
        # Status
        status = ship['status']
        if isinstance(status, dict):
            if 'Traveling' in status:
                dest = status['Traveling']['destination']
                eta = status['Traveling']['eta']
                print(f"🛸 Status: Traveling to ({dest['x']:.1f}, {dest['y']:.1f}, {dest['z']:.1f})")
                print(f"⏰ ETA: T+{eta:.1f}h")
            elif 'Docked' in status:
                print(f"🏭 Status: Docked at {status['Docked']['port']}")
            else:
                print(f"📊 Status: {list(status.keys())[0] if status else 'Unknown'}")
        else:
            print(f"📊 Status: {status}")
        
        # Resources
        print(f"💰 Credits: {ship['credits']:,.2f}")
        print(f"⛽ Fuel: {ship['fuel']:.1f}/{ship['max_fuel']:.1f} ({ship['fuel']/ship['max_fuel']*100:.1f}%)")
        
        # Cargo
        cargo = ship['cargo']
        cargo_percent = (cargo['used'] / cargo['capacity']) * 100
        print(f"📦 Cargo: {cargo['used']:.1f}/{cargo['capacity']:.1f} ({cargo_percent:.1f}%)")
        
        # Materials
        materials = cargo['materials']
        active_materials = {k: v for k, v in materials.items() if v > 0}
        if active_materials:
            print("   Materials:")
            for material, amount in active_materials.items():
                print(f"     • {material}: {amount:.1f} units")
        
        # Parts and robots
        parts = cargo['parts']
        active_parts = {k: v for k, v in parts.items() if v > 0}
        if active_parts:
            print("   Parts:")
            for part, count in active_parts.items():
                print(f"     • {part}: {count} units")
        
        if cargo['robots'] > 0:
            print(f"     • robots: {cargo['robots']} units")
        
        # Crew
        crew = ship['crew']
        total_crew = crew['engineers'] + crew['pilots'] + crew['miners']
        print(f"👥 Crew: {total_crew}/{crew['capacity']} (E:{crew['engineers']}, P:{crew['pilots']}, M:{crew['miners']})")
        
        print("-" * 50)
    
    def display_recent_events(self, events: List[Dict[str, Any]]):
        """Display recent simulation events"""
        print("\n📋 Recent Events:")
        for event in reversed(events[-3:]):  # Show last 3 events
            timestamp = event['timestamp']
            event_type = event['event_type']
            description = event['description']
            
            # Color code by event type
            emoji = self.get_event_emoji(event_type)
            print(f"   {emoji} [T+{timestamp:.1f}h] {description}")
            
            # Show additional data for certain events
            if event_type in ['LoadCargo', 'UnloadCargo', 'Transaction']:
                data = event.get('data', {})
                if 'cost' in data:
                    print(f"      💸 Cost: {data['cost']:,.2f} credits")
                if 'revenue' in data:
                    print(f"      💰 Revenue: {data['revenue']:,.2f} credits")
                if 'fuel_cost' in data:
                    print(f"      ⛽ Fuel Cost: {data['fuel_cost']:.1f} units")
    
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
    
    def print_final_summary(self):
        """Print final simulation summary"""
        runtime = datetime.now() - self.start_time
        
        print("\n" + "="*60)
        print("🎯 SIMULATION COMPLETE - FINAL SUMMARY")
        print("="*60)
        
        print(f"⏱️  Real-time Runtime: {runtime.total_seconds():.1f} seconds")
        print(f"📊 Data Points Collected: {len(self.simulation_data)}")
        
        if self.latest_ship_data:
            ship = self.latest_ship_data
            print(f"\n🚢 Final Ship Status - {ship['name']}:")
            print(f"   💰 Final Credits: {ship['credits']:,.2f}")
            print(f"   ⛽ Fuel Remaining: {ship['fuel']:.1f}/{ship['max_fuel']:.1f}")
            print(f"   📦 Cargo Load: {ship['cargo']['used']:.1f}/{ship['cargo']['capacity']:.1f}")
            
            # Calculate mission profit (rough estimate)
            initial_credits = 50000.0  # Known starting credits
            profit = ship['credits'] - initial_credits
            print(f"   📈 Mission Profit: {profit:,.2f} credits ({profit/initial_credits*100:+.1f}%)")
        
        print("\n📈 Mission Performance Metrics:")
        
        # Analyze events for performance metrics
        events_by_type = {}
        total_revenue = 0.0
        total_expenses = 0.0
        
        for data_point in self.simulation_data:
            events = data_point.get('latest_events', [])
            for event in events:
                event_type = event['event_type']
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                
                # Track financial metrics
                event_data = event.get('data', {})
                if 'revenue' in event_data:
                    total_revenue += event_data['revenue']
                if 'cost' in event_data:
                    total_expenses += event_data['cost']
        
        print(f"   💰 Total Revenue: {total_revenue:,.2f} credits")
        print(f"   💸 Total Expenses: {total_expenses:,.2f} credits")
        print(f"   💵 Net Profit: {total_revenue - total_expenses:,.2f} credits")
        
        print(f"\n📋 Event Summary:")
        for event_type, count in sorted(events_by_type.items()):
            emoji = self.get_event_emoji(event_type)
            print(f"   {emoji} {event_type}: {count} events")
        
        print("\n🎉 Thank you for watching the Space Resource Management Simulation!")
        print("="*60)

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Space Simulation Python Client")
        print("Usage: python3 client.py")
        print("This client monitors the Rust space simulator and displays formatted output.")
        return
    
    client = SimulationClient()
    
    try:
        client.run_simulator()
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user")
        print("Final data collected:", len(client.simulation_data), "data points")
    except Exception as e:
        print(f"\n❌ Client error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()