# Python Client Documentation

## Overview

The Space Simulation Python clients provide two distinct interfaces for monitoring the Rust simulation engine: a **scrolling console client** (`client.py`) and a **real-time TUI dashboard** (`client_curses.py`). This document explains the implementation details, design patterns, and Python-specific techniques used in both clients.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Scrolling Console Client](#scrolling-console-client)
3. [TUI Dashboard Client](#tui-dashboard-client)
4. [Communication Protocol](#communication-protocol)
5. [Data Processing](#data-processing)
6. [Python Programming Patterns](#python-programming-patterns)
7. [Error Handling](#error-handling)
8. [Performance Considerations](#performance-considerations)

## Architecture Overview

### System Design

Both clients follow a similar architectural pattern with different presentation layers:

```
┌─────────────────────────┐
│    Subprocess Manager   │  ← Manages Rust simulator
├─────────────────────────┤
│   Communication Layer   │  ← Parses JSON protocol
├─────────────────────────┤
│     Data Processing     │  ← Transforms simulation data
├─────────────────────────┤
│   Presentation Layer    │  ← Console/TUI output
└─────────────────────────┘
```

### Key Components

| Component | Scrolling Client | TUI Client | Purpose |
|-----------|------------------|------------|---------|
| Process Management | `subprocess.Popen` | `subprocess.Popen` | Rust simulator execution |
| Data Parsing | JSON extraction | JSON extraction | Protocol handling |
| Display Logic | Print statements | Curses panels | User interface |
| Event Loop | Linear processing | Non-blocking loop | Control flow |
| State Management | Event accumulation | Live state updates | Data retention |

## Scrolling Console Client

### Core Implementation

```python
class SimulationClient:
    def __init__(self):
        self.simulation_data = []
        self.latest_ship_data = None
        self.event_count = 0
        self.start_time = datetime.now()
```

**Design Philosophy:**
- **Stateful accumulation**: Collects all simulation data for analysis
- **Comprehensive logging**: Preserves complete mission transcript
- **Post-processing analysis**: Rich summary after completion
- **Linear flow**: Simple, predictable execution model

### Process Management

```python
def run_simulator(self):
    """Run the Rust simulator and capture its output"""
    try:
        # Start Rust process with proper configuration
        process = subprocess.Popen(
            ['cargo', 'run'],                    # Command to execute
            cwd='.',                             # Working directory
            stdout=subprocess.PIPE,              # Capture stdout
            stderr=subprocess.PIPE,              # Capture stderr
            text=True,                          # Text mode (not bytes)
            bufsize=1,                          # Line buffering
            universal_newlines=True             # Handle line endings
        )
        
        # Real-time output processing
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                self.process_simulator_output(line)
        
        process.wait()  # Wait for completion
        
    except FileNotFoundError:
        print("❌ Error: Could not find cargo. Make sure Rust is installed.")
        sys.exit(1)
```

**Subprocess Patterns:**
- **Line buffering**: `bufsize=1` ensures immediate output
- **Text mode**: Automatic encoding handling
- **Blocking I/O**: Simple sequential processing
- **Error separation**: Separate stdout/stderr handling
- **Iterator pattern**: `iter(readline, '')` for clean line processing

### Data Processing Pipeline

```python
def process_simulator_output(self, line: str):
    """Process a line of output from the simulator"""
    if line.startswith("SIM_DATA:"):
        # Extract and parse JSON data
        json_data = line[9:]  # Remove "SIM_DATA:" prefix
        try:
            data = json.loads(json_data)
            self.process_simulation_data(data)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error: {e}")
    else:
        # Regular simulator output - just print it
        print(line)
```

**Protocol Handling:**
- **Prefix filtering**: Separates data from regular output
- **JSON parsing**: Structured data extraction
- **Error resilience**: Continues on malformed JSON
- **Pass-through**: Regular output preserved for user

### Rich Display Formatting

```python
def display_ship_status(self, timestamp: float, ship: Dict[str, Any]):
    """Display current ship status in a formatted way"""
    print(f"\n📡 Real-time Ship Data [T+{timestamp:.1f}h]")
    print("-" * 50)
    
    # Position with semantic formatting
    pos = ship['position']
    print(f"📍 Position: ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) - {pos['sector']}")
    
    # Status with conditional formatting
    status = ship['status']
    if isinstance(status, dict):
        if 'Traveling' in status:
            dest = status['Traveling']['destination']
            eta = status['Traveling']['eta']
            print(f"🛸 Status: Traveling to ({dest['x']:.1f}, {dest['y']:.1f}, {dest['z']:.1f})")
            print(f"⏰ ETA: T+{eta:.1f}h")
        elif 'Docked' in status:
            print(f"🏭 Status: Docked at {status['Docked']['port']}")
    
    # Resource display with calculations
    cargo = ship['cargo']
    cargo_percent = (cargo['used'] / cargo['capacity']) * 100
    print(f"📦 Cargo: {cargo['used']:.1f}/{cargo['capacity']:.1f} ({cargo_percent:.1f}%)")
```

**Formatting Techniques:**
- **Unicode emojis**: Visual categorization and appeal
- **Conditional formatting**: Different displays for different states
- **Calculated metrics**: Percentage displays for intuitive understanding
- **Consistent alignment**: Visual structure with separators
- **Precision control**: Appropriate decimal places for different data types

### Comprehensive Analytics

```python
def print_final_summary(self):
    """Print final simulation summary"""
    runtime = datetime.now() - self.start_time
    
    print("\n" + "="*60)
    print("🎯 SIMULATION COMPLETE - FINAL SUMMARY")
    print("="*60)
    
    # Performance metrics
    print(f"⏱️  Real-time Runtime: {runtime.total_seconds():.1f} seconds")
    print(f"📊 Data Points Collected: {len(self.simulation_data)}")
    
    # Financial analysis
    events_by_type = {}
    total_revenue = 0.0
    total_expenses = 0.0
    
    for data_point in self.simulation_data:
        events = data_point.get('latest_events', [])
        for event in events:
            event_type = event['event_type']
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            
            # Financial tracking
            event_data = event.get('data', {})
            if 'revenue' in event_data:
                total_revenue += event_data['revenue']
            if 'cost' in event_data:
                total_expenses += event_data['cost']
    
    print(f"   💰 Total Revenue: {total_revenue:,.2f} credits")
    print(f"   💸 Total Expenses: {total_expenses:,.2f} credits")
    print(f"   💵 Net Profit: {total_revenue - total_expenses:,.2f} credits")
```

**Analytics Features:**
- **Time tracking**: Real-world execution time
- **Data quantification**: Volume of information processed
- **Financial modeling**: Complete profit/loss analysis
- **Event categorization**: Breakdown by operation type
- **Professional formatting**: Business-style reporting

## TUI Dashboard Client

### Curses Framework Integration

```python
def init_curses(self):
    """Initialize curses display"""
    self.stdscr = curses.initscr()
    curses.noecho()                    # Don't echo keystrokes
    curses.cbreak()                    # React to keys immediately
    self.stdscr.keypad(True)          # Enable special keys
    self.stdscr.nodelay(True)         # Non-blocking input
    curses.curs_set(0)                # Hide cursor
    
    # Color system initialization
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Success
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)     # Error
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Headers
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Special
        curses.init_pair(6, curses.COLOR_BLUE, curses.COLOR_BLACK)    # Values
```

**Curses Configuration:**
- **Input handling**: Immediate key response without Enter
- **Display control**: Hidden cursor, no echo
- **Color management**: Semantic color pairs for different data types
- **Non-blocking I/O**: UI remains responsive during simulation

### Thread-Safe Architecture

```python
class SimulationTUI:
    def __init__(self):
        self.simulation_data = []
        self.latest_ship_data = None
        self.latest_events = []
        self.running = True              # Thread coordination flag
        
    def run_simulator_thread(self):
        """Run the Rust simulator in a separate thread"""
        try:
            self.process = subprocess.Popen(/* same as console client */)
            
            # Monitor output
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:     # Check termination flag
                    break
                line = line.strip()
                if line:
                    self.process_simulator_output(line)
            
            self.process.wait()
        except Exception as e:
            self.running = False
```

**Threading Model:**
- **Background simulation**: Rust process runs in separate thread
- **UI thread**: Main thread handles display and input
- **Coordination flags**: `self.running` for clean shutdown
- **Exception isolation**: Simulation errors don't crash UI

### Real-Time Dashboard Layout

```python
def update_display(self):
    """Update the entire display"""
    try:
        self.stdscr.clear()
        
        # Hierarchical panel rendering
        current_row = self.draw_header()
        current_row = self.draw_ship_status(current_row)
        current_row = self.draw_events(current_row)
        current_row = self.draw_mission_stats(current_row)
        
        self.draw_footer()
        
        self.stdscr.refresh()    # Actually update the screen
    except curses.error:
        pass  # Handle display errors gracefully
```

**Display Architecture:**
- **Panel system**: Modular rendering functions
- **Row tracking**: Automatic layout management
- **Full refresh**: Clear and redraw for consistency
- **Error resilience**: Graceful handling of display issues

### Dynamic Color Coding

```python
def draw_ship_status(self, start_row: int) -> int:
    """Draw ship status panel with dynamic coloring"""
    ship = self.latest_ship_data
    
    # Dynamic color selection based on values
    credits_color = (curses.color_pair(1) if ship['credits'] >= 50000 
                    else curses.color_pair(2))
    
    fuel_percent = ship['fuel'] / ship['max_fuel'] * 100
    fuel_color = (curses.color_pair(1) if fuel_percent > 50 
                 else curses.color_pair(3) if fuel_percent > 25 
                 else curses.color_pair(2))
    
    cargo_percent = (ship['cargo']['used'] / ship['cargo']['capacity']) * 100
    cargo_color = (curses.color_pair(2) if cargo_percent > 90 
                  else curses.color_pair(3) if cargo_percent > 70 
                  else curses.color_pair(1))
    
    # Apply colors to display
    self.stdscr.addstr(row, 4, f"💰 Credits: {ship['credits']:,.2f}", credits_color)
    self.stdscr.addstr(row+1, 4, f"⛽ Fuel: {fuel_percent:.1f}%", fuel_color)
    self.stdscr.addstr(row+2, 4, f"📦 Cargo: {cargo_percent:.1f}%", cargo_color)
```

**Color Strategy:**
- **Semantic colors**: Green=good, Yellow=warning, Red=critical
- **Threshold-based**: Different colors for different value ranges
- **Visual hierarchy**: Headers, values, and status use different colors
- **Accessibility**: High contrast combinations

### Event Management System

```python
def process_simulation_data(self, data: Dict[str, Any]):
    """Process simulation data and display formatted information"""
    self.simulation_data.append(data)
    self.latest_ship_data = data.get('ship')
    
    # Event deduplication and management
    events = data.get('latest_events', [])
    if events:
        for event in events:
            # Avoid duplicate events
            if not any(e['timestamp'] == event['timestamp'] and 
                     e['description'] == event['description'] 
                     for e in self.latest_events):
                self.latest_events.append(event)
        
        # Sliding window - keep only recent events
        self.latest_events = self.latest_events[-20:]
    
    # Trigger display update
    self.update_display()
```

**Event Processing:**
- **Deduplication**: Prevents showing the same event multiple times
- **Sliding window**: Maintains fixed-size event history
- **Incremental updates**: Only processes new events
- **Immediate refresh**: Real-time display updates

### Input Handling and Control

```python
def run(self):
    """Main run loop"""
    try:
        self.init_curses()
        
        # Start background thread
        simulator_thread = threading.Thread(target=self.run_simulator_thread)
        simulator_thread.daemon = True
        simulator_thread.start()
        
        # Main event loop
        while self.running:
            # Non-blocking input check
            try:
                key = self.stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                    break
            except curses.error:
                pass  # No input available
            
            time.sleep(0.1)  # Prevent excessive CPU usage
            
        # Cleanup
        if self.process and self.process.poll() is None:
            self.process.terminate()
            
    finally:
        self.cleanup_curses()
```

**Control Flow:**
- **Event-driven**: Responds to user input and simulation events
- **Non-blocking**: UI remains responsive
- **Graceful shutdown**: Proper cleanup of resources
- **Exception safety**: Finally block ensures cleanup

## Communication Protocol

### JSON Protocol Implementation

Both clients use the same protocol for communication:

```python
def process_simulator_output(self, line: str):
    """Parse simulator output for JSON data"""
    if line.startswith("SIM_DATA:"):
        json_data = line[9:]  # Strip prefix
        try:
            data = json.loads(json_data)
            return data
        except json.JSONDecodeError as e:
            # Log error but continue processing
            if self.debug_mode:
                print(f"JSON Error: {e}")
            return None
    return None
```

### Protocol Advantages

| Feature | Benefit | Implementation |
|---------|---------|----------------|
| **Prefix filtering** | Separate data from logs | `line.startswith("SIM_DATA:")` |
| **JSON structure** | Rich, nested data | `json.loads()` with error handling |
| **Streaming** | Real-time updates | Line-by-line processing |
| **Language agnostic** | Works with any client | Standard JSON format |
| **Extensible** | Easy to add new fields | Nested object structure |

### Data Structure Handling

```python
def extract_ship_data(self, data: Dict[str, Any]) -> Optional[Dict]:
    """Safely extract ship data with validation"""
    ship = data.get('ship')
    if not ship:
        return None
    
    # Validate required fields
    required_fields = ['id', 'name', 'position', 'credits', 'fuel']
    if not all(field in ship for field in required_fields):
        return None
    
    # Type validation
    try:
        position = ship['position']
        assert isinstance(position, dict)
        assert all(coord in position for coord in ['x', 'y', 'z', 'sector'])
        assert isinstance(ship['credits'], (int, float))
        assert isinstance(ship['fuel'], (int, float))
        
        return ship
    except (AssertionError, KeyError, TypeError):
        return None
```

**Validation Strategy:**
- **Presence checking**: Ensure required fields exist
- **Type validation**: Verify expected data types
- **Structure validation**: Nested object validation
- **Graceful degradation**: Continue operation with partial data

## Python Programming Patterns

### Object-Oriented Design

```python
class SimulationClient:
    """Scrolling console client for space simulation"""
    
    def __init__(self):
        # State initialization
        self.simulation_data: List[Dict] = []
        self.latest_ship_data: Optional[Dict] = None
        self.start_time: datetime = datetime.now()
    
    def run_simulator(self) -> None:
        """Main entry point - orchestrates entire simulation"""
        # Implementation here
    
    def process_simulation_data(self, data: Dict[str, Any]) -> None:
        """Process individual data points"""
        # Implementation here
    
    def print_final_summary(self) -> None:
        """Generate comprehensive final report"""
        # Implementation here
```

**OOP Benefits:**
- **Encapsulation**: State and behavior bundled together
- **Type hints**: Clear interface documentation
- **Method organization**: Logical grouping of functionality
- **Inheritance potential**: Base class for different client types

### Error Handling Patterns

```python
def run_simulator(self):
    """Run with comprehensive error handling"""
    try:
        # Primary operation
        process = subprocess.Popen([/* ... */])
        for line in iter(process.stdout.readline, ''):
            self.process_line(line)
            
    except FileNotFoundError:
        # Specific error handling
        print("❌ Error: Could not find cargo. Make sure Rust is installed.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        # User interruption
        print("\n\n⏹️  Simulation interrupted by user")
        
    except Exception as e:
        # Catch-all with context
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup regardless of exit path
        self.cleanup_resources()
```

**Exception Hierarchy:**
- **Specific exceptions first**: Handle known conditions explicitly
- **KeyboardInterrupt**: Special handling for user interruption
- **Generic Exception**: Catch-all for unexpected conditions
- **Finally block**: Guaranteed cleanup

### Context Management

```python
class CursesContext:
    """Context manager for curses initialization/cleanup"""
    
    def __enter__(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.curs_set(0)
        return self.stdscr
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            curses.nocbreak()
            self.stdscr.keypad(False)
            curses.echo()
            curses.endwin()
        except:
            pass  # Ignore cleanup errors

# Usage
def run_with_context(self):
    with CursesContext() as stdscr:
        self.stdscr = stdscr
        # Curses operations here
        # Automatic cleanup on exit
```

### Data Processing Pipelines

```python
def analyze_financial_data(self, events: List[Dict]) -> Dict[str, float]:
    """Process events into financial summary using functional patterns"""
    
    # Filter and map in pipeline style
    financial_events = (
        event for event in events 
        if event.get('event_type') in ['LoadCargo', 'UnloadCargo', 'Transaction']
    )
    
    revenues = sum(
        event.get('data', {}).get('revenue', 0) 
        for event in financial_events
    )
    
    costs = sum(
        event.get('data', {}).get('cost', 0) + 
        event.get('data', {}).get('fee', 0)
        for event in financial_events
    )
    
    return {
        'total_revenue': revenues,
        'total_costs': costs,
        'net_profit': revenues - costs,
        'profit_margin': revenues / costs if costs > 0 else float('inf')
    }
```

**Functional Programming:**
- **Generator expressions**: Memory-efficient filtering
- **Comprehensions**: Concise data transformations
- **Built-in functions**: `sum()`, `max()`, `min()` for aggregation
- **Pipeline style**: Chain operations for clarity

## Error Handling

### Graceful Degradation Strategy

```python
def display_ship_status(self, timestamp: float, ship: Dict[str, Any]):
    """Display ship status with robust error handling"""
    try:
        # Primary display logic
        print(f"📊 Ship: {ship['name']} (ID: {ship['id']})")
        
        # Safe nested access with defaults
        position = ship.get('position', {})
        x = position.get('x', 0.0)
        y = position.get('y', 0.0)
        z = position.get('z', 0.0)
        sector = position.get('sector', 'Unknown')
        
        print(f"📍 Position: ({x:.1f}, {y:.1f}, {z:.1f}) - {sector}")
        
    except KeyError as e:
        print(f"⚠️  Missing ship data field: {e}")
        
    except (TypeError, ValueError) as e:
        print(f"⚠️  Invalid ship data format: {e}")
        
    except Exception as e:
        print(f"⚠️  Unexpected error displaying ship: {e}")
```

**Error Resilience Techniques:**
- **Safe dictionary access**: `.get()` with defaults
- **Type-specific handling**: Different strategies for different error types
- **Partial success**: Display what can be displayed
- **User feedback**: Clear error messages for debugging

### Validation and Sanitization

```python
def sanitize_display_string(self, text: str, max_length: int = 50) -> str:
    """Sanitize strings for safe terminal display"""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text

def validate_numeric_value(self, value: Any, default: float = 0.0) -> float:
    """Safely convert values to numeric"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
```

## Performance Considerations

### Memory Management

```python
class SimulationClient:
    def __init__(self):
        # Bounded collections to prevent memory leaks
        self.simulation_data: List[Dict] = []
        self.max_data_points: int = 1000  # Limit historical data
    
    def add_data_point(self, data: Dict):
        """Add data point with memory management"""
        self.simulation_data.append(data)
        
        # Sliding window to prevent unbounded growth
        if len(self.simulation_data) > self.max_data_points:
            # Remove oldest data points
            excess = len(self.simulation_data) - self.max_data_points
            self.simulation_data = self.simulation_data[excess:]
```

### I/O Optimization

```python
def process_output_efficiently(self, process):
    """Optimized output processing"""
    # Use buffered reading for better performance
    buffer_size = 8192
    partial_line = ""
    
    while True:
        # Read in chunks rather than line by line
        chunk = process.stdout.read(buffer_size)
        if not chunk:
            break
            
        # Handle partial lines at buffer boundaries
        lines = (partial_line + chunk).split('\n')
        partial_line = lines[-1]  # Save incomplete line
        
        # Process complete lines
        for line in lines[:-1]:
            self.process_line(line.strip())
```

### Display Optimization (TUI)

```python
def optimized_display_update(self):
    """Update only changed portions of display"""
    # Track what needs updating
    if self.ship_data_changed:
        self.update_ship_panel()
        self.ship_data_changed = False
    
    if self.events_changed:
        self.update_events_panel()
        self.events_changed = False
    
    # Only refresh if something changed
    if self.ship_data_changed or self.events_changed:
        self.stdscr.refresh()
```

## Testing and Debugging

### Logging Infrastructure

```python
import logging

# Configure logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation_client.log'),
        logging.StreamHandler()
    ]
)

class SimulationClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_line(self, line: str):
        self.logger.debug(f"Processing line: {line[:100]}...")  # Truncate long lines
        
        if line.startswith("SIM_DATA:"):
            self.logger.info("Received simulation data")
            # Process data...
        else:
            self.logger.debug("Regular output line")
```

### Unit Testing Support

```python
import unittest
from unittest.mock import Mock, patch

class TestSimulationClient(unittest.TestCase):
    def setUp(self):
        self.client = SimulationClient()
    
    def test_json_parsing(self):
        """Test JSON data parsing"""
        test_line = 'SIM_DATA:{"timestamp": 1.0, "ship": {"id": 1, "credits": 50000}}'
        
        with patch.object(self.client, 'process_simulation_data') as mock_process:
            self.client.process_simulator_output(test_line)
            
            mock_process.assert_called_once()
            args = mock_process.call_args[0][0]
            self.assertEqual(args['timestamp'], 1.0)
            self.assertEqual(args['ship']['id'], 1)
    
    def test_error_resilience(self):
        """Test handling of malformed JSON"""
        bad_line = 'SIM_DATA:{"invalid": json}'
        
        # Should not raise exception
        self.client.process_simulator_output(bad_line)
    
    @patch('subprocess.Popen')
    def test_process_management(self, mock_popen):
        """Test subprocess handling"""
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = ['test line\n', '']
        mock_popen.return_value = mock_process
        
        self.client.run_simulator()
        
        mock_popen.assert_called_once()
        mock_process.wait.assert_called_once()
```

## Conclusion

The Python clients demonstrate several important Python programming concepts:

### Key Patterns Demonstrated

1. **Process Management**: Robust subprocess handling with proper cleanup
2. **Real-time Processing**: Stream processing of live data
3. **Error Resilience**: Graceful degradation and comprehensive error handling
4. **Threading**: Background processing with UI responsiveness
5. **Protocol Implementation**: JSON-based inter-process communication
6. **Terminal Programming**: Both simple console and advanced TUI interfaces

### Architecture Benefits

- **Separation of Concerns**: Clear division between data processing and presentation
- **Extensibility**: Easy to add new display modes or data processing features
- **Robustness**: Comprehensive error handling prevents crashes
- **Performance**: Efficient I/O and memory management
- **Usability**: Multiple interface options for different user preferences

### Python-Specific Advantages

- **Rapid Development**: Quick prototyping and iteration
- **Rich Libraries**: `curses`, `json`, `subprocess` built-in modules
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Readable Code**: Clear, maintainable implementation
- **Integration**: Easy interfacing with external processes

The implementation serves as a practical example of how Python excels at system integration, data processing, and user interface development, complementing Rust's strengths in systems programming and performance-critical applications.

### Future Enhancements

1. **Web Interface**: Flask/Django web dashboard
2. **Data Export**: CSV/Excel export capabilities
3. **Real-time Graphs**: matplotlib integration for live plotting
4. **Remote Monitoring**: Network protocol for distributed monitoring
5. **Plugin System**: Extensible analysis and display modules
6. **Configuration Management**: YAML/JSON configuration files