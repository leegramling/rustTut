use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
use std::thread;
use std::sync::{mpsc, Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub sector: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ship {
    pub id: u32,
    pub name: String,
    pub position: Position,
    pub credits: f64,
    pub fuel: f64,
    pub max_fuel: f64,
    pub status: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationData {
    pub timestamp: f64,
    pub ship: Ship,
    pub latest_events: Vec<serde_json::Value>,
}

#[derive(Component)]
pub struct Player;

#[derive(Component)]
pub struct Meteor;

#[derive(Component)]
pub struct Planet {
    pub name: String,
}

#[derive(Component)]
pub struct ActivityEffect {
    pub timer: Timer,
}

#[derive(Component)]
pub struct LoadingEffect;

#[derive(Component)]
pub struct RefuelingEffect;

#[derive(Component)]
pub struct DockingEffect;

#[derive(Resource)]
pub struct SimulationReceiver {
    pub receiver: Arc<Mutex<mpsc::Receiver<SimulationData>>>,
}

#[derive(Resource)]
pub struct CurrentSimData {
    pub data: Option<SimulationData>,
}

#[derive(Resource)]
pub struct Stations {
    pub mining_station: Vec3,      // Mining Station Alpha
    pub trade_hub: Vec3,           // Trade Hub Beta  
    pub research_outpost: Vec3,    // Research Outpost Gamma
    pub industrial_complex: Vec3,  // Industrial Complex Delta
    pub mining_outpost: Vec3,      // Mining Outpost Epsilon
    pub space_station: Vec3,       // Space Station Zeta
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Space Simulation - Bevy Client".into(),
                resolution: (1200., 800.).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(CurrentSimData { data: None })
        .insert_resource(Stations {
            mining_station: Vec3::new(250.0, 150.0, 0.0),       // Mining Station Alpha
            trade_hub: Vec3::new(-200.0, 200.0, 0.0),           // Trade Hub Beta
            research_outpost: Vec3::new(300.0, -120.0, 0.0),    // Research Outpost Gamma
            industrial_complex: Vec3::new(-240.0, -100.0, 0.0), // Industrial Complex Delta
            mining_outpost: Vec3::new(160.0, 240.0, 0.0),       // Mining Outpost Epsilon
            space_station: Vec3::new(-80.0, 360.0, 0.0),        // Space Station Zeta
        })
        .add_systems(Startup, (setup_camera, setup_scene, start_simulation))
        .add_systems(Update, update_from_simulation)
        .add_systems(Update, update_ship_position)
        .add_systems(Update, update_activity_effects)
        .add_systems(Update, cleanup_expired_effects)
        .run();
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    stations: Res<Stations>,
) {
    // Background
    commands.spawn(SpriteBundle {
        texture: asset_server.load("starBackground.png"),
        transform: Transform::from_scale(Vec3::splat(2.0)),
        ..default()
    });
    
    // Player ship (starts at origin)
    commands.spawn((
        SpriteBundle {
            texture: asset_server.load("player.png"),
            transform: Transform::from_xyz(0.0, 0.0, 1.0).with_scale(Vec3::splat(0.8)),
            ..default()
        },
        Player,
    ));
    
    // Meteors as decoration (representing asteroid belt)
    for i in 0..4 {
        let angle = (i as f32) * std::f32::consts::TAU / 4.0;
        let radius = 300.0; // Adjusted radius to stay on screen
        let x = angle.cos() * radius;
        let y = angle.sin() * radius;
        
        commands.spawn((
            SpriteBundle {
                texture: asset_server.load("meteorBig.png"),
                transform: Transform::from_xyz(x, y, 0.5).with_scale(Vec3::splat(0.6)),
                ..default()
            },
            Meteor,
        ));
    }
    
    // Planets/Stations represented as colored circles
    // Mining Station Alpha
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.8, 0.4, 0.2), // Brown/orange for mining
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.mining_station.x, stations.mining_station.y, 0.5),
            ..default()
        },
        Planet {
            name: "Mining Station Alpha".to_string(),
        },
    ));
    
    // Trade Hub Beta  
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.2, 0.4, 0.8), // Blue for trade
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.trade_hub.x, stations.trade_hub.y, 0.5),
            ..default()
        },
        Planet {
            name: "Trade Hub Beta".to_string(),
        },
    ));
    
    // Research Outpost Gamma
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.6, 0.2, 0.8), // Purple for research
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.research_outpost.x, stations.research_outpost.y, 0.5),
            ..default()
        },
        Planet {
            name: "Research Outpost Gamma".to_string(),
        },
    ));
    
    // Industrial Complex Delta
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.7, 0.7, 0.2), // Yellow for industrial
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.industrial_complex.x, stations.industrial_complex.y, 0.5),
            ..default()
        },
        Planet {
            name: "Industrial Complex Delta".to_string(),
        },
    ));
    
    // Mining Outpost Epsilon
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.9, 0.3, 0.1), // Red-orange for outer mining
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.mining_outpost.x, stations.mining_outpost.y, 0.5),
            ..default()
        },
        Planet {
            name: "Mining Outpost Epsilon".to_string(),
        },
    ));
    
    // Space Station Zeta
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.9, 0.9, 0.9), // Silver for luxury station
                custom_size: Some(Vec2::new(40.0, 40.0)),
                ..default()
            },
            transform: Transform::from_xyz(stations.space_station.x, stations.space_station.y, 0.5),
            ..default()
        },
        Planet {
            name: "Space Station Zeta".to_string(),
        },
    ));
    
    // UI Text for simulation data
    commands.spawn(
        TextBundle::from_section(
            "Connecting to simulation...",
            TextStyle {
                font_size: 20.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        }),
    );
}

fn start_simulation(mut commands: Commands) {
    let (tx, rx) = mpsc::channel();
    
    // Spawn thread to run simulation and capture output
    thread::spawn(move || {
        let mut child = Command::new("cargo")
            .arg("run")
            .current_dir("../demo") // Run demo from demo directory
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to start simulation");
        
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            
            for line in reader.lines() {
                if let Ok(line) = line {
                    // Look for simulation data lines
                    if line.starts_with("SIM_DATA:") {
                        let json_str = &line[9..]; // Remove "SIM_DATA:" prefix
                        if let Ok(data) = serde_json::from_str::<SimulationData>(json_str) {
                            if tx.send(data).is_err() {
                                break; // Receiver dropped, exit thread
                            }
                        }
                    }
                    // Print regular output for debugging
                    println!("SIM: {}", line);
                }
            }
        }
    });
    
    commands.insert_resource(SimulationReceiver { receiver: Arc::new(Mutex::new(rx)) });
}

fn update_from_simulation(
    mut commands: Commands,
    mut current_data: ResMut<CurrentSimData>,
    sim_receiver: Option<Res<SimulationReceiver>>,
    mut query: Query<&mut Text>,
    asset_server: Res<AssetServer>,
    player_query: Query<&Transform, With<Player>>,
) {
    if let Some(receiver_resource) = sim_receiver {
        // Try to receive latest data (non-blocking)
        if let Ok(receiver) = receiver_resource.receiver.try_lock() {
            while let Ok(data) = receiver.try_recv() {
                // Check for activity changes and spawn effects
                if let Some(ref old_data) = current_data.data {
                    check_and_spawn_effects(&mut commands, &asset_server, &old_data.ship, &data.ship, &player_query);
                }
                current_data.data = Some(data);
            }
        }
        
        // Update UI text with current simulation state
        if let Some(ref data) = current_data.data {
            for mut text in query.iter_mut() {
                let ship = &data.ship;
                let pos = &ship.position;
                
                // Format status
                let status_text = if let Some(status_obj) = ship.status.as_object() {
                    if let Some(_traveling) = status_obj.get("Traveling") {
                        "Traveling".to_string()
                    } else if let Some(docked) = status_obj.get("Docked") {
                        format!("Docked at {}", docked.get("port").unwrap_or(&serde_json::Value::String("Unknown".to_string())).as_str().unwrap_or("Unknown"))
                    } else {
                        format!("{:?}", status_obj.keys().next().unwrap_or(&"Unknown".to_string()))
                    }
                } else {
                    ship.status.as_str().unwrap_or("Unknown").to_string()
                };
                
                text.sections[0].value = format!(
                    "🚀 {} (ID: {})\n📍 Position: ({:.1}, {:.1}, {:.1}) - {}\n📊 Status: {}\n💰 Credits: {:.2}\n⛽ Fuel: {:.1}/{:.1}\n⏰ Time: T+{:.1}h",
                    ship.name,
                    ship.id,
                    pos.x, pos.y, pos.z,
                    pos.sector,
                    status_text,
                    ship.credits,
                    ship.fuel, ship.max_fuel,
                    data.timestamp
                );
            }
        }
    }
}

fn update_ship_position(
    current_data: Res<CurrentSimData>,
    mut query: Query<&mut Transform, With<Player>>,
    stations: Res<Stations>,
) {
    if let Some(ref data) = current_data.data {
        for mut transform in query.iter_mut() {
            let ship = &data.ship;
            let pos = &ship.position;
            
            // Map simulation coordinates to screen coordinates
            let screen_pos = map_sim_to_screen(pos, &stations);
            
            // Get current and target positions
            let current_pos = transform.translation;
            let target_pos = Vec3::new(screen_pos.x, screen_pos.y, current_pos.z);
            
            // Calculate movement direction for rotation
            let movement_vector = target_pos - current_pos;
            if movement_vector.length() > 0.1 { // Only rotate if there's significant movement
                // Calculate angle to face movement direction
                // atan2(y, x) gives angle from positive x-axis
                // We need to adjust because ship sprite faces up (positive y)
                let angle = movement_vector.y.atan2(movement_vector.x) - std::f32::consts::FRAC_PI_2;
                transform.rotation = Quat::from_rotation_z(angle);
            }
            
            // Slower interpolation to new position (20% slower)
            transform.translation = current_pos.lerp(target_pos, 0.08);
        }
    }
}

fn check_and_spawn_effects(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    old_ship: &Ship,
    new_ship: &Ship,
    player_query: &Query<&Transform, With<Player>>,
) {
    if let Ok(player_transform) = player_query.get_single() {
        let ship_pos = player_transform.translation;
        
        // Check for status changes and spawn appropriate effects
        let old_status_str = format!("{:?}", old_ship.status);
        let new_status_str = format!("{:?}", new_ship.status);
        
        // Docking effect
        if new_status_str.contains("Docked") && !old_status_str.contains("Docked") {
            spawn_docking_effect(commands, asset_server, ship_pos);
        }
        
        // Loading effect
        if new_status_str.contains("Loading") {
            spawn_loading_effect(commands, asset_server, ship_pos);
        }
        
        // Unloading effect (selling materials)
        if new_status_str.contains("Unloading") {
            spawn_unloading_effect(commands, asset_server, ship_pos);
        }
        
        // Refueling effect - detect fuel increase
        if new_ship.fuel > old_ship.fuel {
            spawn_refueling_effect(commands, asset_server, ship_pos);
        }
    }
}

fn spawn_docking_effect(commands: &mut Commands, _asset_server: &Res<AssetServer>, pos: Vec3) {
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.0, 1.0, 1.0), // Cyan for docking
                custom_size: Some(Vec2::new(20.0, 20.0)),
                ..default()
            },
            transform: Transform::from_translation(pos + Vec3::new(0.0, 40.0, 1.0)),
            ..default()
        },
        ActivityEffect {
            timer: Timer::from_seconds(2.0, TimerMode::Once),
        },
        DockingEffect,
    ));
}

fn spawn_loading_effect(commands: &mut Commands, _asset_server: &Res<AssetServer>, pos: Vec3) {
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(1.0, 0.5, 0.0), // Orange for loading
                custom_size: Some(Vec2::new(15.0, 15.0)),
                ..default()
            },
            transform: Transform::from_translation(pos + Vec3::new(-25.0, 0.0, 1.0)),
            ..default()
        },
        ActivityEffect {
            timer: Timer::from_seconds(3.0, TimerMode::Once),
        },
        LoadingEffect,
    ));
}

fn spawn_unloading_effect(commands: &mut Commands, _asset_server: &Res<AssetServer>, pos: Vec3) {
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(0.0, 1.0, 0.0), // Green for unloading/selling
                custom_size: Some(Vec2::new(15.0, 15.0)),
                ..default()
            },
            transform: Transform::from_translation(pos + Vec3::new(25.0, 0.0, 1.0)),
            ..default()
        },
        ActivityEffect {
            timer: Timer::from_seconds(3.0, TimerMode::Once),
        },
    ));
}

fn spawn_refueling_effect(commands: &mut Commands, _asset_server: &Res<AssetServer>, pos: Vec3) {
    commands.spawn((
        SpriteBundle {
            sprite: Sprite {
                color: Color::srgb(1.0, 1.0, 0.0), // Yellow for refueling
                custom_size: Some(Vec2::new(12.0, 12.0)),
                ..default()
            },
            transform: Transform::from_translation(pos + Vec3::new(0.0, -30.0, 1.0)),
            ..default()
        },
        ActivityEffect {
            timer: Timer::from_seconds(1.5, TimerMode::Once),
        },
        RefuelingEffect,
    ));
}

fn update_activity_effects(
    time: Res<Time>,
    mut effects_query: Query<(&mut ActivityEffect, &mut Transform)>,
) {
    for (mut effect, mut transform) in effects_query.iter_mut() {
        effect.timer.tick(time.delta());
        
        // Pulse effect
        let pulse = (effect.timer.elapsed_secs() * 5.0).sin() * 0.3 + 1.0;
        transform.scale = Vec3::splat(pulse);
    }
}

fn cleanup_expired_effects(
    mut commands: Commands,
    effects_query: Query<(Entity, &ActivityEffect)>,
) {
    for (entity, effect) in effects_query.iter() {
        if effect.timer.finished() {
            commands.entity(entity).despawn();
        }
    }
}

fn map_sim_to_screen(sim_pos: &Position, stations: &Stations) -> Vec2 {
    // Map simulation coordinates to screen space based on sectors and known station positions
    match sim_pos.sector.as_str() {
        "Home Base" => Vec2::new(0.0, 0.0), // Center of screen
        
        "Asteroid Belt" => {
            // Mining Station Alpha: (100, 50, 25)
            let base_pos = stations.mining_station.truncate();
            let offset = Vec2::new((sim_pos.x - 100.0) as f32 * 2.0, (sim_pos.y - 50.0) as f32 * 2.0);
            base_pos + offset
        },
        
        "Trade Sector" => {
            // Trade Hub Beta: (-75, 100, -30)
            let base_pos = stations.trade_hub.truncate();
            let offset = Vec2::new((sim_pos.x + 75.0) as f32 * 2.0, (sim_pos.y - 100.0) as f32 * 2.0);
            base_pos + offset
        },
        
        "Deep Space" => {
            // Research Outpost Gamma: (150, -80, 60)
            let base_pos = stations.research_outpost.truncate();
            let offset = Vec2::new((sim_pos.x - 150.0) as f32 * 2.0, (sim_pos.y + 80.0) as f32 * 2.0);
            base_pos + offset
        },
        
        "Industrial Zone" => {
            // Industrial Complex Delta: (-120, -50, 20)
            let base_pos = stations.industrial_complex.truncate();
            let offset = Vec2::new((sim_pos.x + 120.0) as f32 * 2.0, (sim_pos.y + 50.0) as f32 * 2.0);
            base_pos + offset
        },
        
        "Outer Asteroids" => {
            // Mining Outpost Epsilon: (80, 120, -40)
            let base_pos = stations.mining_outpost.truncate();
            let offset = Vec2::new((sim_pos.x - 80.0) as f32 * 2.0, (sim_pos.y - 120.0) as f32 * 2.0);
            base_pos + offset
        },
        
        "Central Hub" => {
            // Space Station Zeta: (-40, 180, 80)
            let base_pos = stations.space_station.truncate();
            let offset = Vec2::new((sim_pos.x + 40.0) as f32 * 2.0, (sim_pos.y - 180.0) as f32 * 2.0);
            base_pos + offset
        },
        
        _ => {
            // For any other sector or during travel, use scaled simulation coordinates
            let scale = 3.0; // Moderate scale to keep on screen
            Vec2::new(sim_pos.x as f32 * scale, sim_pos.y as f32 * scale)
        }
    }
}