# Tutorial 10: Procedural Macros

## Learning Objectives

By the end of this tutorial, you will be able to:

1. **Understand macro fundamentals**: Distinguish between declarative and procedural macros and their use cases
2. **Create derive macros**: Build custom derive macros for automatic trait implementations
3. **Implement function-like macros**: Create macros that generate code based on input syntax
4. **Build attribute macros**: Develop macros that modify or annotate existing code
5. **Design domain-specific languages**: Use macros to create DSLs for space simulation configuration
6. **Apply macro hygiene**: Write safe macros that avoid variable capture and namespace pollution
7. **Debug macro expansion**: Use tools and techniques to understand macro-generated code

## Lesson Section

### Introduction to Procedural Macros

Procedural macros in Rust are a powerful metaprogramming tool that allows you to write code that generates other code at compile time. Unlike declarative macros (`macro_rules!`), procedural macros operate on the abstract syntax tree (AST) of Rust code, providing fine-grained control over code generation.

In our space simulation engine, procedural macros enable us to:
- Automatically generate boilerplate code for components and systems
- Create domain-specific languages for configuration and scripting
- Implement custom serialization and validation logic
- Generate efficient code for mathematical operations and data structures

### Industry Context

Procedural macros are widely used in production Rust codebases:

**Game Engines:**
- **Bevy Engine**: Uses extensive derive macros for ECS components, systems, and resources
- **Amethyst**: Leveraged macros for state management and component registration
- **wgpu**: Uses macros for shader binding generation and resource management

**Web Frameworks:**
- **Rocket**: Uses attribute macros for route handlers and request guards
- **Actix-web**: Employs macros for middleware and handler registration
- **Serde**: The gold standard for derive macros, enabling automatic serialization

**Database ORMs:**
- **Diesel**: Uses macros extensively for query building and table definitions
- **SeaORM**: Leverages derive macros for entity definitions and relationships
- **SQLx**: Uses macros for compile-time SQL validation

### Types of Procedural Macros

#### 1. Derive Macros

Derive macros automatically implement traits for structs and enums. They're the most common type of procedural macro.

```rust
// Usage example
#[derive(SimulationComponent, Serialize, Deserialize)]
pub struct SpaceShip {
    pub position: Vec3,
    pub velocity: Vec3,
    pub mass: f32,
}
```

**Real-world Applications:**
- **Serde**: `#[derive(Serialize, Deserialize)]` for JSON/YAML handling
- **Clone/Debug/PartialEq**: Standard library derive macros
- **Custom traits**: Domain-specific implementations like `Component` in ECS systems

#### 2. Function-like Macros

Function-like macros look like function calls but generate code based on their input.

```rust
// Generate entity creation code
let entity = create_entity! {
    SpaceShip {
        position: [0.0, 0.0, 0.0],
        velocity: [1.0, 0.0, 0.0],
        mass: 1000.0,
    },
    RigidBody::dynamic(),
    Collider::cuboid(2.0, 1.0, 3.0)
};
```

**Industry Examples:**
- **vec!**: Standard library macro for vector creation
- **println!**: Compile-time formatted printing
- **html!**: Yew framework's HTML generation macro
- **sql!**: SQLx's compile-time SQL query validation

#### 3. Attribute Macros

Attribute macros can modify or wrap existing items (functions, structs, modules).

```rust
#[system(stage = "update", after = "physics")]
fn movement_system(query: Query<(&mut Transform, &Velocity)>) {
    // System implementation
}
```

**Production Usage:**
- **#[tokio::main]**: Async runtime setup
- **#[test]**: Test framework integration
- **#[wasm_bindgen]**: WebAssembly bindings
- **#[route(...)]**: Web framework route definitions

### Macro Architecture and TokenStream Processing

Procedural macros work by manipulating `TokenStream`s - a representation of Rust code as a sequence of tokens. The process involves:

1. **Parsing**: Convert input tokens into an AST using `syn`
2. **Analysis**: Extract information and validate input
3. **Generation**: Create new code using `quote`
4. **Output**: Return generated code as tokens

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(SimulationComponent)]
pub fn simulation_component_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let generics = &input.generics;
    
    let expanded = quote! {
        impl #generics SimulationComponent for #name #generics {
            fn component_type(&self) -> &'static str {
                stringify!(#name)
            }
        }
    };
    
    TokenStream::from(expanded)
}
```

### Advanced Macro Patterns

#### Generic Type Handling

Modern macro development requires careful handling of generic types and where clauses:

```rust
fn handle_generics(generics: &Generics) -> (TokenStream2, TokenStream2, TokenStream2) {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    (
        quote!(#impl_generics),
        quote!(#ty_generics), 
        quote!(#where_clause)
    )
}
```

#### Error Handling and Diagnostics

Professional macro development includes comprehensive error handling:

```rust
use syn::{Error, Result};

fn validate_struct(input: &DeriveInput) -> Result<()> {
    match &input.data {
        syn::Data::Struct(_) => Ok(()),
        _ => Err(Error::new_spanned(input, "SimulationComponent can only be derived for structs"))
    }
}
```

### Domain-Specific Language Design

Macros enable creating intuitive configuration languages for complex systems:

```rust
// Simulation configuration DSL
simulation_config! {
    world {
        gravity: [0.0, -9.81, 0.0],
        time_step: 0.016,
        bounds: cube(1000.0),
    }
    
    entities {
        fleet "patrol_alpha" {
            count: 5,
            formation: line_formation(spacing: 50.0),
            components: [
                SpaceShip { mass: 1000.0 },
                Engine { thrust: 50000.0 },
                Weapon { damage: 100.0 },
            ]
        }
        
        asteroids "mining_field" {
            count: 100,
            distribution: random_sphere(radius: 500.0),
            resources: weighted {
                iron: 0.6,
                rare_earth: 0.3,
                crystal: 0.1,
            }
        }
    }
    
    systems {
        physics: enabled,
        rendering: enabled { 
            quality: high,
            shadows: true,
        },
        ai: enabled {
            behavior_trees: true,
            pathfinding: a_star,
        }
    }
}
```

### Performance Considerations

Procedural macros run at compile time, so performance impacts compilation speed rather than runtime:

**Compilation Performance:**
- **Minimize parsing**: Cache parsed results when possible
- **Efficient code generation**: Generate minimal, optimized code
- **Conditional compilation**: Use `cfg` attributes to reduce generated code

**Runtime Performance:**
- Generated code should be as efficient as hand-written code
- Use `const` and `inline` appropriately
- Leverage Rust's zero-cost abstractions

### Integration with Game Engine Architecture

In game engines like Bevy, macros integrate deeply with the ECS architecture:

```rust
// Bevy-style system registration
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub enum SimulationSet {
    Input,
    Logic,
    Physics,
    Rendering,
}

#[system_set(SimulationSet::Physics)]
fn physics_systems() -> SystemSet {
    (
        apply_forces,
        integrate_velocities,
        detect_collisions,
        resolve_collisions,
    ).chain()
}
```

### Testing and Debugging Strategies

**Macro Testing:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_macro_expansion() {
        let t = trybuild::TestCases::new();
        t.pass("tests/01-parse.rs");
        t.compile_fail("tests/02-missing-field.rs");
    }
}
```

**Debugging Techniques:**
- Use `cargo expand` to view macro-generated code
- Add `println!` statements in macro code for debugging
- Use `syn::Error` for clear error messages
- Test with `trybuild` for compile-time testing

### Best Practices

1. **Clear Error Messages**: Provide helpful error messages with `syn::Error`
2. **Hygiene**: Use proper scoping to avoid variable capture
3. **Documentation**: Document macro usage with examples
4. **Testing**: Comprehensive test coverage including edge cases
5. **Performance**: Optimize for compilation speed
6. **Maintainability**: Keep macro code simple and well-structured

### Key Takeaways

- **Procedural macros** enable powerful compile-time code generation
- **Three types** serve different purposes: derive, function-like, and attribute macros
- **Industry adoption** is widespread in frameworks, ORMs, and game engines
- **DSL creation** allows intuitive configuration and scripting languages
- **Performance considerations** focus on compile-time rather than runtime
- **Testing and debugging** require specialized tools and techniques
- **Best practices** emphasize clarity, safety, and maintainability

### Next Steps

In the next tutorial, we'll explore **WebAssembly Integration**, learning how to compile Rust code to WebAssembly for browser-based space simulations, including performance optimization and JavaScript interop.

---

*This tutorial provides the foundation for understanding and implementing procedural macros in Rust, with practical applications in game development and simulation systems. The techniques learned here will enable you to create powerful code generation tools and domain-specific languages for your projects.*