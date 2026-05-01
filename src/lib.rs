#![cfg_attr(not(feature = "std"), no_std)]

//! Starter crate for `alani-cognition`.
//!
//! Expand this crate according to `docs/repositories/alani-cognition.md`.

pub mod engine;
pub mod model;
pub mod planner;
pub mod retrieval;

pub const REPOSITORY: &str = "alani-cognition";
pub const VERSION: &str = "0.1.0";
pub const MODULES: &[&str] = &["engine", "model", "planner", "retrieval"];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentStatus {
    Draft,
    Experimental,
    Stable,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInfo {
    pub repository: &'static str,
    pub version: &'static str,
    pub status: ComponentStatus,
}

pub const fn component_info() -> ComponentInfo {
    ComponentInfo {
        repository: REPOSITORY,
        version: VERSION,
        status: ComponentStatus::Draft,
    }
}

pub const fn repository_name() -> &'static str {
    REPOSITORY
}

pub fn module_names() -> &'static [&'static str] {
    MODULES
}
