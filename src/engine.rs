//! `engine` module boundary.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineDescriptor<'a> {
    pub name: &'a str,
    pub version: u32,
}

impl<'a> EngineDescriptor<'a> {
    pub const fn new(name: &'a str, version: u32) -> Self {
        Self { name, version }
    }
}
