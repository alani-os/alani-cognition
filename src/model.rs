//! Model metadata, capability flags, and fixed-capacity registry.
//!
//! The registry is intentionally small and allocation-free so host-mode tests
//! can exercise model lifecycle and selection before real `alani-models` or
//! device integrations are wired.

use crate::{CognitionError, CognitionResult, DataClass};

/// Maximum stable model identifier length.
pub const MAX_MODEL_ID_LEN: usize = 96;

/// Maximum display name length for a model.
pub const MAX_MODEL_NAME_LEN: usize = 128;

/// Maximum revision string length.
pub const MAX_MODEL_REVISION_LEN: usize = 64;

/// Permission to run inference.
pub const MODEL_CAP_INFER: u64 = 1 << 0;
/// Permission to produce embeddings.
pub const MODEL_CAP_EMBED: u64 = 1 << 1;
/// Permission to produce plans.
pub const MODEL_CAP_PLAN: u64 = 1 << 2;
/// Permission to query memory or retrieval context.
pub const MODEL_CAP_RETRIEVE: u64 = 1 << 3;
/// Permission to use accelerator-backed execution.
pub const MODEL_CAP_ACCELERATED: u64 = 1 << 4;
/// Permission to stream output in a future API.
pub const MODEL_CAP_STREAM: u64 = 1 << 5;

/// All model capability bits known by this crate version.
pub const KNOWN_MODEL_CAPABILITIES: u64 = MODEL_CAP_INFER
    | MODEL_CAP_EMBED
    | MODEL_CAP_PLAN
    | MODEL_CAP_RETRIEVE
    | MODEL_CAP_ACCELERATED
    | MODEL_CAP_STREAM;

/// Model class used by routing and policy hooks.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    /// Deterministic host-mode mock model.
    Mock = 1,
    /// Language model.
    Language = 2,
    /// Embedding model.
    Embedding = 3,
    /// Planning model.
    Planner = 4,
    /// Reranker or scorer model.
    Reranker = 5,
    /// Memory retrieval model.
    Memory = 6,
}

impl ModelKind {
    /// Stable model-kind label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Mock => "mock",
            Self::Language => "language",
            Self::Embedding => "embedding",
            Self::Planner => "planner",
            Self::Reranker => "reranker",
            Self::Memory => "memory",
        }
    }
}

/// Model lifecycle state.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelState {
    /// Model metadata is known but not loaded.
    Registered = 1,
    /// Model is loaded and can be used.
    Loaded = 2,
    /// Model was unloaded.
    Unloaded = 3,
    /// Model is unavailable due to device or policy constraints.
    Unavailable = 4,
}

impl ModelState {
    /// Stable state label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::Loaded => "loaded",
            Self::Unloaded => "unloaded",
            Self::Unavailable => "unavailable",
        }
    }
}

/// Model capability bitset.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelCapabilities(pub u64);

impl ModelCapabilities {
    /// No model capabilities.
    pub const EMPTY: Self = Self(0);
    /// All known model capabilities.
    pub const ALL: Self = Self(KNOWN_MODEL_CAPABILITIES);
    /// Standard inference capability.
    pub const INFER: Self = Self(MODEL_CAP_INFER);
    /// Standard retrieval-augmented inference capability.
    pub const RAG: Self = Self(MODEL_CAP_INFER | MODEL_CAP_RETRIEVE);
    /// Standard planning capability.
    pub const PLANNER: Self = Self(MODEL_CAP_INFER | MODEL_CAP_PLAN);

    /// Creates capabilities from raw bits.
    pub const fn from_bits(bits: u64) -> CognitionResult<Self> {
        if bits & !KNOWN_MODEL_CAPABILITIES != 0 {
            Err(CognitionError::ReservedBits)
        } else {
            Ok(Self(bits))
        }
    }

    /// Returns raw capability bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Returns `true` when all required capabilities are present.
    pub const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }

    /// Returns the union of two capability sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Stable model reference and lifecycle metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelRef<'a> {
    /// Stable model identifier.
    pub id: &'a str,
    /// Display name.
    pub name: &'a str,
    /// Model revision or package version.
    pub revision: &'a str,
    /// Model class.
    pub kind: ModelKind,
    /// Capability bitset.
    pub capabilities: ModelCapabilities,
    /// Maximum context tokens accepted by the model.
    pub max_context_tokens: u32,
    /// Maximum output tokens produced by the model.
    pub max_output_tokens: u32,
    /// Sensitivity class of model metadata and outputs.
    pub data_class: DataClass,
    /// Current lifecycle state.
    pub state: ModelState,
}

impl<'a> ModelRef<'a> {
    /// Creates registered model metadata.
    pub const fn new(
        id: &'a str,
        name: &'a str,
        revision: &'a str,
        kind: ModelKind,
        capabilities: ModelCapabilities,
    ) -> Self {
        Self {
            id,
            name,
            revision,
            kind,
            capabilities,
            max_context_tokens: 0,
            max_output_tokens: 0,
            data_class: DataClass::Operational,
            state: ModelState::Registered,
        }
    }

    /// Sets context and output token limits.
    pub const fn with_limits(mut self, max_context_tokens: u32, max_output_tokens: u32) -> Self {
        self.max_context_tokens = max_context_tokens;
        self.max_output_tokens = max_output_tokens;
        self
    }

    /// Sets model data classification.
    pub const fn with_data_class(mut self, data_class: DataClass) -> Self {
        self.data_class = data_class;
        self
    }

    /// Sets lifecycle state.
    pub const fn with_state(mut self, state: ModelState) -> Self {
        self.state = state;
        self
    }

    /// Returns `true` when the model can run inference.
    pub const fn can_infer(self) -> bool {
        matches!(self.state, ModelState::Loaded)
            && self.capabilities.contains(ModelCapabilities::INFER)
    }

    /// Validates model metadata and limits.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.id.is_empty() || self.name.is_empty() || self.revision.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.id.len() > MAX_MODEL_ID_LEN
            || self.name.len() > MAX_MODEL_NAME_LEN
            || self.revision.len() > MAX_MODEL_REVISION_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        if self.max_context_tokens == 0 || self.max_output_tokens == 0 {
            return Err(CognitionError::InvalidModel);
        }
        match ModelCapabilities::from_bits(self.capabilities.bits()) {
            Ok(_) => Ok(()),
            Err(error) => Err(error),
        }
    }
}

/// Model selection request used by planners and engines.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelSelection {
    /// Required model kind.
    pub kind: Option<ModelKind>,
    /// Required model capabilities.
    pub required: ModelCapabilities,
    /// Minimum context tokens needed.
    pub min_context_tokens: u32,
    /// Minimum output tokens needed.
    pub min_output_tokens: u32,
}

impl ModelSelection {
    /// Creates an inference model selection.
    pub const fn inference() -> Self {
        Self {
            kind: None,
            required: ModelCapabilities::INFER,
            min_context_tokens: 0,
            min_output_tokens: 1,
        }
    }

    /// Creates a planner model selection.
    pub const fn planner() -> Self {
        Self {
            kind: Some(ModelKind::Planner),
            required: ModelCapabilities::PLANNER,
            min_context_tokens: 0,
            min_output_tokens: 1,
        }
    }

    /// Sets minimum token limits.
    pub const fn with_limits(mut self, min_context_tokens: u32, min_output_tokens: u32) -> Self {
        self.min_context_tokens = min_context_tokens;
        self.min_output_tokens = min_output_tokens;
        self
    }

    /// Returns `true` when a model satisfies this selection.
    pub const fn matches(self, model: ModelRef<'_>) -> bool {
        if let Some(kind) = self.kind {
            if !same_kind(kind, model.kind) {
                return false;
            }
        }
        matches!(model.state, ModelState::Loaded)
            && model.capabilities.contains(self.required)
            && model.max_context_tokens >= self.min_context_tokens
            && model.max_output_tokens >= self.min_output_tokens
    }
}

const fn same_kind(left: ModelKind, right: ModelKind) -> bool {
    matches!(
        (left, right),
        (ModelKind::Mock, ModelKind::Mock)
            | (ModelKind::Language, ModelKind::Language)
            | (ModelKind::Embedding, ModelKind::Embedding)
            | (ModelKind::Planner, ModelKind::Planner)
            | (ModelKind::Reranker, ModelKind::Reranker)
            | (ModelKind::Memory, ModelKind::Memory)
    )
}

/// Fixed-capacity model registry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelRegistry<'a, const N: usize> {
    models: [Option<ModelRef<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> ModelRegistry<'a, N> {
    /// Creates an empty model registry.
    pub const fn new() -> Self {
        Self {
            models: [None; N],
            len: 0,
        }
    }

    /// Returns the number of registered models.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when no models are registered.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Registers a model after validating metadata.
    pub fn register(&mut self, model: ModelRef<'a>) -> CognitionResult<()> {
        model.validate()?;
        if self.len == N {
            return Err(CognitionError::CapacityExceeded);
        }
        if self.find(model.id).is_some() {
            return Err(CognitionError::DuplicateModel);
        }
        self.models[self.len] = Some(model);
        self.len += 1;
        Ok(())
    }

    /// Finds a model by identifier.
    pub fn find(&self, id: &str) -> Option<&ModelRef<'a>> {
        self.iter().find(|model| model.id == id)
    }

    /// Selects the first loaded model matching the criteria.
    pub fn select(&self, selection: ModelSelection) -> Option<&ModelRef<'a>> {
        self.iter().find(|model| selection.matches(**model))
    }

    /// Updates the lifecycle state for a model.
    pub fn set_state(&mut self, id: &str, state: ModelState) -> CognitionResult<()> {
        match self.find_index(id) {
            Some(index) => {
                if let Some(mut model) = self.models[index] {
                    model.state = state;
                    self.models[index] = Some(model);
                    Ok(())
                } else {
                    Err(CognitionError::ModelNotFound)
                }
            }
            None => Err(CognitionError::ModelNotFound),
        }
    }

    /// Returns a model by zero-based index.
    pub fn model(&self, index: usize) -> Option<&ModelRef<'a>> {
        if index < self.len {
            self.models[index].as_ref()
        } else {
            None
        }
    }

    /// Iterates over registered models.
    pub fn iter(&self) -> impl Iterator<Item = &ModelRef<'a>> {
        self.models[..self.len].iter().filter_map(Option::as_ref)
    }

    fn find_index(&self, id: &str) -> Option<usize> {
        let mut index = 0;
        while index < self.len {
            if let Some(model) = self.models[index] {
                if model.id == id {
                    return Some(index);
                }
            }
            index += 1;
        }
        None
    }
}

impl<'a, const N: usize> Default for ModelRegistry<'a, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelDescriptor<'a> {
    /// Component name.
    pub name: &'a str,
    /// Component version marker.
    pub version: u32,
}

impl<'a> ModelDescriptor<'a> {
    /// Creates a model component descriptor.
    pub const fn new(name: &'a str, version: u32) -> Self {
        Self { name, version }
    }
}
