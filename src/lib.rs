#![cfg_attr(not(feature = "std"), no_std)]

//! Dependency-free cognition contracts for the Alani MVK.
//!
//! This crate owns the public skeleton for model registration, bounded
//! inference, retrieval provenance, planning, evidence metadata, and host-mode
//! deterministic execution. Real sibling integrations remain represented as
//! Cargo metadata until their public APIs stabilize.

pub mod engine;
pub mod model;
pub mod planner;
pub mod retrieval;

pub use engine::{
    BudgetUsage, CognitionEngine, CognitiveContext, EngineDescriptor, EvidenceBundle,
    InferenceBudget, InferenceRequest, InferenceResult, InferenceStatus, MockCognitionEngine,
    PolicyDecision, TraceContext, BUDGET_FLAG_AUDIT_REQUIRED, BUDGET_FLAG_DETERMINISTIC,
    BUDGET_KNOWN_FLAGS, MAX_CONTEXT_LABEL_LEN, MAX_OUTPUT_LEN, MAX_PROMPT_LEN,
    MAX_REQUEST_RETRIEVAL_LIMIT, REQUEST_FLAG_ALLOW_RETRIEVAL, REQUEST_FLAG_CANCELLED,
    REQUEST_KNOWN_FLAGS,
};
pub use model::{
    ModelCapabilities, ModelDescriptor, ModelKind, ModelRef, ModelRegistry, ModelSelection,
    ModelState, KNOWN_MODEL_CAPABILITIES, MAX_MODEL_ID_LEN, MAX_MODEL_NAME_LEN,
    MAX_MODEL_REVISION_LEN, MODEL_CAP_ACCELERATED, MODEL_CAP_EMBED, MODEL_CAP_INFER,
    MODEL_CAP_PLAN, MODEL_CAP_RETRIEVE, MODEL_CAP_STREAM,
};
pub use planner::{
    DeterministicPlanner, Plan, PlanGoal, PlanPriority, PlanStatus, PlanStep, PlanStepKind,
    PlannerConfig, PlannerDescriptor, DEFAULT_MAX_PLAN_STEPS, MAX_GOAL_DESCRIPTION_LEN,
    MAX_PLAN_RESOURCE_LEN, MAX_STEP_DESCRIPTION_LEN,
};
pub use retrieval::{
    KnowledgeRecord, KnowledgeRecordKind, KnowledgeStore, Provenance, RetrievalDescriptor,
    RetrievalHit, RetrievalQuery, RetrievalResult, MAX_NAMESPACE_LEN, MAX_PROVENANCE_SOURCE_LEN,
    MAX_RECORD_KEY_LEN, MAX_RECORD_TEXT_LEN, MAX_RETRIEVAL_HITS,
};

/// Repository name.
pub const REPOSITORY: &str = "alani-cognition";

/// Crate version.
pub const VERSION: &str = "0.1.0";

/// Public module names exposed by this crate.
pub const MODULES: &[&str] = &["engine", "model", "planner", "retrieval"];

/// Feature bit for model registry APIs.
pub const COGNITION_FEATURE_MODEL_REGISTRY: u64 = 1 << 0;
/// Feature bit for bounded inference APIs.
pub const COGNITION_FEATURE_INFERENCE: u64 = 1 << 1;
/// Feature bit for retrieval provenance APIs.
pub const COGNITION_FEATURE_RETRIEVAL: u64 = 1 << 2;
/// Feature bit for deterministic planner APIs.
pub const COGNITION_FEATURE_PLANNER: u64 = 1 << 3;
/// Feature bit for evidence metadata.
pub const COGNITION_FEATURE_EVIDENCE: u64 = 1 << 4;
/// Feature bit for host-mode mock execution.
pub const COGNITION_FEATURE_MOCK_ENGINE: u64 = 1 << 5;

/// All feature bits known by this crate version.
pub const COGNITION_KNOWN_FEATURES: u64 = COGNITION_FEATURE_MODEL_REGISTRY
    | COGNITION_FEATURE_INFERENCE
    | COGNITION_FEATURE_RETRIEVAL
    | COGNITION_FEATURE_PLANNER
    | COGNITION_FEATURE_EVIDENCE
    | COGNITION_FEATURE_MOCK_ENGINE;

/// Result alias used by cognition validation and host-mode APIs.
pub type CognitionResult<T> = Result<T, CognitionError>;

/// Error taxonomy for cognition validation, retrieval, planning, and inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CognitionError {
    /// Required field is missing.
    MissingField,
    /// Bounded string or payload metadata exceeded its documented limit.
    FieldTooLong,
    /// Reserved flag or capability bits were present.
    ReservedBits,
    /// Model metadata failed validation.
    InvalidModel,
    /// Model is not registered.
    ModelNotFound,
    /// Model is unavailable or not loaded for inference.
    ModelUnavailable,
    /// Model identifier already exists.
    DuplicateModel,
    /// Fixed-capacity collection is full.
    CapacityExceeded,
    /// Retrieval query, record, score, or hit metadata is invalid.
    InvalidRetrieval,
    /// Knowledge record identifier already exists.
    DuplicateRecord,
    /// Budget metadata is invalid.
    InvalidBudget,
    /// Request metadata is invalid.
    InvalidRequest,
    /// Trace context is malformed.
    InvalidTrace,
    /// Request was cancelled.
    Cancelled,
    /// Deadline expired before or during execution.
    DeadlineExceeded,
    /// Budget ceiling was exceeded.
    BudgetExceeded,
    /// Evidence metadata is invalid or inconsistent.
    InvalidEvidence,
    /// Planning metadata is invalid.
    InvalidPlan,
    /// Plan step identifier already exists.
    DuplicatePlanStep,
    /// Policy denied execution.
    AccessDenied,
    /// Internal invariant failed.
    Internal,
}

impl CognitionError {
    /// Stable reason label for tests and diagnostics.
    pub const fn reason(self) -> &'static str {
        match self {
            Self::MissingField => "missing_field",
            Self::FieldTooLong => "field_too_long",
            Self::ReservedBits => "reserved_bits",
            Self::InvalidModel => "invalid_model",
            Self::ModelNotFound => "model_not_found",
            Self::ModelUnavailable => "model_unavailable",
            Self::DuplicateModel => "duplicate_model",
            Self::CapacityExceeded => "capacity_exceeded",
            Self::InvalidRetrieval => "invalid_retrieval",
            Self::DuplicateRecord => "duplicate_record",
            Self::InvalidBudget => "invalid_budget",
            Self::InvalidRequest => "invalid_request",
            Self::InvalidTrace => "invalid_trace",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::BudgetExceeded => "budget_exceeded",
            Self::InvalidEvidence => "invalid_evidence",
            Self::InvalidPlan => "invalid_plan",
            Self::DuplicatePlanStep => "duplicate_plan_step",
            Self::AccessDenied => "access_denied",
            Self::Internal => "internal",
        }
    }

    /// Returns `true` when this error is security or audit relevant.
    pub const fn is_security_relevant(self) -> bool {
        matches!(
            self,
            Self::ReservedBits
                | Self::ModelUnavailable
                | Self::Cancelled
                | Self::DeadlineExceeded
                | Self::BudgetExceeded
                | Self::AccessDenied
        )
    }
}

/// Data sensitivity classification.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DataClass {
    /// Public data.
    Public = 0,
    /// Operational data.
    Operational = 1,
    /// Sensitive data requiring redaction.
    Sensitive = 2,
    /// Secret data that must not be exposed.
    Secret = 3,
}

impl DataClass {
    /// Stable data class label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Operational => "operational",
            Self::Sensitive => "sensitive",
            Self::Secret => "secret",
        }
    }

    /// Returns `true` when redaction is required before broad export.
    pub const fn requires_redaction(self) -> bool {
        matches!(self, Self::Sensitive | Self::Secret)
    }
}

/// Redaction state for model inputs, outputs, and evidence.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RedactionState {
    /// Public data only.
    Public = 0,
    /// Operational metadata only.
    Operational = 1,
    /// Sensitive fields were redacted.
    SensitiveRedacted = 2,
    /// Secret fields were redacted.
    SecretRedacted = 3,
    /// Sensitive fields remain present and must not be broadly exported.
    UnredactedSensitive = 4,
}

impl RedactionState {
    /// Stable redaction label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Operational => "operational",
            Self::SensitiveRedacted => "sensitive_redacted",
            Self::SecretRedacted => "secret_redacted",
            Self::UnredactedSensitive => "unredacted_sensitive",
        }
    }
}

/// Implementation maturity marker for generated repository metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComponentStatus {
    /// API is present as a draft skeleton.
    Draft,
    /// API is implemented enough for host-mode experimentation.
    Experimental,
    /// API is compatible and stable.
    Stable,
}

/// Stable component identity record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComponentInfo {
    /// Repository name.
    pub repository: &'static str,
    /// Crate version.
    pub version: &'static str,
    /// Current implementation status.
    pub status: ComponentStatus,
}

/// Returns stable component identity metadata.
pub const fn component_info() -> ComponentInfo {
    ComponentInfo {
        repository: REPOSITORY,
        version: VERSION,
        status: ComponentStatus::Experimental,
    }
}

/// Returns the repository name.
pub const fn repository_name() -> &'static str {
    REPOSITORY
}

/// Returns public module names.
pub fn module_names() -> &'static [&'static str] {
    MODULES
}

/// Compact root view of the cognition crate contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CognitionCatalog {
    /// Repository name.
    pub repository: &'static str,
    /// Crate version.
    pub version: &'static str,
    /// Feature bitmap.
    pub features: u64,
    /// Maximum prompt length.
    pub max_prompt_len: usize,
    /// Maximum retrieval hits.
    pub max_retrieval_hits: usize,
    /// Maximum model id length.
    pub max_model_id_len: usize,
}

impl CognitionCatalog {
    /// Current cognition catalog.
    pub const CURRENT: Self = Self {
        repository: REPOSITORY,
        version: VERSION,
        features: COGNITION_KNOWN_FEATURES,
        max_prompt_len: MAX_PROMPT_LEN,
        max_retrieval_hits: MAX_RETRIEVAL_HITS,
        max_model_id_len: MAX_MODEL_ID_LEN,
    };

    /// Returns the current catalog.
    pub const fn current() -> Self {
        Self::CURRENT
    }

    /// Validates catalog metadata.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.repository.is_empty() || self.version.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.features & !COGNITION_KNOWN_FEATURES != 0 {
            return Err(CognitionError::ReservedBits);
        }
        if self.max_prompt_len == 0 || self.max_retrieval_hits == 0 || self.max_model_id_len == 0 {
            return Err(CognitionError::InvalidRequest);
        }
        Ok(())
    }
}

/// Current cognition catalog.
pub const COGNITION_CATALOG: CognitionCatalog = CognitionCatalog::CURRENT;

/// Returns the current cognition catalog.
pub const fn cognition_catalog() -> CognitionCatalog {
    CognitionCatalog::CURRENT
}
