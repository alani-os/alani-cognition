//! Inference requests, budgets, evidence, and host-mode cognition engine.
//!
//! This module models the request pipeline from validation through deterministic
//! mock inference. Real model runtimes can implement [`CognitionEngine`] after
//! sibling crate contracts stabilize.

use crate::model::{ModelCapabilities, ModelRef};
use crate::{CognitionError, CognitionResult, DataClass, RedactionState};

/// Maximum prompt length accepted by the host-mode skeleton.
pub const MAX_PROMPT_LEN: usize = 4096;

/// Maximum output length produced by the host-mode skeleton.
pub const MAX_OUTPUT_LEN: usize = 4096;

/// Maximum principal, session, policy, and memory-scope label length.
pub const MAX_CONTEXT_LABEL_LEN: usize = 128;

/// Maximum retrieval hits attached to one request.
pub const MAX_REQUEST_RETRIEVAL_LIMIT: usize = 16;

/// Budget flag requesting deterministic execution when possible.
pub const BUDGET_FLAG_DETERMINISTIC: u32 = 1 << 0;
/// Budget flag requiring audit/evidence output.
pub const BUDGET_FLAG_AUDIT_REQUIRED: u32 = 1 << 1;
/// Known budget flags.
pub const BUDGET_KNOWN_FLAGS: u32 = BUDGET_FLAG_DETERMINISTIC | BUDGET_FLAG_AUDIT_REQUIRED;

/// Request flag indicating that the request has been cancelled.
pub const REQUEST_FLAG_CANCELLED: u32 = 1 << 0;
/// Request flag allowing retrieval context.
pub const REQUEST_FLAG_ALLOW_RETRIEVAL: u32 = 1 << 1;
/// Known request flags.
pub const REQUEST_KNOWN_FLAGS: u32 = REQUEST_FLAG_CANCELLED | REQUEST_FLAG_ALLOW_RETRIEVAL;

/// Trace context propagated from runtime and syscalls.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TraceContext {
    /// Stable trace identifier.
    pub trace_id: u64,
    /// Current span identifier.
    pub span_id: u64,
}

impl TraceContext {
    /// Empty trace context.
    pub const EMPTY: Self = Self {
        trace_id: 0,
        span_id: 0,
    };

    /// Creates a trace context.
    pub const fn new(trace_id: u64, span_id: u64) -> Self {
        Self { trace_id, span_id }
    }

    /// Returns `true` when trace identifiers are present.
    pub const fn is_present(self) -> bool {
        self.trace_id != 0 && self.span_id != 0
    }

    /// Validates that trace identifiers are either both present or both absent.
    pub const fn validate(self) -> CognitionResult<()> {
        if (self.trace_id == 0) != (self.span_id == 0) {
            Err(CognitionError::InvalidTrace)
        } else {
            Ok(())
        }
    }
}

/// Explicit model execution budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceBudget {
    /// Maximum output tokens. Zero means unspecified.
    pub max_tokens: u32,
    /// Maximum compute units. Zero means unspecified.
    pub max_compute_units: u32,
    /// Maximum memory bytes. Zero means unspecified.
    pub max_memory_bytes: u64,
    /// Absolute deadline in monotonic nanoseconds. Zero means unset.
    pub deadline_ns: u64,
    /// Budget flags.
    pub flags: u32,
}

impl InferenceBudget {
    /// Creates a bounded budget.
    pub const fn bounded(
        max_tokens: u32,
        max_compute_units: u32,
        max_memory_bytes: u64,
        deadline_ns: u64,
    ) -> Self {
        Self {
            max_tokens,
            max_compute_units,
            max_memory_bytes,
            deadline_ns,
            flags: BUDGET_FLAG_DETERMINISTIC,
        }
    }

    /// Creates an unbounded placeholder. Policy may reject it.
    pub const fn unbounded() -> Self {
        Self {
            max_tokens: 0,
            max_compute_units: 0,
            max_memory_bytes: 0,
            deadline_ns: 0,
            flags: 0,
        }
    }

    /// Sets budget flags.
    pub const fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Returns `true` when at least one ceiling is set.
    pub const fn is_bounded(self) -> bool {
        self.max_tokens != 0
            || self.max_compute_units != 0
            || self.max_memory_bytes != 0
            || self.deadline_ns != 0
    }

    /// Validates budget flags and ceilings.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.flags & !BUDGET_KNOWN_FLAGS != 0 {
            return Err(CognitionError::ReservedBits);
        }
        if !self.is_bounded() {
            return Err(CognitionError::InvalidBudget);
        }
        Ok(())
    }

    /// Returns `true` when deterministic execution is requested.
    pub const fn deterministic(self) -> bool {
        self.flags & BUDGET_FLAG_DETERMINISTIC != 0
    }
}

/// Resource usage reported by an inference result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BudgetUsage {
    /// Output tokens produced.
    pub output_tokens: u32,
    /// Compute units consumed.
    pub compute_units: u32,
    /// Memory bytes used.
    pub memory_bytes: u64,
    /// Elapsed time in nanoseconds.
    pub elapsed_ns: u64,
}

impl BudgetUsage {
    /// Creates usage metadata.
    pub const fn new(
        output_tokens: u32,
        compute_units: u32,
        memory_bytes: u64,
        elapsed_ns: u64,
    ) -> Self {
        Self {
            output_tokens,
            compute_units,
            memory_bytes,
            elapsed_ns,
        }
    }

    /// Validates usage against a budget.
    pub const fn fits(self, budget: InferenceBudget) -> bool {
        (budget.max_tokens == 0 || self.output_tokens <= budget.max_tokens)
            && (budget.max_compute_units == 0 || self.compute_units <= budget.max_compute_units)
            && (budget.max_memory_bytes == 0 || self.memory_bytes <= budget.max_memory_bytes)
    }
}

/// Request execution context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CognitiveContext<'a> {
    /// Principal or service identity.
    pub principal: &'a str,
    /// Session or request group identifier.
    pub session_id: &'a str,
    /// Policy label used by authorization hooks.
    pub policy_label: &'a str,
    /// Memory scope for retrieval.
    pub memory_scope: &'a str,
    /// Current trace context.
    pub trace: TraceContext,
    /// Current monotonic timestamp.
    pub monotonic_ns: u64,
    /// Data class for prompt/output handling.
    pub data_class: DataClass,
    /// Whether policy authorization is required before execution.
    pub requires_authorization: bool,
}

impl<'a> CognitiveContext<'a> {
    /// Creates an execution context.
    pub const fn new(principal: &'a str, session_id: &'a str, policy_label: &'a str) -> Self {
        Self {
            principal,
            session_id,
            policy_label,
            memory_scope: "default",
            trace: TraceContext::EMPTY,
            monotonic_ns: 0,
            data_class: DataClass::Operational,
            requires_authorization: true,
        }
    }

    /// Sets memory scope.
    pub const fn with_memory_scope(mut self, memory_scope: &'a str) -> Self {
        self.memory_scope = memory_scope;
        self
    }

    /// Sets trace context.
    pub const fn with_trace(mut self, trace: TraceContext) -> Self {
        self.trace = trace;
        self
    }

    /// Sets monotonic timestamp.
    pub const fn with_time(mut self, monotonic_ns: u64) -> Self {
        self.monotonic_ns = monotonic_ns;
        self
    }

    /// Sets data classification.
    pub const fn with_data_class(mut self, data_class: DataClass) -> Self {
        self.data_class = data_class;
        self
    }

    /// Validates context labels and trace context.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.principal.is_empty()
            || self.session_id.is_empty()
            || self.policy_label.is_empty()
            || self.memory_scope.is_empty()
        {
            return Err(CognitionError::MissingField);
        }
        if self.principal.len() > MAX_CONTEXT_LABEL_LEN
            || self.session_id.len() > MAX_CONTEXT_LABEL_LEN
            || self.policy_label.len() > MAX_CONTEXT_LABEL_LEN
            || self.memory_scope.len() > MAX_CONTEXT_LABEL_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        self.trace.validate()
    }
}

/// Policy decision recorded in evidence.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyDecision {
    /// Request may execute.
    Allow = 1,
    /// Request is denied.
    Deny = 2,
    /// Request requires an authorization step first.
    NeedsAuthorization = 3,
}

impl PolicyDecision {
    /// Stable decision label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Deny => "deny",
            Self::NeedsAuthorization => "needs_authorization",
        }
    }
}

/// Inference request status.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceStatus {
    /// Request completed.
    Completed = 1,
    /// Request was denied by policy.
    Denied = 2,
    /// Request was cancelled.
    Cancelled = 3,
    /// Request exceeded budget.
    BudgetExceeded = 4,
    /// Request deadline expired.
    DeadlineExceeded = 5,
}

impl InferenceStatus {
    /// Stable status label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Denied => "denied",
            Self::Cancelled => "cancelled",
            Self::BudgetExceeded => "budget_exceeded",
            Self::DeadlineExceeded => "deadline_exceeded",
        }
    }
}

/// Evidence bundle for model-influenced decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvidenceBundle<'a> {
    /// Request identifier.
    pub request_id: u64,
    /// Model identifier.
    pub model_id: &'a str,
    /// Trace context linked to audit records.
    pub trace: TraceContext,
    /// Retrieval hits used by the request.
    pub retrieval_hits: u16,
    /// Plan steps used by the request.
    pub plan_steps: u16,
    /// Audit event identifier, if emitted.
    pub audit_event_id: u128,
    /// Policy decision.
    pub policy_decision: PolicyDecision,
    /// Output redaction state.
    pub redaction: RedactionState,
}

impl<'a> EvidenceBundle<'a> {
    /// Creates evidence metadata.
    pub const fn new(request_id: u64, model_id: &'a str, trace: TraceContext) -> Self {
        Self {
            request_id,
            model_id,
            trace,
            retrieval_hits: 0,
            plan_steps: 0,
            audit_event_id: 0,
            policy_decision: PolicyDecision::Allow,
            redaction: RedactionState::Operational,
        }
    }

    /// Sets retrieval hit count.
    pub const fn with_retrieval_hits(mut self, retrieval_hits: u16) -> Self {
        self.retrieval_hits = retrieval_hits;
        self
    }

    /// Sets plan step count.
    pub const fn with_plan_steps(mut self, plan_steps: u16) -> Self {
        self.plan_steps = plan_steps;
        self
    }

    /// Sets audit event id.
    pub const fn with_audit_event(mut self, audit_event_id: u128) -> Self {
        self.audit_event_id = audit_event_id;
        self
    }

    /// Sets policy decision.
    pub const fn with_policy_decision(mut self, decision: PolicyDecision) -> Self {
        self.policy_decision = decision;
        self
    }

    /// Sets redaction state.
    pub const fn with_redaction(mut self, redaction: RedactionState) -> Self {
        self.redaction = redaction;
        self
    }

    /// Validates evidence metadata.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.request_id == 0 || self.model_id.is_empty() {
            return Err(CognitionError::MissingField);
        }
        self.trace.validate()
    }
}

/// Inference request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceRequest<'a> {
    /// Stable request identifier.
    pub request_id: u64,
    /// Target model.
    pub model: ModelRef<'a>,
    /// Prompt or redacted prompt summary.
    pub prompt: &'a str,
    /// Execution budget.
    pub budget: InferenceBudget,
    /// Request context.
    pub context: CognitiveContext<'a>,
    /// Maximum retrieval hits allowed for this request.
    pub retrieval_limit: usize,
    /// Request flags.
    pub flags: u32,
}

impl<'a> InferenceRequest<'a> {
    /// Creates an inference request.
    pub const fn new(
        request_id: u64,
        model: ModelRef<'a>,
        prompt: &'a str,
        budget: InferenceBudget,
        context: CognitiveContext<'a>,
    ) -> Self {
        Self {
            request_id,
            model,
            prompt,
            budget,
            context,
            retrieval_limit: 0,
            flags: 0,
        }
    }

    /// Enables retrieval context with a hit limit.
    pub const fn with_retrieval_limit(mut self, retrieval_limit: usize) -> Self {
        self.retrieval_limit = retrieval_limit;
        self.flags |= REQUEST_FLAG_ALLOW_RETRIEVAL;
        self
    }

    /// Sets request flags.
    pub const fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }

    /// Returns `true` when the request was cancelled.
    pub const fn is_cancelled(self) -> bool {
        self.flags & REQUEST_FLAG_CANCELLED != 0
    }

    /// Validates request fields, budget, context, and model readiness.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.request_id == 0 || self.prompt.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.prompt.len() > MAX_PROMPT_LEN {
            return Err(CognitionError::FieldTooLong);
        }
        if self.flags & !REQUEST_KNOWN_FLAGS != 0 {
            return Err(CognitionError::ReservedBits);
        }
        if self.retrieval_limit > MAX_REQUEST_RETRIEVAL_LIMIT {
            return Err(CognitionError::InvalidRequest);
        }
        if !self.model.can_infer() {
            return Err(CognitionError::ModelUnavailable);
        }
        match self.model.validate() {
            Ok(()) => match self.budget.validate() {
                Ok(()) => self.context.validate(),
                Err(error) => Err(error),
            },
            Err(error) => Err(error),
        }
    }
}

/// Inference result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceResult<'a> {
    /// Request identifier.
    pub request_id: u64,
    /// Model identifier.
    pub model_id: &'a str,
    /// Completion status.
    pub status: InferenceStatus,
    /// Output text or redacted output summary.
    pub output: &'a str,
    /// Resource usage.
    pub usage: BudgetUsage,
    /// Evidence bundle.
    pub evidence: EvidenceBundle<'a>,
}

impl<'a> InferenceResult<'a> {
    /// Creates an inference result.
    pub const fn new(
        request_id: u64,
        model_id: &'a str,
        status: InferenceStatus,
        output: &'a str,
        usage: BudgetUsage,
        evidence: EvidenceBundle<'a>,
    ) -> Self {
        Self {
            request_id,
            model_id,
            status,
            output,
            usage,
            evidence,
        }
    }

    /// Validates result fields and evidence consistency.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.request_id == 0 || self.model_id.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if matches!(self.status, InferenceStatus::Completed) && self.output.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.output.len() > MAX_OUTPUT_LEN {
            return Err(CognitionError::FieldTooLong);
        }
        if self.evidence.request_id != self.request_id {
            return Err(CognitionError::InvalidEvidence);
        }
        self.evidence.validate()
    }
}

/// Cognition engine contract.
pub trait CognitionEngine {
    /// Executes an inference request.
    fn infer<'a>(&self, request: InferenceRequest<'a>) -> CognitionResult<InferenceResult<'a>>;
}

/// Deterministic host-mode cognition engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MockCognitionEngine {
    /// Redacted deterministic output.
    pub output: &'static str,
    /// Output token estimate.
    pub output_tokens: u32,
    /// Compute-unit estimate.
    pub compute_units: u32,
}

impl MockCognitionEngine {
    /// Default deterministic mock engine.
    pub const DEFAULT: Self = Self {
        output: "mock inference accepted",
        output_tokens: 3,
        compute_units: 10,
    };

    /// Creates a mock engine.
    pub const fn new(output: &'static str, output_tokens: u32, compute_units: u32) -> Self {
        Self {
            output,
            output_tokens,
            compute_units,
        }
    }
}

impl Default for MockCognitionEngine {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl CognitionEngine for MockCognitionEngine {
    fn infer<'a>(&self, request: InferenceRequest<'a>) -> CognitionResult<InferenceResult<'a>> {
        request.validate()?;
        if request.is_cancelled() {
            return Err(CognitionError::Cancelled);
        }
        if request.budget.deadline_ns != 0
            && request.context.monotonic_ns > request.budget.deadline_ns
        {
            return Err(CognitionError::DeadlineExceeded);
        }
        let usage = BudgetUsage::new(self.output_tokens, self.compute_units, 1024, 1_000);
        if !usage.fits(request.budget) {
            return Err(CognitionError::BudgetExceeded);
        }
        if !request
            .model
            .capabilities
            .contains(ModelCapabilities::INFER)
        {
            return Err(CognitionError::ModelUnavailable);
        }
        let evidence =
            EvidenceBundle::new(request.request_id, request.model.id, request.context.trace)
                .with_retrieval_hits(request.retrieval_limit as u16)
                .with_policy_decision(PolicyDecision::Allow)
                .with_redaction(RedactionState::SensitiveRedacted);
        Ok(InferenceResult::new(
            request.request_id,
            request.model.id,
            InferenceStatus::Completed,
            self.output,
            usage,
            evidence,
        ))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineDescriptor<'a> {
    /// Component name.
    pub name: &'a str,
    /// Component version marker.
    pub version: u32,
}

impl<'a> EngineDescriptor<'a> {
    /// Creates an engine component descriptor.
    pub const fn new(name: &'a str, version: u32) -> Self {
        Self { name, version }
    }
}
