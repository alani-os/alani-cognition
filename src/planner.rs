//! Deterministic planning contracts for model-assisted workflows.
//!
//! The planner skeleton emits bounded, inspectable plans that can be audited
//! before any privileged tool, device, or memory operation is attempted.

use crate::engine::{InferenceBudget, TraceContext};
use crate::{CognitionError, CognitionResult, DataClass};

/// Maximum goal description length.
pub const MAX_GOAL_DESCRIPTION_LEN: usize = 512;

/// Maximum plan-step description length.
pub const MAX_STEP_DESCRIPTION_LEN: usize = 256;

/// Maximum plan resource label length.
pub const MAX_PLAN_RESOURCE_LEN: usize = 128;

/// Default maximum plan steps for host-mode planning.
pub const DEFAULT_MAX_PLAN_STEPS: usize = 8;

/// Planning priority.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanPriority {
    /// Low-priority background planning.
    Low = 1,
    /// Normal planning priority.
    Normal = 2,
    /// Urgent planning priority within explicit budget.
    Urgent = 3,
}

impl PlanPriority {
    /// Stable priority label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::Urgent => "urgent",
        }
    }
}

/// Plan step kind.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanStepKind {
    /// Normalize or validate input.
    NormalizeInput = 1,
    /// Retrieve memory context.
    RetrieveContext = 2,
    /// Invoke model inference.
    RunInference = 3,
    /// Call a policy-authorized tool or device.
    ToolCall = 4,
    /// Emit audit/evidence record.
    EmitEvidence = 5,
    /// Finish and return result.
    Complete = 6,
}

impl PlanStepKind {
    /// Stable step-kind label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NormalizeInput => "normalize_input",
            Self::RetrieveContext => "retrieve_context",
            Self::RunInference => "run_inference",
            Self::ToolCall => "tool_call",
            Self::EmitEvidence => "emit_evidence",
            Self::Complete => "complete",
        }
    }
}

/// Plan status.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanStatus {
    /// Plan is being built.
    Draft = 1,
    /// Plan is ready for execution.
    Ready = 2,
    /// Plan requires authorization before execution.
    NeedsAuthorization = 3,
    /// Plan was denied by policy.
    Denied = 4,
}

impl PlanStatus {
    /// Stable status label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Draft => "draft",
            Self::Ready => "ready",
            Self::NeedsAuthorization => "needs_authorization",
            Self::Denied => "denied",
        }
    }
}

/// Planner goal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanGoal<'a> {
    /// Stable goal identifier.
    pub goal_id: u64,
    /// Principal requesting the plan.
    pub principal: &'a str,
    /// Goal description.
    pub description: &'a str,
    /// Policy label.
    pub policy_label: &'a str,
    /// Memory scope.
    pub memory_scope: &'a str,
    /// Budget for the plan.
    pub budget: InferenceBudget,
    /// Trace context.
    pub trace: TraceContext,
    /// Priority.
    pub priority: PlanPriority,
    /// Data sensitivity.
    pub data_class: DataClass,
}

impl<'a> PlanGoal<'a> {
    /// Creates a plan goal.
    pub const fn new(
        goal_id: u64,
        principal: &'a str,
        description: &'a str,
        policy_label: &'a str,
        budget: InferenceBudget,
    ) -> Self {
        Self {
            goal_id,
            principal,
            description,
            policy_label,
            memory_scope: "default",
            budget,
            trace: TraceContext::EMPTY,
            priority: PlanPriority::Normal,
            data_class: DataClass::Operational,
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

    /// Sets priority.
    pub const fn with_priority(mut self, priority: PlanPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets data class.
    pub const fn with_data_class(mut self, data_class: DataClass) -> Self {
        self.data_class = data_class;
        self
    }

    /// Validates goal metadata and budget.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.goal_id == 0
            || self.principal.is_empty()
            || self.description.is_empty()
            || self.policy_label.is_empty()
            || self.memory_scope.is_empty()
        {
            return Err(CognitionError::MissingField);
        }
        if self.description.len() > MAX_GOAL_DESCRIPTION_LEN
            || self.principal.len() > MAX_PLAN_RESOURCE_LEN
            || self.policy_label.len() > MAX_PLAN_RESOURCE_LEN
            || self.memory_scope.len() > MAX_PLAN_RESOURCE_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        match self.budget.validate() {
            Ok(()) => self.trace.validate(),
            Err(error) => Err(error),
        }
    }
}

/// One bounded plan step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanStep<'a> {
    /// Step identifier unique within a plan.
    pub step_id: u16,
    /// Step kind.
    pub kind: PlanStepKind,
    /// Step description.
    pub description: &'a str,
    /// Resource or subsystem label.
    pub resource: &'a str,
    /// Whether this step requires authorization before execution.
    pub requires_authorization: bool,
    /// Estimated compute units.
    pub estimated_compute_units: u32,
}

impl<'a> PlanStep<'a> {
    /// Creates a plan step.
    pub const fn new(
        step_id: u16,
        kind: PlanStepKind,
        description: &'a str,
        resource: &'a str,
    ) -> Self {
        Self {
            step_id,
            kind,
            description,
            resource,
            requires_authorization: false,
            estimated_compute_units: 0,
        }
    }

    /// Marks whether authorization is required.
    pub const fn with_authorization(mut self, requires_authorization: bool) -> Self {
        self.requires_authorization = requires_authorization;
        self
    }

    /// Sets estimated compute units.
    pub const fn with_compute_units(mut self, estimated_compute_units: u32) -> Self {
        self.estimated_compute_units = estimated_compute_units;
        self
    }

    /// Validates step metadata.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.step_id == 0 || self.description.is_empty() || self.resource.is_empty() {
            return Err(CognitionError::MissingField);
        }
        if self.description.len() > MAX_STEP_DESCRIPTION_LEN
            || self.resource.len() > MAX_PLAN_RESOURCE_LEN
        {
            return Err(CognitionError::FieldTooLong);
        }
        Ok(())
    }
}

/// Planner configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlannerConfig {
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Require authorization for tool/device steps.
    pub require_authorization_for_tools: bool,
    /// Emit audit/evidence step.
    pub emit_evidence: bool,
}

impl PlannerConfig {
    /// Default host-mode planner config.
    pub const DEFAULT: Self = Self {
        max_steps: DEFAULT_MAX_PLAN_STEPS,
        require_authorization_for_tools: true,
        emit_evidence: true,
    };

    /// Validates planner config.
    pub const fn validate(self) -> CognitionResult<()> {
        if self.max_steps == 0 {
            Err(CognitionError::InvalidPlan)
        } else {
            Ok(())
        }
    }
}

/// Fixed-capacity plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan<'a, const N: usize> {
    /// Goal metadata.
    pub goal: PlanGoal<'a>,
    /// Plan status.
    pub status: PlanStatus,
    steps: [Option<PlanStep<'a>>; N],
    len: usize,
}

impl<'a, const N: usize> Plan<'a, N> {
    /// Creates an empty draft plan.
    pub const fn new(goal: PlanGoal<'a>) -> Self {
        Self {
            goal,
            status: PlanStatus::Draft,
            steps: [None; N],
            len: 0,
        }
    }

    /// Returns number of steps.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` when the plan has no steps.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Adds a validated step.
    pub fn push(&mut self, step: PlanStep<'a>) -> CognitionResult<()> {
        step.validate()?;
        if self.len == N {
            return Err(CognitionError::CapacityExceeded);
        }
        if self.find_step(step.step_id).is_some() {
            return Err(CognitionError::DuplicatePlanStep);
        }
        self.steps[self.len] = Some(step);
        self.len += 1;
        self.refresh_status();
        Ok(())
    }

    /// Returns `true` when any step requires authorization.
    pub fn requires_authorization(&self) -> bool {
        self.iter().any(|step| step.requires_authorization)
    }

    /// Finds a step by id.
    pub fn find_step(&self, step_id: u16) -> Option<&PlanStep<'a>> {
        self.iter().find(|step| step.step_id == step_id)
    }

    /// Returns a step by zero-based index.
    pub fn step(&self, index: usize) -> Option<&PlanStep<'a>> {
        if index < self.len {
            self.steps[index].as_ref()
        } else {
            None
        }
    }

    /// Iterates over plan steps.
    pub fn iter(&self) -> impl Iterator<Item = &PlanStep<'a>> {
        self.steps[..self.len].iter().filter_map(Option::as_ref)
    }

    /// Validates goal and all steps.
    pub fn validate(&self) -> CognitionResult<()> {
        self.goal.validate()?;
        if self.len == 0 {
            return Err(CognitionError::InvalidPlan);
        }
        for step in self.iter() {
            step.validate()?;
        }
        Ok(())
    }

    fn refresh_status(&mut self) {
        self.status = if self.requires_authorization() {
            PlanStatus::NeedsAuthorization
        } else {
            PlanStatus::Ready
        };
    }
}

/// Deterministic host-mode planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeterministicPlanner {
    /// Planner configuration.
    pub config: PlannerConfig,
}

impl DeterministicPlanner {
    /// Creates a deterministic planner.
    pub const fn new(config: PlannerConfig) -> Self {
        Self { config }
    }

    /// Creates a bounded plan for a goal.
    pub fn plan<'a, const N: usize>(&self, goal: PlanGoal<'a>) -> CognitionResult<Plan<'a, N>> {
        self.config.validate()?;
        goal.validate()?;
        if N == 0 || self.config.max_steps > N {
            return Err(CognitionError::CapacityExceeded);
        }

        let mut plan = Plan::new(goal);
        plan.push(PlanStep::new(
            1,
            PlanStepKind::NormalizeInput,
            "normalize goal input",
            "cognition:input",
        ))?;
        plan.push(
            PlanStep::new(
                2,
                PlanStepKind::RetrieveContext,
                "retrieve scoped memory context",
                goal.memory_scope,
            )
            .with_compute_units(1),
        )?;
        plan.push(
            PlanStep::new(
                3,
                PlanStepKind::RunInference,
                "run bounded model inference",
                "model:selected",
            )
            .with_compute_units(10),
        )?;
        if self.config.emit_evidence {
            plan.push(PlanStep::new(
                4,
                PlanStepKind::EmitEvidence,
                "emit traceable audit evidence",
                "audit:cognition",
            ))?;
        }
        Ok(plan)
    }
}

impl Default for DeterministicPlanner {
    fn default() -> Self {
        Self::new(PlannerConfig::DEFAULT)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannerDescriptor<'a> {
    /// Component name.
    pub name: &'a str,
    /// Component version marker.
    pub version: u32,
}

impl<'a> PlannerDescriptor<'a> {
    /// Creates a planner component descriptor.
    pub const fn new(name: &'a str, version: u32) -> Self {
        Self { name, version }
    }
}
