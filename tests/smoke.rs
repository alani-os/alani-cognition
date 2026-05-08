use alani_cognition::{
    cognition_catalog, BudgetUsage, CognitionEngine, CognitionError, CognitiveContext,
    ComponentStatus, DataClass, DeterministicPlanner, InferenceBudget, InferenceRequest,
    KnowledgeRecord, KnowledgeRecordKind, KnowledgeStore, MockCognitionEngine, ModelCapabilities,
    ModelKind, ModelRef, ModelRegistry, ModelSelection, ModelState, PlanGoal, PlanStep,
    PlanStepKind, PlannerConfig, PolicyDecision, Provenance, RedactionState, RetrievalQuery,
    TraceContext, COGNITION_CATALOG, COGNITION_FEATURE_INFERENCE, COGNITION_KNOWN_FEATURES,
};

fn mock_model() -> ModelRef<'static> {
    ModelRef::new(
        "mock-default",
        "Mock Default",
        "0.1.0",
        ModelKind::Mock,
        ModelCapabilities::RAG,
    )
    .with_limits(2048, 256)
    .with_state(ModelState::Loaded)
}

fn context() -> CognitiveContext<'static> {
    CognitiveContext::new("agent:demo", "session:1", "policy:default")
        .with_memory_scope("demo")
        .with_trace(TraceContext::new(7, 9))
        .with_time(1_000)
}

fn budget() -> InferenceBudget {
    InferenceBudget::bounded(64, 100, 1_000_000, 10_000)
}

fn provenance() -> Provenance<'static> {
    Provenance::new("synthetic-corpus", "internal-test", [3; 32]).with_generation(1)
}

#[test]
fn repository_identity_and_catalog_are_stable() {
    let info = alani_cognition::component_info();

    assert_eq!(alani_cognition::repository_name(), "alani-cognition");
    assert_eq!(info.repository, "alani-cognition");
    assert_eq!(info.status, ComponentStatus::Experimental);
    assert_eq!(
        alani_cognition::module_names(),
        &["engine", "model", "planner", "retrieval"]
    );
    assert_eq!(cognition_catalog(), COGNITION_CATALOG);
    assert_eq!(cognition_catalog().validate(), Ok(()));
    assert_eq!(
        cognition_catalog().features & COGNITION_FEATURE_INFERENCE,
        COGNITION_FEATURE_INFERENCE
    );
    assert_eq!(COGNITION_KNOWN_FEATURES & !cognition_catalog().features, 0);
}

#[test]
fn model_registry_validates_lifecycle_selection_and_duplicates() {
    let mut registry = ModelRegistry::<2>::new();
    let model = mock_model();

    assert_eq!(model.validate(), Ok(()));
    assert!(model.can_infer());
    assert_eq!(registry.register(model), Ok(()));
    assert_eq!(
        registry.register(model),
        Err(CognitionError::DuplicateModel)
    );
    assert_eq!(
        registry
            .select(ModelSelection::inference().with_limits(512, 64))
            .unwrap()
            .id,
        "mock-default"
    );

    assert_eq!(
        registry.set_state("mock-default", ModelState::Unloaded),
        Ok(())
    );
    assert!(registry.select(ModelSelection::inference()).is_none());
    assert_eq!(
        registry.set_state("missing", ModelState::Loaded),
        Err(CognitionError::ModelNotFound)
    );

    let invalid =
        ModelRef::new("", "bad", "0", ModelKind::Mock, ModelCapabilities::INFER).with_limits(1, 1);
    assert_eq!(invalid.validate(), Err(CognitionError::MissingField));
    assert_eq!(
        ModelCapabilities::from_bits(1 << 63),
        Err(CognitionError::ReservedBits)
    );
}

#[test]
fn mock_engine_executes_deterministically_and_reports_evidence() {
    let request = InferenceRequest::new(42, mock_model(), "summarize corpus", budget(), context())
        .with_retrieval_limit(2);
    let result = MockCognitionEngine::DEFAULT.infer(request).unwrap();

    assert_eq!(result.validate(), Ok(()));
    assert_eq!(result.request_id, 42);
    assert_eq!(result.model_id, "mock-default");
    assert_eq!(result.output, "mock inference accepted");
    assert_eq!(result.usage, BudgetUsage::new(3, 10, 1024, 1_000));
    assert_eq!(result.evidence.retrieval_hits, 2);
    assert_eq!(result.evidence.policy_decision, PolicyDecision::Allow);
    assert_eq!(result.evidence.redaction, RedactionState::SensitiveRedacted);
}

#[test]
fn inference_validation_handles_cancel_deadline_and_budget_errors() {
    let cancelled = InferenceRequest::new(43, mock_model(), "cancel me", budget(), context())
        .with_flags(alani_cognition::REQUEST_FLAG_CANCELLED);
    assert_eq!(
        MockCognitionEngine::DEFAULT.infer(cancelled),
        Err(CognitionError::Cancelled)
    );

    let expired_budget = InferenceBudget::bounded(64, 100, 1_000_000, 500);
    let expired = InferenceRequest::new(44, mock_model(), "too late", expired_budget, context());
    assert_eq!(
        MockCognitionEngine::DEFAULT.infer(expired),
        Err(CognitionError::DeadlineExceeded)
    );

    let tiny_budget = InferenceBudget::bounded(1, 1, 1, 10_000);
    let exhausted = InferenceRequest::new(45, mock_model(), "too small", tiny_budget, context());
    assert_eq!(
        MockCognitionEngine::DEFAULT.infer(exhausted),
        Err(CognitionError::BudgetExceeded)
    );

    let invalid_trace = context().with_trace(TraceContext::new(1, 0));
    let invalid_request =
        InferenceRequest::new(46, mock_model(), "bad trace", budget(), invalid_trace);
    assert_eq!(
        invalid_request.validate(),
        Err(CognitionError::InvalidTrace)
    );
}

#[test]
fn retrieval_store_returns_provenance_and_enforces_limits() {
    let mut store = KnowledgeStore::<3>::new();
    let public = KnowledgeRecord::new(
        1,
        "demo",
        "kernel",
        "kernel mediates model access",
        KnowledgeRecordKind::Fact,
        provenance(),
    )
    .with_confidence(9_500)
    .with_data_class(DataClass::Operational);
    let sensitive = KnowledgeRecord::new(
        2,
        "demo",
        "prompt",
        "sensitive prompt summary",
        KnowledgeRecordKind::SemanticChunk,
        provenance(),
    )
    .with_confidence(9_000)
    .with_data_class(DataClass::Sensitive);

    assert_eq!(store.put(public), Ok(()));
    assert_eq!(store.put(sensitive), Ok(()));
    assert_eq!(store.put(public), Err(CognitionError::DuplicateRecord));

    let operational = store
        .query::<4>(
            RetrievalQuery::new("demo", "kernel")
                .with_top_k(4)
                .with_min_confidence(9_000),
        )
        .unwrap();
    assert_eq!(operational.len(), 1);
    assert_eq!(operational.hit(0).unwrap().record.id, 1);
    assert_eq!(
        operational.hit(0).unwrap().record.provenance.source,
        "synthetic-corpus"
    );

    let sensitive_allowed = store
        .query::<1>(
            RetrievalQuery::new("demo", "prompt")
                .with_top_k(2)
                .with_max_data_class(DataClass::Sensitive),
        )
        .unwrap();
    assert_eq!(sensitive_allowed.len(), 1);
    assert!(sensitive_allowed.is_truncated());

    assert_eq!(
        RetrievalQuery::new("", "bad").validate(),
        Err(CognitionError::MissingField)
    );
}

#[test]
fn deterministic_planner_builds_authorization_aware_plan() {
    let goal = PlanGoal::new(
        100,
        "agent:demo",
        "answer using bounded retrieval",
        "policy:default",
        budget(),
    )
    .with_memory_scope("demo")
    .with_trace(TraceContext::new(7, 10));
    let planner = DeterministicPlanner::default();
    let plan = planner.plan::<8>(goal).unwrap();

    assert_eq!(plan.validate(), Ok(()));
    assert_eq!(plan.len(), 4);
    assert_eq!(plan.step(1).unwrap().kind, PlanStepKind::RetrieveContext);
    assert_eq!(plan.status.label(), "ready");

    let mut manual = alani_cognition::Plan::<4>::new(goal);
    assert_eq!(
        manual.push(
            PlanStep::new(
                1,
                PlanStepKind::ToolCall,
                "open accelerator",
                "device:model"
            )
            .with_authorization(true),
        ),
        Ok(())
    );
    assert!(manual.requires_authorization());
    assert_eq!(manual.status.label(), "needs_authorization");
    assert_eq!(
        manual.push(PlanStep::new(1, PlanStepKind::Complete, "done", "result")),
        Err(CognitionError::DuplicatePlanStep)
    );
}

#[test]
fn planner_and_budget_negative_paths_are_typed() {
    let invalid_goal = PlanGoal::new(0, "agent", "bad", "policy", budget());
    assert_eq!(invalid_goal.validate(), Err(CognitionError::MissingField));
    assert_eq!(
        InferenceBudget::unbounded().validate(),
        Err(CognitionError::InvalidBudget)
    );

    let planner = DeterministicPlanner::new(PlannerConfig {
        max_steps: 9,
        require_authorization_for_tools: true,
        emit_evidence: true,
    });
    let goal = PlanGoal::new(101, "agent", "goal", "policy", budget());
    assert_eq!(
        planner.plan::<4>(goal),
        Err(CognitionError::CapacityExceeded)
    );
    assert!(CognitionError::BudgetExceeded.is_security_relevant());
    assert_eq!(DataClass::Secret.label(), "secret");
    assert!(DataClass::Sensitive.requires_redaction());
}
