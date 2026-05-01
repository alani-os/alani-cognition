#[test]
fn repository_identity_is_stable() {
    assert_eq!(alani_cognition::repository_name(), "alani-cognition");
    assert!(!alani_cognition::module_names().is_empty());
}
