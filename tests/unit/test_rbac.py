"""
Unit tests for the AISec RBAC system.

Run with:
    pytest tests/unit/test_rbac.py -v
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile

import pytest

from aisec.security.rbac import (
    ADMIN_SENSITIVE_PERMISSIONS,
    AccessDeniedError,
    AuthorizationDecision,
    Permission,
    Principal,
    PrincipalStatus,
    PrincipalType,
    RBACEnforcer,
    Role,
    ROLE_PERMISSIONS,
    is_admin_sensitive,
)


@pytest.fixture
def enforcer() -> RBACEnforcer:
    return RBACEnforcer()


@pytest.fixture
def viewer() -> Principal:
    return Principal(
        principal_id="viewer_01",
        role=Role.VIEWER,
        display_name="Victor Viewer",
    )


@pytest.fixture
def analyst() -> Principal:
    return Principal(
        principal_id="analyst_01",
        role=Role.ANALYST,
        display_name="Alice Analyst",
    )


@pytest.fixture
def admin() -> Principal:
    return Principal(
        principal_id="admin_01",
        role=Role.ADMIN,
        display_name="Bob Admin",
    )


@pytest.fixture
def system_principal() -> Principal:
    return Principal(
        principal_id="system:engine",
        role=Role.SYSTEM,
        display_name="AISec Engine",
        principal_type=PrincipalType.SYSTEM,
    )


class TestPrincipal:
    def test_creates_valid_principal(self, analyst: Principal) -> None:
        assert analyst.principal_id == "analyst_01"
        assert analyst.role == Role.ANALYST
        assert analyst.display_name == "Alice Analyst"
        assert analyst.principal_type == PrincipalType.HUMAN
        assert analyst.status == PrincipalStatus.ACTIVE

    def test_strips_surrounding_whitespace_from_principal_id(self) -> None:
        principal = Principal(principal_id="  analyst_01  ", role=Role.ANALYST)
        assert principal.principal_id == "analyst_01"

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Principal(principal_id="", role=Role.ANALYST)

    def test_rejects_whitespace_only_id(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Principal(principal_id="   ", role=Role.ANALYST)

    def test_rejects_short_id(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            Principal(principal_id="ab", role=Role.ANALYST)

    def test_rejects_long_id(self) -> None:
        with pytest.raises(ValueError, match="too long"):
            Principal(principal_id="a" * 65, role=Role.ANALYST)

    def test_rejects_unsafe_principal_id_characters(self) -> None:
        bad_ids = [
            "admin\nfake",
            "admin\tfake",
            "admin fake",
            "admin;drop",
            "admin/slash",
            "admin\\slash",
        ]

        for bad_id in bad_ids:
            with pytest.raises(ValueError, match="unsafe characters"):
                Principal(principal_id=bad_id, role=Role.ADMIN)

    def test_allows_safe_principal_id_characters(self) -> None:
        safe_ids = [
            "admin_01",
            "admin-01",
            "admin.01",
            "user@example.com",
            "service:engine",
        ]

        for safe_id in safe_ids:
            principal = Principal(principal_id=safe_id, role=Role.ANALYST)
            assert principal.principal_id == safe_id

    def test_rejects_invalid_role_type(self) -> None:
        with pytest.raises(ValueError, match="role must be"):
            Principal(  # type: ignore[arg-type]
                principal_id="analyst_01",
                role="analyst",
            )

    def test_rejects_invalid_principal_type(self) -> None:
        with pytest.raises(ValueError, match="principal_type"):
            Principal(  # type: ignore[arg-type]
                principal_id="analyst_01",
                role=Role.ANALYST,
                principal_type="human",
            )

    def test_rejects_invalid_status(self) -> None:
        with pytest.raises(ValueError, match="status"):
            Principal(  # type: ignore[arg-type]
                principal_id="analyst_01",
                role=Role.ANALYST,
                status="active",
            )

    def test_rejects_invalid_explicit_permission_type(self) -> None:
        with pytest.raises(ValueError, match="explicit_permissions"):
            Principal(  # type: ignore[arg-type]
                principal_id="analyst_01",
                role=Role.ANALYST,
                explicit_permissions=frozenset({"bad_permission"}),
            )

    def test_principal_is_frozen(self, analyst: Principal) -> None:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            analyst.role = Role.ADMIN  # type: ignore[misc]

    def test_display_name_is_sanitized(self) -> None:
        principal = Principal(
            principal_id="analyst_01",
            role=Role.ANALYST,
            display_name="Alice\nAnalyst\tName",
        )
        assert principal.display_name == "Alice Analyst Name"

    def test_display_name_is_truncated(self) -> None:
        principal = Principal(
            principal_id="analyst_01",
            role=Role.ANALYST,
            display_name="x" * 200,
        )
        assert len(principal.display_name) == 128

    def test_has_permission_returns_true_for_granted(self, analyst: Principal) -> None:
        assert analyst.has_permission(Permission.VIEW_EVENTS) is True

    def test_has_permission_returns_false_for_denied(self, analyst: Principal) -> None:
        assert analyst.has_permission(Permission.MANAGE_SAFE_STATE) is False

    def test_has_permission_returns_false_for_invalid_permission(
        self, analyst: Principal
    ) -> None:
        assert analyst.has_permission("VIEW_EVENTS") is False  # type: ignore[arg-type]

    def test_permissions_property_returns_frozenset(self, analyst: Principal) -> None:
        assert isinstance(analyst.permissions, frozenset)

    def test_explicit_permissions_are_additive(self) -> None:
        principal = Principal(
            principal_id="special_analyst",
            role=Role.ANALYST,
            explicit_permissions=frozenset({Permission.EXPORT_EVENTS}),
        )

        assert Permission.VIEW_EVENTS in principal.permissions
        assert Permission.EXPORT_EVENTS in principal.permissions
        assert Permission.MANAGE_SAFE_STATE not in principal.permissions

    def test_disabled_principal_has_no_effective_permissions(self) -> None:
        principal = Principal(
            principal_id="disabled_admin",
            role=Role.ADMIN,
            status=PrincipalStatus.DISABLED,
        )

        assert principal.is_active is False
        assert principal.permissions == frozenset()
        assert principal.has_permission(Permission.MANAGE_SAFE_STATE) is False

    def test_locked_principal_has_no_effective_permissions(self) -> None:
        principal = Principal(
            principal_id="locked_admin",
            role=Role.ADMIN,
            status=PrincipalStatus.LOCKED,
        )

        assert principal.is_active is False
        assert principal.permissions == frozenset()
        assert principal.has_permission(Permission.MANAGE_SAFE_STATE) is False

    def test_to_audit_dict_contains_safe_identity_fields(
        self, analyst: Principal
    ) -> None:
        audit_dict = analyst.to_audit_dict()

        assert audit_dict["principal_id"] == "analyst_01"
        assert audit_dict["role"] == "analyst"
        assert audit_dict["principal_type"] == "human"
        assert audit_dict["status"] == "active"
        assert audit_dict["display_name"] == "Alice Analyst"


class TestRolePermissions:
    def test_viewer_has_read_only_permissions(self) -> None:
        perms = ROLE_PERMISSIONS[Role.VIEWER]

        assert Permission.VIEW_EVENTS in perms
        assert Permission.VIEW_METRICS in perms
        assert Permission.VIEW_QUEUE in perms
        assert Permission.VIEW_SAFE_STATE in perms
        assert Permission.RESOLVE_QUEUE not in perms
        assert Permission.MANAGE_SAFE_STATE not in perms

    def test_analyst_has_view_and_resolution_permissions(self) -> None:
        perms = ROLE_PERMISSIONS[Role.ANALYST]

        assert Permission.VIEW_EVENTS in perms
        assert Permission.VIEW_AUDIT_LOG in perms
        assert Permission.VIEW_QUEUE in perms
        assert Permission.RESOLVE_QUEUE in perms
        assert Permission.RESOLVE_ESCALATION in perms
        assert Permission.ACKNOWLEDGE_ALERT in perms

    def test_analyst_cannot_manage_safe_state(self) -> None:
        perms = ROLE_PERMISSIONS[Role.ANALYST]

        assert Permission.MANAGE_SAFE_STATE not in perms
        assert Permission.RESET_SAFE_STATE not in perms
        assert Permission.CONFIGURE_THRESHOLDS not in perms
        assert Permission.EMERGENCY_SHUTDOWN not in perms

    def test_admin_has_all_analyst_permissions(self) -> None:
        analyst_perms = ROLE_PERMISSIONS[Role.ANALYST]
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]

        for permission in analyst_perms:
            assert (
                permission in admin_perms
            ), f"Admin missing analyst permission: {permission.name}"

    def test_admin_has_admin_sensitive_permissions(self) -> None:
        perms = ROLE_PERMISSIONS[Role.ADMIN]

        assert Permission.MANAGE_SAFE_STATE in perms
        assert Permission.RESET_SAFE_STATE in perms
        assert Permission.CONFIGURE_THRESHOLDS in perms
        assert Permission.EXPORT_AUDIT_LOG in perms
        assert Permission.EMERGENCY_SHUTDOWN in perms

    def test_system_role_is_not_full_admin(self) -> None:
        perms = ROLE_PERMISSIONS[Role.SYSTEM]

        assert Permission.VIEW_EVENTS in perms
        assert Permission.VERIFY_AUDIT_CHAIN in perms
        assert Permission.EXPORT_METRICS in perms
        assert Permission.MANAGE_SAFE_STATE not in perms
        assert Permission.MANAGE_ROLES not in perms
        assert Permission.EMERGENCY_SHUTDOWN not in perms

    def test_admin_sensitive_permissions_are_classified(self) -> None:
        assert Permission.MANAGE_SAFE_STATE in ADMIN_SENSITIVE_PERMISSIONS
        assert Permission.CONFIGURE_API in ADMIN_SENSITIVE_PERMISSIONS
        assert Permission.EMERGENCY_SHUTDOWN in ADMIN_SENSITIVE_PERMISSIONS

    def test_read_only_permissions_are_not_admin_sensitive(self) -> None:
        assert is_admin_sensitive(Permission.VIEW_EVENTS) is False
        assert is_admin_sensitive(Permission.VIEW_METRICS) is False

    def test_admin_permissions_are_admin_sensitive(self) -> None:
        assert is_admin_sensitive(Permission.MANAGE_SAFE_STATE) is True
        assert is_admin_sensitive(Permission.EXPORT_AUDIT_LOG) is True


class TestRBACEnforcer:
    def test_authorize_returns_decision_for_allowed(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        decision = enforcer.authorize(
            admin,
            Permission.MANAGE_SAFE_STATE,
            "release agent",
        )

        assert isinstance(decision, AuthorizationDecision)
        assert decision.allowed is True
        assert decision.principal_id == "admin_01"
        assert decision.role == "admin"
        assert decision.permission == "MANAGE_SAFE_STATE"
        assert decision.operation == "release agent"
        assert decision.reason == "allowed"
        assert decision.admin_sensitive is True

    def test_authorize_returns_decision_for_denied(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        decision = enforcer.authorize(
            analyst,
            Permission.MANAGE_SAFE_STATE,
            "release agent",
        )

        assert isinstance(decision, AuthorizationDecision)
        assert decision.allowed is False
        assert decision.principal_id == "analyst_01"
        assert decision.role == "analyst"
        assert decision.permission == "MANAGE_SAFE_STATE"
        assert decision.operation == "release agent"
        assert decision.reason == "missing_permission"
        assert decision.admin_sensitive is True

    def test_require_returns_decision_for_authorised_principal(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        decision = enforcer.require(
            admin,
            Permission.MANAGE_SAFE_STATE,
            "release agent",
        )

        assert isinstance(decision, AuthorizationDecision)
        assert decision.allowed is True

    def test_require_denies_permission_to_unauthorised_principal(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(
                analyst,
                Permission.MANAGE_SAFE_STATE,
                "release agent",
            )

        err = exc_info.value
        assert err.principal.principal_id == "analyst_01"
        assert err.permission == Permission.MANAGE_SAFE_STATE
        assert err.operation == "release agent"
        assert err.reason == "missing_permission"

    def test_check_returns_true_for_authorised(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        assert enforcer.check(admin, Permission.MANAGE_SAFE_STATE) is True

    def test_check_returns_false_for_unauthorised(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        assert enforcer.check(analyst, Permission.MANAGE_SAFE_STATE) is False

    def test_check_returns_false_for_invalid_principal(
        self, enforcer: RBACEnforcer
    ) -> None:
        assert enforcer.check(None, Permission.VIEW_EVENTS) is False  # type: ignore[arg-type]

    def test_check_returns_false_for_invalid_permission(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        assert enforcer.check(admin, "VIEW_EVENTS") is False  # type: ignore[arg-type]

    def test_analyst_can_resolve_queue(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        decision = enforcer.require(
            analyst,
            Permission.RESOLVE_QUEUE,
            "approve event",
        )
        assert decision.allowed is True

    def test_analyst_can_view_metrics(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        decision = enforcer.require(
            analyst,
            Permission.VIEW_METRICS,
            "view dashboard",
        )
        assert decision.allowed is True

    def test_analyst_cannot_export_audit_log(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(analyst, Permission.EXPORT_AUDIT_LOG)

    def test_analyst_cannot_configure_thresholds(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(analyst, Permission.CONFIGURE_THRESHOLDS)

    def test_viewer_can_view_events(
        self, enforcer: RBACEnforcer, viewer: Principal
    ) -> None:
        decision = enforcer.require(
            viewer,
            Permission.VIEW_EVENTS,
            "view events",
        )
        assert decision.allowed is True

    def test_viewer_cannot_resolve_queue(
        self, enforcer: RBACEnforcer, viewer: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(viewer, Permission.RESOLVE_QUEUE)

    def test_system_principal_can_verify_audit_chain(
        self, enforcer: RBACEnforcer, system_principal: Principal
    ) -> None:
        decision = enforcer.require(
            system_principal,
            Permission.VERIFY_AUDIT_CHAIN,
            "verify audit chain",
        )
        assert decision.allowed is True

    def test_system_principal_cannot_manage_roles(
        self, enforcer: RBACEnforcer, system_principal: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require(system_principal, Permission.MANAGE_ROLES)

    def test_disabled_principal_is_denied_even_if_admin(
        self, enforcer: RBACEnforcer
    ) -> None:
        disabled_admin = Principal(
            principal_id="disabled_admin",
            role=Role.ADMIN,
            status=PrincipalStatus.DISABLED,
        )

        decision = enforcer.authorize(
            disabled_admin,
            Permission.MANAGE_SAFE_STATE,
            "release agent",
        )

        assert decision.allowed is False
        assert decision.reason == "principal_disabled"

        with pytest.raises(AccessDeniedError):
            enforcer.require(disabled_admin, Permission.MANAGE_SAFE_STATE)

    def test_locked_principal_is_denied_even_if_admin(
        self, enforcer: RBACEnforcer
    ) -> None:
        locked_admin = Principal(
            principal_id="locked_admin",
            role=Role.ADMIN,
            status=PrincipalStatus.LOCKED,
        )

        decision = enforcer.authorize(
            locked_admin,
            Permission.MANAGE_SAFE_STATE,
            "release agent",
        )

        assert decision.allowed is False
        assert decision.reason == "principal_locked"

    def test_explicit_permission_allows_limited_escalation(
        self, enforcer: RBACEnforcer
    ) -> None:
        principal = Principal(
            principal_id="metrics_exporter",
            role=Role.VIEWER,
            principal_type=PrincipalType.SERVICE_ACCOUNT,
            explicit_permissions=frozenset({Permission.EXPORT_METRICS}),
        )

        assert enforcer.check(principal, Permission.VIEW_EVENTS) is True
        assert enforcer.check(principal, Permission.EXPORT_METRICS) is True
        assert enforcer.check(principal, Permission.EXPORT_AUDIT_LOG) is False

    def test_operation_is_sanitized_in_decision(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        decision = enforcer.authorize(
            admin,
            Permission.VIEW_EVENTS,
            "view\n\t events !!!",
        )

        assert decision.allowed is True
        assert "\n" not in decision.operation
        assert "\t" not in decision.operation

    def test_require_any_allows_if_one_permission_matches(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        decision = enforcer.require_any(
            analyst,
            [
                Permission.MANAGE_SAFE_STATE,
                Permission.RESOLVE_QUEUE,
            ],
            "resolve or release",
        )

        assert decision.allowed is True
        assert decision.permission == "RESOLVE_QUEUE"

    def test_require_any_denies_if_none_match(
        self, enforcer: RBACEnforcer, viewer: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require_any(
                viewer,
                [
                    Permission.RESOLVE_QUEUE,
                    Permission.MANAGE_SAFE_STATE,
                ],
                "resolve or release",
            )

        assert exc_info.value.reason == "missing_any_required_permission"

    def test_require_any_denies_empty_permission_set(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require_any(admin, [], "empty permission operation")

        assert exc_info.value.reason == "empty_permission_set"

    def test_require_all_allows_if_all_permissions_match(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        decisions = enforcer.require_all(
            admin,
            [
                Permission.VIEW_EVENTS,
                Permission.MANAGE_SAFE_STATE,
                Permission.CONFIGURE_THRESHOLDS,
            ],
            "admin operation",
        )

        assert len(decisions) == 3
        assert all(decision.allowed for decision in decisions)

    def test_require_all_denies_if_one_permission_missing(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError):
            enforcer.require_all(
                analyst,
                [
                    Permission.VIEW_EVENTS,
                    Permission.MANAGE_SAFE_STATE,
                ],
                "mixed operation",
            )

    def test_list_permissions_returns_sorted_list(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        permissions = enforcer.list_permissions(analyst)

        assert isinstance(permissions, list)
        assert len(permissions) > 0
        names = [permission.name for permission in permissions]
        assert names == sorted(names)

    def test_list_permissions_returns_empty_for_invalid_principal(
        self, enforcer: RBACEnforcer
    ) -> None:
        permissions = enforcer.list_permissions(None)  # type: ignore[arg-type]
        assert permissions == []

    def test_list_role_permissions_returns_sorted_list(
        self, enforcer: RBACEnforcer
    ) -> None:
        permissions = enforcer.list_role_permissions(Role.ADMIN)

        assert isinstance(permissions, list)
        assert Permission.MANAGE_SAFE_STATE in permissions
        names = [permission.name for permission in permissions]
        assert names == sorted(names)

    def test_list_role_permissions_returns_empty_for_invalid_role(
        self, enforcer: RBACEnforcer
    ) -> None:
        permissions = enforcer.list_role_permissions("admin")  # type: ignore[arg-type]
        assert permissions == []

    def test_has_admin_sensitive_permission_for_admin(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        assert enforcer.has_admin_sensitive_permission(admin) is True

    def test_has_admin_sensitive_permission_for_analyst(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        assert enforcer.has_admin_sensitive_permission(analyst) is False

    def test_has_admin_sensitive_permission_fail_closed_for_invalid(
        self, enforcer: RBACEnforcer
    ) -> None:
        assert enforcer.has_admin_sensitive_permission(None) is False  # type: ignore[arg-type]


class TestAccessDeniedError:
    def test_error_message_contains_principal_id(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(analyst, Permission.MANAGE_SAFE_STATE)

        assert "analyst_01" in str(exc_info.value)

    def test_error_message_contains_role(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(analyst, Permission.MANAGE_SAFE_STATE)

        assert "analyst" in str(exc_info.value)

    def test_error_message_contains_permission(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(analyst, Permission.MANAGE_SAFE_STATE)

        assert "MANAGE_SAFE_STATE" in str(exc_info.value)

    def test_error_message_contains_reason(
        self, enforcer: RBACEnforcer, analyst: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(analyst, Permission.MANAGE_SAFE_STATE)

        assert "reason=missing_permission" in str(exc_info.value)

    def test_access_denied_error_handles_invalid_principal(self) -> None:
        error = AccessDeniedError(
            principal=None,
            permission=Permission.VIEW_EVENTS,
            operation="view events",
            reason="invalid_principal",
        )

        assert "unknown" in str(error)
        assert "VIEW_EVENTS" in str(error)
        assert error.reason == "invalid_principal"

    def test_access_denied_error_handles_invalid_permission(
        self, analyst: Principal
    ) -> None:
        error = AccessDeniedError(
            principal=analyst,
            permission="BAD_PERMISSION",
            operation="bad operation",
            reason="invalid_permission",
        )

        assert "BAD_PERMISSION" in str(error)
        assert error.reason == "invalid_permission"


class TestRBACFailClosed:
    def test_authorize_denies_invalid_principal_object(
        self, enforcer: RBACEnforcer
    ) -> None:
        class FakePrincipal:
            principal_id = "fake_01"
            role = "admin"

        decision = enforcer.authorize(
            FakePrincipal(),  # type: ignore[arg-type]
            Permission.VIEW_EVENTS,
            "view events",
        )

        assert decision.allowed is False
        assert decision.reason == "invalid_principal"

    def test_require_denies_invalid_principal_object(
        self, enforcer: RBACEnforcer
    ) -> None:
        class FakePrincipal:
            principal_id = "fake_01"
            role = "admin"

        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(
                FakePrincipal(),  # type: ignore[arg-type]
                Permission.VIEW_EVENTS,
                "view events",
            )

        assert exc_info.value.reason == "invalid_principal"

    def test_authorize_denies_invalid_permission(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        decision = enforcer.authorize(
            admin,
            "MANAGE_SAFE_STATE",  # type: ignore[arg-type]
            "release agent",
        )

        assert decision.allowed is False
        assert decision.reason == "invalid_permission"

    def test_require_denies_invalid_permission(
        self, enforcer: RBACEnforcer, admin: Principal
    ) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(
                admin,
                "MANAGE_SAFE_STATE",  # type: ignore[arg-type]
                "release agent",
            )

        assert exc_info.value.reason == "invalid_permission"


class TestRBACWithSafeState:
    """
    Integration tests: RBAC enforcement on safe-state operations.

    Critical rule:
        Only principals with MANAGE_SAFE_STATE can release agents from safe state.
    """

    def test_admin_can_exit_safe_state(self) -> None:
        from aisec.security.safe_state import SafeStateEnforcer
        from aisec.storage.audit import AuditLogger

        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(Path(tmp) / "rbac_test.jsonl")
            enforcer = RBACEnforcer()
            safe_state = SafeStateEnforcer(audit_logger=logger)
            admin = Principal("admin_01", Role.ADMIN)

            safe_state.enter_safe_state("bot_v1", "test", "BURST_ATTACK")

            decision = enforcer.require(
                admin,
                Permission.MANAGE_SAFE_STATE,
                "release agent from safe state",
            )
            assert decision.allowed is True

            released = safe_state.exit_safe_state(
                "bot_v1",
                admin.principal_id,
            )
            assert released is True

    def test_analyst_cannot_exit_safe_state(self) -> None:
        enforcer = RBACEnforcer()
        analyst = Principal("analyst_01", Role.ANALYST)

        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(
                analyst,
                Permission.MANAGE_SAFE_STATE,
                "release agent from safe state",
            )

        assert exc_info.value.permission == Permission.MANAGE_SAFE_STATE
        assert exc_info.value.reason == "missing_permission"

    def test_viewer_cannot_exit_safe_state(self) -> None:
        enforcer = RBACEnforcer()
        viewer = Principal("viewer_01", Role.VIEWER)

        with pytest.raises(AccessDeniedError):
            enforcer.require(
                viewer,
                Permission.MANAGE_SAFE_STATE,
                "release agent from safe state",
            )

    def test_disabled_admin_cannot_exit_safe_state(self) -> None:
        enforcer = RBACEnforcer()
        disabled_admin = Principal(
            "disabled_admin",
            Role.ADMIN,
            status=PrincipalStatus.DISABLED,
        )

        with pytest.raises(AccessDeniedError) as exc_info:
            enforcer.require(
                disabled_admin,
                Permission.MANAGE_SAFE_STATE,
                "release agent from safe state",
            )

        assert exc_info.value.reason == "principal_disabled"
