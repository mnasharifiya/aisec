"""
Unit tests for AISec RBAC.
Run with: pytest tests/unit/test_rbac.py -v
"""

from __future__ import annotations

import pytest

from aisec.security.rbac import (
    AccessDeniedError,
    Permission,
    Principal,
    RBACEnforcer,
    Role,
    ROLE_PERMISSIONS,
    create_principal,
)

# ── Principal construction tests ──────────────────────────────────────────────


class TestPrincipalConstruction:

    def test_creates_valid_analyst(self) -> None:
        p = Principal("analyst_01", Role.ANALYST)
        assert p.principal_id == "analyst_01"
        assert p.role == Role.ANALYST

    def test_creates_valid_admin(self) -> None:
        p = Principal("admin_01", Role.ADMIN)
        assert p.role == Role.ADMIN

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Principal("", Role.ANALYST)

    def test_rejects_short_safe_id(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            Principal(";;", Role.ANALYST)

    def test_sanitises_dangerous_characters(self) -> None:
        p = Principal("analyst;DROP--TABLE", Role.ANALYST)
        assert ";" not in p.principal_id
        assert "--" not in p.principal_id
        # Only alphanumeric, underscore, dot allowed
        assert all(c.isalnum() or c in "_." for c in p.principal_id)

    def test_truncates_long_id(self) -> None:
        p = Principal("a" * 200, Role.ANALYST)
        assert len(p.principal_id) <= 64

    def test_principal_is_immutable(self) -> None:
        p = Principal("analyst_01", Role.ANALYST)
        with pytest.raises((AttributeError, TypeError)):
            p.role = Role.ADMIN  # type: ignore

    def test_principal_id_is_immutable(self) -> None:
        p = Principal("analyst_01", Role.ANALYST)
        with pytest.raises((AttributeError, TypeError)):
            p.principal_id = "attacker"  # type: ignore

    def test_rejects_invalid_role_type(self) -> None:
        with pytest.raises(ValueError, match="Role enum"):
            Principal("analyst_01", "analyst")  # type: ignore


# ── Permission tests ──────────────────────────────────────────────────────────


class TestAnalystPermissions:

    def setup_method(self) -> None:
        self.analyst = Principal("analyst_01", Role.ANALYST)

    def test_analyst_can_view_queue(self) -> None:
        assert self.analyst.has_permission(Permission.VIEW_QUEUE)

    def test_analyst_can_approve_events(self) -> None:
        assert self.analyst.has_permission(Permission.APPROVE_EVENT)

    def test_analyst_can_block_events(self) -> None:
        assert self.analyst.has_permission(Permission.BLOCK_EVENT)

    def test_analyst_can_escalate_events(self) -> None:
        assert self.analyst.has_permission(Permission.ESCALATE_EVENT)

    def test_analyst_can_view_audit_log(self) -> None:
        assert self.analyst.has_permission(Permission.VIEW_AUDIT_LOG)

    def test_analyst_can_verify_chain(self) -> None:
        assert self.analyst.has_permission(Permission.VERIFY_AUDIT_CHAIN)

    def test_analyst_cannot_export_audit_log(self) -> None:
        assert not self.analyst.has_permission(Permission.EXPORT_AUDIT_LOG)

    def test_analyst_cannot_manage_roles(self) -> None:
        assert not self.analyst.has_permission(Permission.MANAGE_ROLES)

    def test_analyst_cannot_modify_thresholds(self) -> None:
        assert not self.analyst.has_permission(Permission.MODIFY_THRESHOLDS)

    def test_analyst_cannot_modify_rules(self) -> None:
        assert not self.analyst.has_permission(Permission.MODIFY_RULES)

    def test_analyst_cannot_clear_queue(self) -> None:
        assert not self.analyst.has_permission(Permission.CLEAR_QUEUE)

    def test_analyst_cannot_view_system_config(self) -> None:
        assert not self.analyst.has_permission(Permission.VIEW_SYSTEM_CONFIG)


class TestAdminPermissions:

    def setup_method(self) -> None:
        self.admin = Principal("admin_01", Role.ADMIN)

    def test_admin_has_all_analyst_permissions(self) -> None:
        analyst_perms = ROLE_PERMISSIONS[Role.ANALYST]
        for perm in analyst_perms:
            assert self.admin.has_permission(
                perm
            ), f"Admin missing analyst permission: {perm.name}"

    def test_admin_can_export_audit_log(self) -> None:
        assert self.admin.has_permission(Permission.EXPORT_AUDIT_LOG)

    def test_admin_can_manage_roles(self) -> None:
        assert self.admin.has_permission(Permission.MANAGE_ROLES)

    def test_admin_can_modify_thresholds(self) -> None:
        assert self.admin.has_permission(Permission.MODIFY_THRESHOLDS)

    def test_admin_can_modify_rules(self) -> None:
        assert self.admin.has_permission(Permission.MODIFY_RULES)

    def test_admin_can_clear_queue(self) -> None:
        assert self.admin.has_permission(Permission.CLEAR_QUEUE)

    def test_admin_can_view_system_config(self) -> None:
        assert self.admin.has_permission(Permission.VIEW_SYSTEM_CONFIG)


# ── Enforcer tests ────────────────────────────────────────────────────────────


class TestRBACEnforcer:

    def setup_method(self) -> None:
        self.enforcer = RBACEnforcer()
        self.analyst = Principal("analyst_01", Role.ANALYST)
        self.admin = Principal("admin_01", Role.ADMIN)

    def test_enforce_grants_permitted_action(self) -> None:
        # Must not raise
        self.enforcer.enforce(self.analyst, Permission.APPROVE_EVENT)

    def test_enforce_raises_for_denied_action(self) -> None:
        with pytest.raises(AccessDeniedError) as exc_info:
            self.enforcer.enforce(self.analyst, Permission.EXPORT_AUDIT_LOG)
        err = exc_info.value
        assert err.principal_id == "analyst_01"
        assert err.role == Role.ANALYST
        assert err.permission == Permission.EXPORT_AUDIT_LOG

    def test_check_returns_true_for_permitted(self) -> None:
        assert self.enforcer.check(self.analyst, Permission.VIEW_QUEUE)

    def test_check_returns_false_for_denied(self) -> None:
        assert not self.enforcer.check(self.analyst, Permission.MANAGE_ROLES)

    def test_admin_enforce_all_permissions(self) -> None:
        for perm in Permission:
            self.enforcer.enforce(self.admin, perm)

    def test_analyst_permitted_commands(self) -> None:
        commands = self.enforcer.get_permitted_commands(self.analyst)
        assert "queue" in commands
        assert "approve" in commands
        assert "block" in commands
        assert "escalate" in commands
        assert "export" not in commands
        assert "roles" not in commands

    def test_admin_permitted_commands(self) -> None:
        commands = self.enforcer.get_permitted_commands(self.admin)
        assert "export" in commands
        assert "roles" in commands
        assert "config" in commands


# ── Access denied error tests ─────────────────────────────────────────────────


class TestAccessDeniedError:

    def test_error_message_contains_principal_id(self) -> None:
        err = AccessDeniedError("analyst_01", Role.ANALYST, Permission.EXPORT_AUDIT_LOG)
        assert "analyst_01" in str(err)

    def test_error_message_contains_role(self) -> None:
        err = AccessDeniedError("analyst_01", Role.ANALYST, Permission.EXPORT_AUDIT_LOG)
        assert "analyst" in str(err)

    def test_error_message_contains_permission(self) -> None:
        err = AccessDeniedError("analyst_01", Role.ANALYST, Permission.EXPORT_AUDIT_LOG)
        assert "EXPORT_AUDIT_LOG" in str(err)


# ── create_principal factory tests ────────────────────────────────────────────


class TestCreatePrincipal:

    def test_creates_analyst_from_string(self) -> None:
        p = create_principal("analyst_01", "analyst")
        assert p.role == Role.ANALYST

    def test_creates_admin_from_string(self) -> None:
        p = create_principal("admin_01", "admin")
        assert p.role == Role.ADMIN

    def test_case_insensitive_role(self) -> None:
        p = create_principal("analyst_01", "ANALYST")
        assert p.role == Role.ANALYST

    def test_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError, match="Invalid role"):
            create_principal("analyst_01", "superuser")

    def test_sanitises_principal_id(self) -> None:
        p = create_principal("analyst;01", "analyst")
        assert ";" not in p.principal_id


# ── No privilege escalation tests ─────────────────────────────────────────────


class TestNoPrivilegeEscalation:
    """
    Critical security tests: no role can escalate its own privileges.
    """

    def test_analyst_cannot_grant_admin_permission(self) -> None:
        analyst = Principal("analyst_01", Role.ANALYST)
        enforcer = RBACEnforcer()
        # Analyst cannot perform MANAGE_ROLES
        with pytest.raises(AccessDeniedError):
            enforcer.enforce(analyst, Permission.MANAGE_ROLES)

    def test_analyst_cannot_modify_own_role(self) -> None:
        analyst = Principal("analyst_01", Role.ANALYST)
        # Frozen dataclass — role is immutable
        with pytest.raises((AttributeError, TypeError)):
            analyst.role = Role.ADMIN  # type: ignore

    def test_different_principals_have_independent_permissions(self) -> None:
        a1 = Principal("analyst_01", Role.ANALYST)
        a2 = Principal("admin_01", Role.ADMIN)
        # Admin permissions do not leak to analyst
        assert not a1.has_permission(Permission.EXPORT_AUDIT_LOG)
        assert a2.has_permission(Permission.EXPORT_AUDIT_LOG)
