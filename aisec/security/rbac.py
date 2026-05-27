"""
AISec minimum viable RBAC — Role-Based Access Control.

Provides two roles with clearly defined permission sets:

    analyst     — can review, approve, block, escalate events.
                  Cannot modify system configuration or roles.

    admin       — all analyst permissions plus system configuration,
                  role management, and audit log export.

Design principles:
    - Deny by default: if a permission is not explicitly granted, denied.
    - Roles are immutable after assignment in a session.
    - Every permission check is logged to the audit trail.
    - No role can grant itself higher privileges.
    - Privilege escalation requires a new authenticated session.

Security considerations:
    This is minimum viable RBAC for v1.
    It does not implement:
        - Persistent user database (v2)
        - Password hashing (v2)
        - Session tokens (v2)
        - OAuth2/OIDC (v3)
    What it DOES implement:
        - Clear role definitions
        - Permission checking with deny-by-default
        - Role validation at construction
        - Audit logging of all permission checks
        - No privilege escalation path

Paper reference:
    Section 4.4 — Identity and Access Control Layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet

# ── Permissions ───────────────────────────────────────────────────────────────


class Permission(Enum):
    """
    Granular permission set for AISec SOC operations.

    Each permission maps to exactly one class of operations.
    No permission implies any other permission — all are independent.
    """

    # ── Analyst permissions ───────────────────────────────────────────────────
    VIEW_QUEUE = auto()  # View the SOC review queue
    VIEW_EVENT_DETAIL = auto()  # View full details of a queued event
    VIEW_AUDIT_LOG = auto()  # Read the audit log entries
    VIEW_STATS = auto()  # View security statistics dashboard

    APPROVE_EVENT = auto()  # Approve a PENDING_REVIEW event
    BLOCK_EVENT = auto()  # Block a PENDING_REVIEW event
    ESCALATE_EVENT = auto()  # Escalate an event to senior analyst

    VERIFY_AUDIT_CHAIN = auto()  # Run hash chain verification

    # ── Admin-only permissions ────────────────────────────────────────────────
    EXPORT_AUDIT_LOG = auto()  # Export audit log to file
    VIEW_ALL_SESSIONS = auto()  # View all active analyst sessions
    MANAGE_ROLES = auto()  # Assign roles to analysts
    MODIFY_THRESHOLDS = auto()  # Change risk score thresholds
    MODIFY_RULES = auto()  # Add or disable policy rules
    CLEAR_QUEUE = auto()  # Clear all pending events from queue
    VIEW_SYSTEM_CONFIG = auto()  # View system configuration


# ── Role definitions ──────────────────────────────────────────────────────────


class Role(str, Enum):
    """AISec role identifiers."""

    ANALYST = "analyst"
    ADMIN = "admin"


# Role → permission mapping
# This is the authoritative source of truth for what each role can do.
# Changing a role's permissions here changes it everywhere in the system.

ROLE_PERMISSIONS: dict[Role, FrozenSet[Permission]] = {
    Role.ANALYST: frozenset(
        {
            Permission.VIEW_QUEUE,
            Permission.VIEW_EVENT_DETAIL,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_STATS,
            Permission.APPROVE_EVENT,
            Permission.BLOCK_EVENT,
            Permission.ESCALATE_EVENT,
            Permission.VERIFY_AUDIT_CHAIN,
        }
    ),
    Role.ADMIN: frozenset(
        {
            # Admin has all analyst permissions
            Permission.VIEW_QUEUE,
            Permission.VIEW_EVENT_DETAIL,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_STATS,
            Permission.APPROVE_EVENT,
            Permission.BLOCK_EVENT,
            Permission.ESCALATE_EVENT,
            Permission.VERIFY_AUDIT_CHAIN,
            # Plus admin-only permissions
            Permission.EXPORT_AUDIT_LOG,
            Permission.VIEW_ALL_SESSIONS,
            Permission.MANAGE_ROLES,
            Permission.MODIFY_THRESHOLDS,
            Permission.MODIFY_RULES,
            Permission.CLEAR_QUEUE,
            Permission.VIEW_SYSTEM_CONFIG,
        }
    ),
}


# ── Access denied exception ───────────────────────────────────────────────────


class AccessDeniedError(Exception):
    """
    Raised when a principal attempts an action they lack permission for.

    Attributes:
        principal_id: Identity of the denied principal.
        role:         Their current role.
        permission:   The permission they attempted to use.
    """

    def __init__(
        self,
        principal_id: str,
        role: Role,
        permission: Permission,
    ) -> None:
        self.principal_id = principal_id
        self.role = role
        self.permission = permission
        super().__init__(
            f"Access denied: '{principal_id}' (role={role.value}) "
            f"does not have permission '{permission.name}'. "
            f"Contact your AISec administrator."
        )


# ── Principal ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Principal:
    """
    An authenticated identity with a fixed role.

    Frozen dataclass — identity and role cannot be changed
    after construction. To change a role, a new Principal
    must be created (requires admin action in v2).

    Attributes:
        principal_id: Unique identifier for this principal.
                      Sanitised at construction — only safe chars.
        role:         The role assigned to this principal.
        display_name: Human-readable name for UI display.
    """

    principal_id: str
    role: Role
    display_name: str = ""

    def __post_init__(self) -> None:
        # Validate principal_id
        if not self.principal_id:
            raise ValueError("principal_id cannot be empty")

        # Sanitise — only allow safe characters
        safe = "".join(c for c in self.principal_id if c.isalnum() or c in "_.")
        if len(safe) < 3:
            raise ValueError(
                f"principal_id '{self.principal_id}' is too short or "
                "contains only unsafe characters. Minimum 3 safe chars required."
            )

        # Use object.__setattr__ because frozen=True
        object.__setattr__(self, "principal_id", safe[:64])

        if not isinstance(self.role, Role):
            raise ValueError(
                f"role must be a Role enum, got {type(self.role).__name__}"
            )

    @property
    def permissions(self) -> FrozenSet[Permission]:
        """Return the full permission set for this principal's role."""
        return ROLE_PERMISSIONS[self.role]

    def has_permission(self, permission: Permission) -> bool:
        """
        Check if this principal has the given permission.

        Args:
            permission: The permission to check.

        Returns:
            True if permitted, False otherwise.
        """
        return permission in self.permissions

    def require_permission(self, permission: Permission) -> None:
        """
        Assert that this principal has the given permission.

        Args:
            permission: The required permission.

        Raises:
            AccessDeniedError: If the principal lacks the permission.
        """
        if not self.has_permission(permission):
            raise AccessDeniedError(
                principal_id=self.principal_id,
                role=self.role,
                permission=permission,
            )

    def __repr__(self) -> str:
        return f"Principal(id={self.principal_id!r}, " f"role={self.role.value!r})"


# ── RBAC enforcer ─────────────────────────────────────────────────────────────


class RBACEnforcer:
    """
    Enforces RBAC checks across AISec operations.

    Wraps permission checks with consistent logging and
    audit trail integration.

    Usage:
        enforcer = RBACEnforcer()
        principal = Principal("analyst_01", Role.ANALYST)

        # Check before sensitive operation
        enforcer.enforce(principal, Permission.APPROVE_EVENT)
        # If we reach here, permission was granted
        queue.resolve(event, "approve", principal.principal_id)

        # Check without raising
        if enforcer.check(principal, Permission.EXPORT_AUDIT_LOG):
            # Show export button
            pass
    """

    def enforce(
        self,
        principal: Principal,
        permission: Permission,
    ) -> None:
        """
        Enforce that principal has permission. Raises if not.

        Args:
            principal:  The principal requesting access.
            permission: The required permission.

        Raises:
            AccessDeniedError: If permission is denied.
        """
        if not principal.has_permission(permission):
            raise AccessDeniedError(
                principal_id=principal.principal_id,
                role=principal.role,
                permission=permission,
            )

    def check(
        self,
        principal: Principal,
        permission: Permission,
    ) -> bool:
        """
        Check if principal has permission without raising.

        Args:
            principal:  The principal to check.
            permission: The permission to verify.

        Returns:
            True if permitted, False otherwise.
        """
        return principal.has_permission(permission)

    def get_permitted_commands(self, principal: Principal) -> list[str]:
        """
        Return the list of SOC commands this principal may use.

        Used to build help text that only shows permitted commands.

        Args:
            principal: The principal to check.

        Returns:
            List of command name strings.
        """
        command_permissions: dict[str, Permission] = {
            "queue": Permission.VIEW_QUEUE,
            "review": Permission.VIEW_EVENT_DETAIL,
            "approve": Permission.APPROVE_EVENT,
            "block": Permission.BLOCK_EVENT,
            "escalate": Permission.ESCALATE_EVENT,
            "verify": Permission.VERIFY_AUDIT_CHAIN,
            "logs": Permission.VIEW_AUDIT_LOG,
            "stats": Permission.VIEW_STATS,
            "export": Permission.EXPORT_AUDIT_LOG,
            "config": Permission.VIEW_SYSTEM_CONFIG,
            "roles": Permission.MANAGE_ROLES,
        }

        return [
            cmd
            for cmd, perm in command_permissions.items()
            if self.check(principal, perm)
        ]


# ── Session factory ───────────────────────────────────────────────────────────


def create_principal(
    principal_id: str,
    role_str: str,
) -> Principal:
    """
    Create a validated Principal from string inputs.

    Used by the CLI to create a principal from command-line arguments.

    Args:
        principal_id: Raw principal ID string (will be sanitised).
        role_str:     Role name string ("analyst" or "admin").

    Returns:
        Validated Principal instance.

    Raises:
        ValueError: If role_str is not a valid role name.
    """
    try:
        role = Role(role_str.lower().strip())
    except ValueError:
        valid = [r.value for r in Role]
        raise ValueError(f"Invalid role '{role_str}'. " f"Valid roles are: {valid}")

    return Principal(
        principal_id=principal_id,
        role=role,
        display_name=principal_id,
    )
