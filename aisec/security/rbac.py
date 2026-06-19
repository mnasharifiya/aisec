"""
AISec Role-Based Access Control (RBAC).

This module enforces least-privilege authorization for AISec platform
operations. Every sensitive operation requires an explicit permission grant.
The default behavior is deny.

Security design:
    - Deny by default.
    - Fail closed on invalid principals, invalid permissions, and internal errors.
    - Principals are immutable after creation.
    - Principal identifiers are validated and sanitized against log injection.
    - Permissions are additive: role permissions are unioned with explicit grants.
    - No negative permissions or inheritance chains.
    - Admin-sensitive permissions are explicitly classified.
    - Authorization decisions are structured and audit-friendly.
    - RBAC authorizes actions; callers are responsible for recording the
      resulting business operation in the audit log.

Important production note:
    RBAC is not authentication. Authentication must happen before this layer.
    This layer assumes a caller has already identified the principal and now
    asks whether that principal is authorized to perform a specific operation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import FrozenSet, Iterable

from aisec.utils.logger import get_logger

log = get_logger("aisec.security.rbac")


# ── Constants ─────────────────────────────────────────────────────────────────

MIN_PRINCIPAL_ID_LEN = 3
MAX_PRINCIPAL_ID_LEN = 64
MAX_DISPLAY_NAME_LEN = 128
MAX_OPERATION_LEN = 160

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:@-]+$")
_OPERATION_SAFE_RE = re.compile(r"[^A-Za-z0-9_.:@/\- ]")


# ── Permissions ───────────────────────────────────────────────────────────────


class Permission(Enum):
    """
    Granular permissions for AISec operations.

    Permission groups:
        VIEW_*        — read-only operations
        RESOLVE_*     — analyst queue decisions
        MANAGE_*      — administrative lifecycle operations
        EXPORT_*      — sensitive data export operations
        CONFIGURE_*   — system configuration changes
        VERIFY_*      — integrity verification operations
        RESET_*       — emergency operations
    """

    # Read-only / analyst permissions
    VIEW_EVENTS = auto()
    VIEW_AUDIT_LOG = auto()
    VIEW_METRICS = auto()
    VIEW_QUEUE = auto()
    VIEW_SAFE_STATE = auto()
    VIEW_CORRELATION_ALERTS = auto()
    VIEW_TEMPORAL_ALERTS = auto()

    # Analyst decision permissions
    RESOLVE_QUEUE = auto()
    RESOLVE_ESCALATION = auto()
    ACKNOWLEDGE_ALERT = auto()

    # Admin permissions
    MANAGE_SAFE_STATE = auto()
    MANAGE_SCENARIOS = auto()
    MANAGE_AGENTS = auto()
    MANAGE_ROLES = auto()
    MANAGE_API_KEYS = auto()

    # Sensitive export permissions
    EXPORT_AUDIT_LOG = auto()
    EXPORT_EVENTS = auto()
    EXPORT_METRICS = auto()

    # Configuration permissions
    CONFIGURE_THRESHOLDS = auto()
    CONFIGURE_WEBHOOKS = auto()
    CONFIGURE_API = auto()
    CONFIGURE_TEMPORAL_DETECTION = auto()
    CONFIGURE_CORRELATION_DETECTION = auto()

    # Integrity / emergency permissions
    VERIFY_AUDIT_CHAIN = auto()
    RESET_SAFE_STATE = auto()
    EMERGENCY_SHUTDOWN = auto()


# ── Role definitions ──────────────────────────────────────────────────────────


class Role(Enum):
    """Built-in AISec roles."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SYSTEM = "system"


class PrincipalType(Enum):
    """Type of authenticated identity."""

    HUMAN = "human"
    SERVICE_ACCOUNT = "service_account"
    SYSTEM = "system"


class PrincipalStatus(Enum):
    """Operational status of a principal."""

    ACTIVE = "active"
    DISABLED = "disabled"
    LOCKED = "locked"


# Role → Permission mapping
# This is the authoritative source of truth for built-in roles.

ROLE_PERMISSIONS: dict[Role, FrozenSet[Permission]] = {
    Role.VIEWER: frozenset(
        {
            Permission.VIEW_EVENTS,
            Permission.VIEW_METRICS,
            Permission.VIEW_QUEUE,
            Permission.VIEW_SAFE_STATE,
            Permission.VIEW_CORRELATION_ALERTS,
            Permission.VIEW_TEMPORAL_ALERTS,
            Permission.VERIFY_AUDIT_CHAIN,
        }
    ),
    Role.ANALYST: frozenset(
        {
            Permission.VIEW_EVENTS,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_METRICS,
            Permission.VIEW_QUEUE,
            Permission.VIEW_SAFE_STATE,
            Permission.VIEW_CORRELATION_ALERTS,
            Permission.VIEW_TEMPORAL_ALERTS,
            Permission.RESOLVE_QUEUE,
            Permission.RESOLVE_ESCALATION,
            Permission.ACKNOWLEDGE_ALERT,
            Permission.VERIFY_AUDIT_CHAIN,
        }
    ),
    Role.ADMIN: frozenset(
        {
            # Viewer / analyst permissions
            Permission.VIEW_EVENTS,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_METRICS,
            Permission.VIEW_QUEUE,
            Permission.VIEW_SAFE_STATE,
            Permission.VIEW_CORRELATION_ALERTS,
            Permission.VIEW_TEMPORAL_ALERTS,
            Permission.RESOLVE_QUEUE,
            Permission.RESOLVE_ESCALATION,
            Permission.ACKNOWLEDGE_ALERT,
            Permission.VERIFY_AUDIT_CHAIN,
            # Admin permissions
            Permission.MANAGE_SAFE_STATE,
            Permission.MANAGE_SCENARIOS,
            Permission.MANAGE_AGENTS,
            Permission.MANAGE_ROLES,
            Permission.MANAGE_API_KEYS,
            Permission.EXPORT_AUDIT_LOG,
            Permission.EXPORT_EVENTS,
            Permission.EXPORT_METRICS,
            Permission.CONFIGURE_THRESHOLDS,
            Permission.CONFIGURE_WEBHOOKS,
            Permission.CONFIGURE_API,
            Permission.CONFIGURE_TEMPORAL_DETECTION,
            Permission.CONFIGURE_CORRELATION_DETECTION,
            Permission.RESET_SAFE_STATE,
            Permission.EMERGENCY_SHUTDOWN,
        }
    ),
    Role.SYSTEM: frozenset(
        {
            # Internal service-level access. Use carefully.
            Permission.VIEW_EVENTS,
            Permission.VIEW_AUDIT_LOG,
            Permission.VIEW_METRICS,
            Permission.VIEW_QUEUE,
            Permission.VIEW_SAFE_STATE,
            Permission.VIEW_CORRELATION_ALERTS,
            Permission.VIEW_TEMPORAL_ALERTS,
            Permission.RESOLVE_QUEUE,
            Permission.ACKNOWLEDGE_ALERT,
            Permission.VERIFY_AUDIT_CHAIN,
            Permission.EXPORT_METRICS,
        }
    ),
}


ADMIN_SENSITIVE_PERMISSIONS: FrozenSet[Permission] = frozenset(
    {
        Permission.MANAGE_SAFE_STATE,
        Permission.MANAGE_SCENARIOS,
        Permission.MANAGE_AGENTS,
        Permission.MANAGE_ROLES,
        Permission.MANAGE_API_KEYS,
        Permission.EXPORT_AUDIT_LOG,
        Permission.EXPORT_EVENTS,
        Permission.CONFIGURE_THRESHOLDS,
        Permission.CONFIGURE_WEBHOOKS,
        Permission.CONFIGURE_API,
        Permission.CONFIGURE_TEMPORAL_DETECTION,
        Permission.CONFIGURE_CORRELATION_DETECTION,
        Permission.RESET_SAFE_STATE,
        Permission.EMERGENCY_SHUTDOWN,
    }
)


# ── Helper functions ──────────────────────────────────────────────────────────


def _sanitize_operation(operation: str) -> str:
    """Return an audit-safe operation name."""
    if not operation:
        return ""

    text = str(operation)

    # Replace control/line-break characters with spaces so words do not merge.
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Remove unsafe characters but keep audit-readable separators.
    text = _OPERATION_SAFE_RE.sub("", text)

    # Collapse repeated whitespace to one space.
    text = " ".join(text.split())

    return text[:MAX_OPERATION_LEN]


def _validate_principal_id(principal_id: str) -> str:
    """Validate and return a safe principal identifier."""
    if not isinstance(principal_id, str):
        raise ValueError("principal_id must be a string")

    principal_id = principal_id.strip()

    if not principal_id:
        raise ValueError("principal_id cannot be empty")

    if len(principal_id) < MIN_PRINCIPAL_ID_LEN:
        raise ValueError(
            f"principal_id must be at least {MIN_PRINCIPAL_ID_LEN} characters"
        )

    if len(principal_id) > MAX_PRINCIPAL_ID_LEN:
        raise ValueError(
            f"principal_id too long: {len(principal_id)} chars "
            f"(max {MAX_PRINCIPAL_ID_LEN})"
        )

    if not _SAFE_ID_RE.fullmatch(principal_id):
        raise ValueError(
            "principal_id contains unsafe characters. Allowed characters: "
            "letters, digits, underscore, hyphen, dot, colon, at-sign"
        )

    return principal_id


def _sanitize_display_name(display_name: str) -> str:
    """Sanitize display name for logging/UI."""
    if not display_name:
        return ""

    text = str(display_name)

    # Replace control/line-break characters with spaces so words do not merge.
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    # Collapse repeated whitespace to one space.
    text = " ".join(text.split())

    return text[:MAX_DISPLAY_NAME_LEN]


def is_admin_sensitive(permission: Permission) -> bool:
    """Return True if permission is admin-sensitive."""
    return permission in ADMIN_SENSITIVE_PERMISSIONS


# ── Principal ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Principal:
    """
    Authenticated identity in AISec.

    Frozen:
        A Principal cannot be mutated after creation.

    explicit_permissions:
        Optional additional permissions granted to this identity. These are
        additive only. They cannot remove role permissions.

    status:
        DISABLED and LOCKED principals are always denied.
    """

    principal_id: str
    role: Role
    display_name: str = ""
    principal_type: PrincipalType = PrincipalType.HUMAN
    status: PrincipalStatus = PrincipalStatus.ACTIVE
    explicit_permissions: FrozenSet[Permission] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        safe_id = _validate_principal_id(self.principal_id)
        safe_display_name = _sanitize_display_name(self.display_name)

        if not isinstance(self.role, Role):
            raise ValueError("role must be an instance of Role")

        if not isinstance(self.principal_type, PrincipalType):
            raise ValueError("principal_type must be an instance of PrincipalType")

        if not isinstance(self.status, PrincipalStatus):
            raise ValueError("status must be an instance of PrincipalStatus")

        for permission in self.explicit_permissions:
            if not isinstance(permission, Permission):
                raise ValueError("explicit_permissions must contain Permission values")

        object.__setattr__(self, "principal_id", safe_id)
        object.__setattr__(self, "display_name", safe_display_name)
        object.__setattr__(
            self,
            "explicit_permissions",
            frozenset(self.explicit_permissions),
        )

    @property
    def is_active(self) -> bool:
        """Return True if this principal is allowed to be authorized."""
        return self.status == PrincipalStatus.ACTIVE

    @property
    def role_permissions(self) -> FrozenSet[Permission]:
        """Return permissions granted by the principal's role."""
        return ROLE_PERMISSIONS.get(self.role, frozenset())

    @property
    def permissions(self) -> FrozenSet[Permission]:
        """Return effective permissions for this principal."""
        if not self.is_active:
            return frozenset()
        return self.role_permissions | self.explicit_permissions

    def has_permission(self, permission: Permission) -> bool:
        """Return True if this principal has the given permission."""
        if not isinstance(permission, Permission):
            return False
        return permission in self.permissions

    def to_audit_dict(self) -> dict[str, str]:
        """Return safe identity details for audit payloads."""
        return {
            "principal_id": self.principal_id,
            "role": self.role.value,
            "principal_type": self.principal_type.value,
            "status": self.status.value,
            "display_name": self.display_name,
        }

    def __repr__(self) -> str:
        return (
            f"Principal(id={self.principal_id!r}, "
            f"role={self.role.value!r}, "
            f"type={self.principal_type.value!r}, "
            f"status={self.status.value!r})"
        )


# ── Authorization result ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthorizationDecision:
    """Structured result of an authorization check."""

    allowed: bool
    principal_id: str
    role: str
    permission: str
    operation: str
    reason: str
    admin_sensitive: bool
    timestamp: float = field(default_factory=time.time)


# ── Access denied error ───────────────────────────────────────────────────────


class AccessDeniedError(Exception):
    """
    Raised when authorization fails.

    This class is defensive: it does not assume the principal object is valid.
    """

    def __init__(
        self,
        principal: Principal | object | None,
        permission: Permission | object,
        operation: str = "",
        reason: str = "missing_permission",
    ) -> None:
        self.principal = principal
        self.permission = permission
        self.operation = _sanitize_operation(operation)
        self.reason = reason

        principal_id = self._safe_principal_id(principal)
        role = self._safe_role(principal)
        permission_name = (
            permission.name if isinstance(permission, Permission) else str(permission)
        )

        super().__init__(
            f"Access denied: principal '{principal_id}' "
            f"(role={role}) lacks permission '{permission_name}'"
            + (f" for operation '{self.operation}'" if self.operation else "")
            + f" reason={reason}"
        )

    @staticmethod
    def _safe_principal_id(principal: Principal | object | None) -> str:
        try:
            return str(getattr(principal, "principal_id", "unknown"))[
                :MAX_PRINCIPAL_ID_LEN
            ]
        except Exception:
            return "unknown"

    @staticmethod
    def _safe_role(principal: Principal | object | None) -> str:
        try:
            role = getattr(principal, "role", "unknown")
            return role.value if isinstance(role, Role) else str(role)
        except Exception:
            return "unknown"


# ── RBAC enforcer ─────────────────────────────────────────────────────────────


class RBACEnforcer:
    """
    Enforces role-based access control for AISec operations.

    Thread-safety:
        Stateless. All checks are pure functions except structured logging.

    Fail-closed:
        Unexpected errors deny access.
    """

    def require(
        self,
        principal: Principal,
        permission: Permission,
        operation: str = "",
    ) -> AuthorizationDecision:
        """
        Require a single permission.

        Returns:
            AuthorizationDecision if allowed.

        Raises:
            AccessDeniedError if denied.
        """
        decision = self.authorize(principal, permission, operation)

        if not decision.allowed:
            raise AccessDeniedError(
                principal=principal,
                permission=permission,
                operation=operation,
                reason=decision.reason,
            )

        return decision

    def require_any(
        self,
        principal: Principal,
        permissions: Iterable[Permission],
        operation: str = "",
    ) -> AuthorizationDecision:
        """
        Require at least one permission from a set.

        Useful for operations that can be performed through multiple roles.
        """
        permissions = list(permissions)
        operation = _sanitize_operation(operation)

        if not permissions:
            raise AccessDeniedError(
                principal=principal,
                permission=Permission.VIEW_EVENTS,
                operation=operation,
                reason="empty_permission_set",
            )

        for permission in permissions:
            decision = self.authorize(principal, permission, operation)
            if decision.allowed:
                return decision

        first_permission = permissions[0]
        raise AccessDeniedError(
            principal=principal,
            permission=first_permission,
            operation=operation,
            reason="missing_any_required_permission",
        )

    def require_all(
        self,
        principal: Principal,
        permissions: Iterable[Permission],
        operation: str = "",
    ) -> list[AuthorizationDecision]:
        """Require all permissions from a set."""
        decisions: list[AuthorizationDecision] = []

        for permission in permissions:
            decisions.append(self.require(principal, permission, operation))

        return decisions

    def authorize(
        self,
        principal: Principal,
        permission: Permission,
        operation: str = "",
    ) -> AuthorizationDecision:
        """
        Check permission and return a structured decision.

        Does not raise on denial. Unexpected internal errors are converted into
        denied AuthorizationDecision objects.
        """
        operation = _sanitize_operation(operation)

        try:
            principal_id, role = self._principal_identity(principal)

            if not isinstance(principal, Principal):
                return self._deny(
                    principal_id=principal_id,
                    role=role,
                    permission=permission,
                    operation=operation,
                    reason="invalid_principal",
                )

            if not isinstance(permission, Permission):
                return self._deny(
                    principal_id=principal_id,
                    role=role,
                    permission=permission,
                    operation=operation,
                    reason="invalid_permission",
                )

            if not principal.is_active:
                return self._deny(
                    principal_id=principal.principal_id,
                    role=principal.role.value,
                    permission=permission,
                    operation=operation,
                    reason=f"principal_{principal.status.value}",
                )

            if not principal.has_permission(permission):
                return self._deny(
                    principal_id=principal.principal_id,
                    role=principal.role.value,
                    permission=permission,
                    operation=operation,
                    reason="missing_permission",
                )

            decision = AuthorizationDecision(
                allowed=True,
                principal_id=principal.principal_id,
                role=principal.role.value,
                permission=permission.name,
                operation=operation,
                reason="allowed",
                admin_sensitive=is_admin_sensitive(permission),
            )

            log.info(
                "access_granted",
                principal_id=decision.principal_id,
                role=decision.role,
                permission=decision.permission,
                operation=decision.operation,
                admin_sensitive=decision.admin_sensitive,
            )

            return decision

        except Exception as exc:
            log.error(
                "rbac_authorize_error",
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
            )

            principal_id, role = self._principal_identity(principal)
            return AuthorizationDecision(
                allowed=False,
                principal_id=principal_id,
                role=role,
                permission=(
                    permission.name
                    if isinstance(permission, Permission)
                    else str(permission)
                ),
                operation=operation,
                reason="rbac_internal_error",
                admin_sensitive=False,
            )

    def check(
        self,
        principal: Principal,
        permission: Permission,
    ) -> bool:
        """
        Check permission without raising.

        Returns False on all errors.
        """
        try:
            return self.authorize(principal, permission).allowed
        except Exception:
            return False

    def list_permissions(self, principal: Principal) -> list[Permission]:
        """Return effective permissions for a principal."""
        try:
            return sorted(
                principal.permissions,
                key=lambda permission: permission.name,
            )
        except Exception:
            return []

    def list_role_permissions(self, role: Role) -> list[Permission]:
        """Return permissions for a built-in role."""
        if not isinstance(role, Role):
            return []

        return sorted(
            ROLE_PERMISSIONS.get(role, frozenset()),
            key=lambda permission: permission.name,
        )

    def has_admin_sensitive_permission(self, principal: Principal) -> bool:
        """Return True if principal has any admin-sensitive permission."""
        try:
            return bool(principal.permissions & ADMIN_SENSITIVE_PERMISSIONS)
        except Exception:
            return False

    def _deny(
        self,
        *,
        principal_id: str,
        role: str,
        permission: Permission | object,
        operation: str,
        reason: str,
    ) -> AuthorizationDecision:
        permission_name = (
            permission.name if isinstance(permission, Permission) else str(permission)
        )
        admin_sensitive = (
            is_admin_sensitive(permission)
            if isinstance(permission, Permission)
            else False
        )

        decision = AuthorizationDecision(
            allowed=False,
            principal_id=principal_id,
            role=role,
            permission=permission_name,
            operation=operation,
            reason=reason,
            admin_sensitive=admin_sensitive,
        )

        log.warning(
            "access_denied",
            principal_id=decision.principal_id,
            role=decision.role,
            permission=decision.permission,
            operation=decision.operation,
            reason=decision.reason,
            admin_sensitive=decision.admin_sensitive,
        )

        return decision

    @staticmethod
    def _principal_identity(principal: Principal | object | None) -> tuple[str, str]:
        try:
            principal_id = str(getattr(principal, "principal_id", "unknown"))[
                :MAX_PRINCIPAL_ID_LEN
            ]
        except Exception:
            principal_id = "unknown"

        try:
            role_obj = getattr(principal, "role", "unknown")
            role = role_obj.value if isinstance(role_obj, Role) else str(role_obj)
        except Exception:
            role = "unknown"

        return principal_id or "unknown", role or "unknown"
