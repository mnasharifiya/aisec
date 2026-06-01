"""
AISec configuration loader.

Loads configuration from aisec.yaml with environment
variable overrides. All settings have documented defaults.

Priority order (highest to lowest):
    1. Environment variables (AISEC_*)
    2. aisec.yaml in current directory
    3. Built-in defaults

Usage:
    config = load_config()
    config = load_config(Path("custom/aisec.yaml"))

    print(config["engine"]["log_path"])
    print(config["thresholds"]["block"])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from aisec.utils.logger import get_logger

log = get_logger("aisec.config")

DEFAULT_CONFIG_PATH = Path("aisec.yaml")

# ── Built-in defaults ─────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "engine": {
        "log_path": ".aisec/audit.jsonl",
        "enable_temporal": True,
    },
    "thresholds": {
        "block": 0.80,
        "review": 0.60,
        "watch": 0.30,
    },
    "temporal": {
        "window_seconds": 60.0,
        "burst_threshold": 20,
        "probe_threshold": 5,
        "escalation_delta": 0.15,
        "escalation_min_events": 5,
        "cumulative_amount_threshold": 5_000_000.0,
        "evasion_threshold": 2,
    },
    "api": {
        "host": "127.0.0.1",
        "port": 8000,
        "enable_cors": True,
        "allowed_origins": ["*"],
    },
    "logging": {
        "level": "INFO",
        "output": "stderr",
    },
    "webhooks": [],
    "scenarios": {
        "directory": "scenarios",
        "load_all": True,
    },
    "safe_state": {
        "auto_enter_on_critical": True,
        "require_admin_release": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge override into base dict.
    Override values take precedence over base values.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides.

    Convention: AISEC_SECTION_KEY maps to config[section][key]
    Example: AISEC_ENGINE_LOG_PATH → config["engine"]["log_path"]
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("AISEC_"):
            continue

        parts = env_key[6:].lower().split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
        if section in config and isinstance(config[section], dict):
            # Type-coerce based on existing value
            existing = config[section].get(key)
            if isinstance(existing, bool):
                config[section][key] = env_val.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                try:
                    config[section][key] = int(env_val)
                except ValueError:
                    pass
            elif isinstance(existing, float):
                try:
                    config[section][key] = float(env_val)
                except ValueError:
                    pass
            else:
                config[section][key] = env_val

    return config


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load AISec configuration.

    Args:
        path: Path to the YAML configuration file.
              Defaults to aisec.yaml in current directory.

    Returns:
        Configuration dict with all settings populated.
        Missing settings fall back to built-in defaults.
    """
    config = dict(DEFAULTS)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                file_config = yaml.safe_load(fh) or {}
            config = _deep_merge(config, file_config)
            log.info("config_loaded", path=str(path))
        except Exception as exc:
            log.warning(
                "config_load_failed",
                path=str(path),
                exc_type=type(exc).__name__,
                detail=str(exc)[:200],
                fallback="using defaults",
            )
    else:
        log.info("config_file_not_found", path=str(path), fallback="using defaults")

    config = _apply_env_overrides(config)

    return config


def get_config() -> dict[str, Any]:
    """
    Return the current configuration.
    Loads from default path if not already loaded.
    """
    return load_config()
