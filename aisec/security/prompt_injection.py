"""
AISec Prompt Injection Detector.

Multi-stage heuristic detector for identifying prompt-injection attempts
in autonomous AI agent inputs, retrieved content, and tool-call arguments.

Security design:
    - Privacy-preserving by default: raw text is not logged.
    - Evasion-aware normalization:
        * Unicode NFKC normalization
        * zero-width/control character removal
        * selected homoglyph folding
        * whitespace and separator normalization
        * cautious base64 decoding only when decoded content is instruction-like
    - Context-aware severity:
        the same suspicious phrase is more dangerous inside tool arguments
        for execute_python/delete_file/execute_trade than inside normal text.
    - Nested payload analysis:
        checks dictionary keys, values, lists, tuples, and nested structures.
    - Thread-safe cache for API/server use.
    - Returns recommended enforcement action:
        ALLOW, WATCH, REVIEW, or BLOCK.

Important limitation:
    This module does not "solve" prompt injection. It is a hardened
    first-generation runtime detector. Research claims require evaluation
    against real prompt-injection datasets, red-team attempts, and baselines.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from aisec.utils.logger import get_logger

log = get_logger("aisec.security.prompt_injection")


class InjectionType(str, Enum):
    """Classification of prompt-injection attack types."""

    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    JAILBREAK_PATTERN = "jailbreak_pattern"
    ROLE_OVERRIDE = "role_override"
    INSTRUCTION_SMUGGLING = "instruction_smuggling"
    CONTEXT_MANIPULATION = "context_manipulation"
    TOOL_ABUSE_PATTERN = "tool_abuse_pattern"
    ENCODED_OBFUSCATION = "encoded_obfuscation"
    UNICODE_EVASION = "unicode_evasion"
    UNKNOWN = "unknown"


class InjectionSeverity(str, Enum):
    """Severity assigned to the detection result."""

    CLEAN = "clean"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendedAction(str, Enum):
    """Recommended AISec enforcement action."""

    ALLOW = "allow"
    WATCH = "watch"
    REVIEW = "review"
    BLOCK = "block"


class InputSource(str, Enum):
    """Where the analysed text came from."""

    USER_INPUT = "user_input"
    RETRIEVED_DOCUMENT = "retrieved_document"
    WEB_PAGE = "web_page"
    EMAIL = "email"
    TOOL_ARGUMENTS = "tool_arguments"
    API_RESPONSE = "api_response"
    SYSTEM_OUTPUT = "system_output"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PromptInjectionContext:
    """
    Runtime context for prompt-injection analysis.

    source:
        Where the input came from.

    tool_name:
        Tool/function being called, if applicable.

    scenario:
        AISec scenario name, e.g. trading_ai or urban_ai.

    high_risk_tool:
        Explicit override for tools known to be high risk.
    """

    source: InputSource = InputSource.UNKNOWN
    tool_name: str | None = None
    scenario: str | None = None
    high_risk_tool: bool = False


@dataclass(frozen=True)
class PatternRule:
    """Single prompt-injection detection rule."""

    rule_id: str
    injection_type: InjectionType
    pattern: re.Pattern[str]
    base_confidence: float
    severity: InjectionSeverity
    description: str


@dataclass
class InjectionDetectionResult:
    """
    Result of prompt-injection analysis.

    This result intentionally stores rule IDs and hashes, not raw prompt text.
    """

    is_injection: bool
    injection_type: InjectionType | None
    severity: InjectionSeverity
    confidence: float
    risk_score: float
    recommended_action: RecommendedAction
    matched_rule_ids: list[str] = field(default_factory=list)
    matched_categories: list[str] = field(default_factory=list)
    semantic_score: float = 0.0
    text_hash: str = ""
    text_length: int = 0
    normalized_length: int = 0
    detection_time_ms: float = 0.0
    explanation: str = ""

    @classmethod
    def clean(
        cls,
        *,
        text_hash: str = "",
        text_length: int = 0,
        normalized_length: int = 0,
        detection_time_ms: float = 0.0,
    ) -> "InjectionDetectionResult":
        return cls(
            is_injection=False,
            injection_type=None,
            severity=InjectionSeverity.CLEAN,
            confidence=0.0,
            risk_score=0.0,
            recommended_action=RecommendedAction.ALLOW,
            text_hash=text_hash,
            text_length=text_length,
            normalized_length=normalized_length,
            detection_time_ms=detection_time_ms,
            explanation="No prompt-injection indicators detected.",
        )


class TextNormalizer:
    """
    Normalizes text before detection.

    The goal is not to decode every possible obfuscation. The goal is to
    remove common evasion layers without creating excessive false positives.
    """

    ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
    CONTROL_RE = re.compile(r"[\u0000-\u0008\u000e-\u001f\u007f]")
    WHITESPACE_RE = re.compile(r"\s+")

    HOMOGLYPH_MAP = str.maketrans(
        {
            "а": "a",
            "е": "e",
            "о": "o",
            "р": "p",
            "с": "c",
            "х": "x",
            "і": "i",
            "ј": "j",
            "у": "y",
            "А": "A",
            "В": "B",
            "Е": "E",
            "К": "K",
            "М": "M",
            "Н": "H",
            "О": "O",
            "Р": "P",
            "С": "C",
            "Т": "T",
            "Х": "X",
            "𝚊": "a",
            "𝚎": "e",
            "𝚒": "i",
            "𝚗": "n",
            "𝚘": "o",
            "𝚌": "c",
            "𝗂": "i",
            "𝗇": "n",
            "𝗌": "s",
            "𝗍": "t",
            "𝗋": "r",
            "𝗎": "u",
            "𝖼": "c",
        }
    )

    INSTRUCTION_KEYWORDS = {
        "ignore",
        "previous",
        "instructions",
        "instruction",
        "system",
        "prompt",
        "override",
        "bypass",
        "developer",
        "jailbreak",
        "unrestricted",
        "execute",
        "run",
        "delete",
        "disable",
        "safety",
        "security",
        "admin",
        "tool",
        "function",
    }

    BASE64_BLOCK_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")

    def __init__(self, max_text_len: int = 10_000, decode_base64: bool = True) -> None:
        self.max_text_len = max_text_len
        self.decode_base64 = decode_base64

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text[: self.max_text_len]
        text = unicodedata.normalize("NFKC", text)
        text = text.translate(self.HOMOGLYPH_MAP)
        text = self.ZERO_WIDTH_RE.sub("", text)
        text = self.CONTROL_RE.sub("", text)

        # Treat separators often used in keys/tool names as spaces so that
        # ignore_previous_instructions becomes detectable.
        text = re.sub(r"[_\-]+", " ", text)

        text = self.WHITESPACE_RE.sub(" ", text).strip()

        if self.decode_base64:
            text = self._decode_suspicious_base64_blocks(text)

        return text

    def _decode_suspicious_base64_blocks(self, text: str) -> str:
        """
        Decode base64 only when decoded text looks printable and instruction-like.

        This avoids flagging harmless IDs, hashes, JWT fragments, or tokens
        merely because they look base64-like.
        """

        def replace_block(match: re.Match[str]) -> str:
            block = match.group(0)
            decoded = self._safe_b64decode(block)
            if decoded is None:
                return block

            decoded_norm = unicodedata.normalize("NFKC", decoded)
            decoded_lower = decoded_norm.lower()

            tokens = set(re.findall(r"[a-zA-Z]{3,}", decoded_lower))
            keyword_hits = tokens & self.INSTRUCTION_KEYWORDS
            printable_ratio = self._printable_ratio(decoded_norm)

            if keyword_hits and printable_ratio >= 0.85:
                return f" {decoded_norm} "

            return block

        return self.BASE64_BLOCK_RE.sub(replace_block, text)

    @staticmethod
    def _safe_b64decode(block: str) -> str | None:
        try:
            padded = block + "=" * ((4 - len(block) % 4) % 4)
            raw = base64.b64decode(padded, validate=True)
            if len(raw) > 4096:
                return None
            decoded = raw.decode("utf-8", errors="ignore")
            return decoded if decoded else None
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _printable_ratio(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for ch in text if ch.isprintable()) / len(text)


class PromptInjectionPatterns:
    """Centralized pattern database."""

    @staticmethod
    def build() -> list[PatternRule]:
        flags = re.IGNORECASE | re.DOTALL

        return [
            PatternRule(
                "PI-DIRECT-001",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"\b(ignore|disregard|forget)\s+(all\s+)?"
                    r"(previous|prior|above|earlier)\s+"
                    r"(instructions?|prompts?|rules?|constraints?)\b",
                    flags,
                ),
                0.95,
                InjectionSeverity.CRITICAL,
                "Classic instruction override.",
            ),
            PatternRule(
                "PI-DIRECT-002",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"\b(new|your\s+new|actual)\s+"
                    r"(instructions?|task|objective|goal|mission)\s*:",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "New instruction block injected into content.",
            ),
            PatternRule(
                "PI-DIRECT-003",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"\boverride\s+(all\s+)?(safety|security)\s+"
                    r"(measures?|controls?|restrictions?|filters?)\b",
                    flags,
                ),
                0.95,
                InjectionSeverity.CRITICAL,
                "Security override request.",
            ),
            PatternRule(
                "PI-DIRECT-004",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"\b(stop|cease|abort)\s+"
                    r"(following|obeying|listening\s+to)\s+"
                    r"(instructions?|commands?|rules?)\b",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Request to stop obeying prior instructions.",
            ),
            PatternRule(
                "PI-JAILBREAK-001",
                InjectionType.JAILBREAK_PATTERN,
                re.compile(r"\bDAN\s+(mode|activated|enabled|prompt)\b", flags),
                0.95,
                InjectionSeverity.CRITICAL,
                "Known DAN jailbreak marker.",
            ),
            PatternRule(
                "PI-JAILBREAK-002",
                InjectionType.JAILBREAK_PATTERN,
                re.compile(
                    r"\b(developer\s+mode|jailbreak|unrestricted\s+mode)\b",
                    flags,
                ),
                0.90,
                InjectionSeverity.HIGH,
                "Known jailbreak phrase.",
            ),
            PatternRule(
                "PI-JAILBREAK-003",
                InjectionType.JAILBREAK_PATTERN,
                re.compile(
                    r"\b(bypass|disable)\s+(all\s+)?"
                    r"(safety|security|content|ethical)\s*"
                    r"(rules?|filters?|controls?|restrictions?)?\b",
                    flags,
                ),
                0.90,
                InjectionSeverity.HIGH,
                "Safety bypass attempt.",
            ),
            PatternRule(
                "PI-JAILBREAK-004",
                InjectionType.JAILBREAK_PATTERN,
                re.compile(
                    r"\b(no\s+(restrictions?|limits?|safety|filters?|rules?))\b",
                    flags,
                ),
                0.75,
                InjectionSeverity.MEDIUM,
                "No-restrictions jailbreak wording.",
            ),
            PatternRule(
                "PI-ROLE-001",
                InjectionType.ROLE_OVERRIDE,
                re.compile(
                    r"\byou\s+"
                    r"(are\s+now|will\s+now\s+act\s+as|must\s+act\s+as)\s+"
                    r"(an?\s+)?(unrestricted|free|different|new)\b",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Role override attempt.",
            ),
            PatternRule(
                "PI-ROLE-002",
                InjectionType.ROLE_OVERRIDE,
                re.compile(
                    r"\b(from\s+now\s+on|starting\s+now)\s*,?\s*"
                    r"(you\s+(are|will\s+be|must\s+be))\b",
                    flags,
                ),
                0.75,
                InjectionSeverity.MEDIUM,
                "Attempt to reset assistant role.",
            ),
            PatternRule(
                "PI-ROLE-003",
                InjectionType.ROLE_OVERRIDE,
                re.compile(
                    r"\bpretend\s+(you\s+are|to\s+be)\s+(an?\s+)?"
                    r"(AI|assistant|agent|system)"
                    r"(\s+(assistant|agent|system))?"
                    r"\s+(without|with\s+no)\b",
                    flags,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Pretend-role override attempt.",
            ),
            PatternRule(
                "PI-INDIRECT-001",
                InjectionType.INDIRECT_INJECTION,
                re.compile(
                    r"(\[SYSTEM\]|<SYSTEM>|###\s*SYSTEM|<<SYS>>|<</SYS>>)",
                    flags,
                ),
                0.90,
                InjectionSeverity.HIGH,
                "System-prompt marker inside untrusted content.",
            ),
            PatternRule(
                "PI-INDIRECT-002",
                InjectionType.INDIRECT_INJECTION,
                re.compile(
                    r"(<\|im_start\|>|<\|im_end\|>|\[INST\]|\[/INST\])",
                    flags,
                ),
                0.90,
                InjectionSeverity.HIGH,
                "Chat-template marker inside untrusted content.",
            ),
            PatternRule(
                "PI-INDIRECT-003",
                InjectionType.INDIRECT_INJECTION,
                re.compile(
                    r"\b(assistant|ai|system|bot)\s*:\s*"
                    r"(ignore|override|execute|run|delete|disable)\b",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Instruction-like content attributed to system/assistant.",
            ),
            PatternRule(
                "PI-INDIRECT-004",
                InjectionType.INDIRECT_INJECTION,
                re.compile(
                    r"(<!--.*?inject.*?-->|/\*.*?inject.*?\*/)",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Hidden injection marker in comment-like content.",
            ),
            PatternRule(
                "PI-CONTEXT-001",
                InjectionType.CONTEXT_MANIPULATION,
                re.compile(
                    r"\b(previous|above|earlier)\s+"
                    r"(conversation|messages?|instructions?)\s+"
                    r"(never\s+happened|did\s+not\s+exist|were\s+wrong)\b",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Context manipulation attempt.",
            ),
            PatternRule(
                "PI-CONTEXT-002",
                InjectionType.CONTEXT_MANIPULATION,
                re.compile(
                    r"\b(reset|clear|erase|delete)\s+(your|all)?\s*"
                    r"(context|memory|instructions?|training|history|logs)\b",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Memory/context reset attempt.",
            ),
            PatternRule(
                "PI-SMUGGLE-001",
                InjectionType.INSTRUCTION_SMUGGLING,
                re.compile(
                    r"[;|&]\s*(execute|run|call|invoke|delete|disable|transfer)\s+"
                    r"[\w. -]+",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Command smuggling inside data field.",
            ),
            PatternRule(
                "PI-SMUGGLE-002",
                InjectionType.INSTRUCTION_SMUGGLING,
                re.compile(
                    r"\{\{.*?"
                    r"(exec|eval|system|subprocess|os\.|import\s+os)"
                    r".*?\}\}",
                    flags,
                ),
                0.90,
                InjectionSeverity.CRITICAL,
                "Template/code execution smuggling pattern.",
            ),
            PatternRule(
                "PI-SMUGGLE-003",
                InjectionType.INSTRUCTION_SMUGGLING,
                re.compile(
                    r"\b(amount|value|quantity|price|symbol)\s*=\s*"
                    r"[^|]{0,80}\|{1,2}\s*\w+",
                    flags,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Instruction smuggled into financial/data argument.",
            ),
            PatternRule(
                "PI-TOOL-001",
                InjectionType.TOOL_ABUSE_PATTERN,
                re.compile(
                    r"\b(call|invoke|use)\s+(the\s+)?"
                    r"(tool|function|api)\s+[`'\"]?"
                    r"(execute|delete|send|transfer|admin|shell|python)[\w. -]*",
                    flags,
                ),
                0.85,
                InjectionSeverity.HIGH,
                "Direct request to invoke dangerous tool.",
            ),
        ]


class PromptInjectionDetector:
    """
    Hardened prompt-injection detector for AISec.

    Default mode is deterministic and dependency-free. A semantic scorer can
    be injected for research mode, but no external ML dependency is required.
    """

    DEFAULT_DANGEROUS_TOOL_KEYWORDS = frozenset(
        {
            "execute",
            "python",
            "shell",
            "bash",
            "cmd",
            "delete",
            "remove",
            "send_email",
            "email",
            "transfer",
            "trade",
            "withdraw",
            "admin",
            "modify",
            "config",
            "deploy",
            "network",
            "http",
            "request",
            "write_file",
            "file_write",
            "subprocess",
            "policy",
            "permission",
            "credential",
            "secret",
        }
    )

    def __init__(
        self,
        *,
        max_text_len: int = 10_000,
        cache_size: int = 2048,
        pattern_threshold: float = 0.70,
        semantic_threshold: float = 0.80,
        semantic_scorer: Callable[[str], float] | None = None,
        log_raw_text: bool = False,
    ) -> None:
        self.max_text_len = max_text_len
        self.pattern_threshold = pattern_threshold
        self.semantic_threshold = semantic_threshold
        self.semantic_scorer = semantic_scorer
        self.log_raw_text = log_raw_text

        self.normalizer = TextNormalizer(max_text_len=max_text_len)
        self.rules = list(PromptInjectionPatterns.build())

        Rule = type(self.rules[0])
        self.rules.extend([
            Rule(
                "real_agent_override_previous_safety_policy",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"override\s+(the\s+)?(previous|prior|existing|normal)\s+"
                    r"(safety\s+)?(policy|policies|rules?|instructions?)",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to override an existing safety policy or instruction set.",
            ),
            Rule(
                "real_agent_ignore_all_safety_instructions",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"ignore\s+(all\s+)?(safety|security|system)\s+"
                    r"(instructions?|rules?|polic(?:y|ies))",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to ignore safety or system instructions.",
            ),
            Rule(
                "real_agent_forget_safety_instructions",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"forget\s+(the\s+)?(safety|security|system|previous|prior)\s+"
                    r"(instructions?|rules?|polic(?:y|ies))",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to make the agent forget safety instructions.",
            ),
            Rule(
                "real_agent_bypass_previous_rule",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"bypass\s+(the\s+)?(previous|prior|existing|normal)\s+"
                    r"(rule|rules|policy|policies|instruction|instructions)",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to bypass an existing rule or policy.",
            ),
            Rule(
                "real_agent_policy_engine_override",
                InjectionType.CONTEXT_MANIPULATION,
                re.compile(
                    r"override\s+(the\s+)?policy\s+engine|"
                    r"approve\s+(this\s+)?change\s+without\s+review",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to manipulate policy-engine or review context.",
            ),
            Rule(
                "real_agent_instruction_hierarchy_bypass",
                InjectionType.CONTEXT_MANIPULATION,
                re.compile(
                    r"bypass\s+(the\s+)?(normal\s+)?instruction\s+hierarchy|"
                    r"treat\s+(this|the)\s+(note|report|document|page)\s+as\s+"
                    r"(controlling|higher\s+priority)",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to replace the instruction hierarchy with untrusted content.",
            ),
            Rule(
                "real_agent_system_instruction_trust_only_content",
                InjectionType.DIRECT_INJECTION,
                re.compile(
                    r"ignore\s+(the\s+)?system\s+instructions?.*trust\s+only\s+"
                    r"(this|the)\s+(report|note|document|page)",
                    re.IGNORECASE,
                ),
                0.80,
                InjectionSeverity.HIGH,
                "Attempt to ignore system instructions and trust untrusted content.",
            ),
        ])

        self._cache_size = max(0, cache_size)
        self._cache: OrderedDict[str, InjectionDetectionResult] = OrderedDict()
        self._cache_lock = threading.RLock()

    def analyse(
        self,
        text: str,
        *,
        context: PromptInjectionContext | None = None,
    ) -> InjectionDetectionResult:
        start = time.perf_counter()
        context = context or PromptInjectionContext()

        if not isinstance(text, str) or not text.strip():
            return InjectionDetectionResult.clean(
                detection_time_ms=self._elapsed_ms(start)
            )

        original_len = len(text)
        normalized = self.normalizer.normalize(text)

        # text_hash is evidence of the text itself.
        # cache_key includes context because the same text can have different
        # severity inside user input vs. dangerous tool arguments.
        text_hash = self._hash_text(normalized)
        cache_key = self._cache_key(normalized, context)

        cached = self._cache_get(cache_key)
        if cached is not None:
            return self._copy_with_time(cached, self._elapsed_ms(start))

        result = self._analyse_normalized(
            normalized,
            text_hash=text_hash,
            original_len=original_len,
            context=context,
            start=start,
        )

        self._cache_put(cache_key, result)
        return result

    def analyse_kwargs(
        self,
        kwargs: dict[str, Any],
        *,
        context: PromptInjectionContext | None = None,
    ) -> InjectionDetectionResult:
        return self.analyse_payload(kwargs, context=context)

    def analyse_payload(
        self,
        payload: Any,
        *,
        context: PromptInjectionContext | None = None,
    ) -> InjectionDetectionResult:
        text = self._flatten_payload(payload)
        return self.analyse(text, context=context)

    def get_risk_boost(
        self,
        text: str,
        base_score: float,
        *,
        context: PromptInjectionContext | None = None,
    ) -> float:
        result = self.analyse(text, context=context)

        if not result.is_injection:
            return base_score

        boosted = max(base_score, result.risk_score)

        if result.recommended_action == RecommendedAction.BLOCK:
            boosted = max(boosted, 0.90)
        elif result.recommended_action == RecommendedAction.REVIEW:
            boosted = max(boosted, 0.70)
        elif result.recommended_action == RecommendedAction.WATCH:
            boosted = max(boosted, 0.45)

        boosted = min(1.0, boosted)

        log.info(
            "risk_score_boosted_by_prompt_injection",
            base_score=base_score,
            boosted_score=boosted,
            recommended_action=result.recommended_action.value,
            injection_type=(
                result.injection_type.value if result.injection_type else None
            ),
            text_hash=result.text_hash,
        )

        return boosted

    def should_block(
        self,
        payload: Any,
        *,
        context: PromptInjectionContext | None = None,
    ) -> bool:
        """Convenience method for adapters and API middleware."""
        result = self.analyse_payload(payload, context=context)
        return result.recommended_action == RecommendedAction.BLOCK

    def _analyse_normalized(
        self,
        normalized: str,
        *,
        text_hash: str,
        original_len: int,
        context: PromptInjectionContext,
        start: float,
    ) -> InjectionDetectionResult:
        matched_rules = [rule for rule in self.rules if rule.pattern.search(normalized)]

        pattern_confidence = max(
            (rule.base_confidence for rule in matched_rules),
            default=0.0,
        )

        best_rule = max(
            matched_rules,
            key=lambda rule: rule.base_confidence,
            default=None,
        )

        semantic_score = self._semantic_score(normalized, text_hash)

        confidence = max(pattern_confidence, semantic_score)

        unique_categories = {rule.injection_type.value for rule in matched_rules}

        if len(unique_categories) >= 2:
            confidence = min(1.0, confidence + 0.05 * (len(unique_categories) - 1))

        is_injection = (
            pattern_confidence >= self.pattern_threshold
            or semantic_score >= self.semantic_threshold
        )

        if not is_injection:
            result = InjectionDetectionResult.clean(
                text_hash=text_hash,
                text_length=original_len,
                normalized_length=len(normalized),
                detection_time_ms=self._elapsed_ms(start),
            )
            self._log_detection(result, context)
            return result

        severity = self._combine_severity(matched_rules, confidence, context)
        action = self._recommended_action(severity, context)
        risk_score = self._risk_score(severity, confidence, context)

        injection_type = (
            best_rule.injection_type if best_rule else InjectionType.UNKNOWN
        )

        result = InjectionDetectionResult(
            is_injection=True,
            injection_type=injection_type,
            severity=severity,
            confidence=round(confidence, 4),
            risk_score=round(risk_score, 4),
            recommended_action=action,
            matched_rule_ids=[rule.rule_id for rule in matched_rules],
            matched_categories=sorted(unique_categories),
            semantic_score=round(semantic_score, 4),
            text_hash=text_hash,
            text_length=original_len,
            normalized_length=len(normalized),
            detection_time_ms=self._elapsed_ms(start),
            explanation=(
                "Prompt-injection indicators detected. "
                f"type={injection_type.value}, "
                f"severity={severity.value}, "
                f"confidence={confidence:.2f}, "
                f"recommended_action={action.value}, "
                f"rules_matched={len(matched_rules)}."
            ),
        )

        self._log_detection(
            result,
            context,
            raw_preview=normalized[:200] if self.log_raw_text else None,
        )
        return result

    def _semantic_score(self, normalized: str, text_hash: str) -> float:
        if self.semantic_scorer is None:
            return 0.0

        try:
            return max(0.0, min(1.0, float(self.semantic_scorer(normalized))))
        except Exception as exc:
            log.warning(
                "semantic_prompt_injection_scorer_failed",
                exc_type=type(exc).__name__,
                text_hash=text_hash,
            )
            return 0.0

    def _combine_severity(
        self,
        rules: list[PatternRule],
        confidence: float,
        context: PromptInjectionContext,
    ) -> InjectionSeverity:
        severity_order = {
            InjectionSeverity.CLEAN: 0,
            InjectionSeverity.LOW: 1,
            InjectionSeverity.MEDIUM: 2,
            InjectionSeverity.HIGH: 3,
            InjectionSeverity.CRITICAL: 4,
        }

        base = max(
            (rule.severity for rule in rules),
            key=lambda severity: severity_order[severity],
            default=InjectionSeverity.LOW,
        )

        score = severity_order[base]

        if confidence >= 0.95:
            score = max(score, severity_order[InjectionSeverity.CRITICAL])
        elif confidence >= 0.85:
            score = max(score, severity_order[InjectionSeverity.HIGH])
        elif confidence >= 0.70:
            score = max(score, severity_order[InjectionSeverity.MEDIUM])

        if self._is_dangerous_context(context):
            score = min(4, score + 1)

        inverse = {value: key for key, value in severity_order.items()}
        return inverse[score]

    def _recommended_action(
        self,
        severity: InjectionSeverity,
        context: PromptInjectionContext,
    ) -> RecommendedAction:
        dangerous_context = self._is_dangerous_context(context)

        if severity == InjectionSeverity.CRITICAL:
            return (
                RecommendedAction.BLOCK
                if dangerous_context
                else RecommendedAction.REVIEW
            )

        if severity == InjectionSeverity.HIGH:
            return (
                RecommendedAction.BLOCK
                if dangerous_context
                else RecommendedAction.REVIEW
            )

        if severity == InjectionSeverity.MEDIUM:
            return (
                RecommendedAction.REVIEW
                if dangerous_context
                else RecommendedAction.WATCH
            )

        if severity == InjectionSeverity.LOW:
            return RecommendedAction.WATCH

        return RecommendedAction.ALLOW

    def _risk_score(
        self,
        severity: InjectionSeverity,
        confidence: float,
        context: PromptInjectionContext,
    ) -> float:
        base = {
            InjectionSeverity.CLEAN: 0.0,
            InjectionSeverity.LOW: 0.25,
            InjectionSeverity.MEDIUM: 0.50,
            InjectionSeverity.HIGH: 0.75,
            InjectionSeverity.CRITICAL: 0.95,
        }[severity]

        risk = max(base, confidence)

        if self._is_dangerous_context(context):
            risk = min(1.0, risk + 0.10)

        return risk

    def _is_dangerous_context(self, context: PromptInjectionContext) -> bool:
        if context.high_risk_tool:
            return True

        if context.source == InputSource.TOOL_ARGUMENTS:
            return True

        if not context.tool_name:
            return False

        tool_name = context.tool_name.lower()
        return any(
            keyword in tool_name for keyword in self.DEFAULT_DANGEROUS_TOOL_KEYWORDS
        )

    def _flatten_payload(
        self,
        payload: Any,
        *,
        max_depth: int = 5,
        max_items: int = 100,
    ) -> str:
        """
        Flatten nested data into bounded text.

        Includes dictionary keys and values because malicious instructions
        can be hidden in JSON keys, not only values.
        """

        parts: list[str] = []
        seen = 0

        def visit(obj: Any, depth: int) -> None:
            nonlocal seen

            if seen >= max_items or depth > max_depth:
                return

            seen += 1

            if isinstance(obj, dict):
                for key, value in list(obj.items())[:max_items]:
                    parts.append(str(key))
                    visit(value, depth + 1)
            elif isinstance(obj, (list, tuple, set)):
                for item in list(obj)[:max_items]:
                    visit(item, depth + 1)
            elif isinstance(obj, (str, int, float, bool)):
                parts.append(str(obj))
            elif obj is None:
                parts.append("None")
            else:
                parts.append(type(obj).__name__)

        visit(payload, 0)
        return " ".join(parts)[: self.max_text_len]

    def _cache_get(self, key: str) -> InjectionDetectionResult | None:
        if self._cache_size <= 0:
            return None

        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)
            return result

    def _cache_put(self, key: str, result: InjectionDetectionResult) -> None:
        if self._cache_size <= 0:
            return

        with self._cache_lock:
            self._cache[key] = result
            self._cache.move_to_end(key)

            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    @staticmethod
    def _copy_with_time(
        result: InjectionDetectionResult,
        detection_time_ms: float,
    ) -> InjectionDetectionResult:
        return InjectionDetectionResult(
            is_injection=result.is_injection,
            injection_type=result.injection_type,
            severity=result.severity,
            confidence=result.confidence,
            risk_score=result.risk_score,
            recommended_action=result.recommended_action,
            matched_rule_ids=list(result.matched_rule_ids),
            matched_categories=list(result.matched_categories),
            semantic_score=result.semantic_score,
            text_hash=result.text_hash,
            text_length=result.text_length,
            normalized_length=result.normalized_length,
            detection_time_ms=detection_time_ms,
            explanation=result.explanation,
        )

    @classmethod
    def _cache_key(
        cls,
        normalized: str,
        context: PromptInjectionContext,
    ) -> str:
        """
        Build a context-aware cache key.

        The same text may be low risk in normal user input but high risk inside
        dangerous tool arguments. Therefore, cache entries must include context.
        """
        context_part = "|".join(
            [
                context.source.value,
                context.tool_name or "",
                context.scenario or "",
                str(context.high_risk_tool),
            ]
        )
        return cls._hash_text(normalized + "|" + context_part)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 4)

    def _log_detection(
        self,
        result: InjectionDetectionResult,
        context: PromptInjectionContext,
        *,
        raw_preview: str | None = None,
    ) -> None:
        log_fields = {
            "is_injection": result.is_injection,
            "injection_type": (
                result.injection_type.value if result.injection_type else None
            ),
            "severity": result.severity.value,
            "confidence": result.confidence,
            "risk_score": result.risk_score,
            "recommended_action": result.recommended_action.value,
            "matched_rule_ids": result.matched_rule_ids,
            "text_hash": result.text_hash,
            "text_length": result.text_length,
            "normalized_length": result.normalized_length,
            "detection_time_ms": result.detection_time_ms,
            "source": context.source.value,
            "tool_name": context.tool_name,
            "scenario": context.scenario,
        }

        if self.log_raw_text and raw_preview is not None:
            log_fields["raw_preview"] = raw_preview

        if result.is_injection:
            log.warning("prompt_injection_detected", **log_fields)
        else:
            log.debug("prompt_injection_clean", **log_fields)
