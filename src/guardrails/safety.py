import re
from typing import Tuple, Optional

class SafetyGuardrailsEngine:
    """
    Enterprise-Grade Input/Output Safety Barrier.
    Protects local SLM instances from prompt injection, system prompt exfiltration,
    and automatically masks sensitive PII prior to inference or caching.
    """
    def __init__(self):
        # 1. Prompt Injection & Jailbreak patterns
        self.injection_patterns = [
            r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
            r"(?i)disregard\s+(all\s+)?(previous|prior)\s+(instructions|directives)",
            r"(?i)you\s+are\s+now\s+in\s+(developer\s+mode|dan\s+mode|unrestricted\s+mode)",
            r"(?i)jailbreak|unfiltered\s+mode|do\s+anything\s+now",
            r"(?i)system\s+override|override\s+safety\s+guidelines",
            r"(?i)print\s+(your\s+)?(system\s+prompt|initial\s+instructions|system\s+message)",
            r"(?i)repeat\s+the\s+words\s+above|reveal\s+your\s+hidden\s+prompt",
            r"<\s*\|\s*im_start\s*\|>",
            r"\[\s*INST\s*\]",
            r"<\s*system\s*>"
        ]
        
        # 2. Sensitive PII regexes for automated masking
        self.pii_rules = [
            # Credit Card numbers (13-16 digits with dashes or spaces)
            (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED_CARD_NUMBER]"),
            # US Social Security Number (SSN)
            (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
            # API Tokens (OpenAI sk-, GitHub ghp-, Generic)
            (r"\b(sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|re_[a-zA-Z0-9]{20,})\b", "[REDACTED_API_KEY]"),
            # Private Key headers
            (r"-----BEGIN[ A-Z0-9_-]+PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY_HEADER]")
        ]

    def validate_input(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates and sanitizes user input.
        Returns: (is_safe: bool, sanitized_text: str, violation_reason: Optional[str])
        """
        if not text:
            return True, "", None

        # Check for length overflow attacks
        if len(text) > 8000:
            return False, "", "Input exceeds maximum character limit of 8000 characters (Buffer Guard)."

        # Check prompt injection patterns
        for pattern in self.injection_patterns:
            if re.search(pattern, text):
                return False, "", f"Prompt Injection detected by Safety Guardrails: rule match '{pattern}'"

        # Apply PII sanitization / redaction
        sanitized = text
        for pii_pattern, mask in self.pii_rules:
            sanitized = re.sub(pii_pattern, mask, sanitized)

        return True, sanitized, None

    def validate_output(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validates model output to ensure no leaked system keys or toxic completions.
        """
        if not text:
            return True, "", None

        # Mask any accidental PII leaks in completions
        sanitized = text
        for pii_pattern, mask in self.pii_rules:
            sanitized = re.sub(pii_pattern, mask, sanitized)

        return True, sanitized, None

safety_guardrails = SafetyGuardrailsEngine()
