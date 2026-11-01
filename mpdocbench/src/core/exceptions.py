"""Pipeline exceptions."""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class LLMError(PipelineError):
    """An LLM call failed after retries."""


class StructuredOutputError(PipelineError):
    """Model output could not be parsed / validated into the expected contract."""


class UnresolvedContinuationError(PipelineError):
    """Continuation remained 'uncertain' and unresolved_policy = stop (spec §22)."""


class UnresolvedExtractionError(PipelineError):
    """A page could not be resolved (used by the deferred Phase B failure path)."""


class SkillRegistryError(PipelineError):
    """Skill registry / record loading problem."""
