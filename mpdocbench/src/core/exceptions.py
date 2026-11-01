from __future__ import annotations


class PipelineError(Exception):
    pass


class LLMError(PipelineError):
    pass


class StructuredOutputError(PipelineError):
    pass


class UnresolvedContinuationError(PipelineError):
    pass


class UnresolvedExtractionError(PipelineError):
    pass


class SkillRegistryError(PipelineError):
    pass
