"""Evaluation — adapt the pipeline's FinalDocument into the scorer's shape, score it
against gold (benchmark/), and run per-page failure detectors. Driven from run.py via
RUN_EVAL (adapt -> evaluate -> failures, in that order). Paths come from src.core.config."""
