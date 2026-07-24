"""Root conftest.

Its presence puts the repository root on ``sys.path`` for pytest, so tests can
``from src.graph import builder`` without any installation step.
"""
