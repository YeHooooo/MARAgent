"""MARAgent package."""

__all__ = ["MARAgentPipeline"]


def __getattr__(name):
    if name == "MARAgentPipeline":
        from .pipeline import MARAgentPipeline

        return MARAgentPipeline
    raise AttributeError(name)
