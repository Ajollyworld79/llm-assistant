"""Compatibility shim: re-export `lifecycle` from `backend.app.module.lifecycle`."""

from .module.lifecycle import lifecycle

__all__ = ["lifecycle"]
