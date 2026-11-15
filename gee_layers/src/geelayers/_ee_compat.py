"""Fallback stub when earthengine-api is not installed (allows offline tests and Meteomatics-only runs)."""
from __future__ import annotations

try:
    import ee  # type: ignore[import-untyped]
except ModuleNotFoundError:
    class _MissingEE:
        class EEException(Exception):
            pass

        class Geometry:
            @staticmethod
            def Rectangle(*args, **kwargs):
                return None

        class Image:
            @staticmethod
            def cat(imgs):
                raise RuntimeError("earthengine-api is not installed; install it with `pip install earthengine-api`")

        class oauth:
            @staticmethod
            def get_credentials_path() -> str:
                return ""

        @staticmethod
        def Authenticate(*args, **kwargs):
            raise RuntimeError("earthengine-api is not installed; install it with `pip install earthengine-api`")

        @staticmethod
        def Initialize(*args, **kwargs):
            raise RuntimeError("earthengine-api is not installed; install it with `pip install earthengine-api`")

        @staticmethod
        def ServiceAccountCredentials(*args, **kwargs):
            raise RuntimeError("earthengine-api is not installed; install it with `pip install earthengine-api`")

    ee = _MissingEE()  # type: ignore[assignment]
