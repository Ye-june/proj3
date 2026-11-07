"""Production entry point for Gunicorn: `gunicorn wsgi:app`."""

from app import app

__all__ = ["app"]
