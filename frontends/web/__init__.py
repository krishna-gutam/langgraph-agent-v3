"""FastAPI frontend: a browser UI served over HTTP.

`from frontends.web import app` gives the ASGI application; run it with
`uvicorn frontends.web:app` or `python -m frontends.web`.
"""

from frontends.web.api import app

__all__ = ["app"]
