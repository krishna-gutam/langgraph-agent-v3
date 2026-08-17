"""`python -m frontends.web` - serve the browser UI on localhost."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "frontends.web.api:app",
        host=os.getenv("WEB_HOST", "127.0.0.1"),
        port=int(os.getenv("WEB_PORT", "8000")),
        reload=bool(os.getenv("WEB_RELOAD")),
    )


if __name__ == "__main__":
    main()
