"""Root-level OpenEnv app wrapper for repo-level validation and deployment."""

from wildfire_env.server.app import app as app
from wildfire_env.server.app import main as _wildfire_main


def main() -> None:
    """Run the wildfire FastAPI app through the root submission entry point."""
    _wildfire_main()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
