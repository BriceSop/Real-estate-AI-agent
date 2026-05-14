from pathlib import Path
import sys
from loguru import logger

def setup_logging() -> None:
    Path("logs").mkdir(exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    logger.add(
        "logs/app.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        encoding="utf8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )