#!/usr/bin/env python3
"""
Main entry point for the autonomous Twitter bot system.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.scheduler import BotScheduler
from src.config.settings import BotConfig
from loguru import logger


async def main() -> None:
    """Main application entry point."""
    logger.info("Starting Autonomous Twitter Bot System")

    try:
        config = BotConfig()
        scheduler = BotScheduler(config)
        await scheduler.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping bot...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Bot system stopped")


if __name__ == "__main__":
    asyncio.run(main())
