# Event-Driven ML Pipeline - Application Entry Point
# Author: Maharshi Soni | License: MIT
#
# Usage:
#   python main.py              -- start the FastAPI server with live event generation
#   python main.py --no-gen     -- start the server without the synthetic generator
#   python main.py --train      -- retrain the model and exit

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import uvicorn

from src.api.dashboard import app, broker, _build_pipeline, _process_event
from src.pipeline.predictor import OnlinePredictor
from src.synthetic import generate_transaction_stream, generate_sensor_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _feed_events() -> None:
    """Background task that pushes synthetic events into the broker."""
    from src.api.dashboard import broker as _broker

    if _broker is None:
        return

    tx_gen = generate_transaction_stream(rate_per_second=100)
    sensor_gen = generate_sensor_stream(rate_per_second=50)

    async def _pump(gen):  # type: ignore[no-untyped-def]
        async for event in gen:
            if _broker is not None:
                await _broker.publish(event)

    await asyncio.gather(_pump(tx_gen), _pump(sensor_gen))


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-Driven ML Pipeline")
    parser.add_argument("--train", action="store_true", help="Retrain model and exit")
    parser.add_argument("--no-gen", action="store_true", help="Disable synthetic event generation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.train:
        predictor = OnlinePredictor()
        result = predictor.train()
        print(f"Training complete: {result}")
        return

    logger.info("Starting Event-Driven ML Pipeline on %s:%s", args.host, args.port)
    uvicorn.run(
        "src.api.dashboard:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
