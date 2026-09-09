# Event-Driven ML Pipeline - In-Process Async Message Broker
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.pipeline.events import Event

logger = logging.getLogger(__name__)

# Type alias for subscriber callbacks.
SubscriberCallback = Callable[[Event], Awaitable[None]]


class TopicStats:
    """Lightweight counters for a single topic."""

    def __init__(self) -> None:
        self.published: int = 0
        self.delivered: int = 0
        self.failed: int = 0
        self.total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.delivered == 0:
            return 0.0
        return self.total_latency_ms / self.delivered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "published": self.published,
            "delivered": self.delivered,
            "failed": self.failed,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
        }


class MessageBroker:
    """In-process async pub/sub message broker.

    * Publishers push :class:`Event` objects to named topics.
    * Subscribers register async callbacks on one or more topics.
    * Delivery is concurrent within a topic (all subscribers fan-out).
    * Failed deliveries are routed to an optional dead-letter handler.
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        self._subscribers: Dict[str, List[SubscriberCallback]] = defaultdict(list)
        self._queues: Dict[str, asyncio.Queue[Event]] = {}
        self._stats: Dict[str, TopicStats] = defaultdict(TopicStats)
        self._max_queue_size = max_queue_size
        self._running = False
        self._dispatch_tasks: List[asyncio.Task[None]] = []
        self._dead_letter_handler: Optional[Callable[[Event, Exception], Awaitable[None]]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: SubscriberCallback) -> None:
        """Register *callback* to receive events published on *topic*."""
        self._subscribers[topic].append(callback)
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue(maxsize=self._max_queue_size)
        logger.info("Subscriber added to topic '%s'", topic)

    def set_dead_letter_handler(
        self,
        handler: Callable[[Event, Exception], Awaitable[None]],
    ) -> None:
        """Set a handler that receives events whose delivery failed."""
        self._dead_letter_handler = handler

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, event: Event) -> None:
        """Enqueue *event* on its topic for asynchronous delivery."""
        topic = event.topic
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue(maxsize=self._max_queue_size)
        await self._queues[topic].put(event)
        self._stats[topic].published += 1

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start background dispatch loops for every registered topic."""
        if self._running:
            return
        self._running = True
        for topic in list(self._queues):
            task = asyncio.create_task(self._dispatch_loop(topic))
            self._dispatch_tasks.append(task)
        logger.info("Broker started with %d topic(s)", len(self._dispatch_tasks))

    async def stop(self) -> None:
        """Gracefully shut down all dispatch loops."""
        self._running = False
        for task in self._dispatch_tasks:
            task.cancel()
        await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)
        self._dispatch_tasks.clear()
        logger.info("Broker stopped")

    async def _dispatch_loop(self, topic: str) -> None:
        """Continuously drain the queue for *topic* and fan-out to subs."""
        queue = self._queues[topic]
        while self._running:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            callbacks = self._subscribers.get(topic, [])
            for cb in callbacks:
                start = time.perf_counter()
                try:
                    await cb(event)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    self._stats[topic].delivered += 1
                    self._stats[topic].total_latency_ms += elapsed_ms
                except Exception as exc:
                    self._stats[topic].failed += 1
                    logger.warning(
                        "Delivery failed on topic '%s': %s", topic, exc
                    )
                    if self._dead_letter_handler:
                        try:
                            await self._dead_letter_handler(event, exc)
                        except Exception:
                            logger.exception("Dead-letter handler itself failed")

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return per-topic delivery statistics."""
        return {topic: stats.to_dict() for topic, stats in self._stats.items()}

    def get_queue_depths(self) -> Dict[str, int]:
        """Return the current queue depth for each topic."""
        return {topic: q.qsize() for topic, q in self._queues.items()}
