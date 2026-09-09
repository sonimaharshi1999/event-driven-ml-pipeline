# Event-Driven ML Pipeline - FastAPI Monitoring Dashboard
# Author: Maharshi Soni | License: MIT

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.pipeline.broker import MessageBroker
from src.pipeline.dead_letter import DeadLetterQueue
from src.pipeline.events import Event, EventType
from src.pipeline.features import FeatureEngine
from src.pipeline.predictor import OnlinePredictor
from src.pipeline.replay import EventReplayStore
from src.pipeline.windows import SlidingWindow, TumblingWindow

# ---------------------------------------------------------------------------
# Pydantic models for request / response
# ---------------------------------------------------------------------------


class EventIn(BaseModel):
    event_type: str
    topic: str
    payload: Dict[str, Any]


class PredictionOut(BaseModel):
    label: int
    probability: float
    is_anomaly: bool
    latency_ms: float


class StatsOut(BaseModel):
    broker: Dict[str, Any]
    predictor: Dict[str, Any]
    dead_letter: Dict[str, Any]
    replay: Dict[str, Any]
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Shared pipeline components (initialized in lifespan)
# ---------------------------------------------------------------------------

broker: Optional[MessageBroker] = None
dlq: Optional[DeadLetterQueue] = None
replay_store: Optional[EventReplayStore] = None
feature_engine: Optional[FeatureEngine] = None
predictor: Optional[OnlinePredictor] = None
tumbling: Optional[TumblingWindow] = None
sliding: Optional[SlidingWindow] = None

_start_time: float = 0.0


def _build_pipeline() -> None:
    """Construct all pipeline components and wire them together."""
    global broker, dlq, replay_store, feature_engine, predictor
    global tumbling, sliding, _start_time

    _start_time = time.time()

    broker = MessageBroker()
    dlq = DeadLetterQueue()
    replay_store = EventReplayStore()
    feature_engine = FeatureEngine(group_key="user_id")
    predictor = OnlinePredictor()
    tumbling = TumblingWindow(window_seconds=5.0)
    sliding = SlidingWindow(window_seconds=10.0, slide_seconds=2.0)

    # Wire dead-letter handler
    broker.set_dead_letter_handler(dlq.handle)

    # Ensure model is ready
    if not predictor.load():
        predictor.train()


async def _process_event(event: Event) -> None:
    """Central subscriber: features -> predict -> windows -> store."""
    assert feature_engine is not None
    assert predictor is not None
    assert tumbling is not None
    assert sliding is not None
    assert replay_store is not None

    feature_engine.ingest(event)
    features = feature_engine.compute(event)
    predictor.predict(features)
    tumbling.add(event)
    sliding.add(event)
    replay_store.store(event)


# ---------------------------------------------------------------------------
# Lifespan & FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Startup: build pipeline and start broker.  Shutdown: stop broker."""
    _build_pipeline()
    assert broker is not None
    broker.subscribe("transactions", _process_event)
    broker.subscribe("sensors", _process_event)
    broker.subscribe("metrics", _process_event)
    await broker.start()
    yield
    if broker:
        await broker.stop()


app = FastAPI(
    title="Event-Driven ML Pipeline",
    description="Real-time stream processing with online ML inference",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Minimal HTML dashboard served inline."""
    return _DASHBOARD_HTML


@app.post("/events", response_model=Dict[str, Any])
async def ingest_event(event_in: EventIn) -> Dict[str, Any]:
    """Publish a new event into the pipeline."""
    assert broker is not None
    event = Event(
        event_type=EventType(event_in.event_type),
        topic=event_in.topic,
        payload=event_in.payload,
    )
    await broker.publish(event)
    return {"status": "accepted", "event_id": event.event_id}


@app.get("/stats", response_model=StatsOut)
async def stats() -> StatsOut:
    """Return aggregated pipeline statistics."""
    return StatsOut(
        broker=broker.get_stats() if broker else {},
        predictor=predictor.get_stats() if predictor else {},
        dead_letter=dlq.get_stats() if dlq else {},
        replay=replay_store.get_stats() if replay_store else {},
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/dead-letters", response_model=List[Dict[str, Any]])
async def dead_letters(n: int = Query(10, ge=1, le=100)) -> List[Dict[str, Any]]:
    """Peek at the most recent dead-letter entries."""
    return dlq.peek(n) if dlq else []


@app.get("/replay", response_model=List[Dict[str, Any]])
async def replay(
    n: int = Query(20, ge=1, le=500),
    topic: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Replay the last *n* events from the store."""
    if not replay_store:
        return []
    events = replay_store.replay_last(n, topic=topic)
    return [e.model_dump() for e in events]


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Inline dashboard HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML Pipeline Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}
  h1{font-size:1.6rem;margin-bottom:16px;color:#38bdf8}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:24px}
  .card{background:#1e293b;border-radius:12px;padding:20px;border:1px solid #334155}
  .card h3{font-size:.85rem;color:#94a3b8;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}
  .card .val{font-size:1.8rem;font-weight:700;color:#38bdf8}
  table{width:100%;border-collapse:collapse;margin-top:8px}
  th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #334155}
  th{color:#94a3b8;font-size:.8rem;text-transform:uppercase}
  td{font-size:.9rem}
  .badge-ok{color:#4ade80} .badge-err{color:#f87171}
  #log{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;
       max-height:200px;overflow-y:auto;font-family:monospace;font-size:.8rem;margin-top:16px}
  button{background:#2563eb;color:#fff;border:none;padding:8px 18px;border-radius:6px;
         cursor:pointer;font-size:.9rem;margin-top:12px}
  button:hover{background:#1d4ed8}
</style></head><body>
<h1>Event-Driven ML Pipeline &mdash; Live Dashboard</h1>
<div class="grid">
  <div class="card"><h3>Events Published</h3><div class="val" id="pub">-</div></div>
  <div class="card"><h3>Events Delivered</h3><div class="val" id="del">-</div></div>
  <div class="card"><h3>Avg Latency</h3><div class="val" id="lat">-</div></div>
  <div class="card"><h3>Predictions</h3><div class="val" id="pred">-</div></div>
  <div class="card"><h3>Dead Letters</h3><div class="val" id="dlq">-</div></div>
  <div class="card"><h3>Replay Store</h3><div class="val" id="replay">-</div></div>
</div>
<button onclick="refresh()">Refresh</button>
<div id="log">Waiting for data...</div>
<script>
async function refresh(){
  try{
    const r=await fetch('/stats');const d=await r.json();
    let pub=0,del_=0,lat=0,cnt=0;
    for(const t of Object.values(d.broker)){pub+=t.published;del_+=t.delivered;if(t.avg_latency_ms){lat+=t.avg_latency_ms;cnt++}}
    document.getElementById('pub').textContent=pub;
    document.getElementById('del').textContent=del_;
    document.getElementById('lat').textContent=cnt?(lat/cnt).toFixed(2)+'ms':'0ms';
    document.getElementById('pred').textContent=d.predictor.predictions||0;
    document.getElementById('dlq').textContent=d.dead_letter.current_size||0;
    document.getElementById('replay').textContent=d.replay.stored_events||0;
    document.getElementById('log').textContent=JSON.stringify(d,null,2);
  }catch(e){document.getElementById('log').textContent='Error: '+e}
}
refresh();setInterval(refresh,3000);
</script></body></html>"""
