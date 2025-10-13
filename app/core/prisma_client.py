"""Shared Prisma client with background event loop for thread-safe access."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from prisma import Prisma

logger = logging.getLogger(__name__)


_client: Prisma = Prisma()
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_ready = threading.Event()
_client_lock = threading.Lock()


def _loop_worker(loop: asyncio.AbstractEventLoop) -> None:
    """Run Prisma client on dedicated event loop thread."""
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_client.connect())
        logger.info("✅ Prisma client connected")
        _loop_ready.set()
        loop.run_forever()
    except Exception as exc:  # pragma: no cover
        logger.error(f"❌ Failed to start Prisma event loop: {exc}", exc_info=True)
        _loop_ready.set()
    finally:
        if _client.is_connected():
            loop.run_until_complete(_client.disconnect())
        loop.close()
        logger.info("🛑 Prisma event loop stopped")


def ensure_client_connected() -> Prisma:
    """Ensure the Prisma client is connected and return it."""
    global _loop, _loop_thread

    if _client.is_connected() and _loop is not None:
        return _client

    with _client_lock:
        if _loop is None or _loop_thread is None or not _loop_thread.is_alive():
            _loop_ready.clear()
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop_worker,
                args=(_loop,),
                daemon=True,
                name="prisma-event-loop",
            )
            _loop_thread.start()

        # Wait until the loop signals readiness (connected or failed)
        _loop_ready.wait()

        if not _client.is_connected():
            raise RuntimeError("Failed to connect Prisma client")

    return _client


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Return the dedicated Prisma event loop."""
    ensure_client_connected()
    assert _loop is not None
    return _loop


def run_async(coro: asyncio.Future) -> any:
    """Execute a coroutine on the Prisma event loop and return the result."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def disconnect_client() -> None:
    """Disconnect Prisma client and stop background loop."""
    global _loop, _loop_thread

    with _client_lock:
        if _loop is None or _loop_thread is None:
            return
        if _client.is_connected():
            try:
                run_async(_client.disconnect())
                logger.info("✅ Prisma client disconnected")
            except Exception as exc:  # pragma: no cover
                logger.warning(f"⚠️ Failed to disconnect Prisma client cleanly: {exc}")
        if _loop_thread.is_alive():
            try:
                _loop.call_soon_threadsafe(_loop.stop)
                _loop_thread.join(timeout=5)
            except Exception:  # pragma: no cover
                pass
        _loop = None
        _loop_thread = None
        _loop_ready.clear()
