# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Model lifecycle managers — Phase 1, 2, 3.

Phase 1: SingleModelManager — one engine, one asyncio.Lock.
Phase 2: SingleModelManager gains idle-timeout auto-unload.
Phase 3: ModelSlotManager — multi-model, refcount + LRU eviction.

API routes depend only on the ModelManager protocol, so Phase 1 → Phase 3
migration is a drop-in swap at startup.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..inference import InferenceEngine


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelManager(Protocol):
    """Interface for model lifecycle management.

    Routes depend only on this interface; implementations are injected at
    startup (SingleModelManager for Phase 1/2, ModelSlotManager for Phase 3).
    """

    @asynccontextmanager
    def borrow(self, model_id: str, task: str | None = None) -> AsyncIterator[InferenceEngine]:
        """Context manager that yields a ready InferenceEngine.

        Blocks until the engine is available (respects concurrency limits).
        ``task`` is a routing hint used by ModelSlotManager when ``model_id``
        is the sentinel ``"_"``.
        """

    async def list_models(self) -> list[dict]:
        """Return metadata for all currently registered/loaded models."""

    def get_engine(self, model_id: str | None = None) -> InferenceEngine | None:
        """Get engine for a model, or the first available engine."""

    def get_all_engines(self) -> list[InferenceEngine]:
        """Get all loaded engines."""

    async def get_model_stats(self, model_id: str) -> tuple[InferenceEngine, str]:
        """Get (engine, status) for a model. Raises KeyError if not found."""

    def shutdown(self) -> None:
        """Release all resources on server shutdown."""


# ---------------------------------------------------------------------------
# Phase 1 + 2: SingleModelManager
# ---------------------------------------------------------------------------


class SingleModelManager:
    """One engine, one asyncio.Lock (Phase 1/2).

    Phase 2 addition: ``idle_timeout_sec`` — when > 0, unloads the session
    after that many seconds of inactivity and reloads on the next request.

    Usage::

        mgr = SingleModelManager(engine, idle_timeout_sec=300)
        async with mgr.borrow("_") as engine:
            result = engine.predict(inputs={"image": data})
    """

    def __init__(
        self,
        engine: InferenceEngine,
        idle_timeout_sec: float = 0.0,
    ) -> None:
        self._engine = engine
        self._lock = asyncio.Lock()
        self._idle_timeout = idle_timeout_sec
        self._idle_task: asyncio.Task | None = None
        self._last_release: float = time.monotonic()

    @asynccontextmanager
    async def borrow(
        self, model_id: str = "_", task: str | None = None
    ) -> AsyncIterator[InferenceEngine]:
        """Acquire the lock, reload if idle-unloaded, yield engine, then start idle timer."""
        async with self._lock:
            # Cancel any pending idle timer while we hold the lock
            self._cancel_idle_timer()

            # Reload if unloaded by the idle timer
            if not self._engine.is_loaded:
                logger.info("SingleModelManager: reloading after idle unload")
                self._engine.reload()

            yield self._engine

            self._last_release = time.monotonic()

        # Start idle timer after releasing lock so new requests can acquire it
        if self._idle_timeout > 0:
            self._start_idle_timer()

    async def list_models(self) -> list[dict]:
        """Return metadata for the single managed model."""
        status = "ready" if self._engine.is_loaded else "unloaded"
        return [
            {
                "model_id": self._engine.model_id or "_",
                "task": self._engine.task,
                "device": self._engine.device,
                "ep": self._engine.ep,
                "status": status,
                "request_count": self._engine.request_count,
                "memory_mb": self._engine.memory_mb,
            }
        ]

    async def switch_ep(self, ep: str) -> None:
        """Switch the engine's execution provider (Phase 1)."""
        async with self._lock:
            self._engine.switch_ep(ep)

    def get_engine(self, model_id: str | None = None) -> InferenceEngine | None:
        """Get the managed engine."""
        return self._engine if self._engine.is_loaded else None

    def get_all_engines(self) -> list[InferenceEngine]:
        """Get all loaded engines."""
        return [self._engine] if self._engine.is_loaded else []

    async def get_model_stats(self, model_id: str) -> tuple[InferenceEngine, str]:
        """Get (engine, status) for the managed model."""
        status = "ready" if self._engine.is_loaded else "unloaded"
        return self._engine, status

    def shutdown(self) -> None:
        """Release resources on server shutdown."""
        self._cancel_idle_timer()
        self._engine.unload()

    # ------------------------------------------------------------------
    # Phase 2: idle timeout
    # ------------------------------------------------------------------

    def _start_idle_timer(self) -> None:
        loop = asyncio.get_running_loop()
        self._idle_task = loop.create_task(self._idle_unload_after())

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_unload_after(self) -> None:
        try:
            await asyncio.sleep(self._idle_timeout)
            async with self._lock:
                if self._engine.is_loaded:
                    logger.info(
                        "SingleModelManager: idle timeout (%.0fs) — unloading session",
                        self._idle_timeout,
                    )
                    self._engine.unload()
        except asyncio.CancelledError:
            pass  # Timer was cancelled because a new request arrived — expected


# ---------------------------------------------------------------------------
# Phase 3: ModelSlotManager
# ---------------------------------------------------------------------------


@dataclass
class ModelSlot:
    """In-memory slot for one loaded model (inspired by Ollama's runnerRef)."""

    model_id: str
    engine: InferenceEngine
    refcount: int = 0
    last_used: float = field(default_factory=time.monotonic)
    expiry_task: asyncio.Task | None = field(default=None, compare=False, repr=False)
    alias: str | None = None
    description: str | None = None

    @property
    def memory_mb(self) -> float:
        """Delegate to engine.memory_mb."""
        return self.engine.memory_mb


class ModelSlotManager:
    """Multi-model manager with refcount + LRU eviction (Phase 3).

    Maintains a pool of loaded InferenceEngines.  When ``memory_budget_mb``
    is exceeded, evicts the least-recently-used slot with refcount == 0.

    Usage::

        mgr = ModelSlotManager(memory_budget_mb=4096, idle_timeout_sec=300)
        async with mgr.borrow("microsoft/resnet-50") as engine:
            result = engine.predict(inputs={"image": data})
    """

    def __init__(
        self,
        memory_budget_mb: float = 4096.0,
        idle_timeout_sec: float = 300.0,
        default_device: str = "auto",
    ) -> None:
        self._budget = memory_budget_mb
        self._idle_timeout = idle_timeout_sec
        self._default_device = default_device
        self._slots: dict[str, ModelSlot] = {}
        self._lock = asyncio.Lock()  # guards _slots mutations
        self._loading: set[str] = set()  # model IDs currently being loaded

    @asynccontextmanager
    async def borrow(
        self, model_id: str, task: str | None = None
    ) -> AsyncIterator[InferenceEngine]:
        """Acquire an engine for model_id, loading it if necessary.

        Resolution priority (when model_id is the sentinel ``"_"``):
          1. ``task`` hint matches exactly one slot → use it
          2. ``task`` hint matches multiple slots → use most recently used
          3. Only one slot loaded → use it regardless of task
          4. Otherwise → raise ValueError listing available (model_id, task) pairs
        """
        resolved = await self._resolve(model_id, task)
        slot = await self._acquire_slot(resolved)
        try:
            yield slot.engine
        finally:
            await self._release_slot(resolved)

    async def list_models(self) -> list[dict]:
        """Return metadata for all currently loaded model slots."""
        async with self._lock:
            return [
                {
                    "model_id": s.model_id,
                    "task": s.engine.task,
                    "device": s.engine.device,
                    "ep": s.engine.ep,
                    "status": "ready" if s.engine.is_loaded else "unloaded",
                    "refcount": s.refcount,
                    "memory_mb": s.memory_mb,
                    "request_count": s.engine.request_count,
                    "last_used_at": _fmt_monotonic(s.last_used),
                    "alias": s.alias,
                    "description": s.description,
                }
                for s in self._slots.values()
            ]

    async def unload_model(self, model_id: str) -> None:
        """Unload a model from the slot manager.

        Raises:
            KeyError: If model_id is not loaded.
            RuntimeError: If model is still in use (refcount > 0).
        """
        async with self._lock:
            slot = self._slots.get(model_id)
            if slot is None:
                raise KeyError(model_id)
            if slot.refcount > 0:
                raise RuntimeError(f"Model '{model_id}' is in use (refcount={slot.refcount})")
            slot.engine.unload()
            del self._slots[model_id]

    def get_engine(self, model_id: str | None = None) -> InferenceEngine | None:
        """Get engine for a model_id, or the first available engine."""
        if model_id:
            slot = self._slots.get(model_id)
            if slot:
                return slot.engine
            # Fallback: match by engine.model_id (handles URL encoding quirks)
            mid = model_id.strip()
            for s in self._slots.values():
                if s.engine.model_id == mid:
                    return s.engine
            return None
        if self._slots:
            return next(iter(self._slots.values())).engine
        return None

    def get_all_engines(self) -> list[InferenceEngine]:
        """Get all loaded engines."""
        return [s.engine for s in self._slots.values() if s.engine.is_loaded]

    async def get_model_stats(self, model_id: str) -> tuple[InferenceEngine, str]:
        """Get (engine, status) for a model. Raises KeyError if not found."""
        async with self._lock:
            slot = self._slots.get(model_id)
        if slot is None:
            raise KeyError(model_id)
        status = "ready" if slot.engine.is_loaded else "unloaded"
        return slot.engine, status

    def shutdown(self) -> None:
        """Release all resources on server shutdown."""
        for slot in self._slots.values():
            if slot.expiry_task and not slot.expiry_task.done():
                slot.expiry_task.cancel()
            slot.engine.unload()
        self._slots.clear()

    async def load_model(
        self,
        model_id: str,
        *,
        task: str | None = None,
        device: str | None = None,
        ep: str | None = None,
        alias: str | None = None,
        description: str | None = None,
    ) -> None:
        """Explicitly load a model with optional alias and description metadata.

        If the model is already in a slot, only updates alias/description.
        Use this instead of ``borrow()`` when metadata must be attached at load
        time (e.g. from ``POST /v1/models``).
        """
        async with self._lock:
            slot = self._slots.get(model_id)
            if slot is not None:
                # Already loaded — update metadata only
                if alias is not None:
                    slot.alias = alias
                if description is not None:
                    slot.description = description
                logger.info("ModelSlotManager: updated metadata for %s (alias=%s)", model_id, alias)
                return

            await self._maybe_evict(exclude=model_id)

        # Load outside the lock — this can take seconds/minutes
        engine = InferenceEngine()
        _device = device or self._default_device
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: engine.load(model_id, task=task, device=_device, ep=ep)
        )

        async with self._lock:
            slot = ModelSlot(
                model_id=model_id,
                engine=engine,
                alias=alias,
                description=description,
            )
            self._slots[model_id] = slot
            logger.info("ModelSlotManager: loaded %s (alias=%s)", model_id, alias)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _resolve(self, model_id: str, task: str | None = None) -> str:
        """Resolve model_id to a concrete slot key.

        When model_id is the sentinel ``"_"``:
          1. task hint matches an alias exactly → that slot (programmatic routing)
          2. task hint matches engine.task, unique → that slot
          3. task hint matches engine.task, multiple → most-recently-used
          4. task hint provided, no match → ValueError listing available (model_id, task)
          5. no task hint, single slot → that slot
          6. no task hint, multiple slots → ValueError listing options
        """
        if model_id != "_":
            return model_id

        async with self._lock:
            if not self._slots:
                raise ValueError("No models loaded. POST /v1/models to load one first.")

            slots = list(self._slots.values())

            if task:
                # 1. Alias match — exact, unambiguous (programmatic agent path)
                alias_matches = [s for s in slots if s.alias == task]
                if alias_matches:
                    return max(alias_matches, key=lambda s: s.last_used).model_id

                # 2/3. Task match — MRU wins when multiple models share the same task
                matches = [s for s in slots if s.engine.task == task]
                if not matches:
                    available = [(s.model_id, s.engine.task) for s in slots]
                    raise ValueError(
                        f"No loaded model handles task '{task}'. Available: {available}"
                    )
                return max(matches, key=lambda s: s.last_used).model_id

            if len(slots) == 1:
                return slots[0].model_id

            available = [(s.model_id, s.engine.task) for s in slots]
            raise ValueError(
                f"Multiple models loaded — specify model_id or task. Available: {available}"
            )

    async def _acquire_slot(self, model_id: str) -> ModelSlot:
        # Fast path: model already loaded — just bump refcount under lock.
        # Also decides whether *this* coroutine is the designated loader.
        am_loader = False
        async with self._lock:
            slot = self._slots.get(model_id)
            if slot is not None:
                if slot.expiry_task and not slot.expiry_task.done():
                    slot.expiry_task.cancel()
                    slot.expiry_task = None
                slot.refcount += 1
                slot.last_used = time.monotonic()
                return slot

            if model_id not in self._loading:
                # We are the designated loader for this model_id
                am_loader = True
                self._loading.add(model_id)
                await self._maybe_evict(exclude=model_id)
            # else: another coroutine is already loading — fall through to poll

        if am_loader:
            # Do the slow load *outside* the lock
            try:
                engine = InferenceEngine()
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, lambda: engine.load(model_id, device=self._default_device)
                )
            except BaseException:
                async with self._lock:
                    self._loading.discard(model_id)
                raise

            # Atomically publish the slot and remove from _loading
            async with self._lock:
                self._loading.discard(model_id)
                slot = ModelSlot(model_id=model_id, engine=engine)
                self._slots[model_id] = slot
                logger.info("ModelSlotManager: loaded %s", model_id)
                slot.refcount += 1
                slot.last_used = time.monotonic()
                return slot

        # Another coroutine is loading this model —
        # poll until the slot appears (the loader will finish shortly)
        while True:
            await asyncio.sleep(0.1)
            async with self._lock:
                slot = self._slots.get(model_id)
                if slot is not None:
                    if slot.expiry_task and not slot.expiry_task.done():
                        slot.expiry_task.cancel()
                        slot.expiry_task = None
                    slot.refcount += 1
                    slot.last_used = time.monotonic()
                    return slot
                if model_id not in self._loading:
                    raise RuntimeError(f"Model '{model_id}' failed to load")

    async def _release_slot(self, model_id: str) -> None:
        async with self._lock:
            slot = self._slots.get(model_id)
            if slot is None:
                return
            slot.refcount = max(0, slot.refcount - 1)
            slot.last_used = time.monotonic()

            if slot.refcount == 0 and self._idle_timeout > 0:
                # Cancel any previous expiry task to avoid leaked timers
                if slot.expiry_task and not slot.expiry_task.done():
                    slot.expiry_task.cancel()
                slot.expiry_task = asyncio.get_running_loop().create_task(
                    self._expire_slot(model_id, self._idle_timeout)
                )

    async def _expire_slot(self, model_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            async with self._lock:
                slot = self._slots.get(model_id)
                if slot and slot.refcount == 0:
                    logger.info("ModelSlotManager: idle timeout — evicting %s", model_id)
                    slot.engine.unload()
                    del self._slots[model_id]
        except asyncio.CancelledError:
            pass  # Slot was re-acquired before expiry — expected

    async def _maybe_evict(self, exclude: str) -> None:
        """Evict LRU idle slot if total memory would exceed budget."""
        total = sum(s.memory_mb for s in self._slots.values())
        if total < self._budget:
            return

        # Candidates: idle slots (refcount == 0), sorted LRU first
        candidates = sorted(
            (s for s in self._slots.values() if s.refcount == 0 and s.model_id != exclude),
            key=lambda s: s.last_used,
        )
        for slot in candidates:
            logger.info(
                "ModelSlotManager: evicting %s (LRU, memory budget exceeded)",
                slot.model_id,
            )
            slot.engine.unload()
            del self._slots[slot.model_id]
            if sum(s.memory_mb for s in self._slots.values()) < self._budget:
                break


def _fmt_monotonic(t: float) -> str:
    """Convert monotonic timestamp to approximate ISO string (for display only)."""
    elapsed = time.monotonic() - t
    approx = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(seconds=elapsed)
    return approx.isoformat()
