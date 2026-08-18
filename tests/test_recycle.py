from __future__ import annotations

import unittest

from cortex_deployer.recycle import (
    EngineSnapshot,
    RecycleState,
    parse_metrics,
    should_recycle,
)


SAMPLE = """
# TYPE rapid_mlx_requests_running gauge
rapid_mlx_requests_running 0
# TYPE rapid_mlx_requests_waiting gauge
rapid_mlx_requests_waiting 0
# TYPE rapid_mlx_metal_active_memory_bytes gauge
rapid_mlx_metal_active_memory_bytes 101010000000
# TYPE rapid_mlx_metal_cap_violations_total counter
rapid_mlx_metal_cap_violations_total 972
rapid_mlx_kv_cache_dtype{dtype="bf16"} 1
"""


class ParseMetricsTests(unittest.TestCase):
    def test_unlabeled_gauges(self):
        snap = parse_metrics(SAMPLE)
        self.assertEqual(snap.running, 0.0)
        self.assertEqual(snap.waiting, 0.0)
        self.assertEqual(snap.metal_active, 101010000000.0)
        self.assertEqual(snap.cap_violations, 972.0)

    def test_empty(self):
        snap = parse_metrics("")
        self.assertIsNone(snap.running)
        self.assertIsNone(snap.cap_violations)


class RecycleDecisionTests(unittest.TestCase):
    def test_busy_never_recycles(self):
        state = RecycleState(last_violations=10)
        snap = EngineSnapshot(running=1, waiting=0, cap_violations=99)
        fire, reason = should_recycle(snap, state, now=100.0, min_hits_needed=1)
        self.assertFalse(fire)
        self.assertEqual(reason, "busy")
        self.assertEqual(state.hits, 0)

    def test_waiting_never_recycles(self):
        state = RecycleState(last_violations=10)
        snap = EngineSnapshot(running=0, waiting=2, cap_violations=99)
        fire, reason = should_recycle(snap, state, now=100.0, min_hits_needed=1)
        self.assertFalse(fire)
        self.assertEqual(reason, "busy")

    def test_idle_violation_climb_recycles_after_hits(self):
        state = RecycleState(last_violations=970)
        snap = EngineSnapshot(running=0, waiting=0, cap_violations=971)
        fire, reason = should_recycle(snap, state, now=50.0, min_hits_needed=2)
        self.assertFalse(fire)
        self.assertEqual(reason, "accumulating")
        snap.cap_violations = 972
        fire, reason = should_recycle(snap, state, now=65.0, min_hits_needed=2)
        self.assertTrue(fire)
        self.assertEqual(reason, "idle-metal-cap")

    def test_idle_stable_does_not_recycle(self):
        state = RecycleState(last_violations=972)
        snap = EngineSnapshot(running=0, waiting=0, cap_violations=972)
        fire, reason = should_recycle(snap, state, now=50.0, min_hits_needed=1)
        self.assertFalse(fire)
        self.assertEqual(reason, "stable")

    def test_cooldown_blocks(self):
        state = RecycleState(last_violations=10, last_recycle_mono=10.0)
        snap = EngineSnapshot(running=0, waiting=0, cap_violations=99)
        fire, reason = should_recycle(
            snap, state, now=100.0, min_hits_needed=1, cooldown_s=900.0
        )
        self.assertFalse(fire)
        self.assertEqual(reason, "cooldown")

    def test_prime_then_first_climb(self):
        state = RecycleState()
        snap = EngineSnapshot(running=0, waiting=0, cap_violations=10)
        fire, reason = should_recycle(snap, state, now=1.0, min_hits_needed=1)
        self.assertFalse(fire)
        self.assertEqual(reason, "prime")
        snap.cap_violations = 11
        fire, reason = should_recycle(snap, state, now=20.0, min_hits_needed=1)
        self.assertTrue(fire)
        self.assertEqual(reason, "idle-metal-cap")
