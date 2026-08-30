from __future__ import annotations

import unittest

from cortex_deployer.engines.comfyui import (
    DIT,
    TEXT_ENCODER,
    TURBO_LORA,
    apply_resolved_status,
    archive_mp4_path,
    decode_image_payload,
    h3_frame_length,
    import_comfy_mp4s,
    max_seconds_for_size,
    parse_seconds,
    parse_size,
    parse_target_seconds,
    resolve_video_status,
    t2v_api_prompt,
)
from cortex_deployer.recipes import load_recipe, list_examples


class ComfyuiWorkflowTests(unittest.TestCase):
    def test_frame_length_snaps_to_17k5(self):
        self.assertEqual(h3_frame_length(5), 124)
        self.assertEqual(h3_frame_length(15), 362)
        self.assertEqual(h3_frame_length(0), 5)
        self.assertEqual((h3_frame_length(5) - 5) % 17, 0)
        self.assertEqual((h3_frame_length(15) - 5) % 17, 0)

    def test_parse_size(self):
        self.assertEqual(parse_size("864x480"), (864, 480))
        self.assertEqual(parse_size("16:9"), (864, 480))
        self.assertEqual(parse_size("9:16"), (480, 864))
        self.assertEqual(parse_size("720p"), (1344, 768))
        self.assertEqual(parse_size("", resolution="1080p", aspect="16:9"), (1920, 1088))
        self.assertEqual(parse_size("3840x2160"), (1920, 1920))  # 4K is not local; clamp to 1080p edge
        self.assertEqual(parse_size("bad"), (864, 480))

    def test_parse_seconds_clamps_to_3090_envelope(self):
        self.assertEqual(parse_seconds(5), 5)
        self.assertEqual(parse_seconds(1), 4)
        self.assertEqual(parse_seconds(15), 15)
        self.assertEqual(parse_seconds(30), 15)
        self.assertEqual(parse_seconds(60), 15)
        self.assertEqual(parse_seconds("nope"), 5)
        self.assertEqual(parse_seconds(10, width=1920, height=1088), 6)
        self.assertEqual(parse_seconds(15, width=1344, height=768), 10)
        self.assertEqual(max_seconds_for_size(864, 480), 15)
        self.assertEqual(max_seconds_for_size(1344, 768), 10)
        self.assertEqual(max_seconds_for_size(1920, 1088), 6)

    def test_missing_history_is_failed_not_queued(self):
        """Comfy returns 200 {{}} for unknown ids — that is not a live queue."""
        ghost = resolve_video_status("missing-id", history={}, queue={"queue_running": [], "queue_pending": []})
        self.assertEqual(ghost["status"], "failed")
        self.assertIn("not on the engine", ghost["error"])

    def test_queue_running_is_in_progress(self):
        pid = "live-1"
        meta = resolve_video_status(
            pid,
            local={"id": pid, "submitted_at": 1.0},
            history={},
            queue={"queue_running": [[0, pid, {}]], "queue_pending": []},
            now=100.0,
        )
        self.assertEqual(meta["status"], "in_progress")

    def test_just_submitted_stays_queued_during_grace(self):
        pid = "new-1"
        meta = resolve_video_status(
            pid,
            local={"id": pid, "submitted_at": 90.0},
            history={},
            queue={"queue_running": [], "queue_pending": []},
            now=100.0,
        )
        self.assertEqual(meta["status"], "queued")

    def test_compose_parent_stays_in_progress_after_grace(self):
        """Studio GET /v1/videos/{parent} after 20s. Parent UUID is not a Comfy prompt.

        Production order (30s stitch): POST compose → composer sets in_progress
        segment 1 → Studio polls parent id while Comfy runs a different prompt_id
        → empty history + empty queue. Must stay in_progress, not 'not on the engine'.
        """
        parent = "496247e5-4b3f-4448-b6a0-007bd9ca2b06"
        local = {
            "id": parent,
            "compose": True,
            "status": "in_progress",
            "segment": 1,
            "segments": 2,
            "submitted_at": 1.0,
        }
        first = resolve_video_status(
            parent,
            local=local,
            history={},
            queue={"queue_running": [[0, "comfy-seg-1", {}]], "queue_pending": []},
            now=25.0,
        )
        self.assertEqual(first["status"], "in_progress")
        self.assertNotIn("error", first)
        second = resolve_video_status(
            parent,
            local=dict(first, segment=2),
            history={},
            queue={"queue_running": [], "queue_pending": []},
            now=550.0,
        )
        self.assertEqual(second["status"], "in_progress")
        self.assertNotIn("error", second)

    def test_compose_poll_must_not_poison_live_row(self):
        """GET merge used to stamp failed onto the composer row mid-stitch."""
        parent = "job-30s"
        live = {
            "id": parent,
            "compose": True,
            "status": "in_progress",
            "segment": 2,
            "submitted_at": 1.0,
        }
        poisoned = resolve_video_status(
            parent,
            local={"id": parent, "submitted_at": 1.0},
            history={},
            queue={"queue_running": [], "queue_pending": []},
            now=100.0,
        )
        self.assertEqual(poisoned["status"], "failed")
        kept = apply_resolved_status(live, poisoned)
        self.assertEqual(kept["status"], "in_progress")
        self.assertEqual(kept["segment"], 2)

    def test_compose_failed_keeps_composer_error(self):
        parent = "job-fail"
        local = {
            "id": parent,
            "compose": True,
            "status": "failed",
            "error": "segment produced no video",
            "submitted_at": 1.0,
        }
        meta = resolve_video_status(
            parent,
            local=local,
            history={},
            queue={"queue_running": [], "queue_pending": []},
            now=100.0,
        )
        self.assertEqual(meta["status"], "failed")
        self.assertEqual(meta["error"], "segment produced no video")

    def test_history_completed_wins(self):
        pid = "done-1"
        meta = resolve_video_status(
            pid,
            history={
                pid: {
                    "status": {"completed": True, "status_str": "success"},
                    "outputs": {"save": {"videos": [{"filename": "x.mp4", "type": "output"}]}},
                }
            },
            queue={"queue_running": [], "queue_pending": []},
        )
        self.assertEqual(meta["status"], "completed")
        self.assertEqual(meta["asset"]["filename"], "x.mp4")

    def test_parse_target_seconds_allows_hour(self):
        self.assertEqual(parse_target_seconds(3600), 3600)
        self.assertEqual(parse_target_seconds(7200), 3600)
        self.assertEqual(parse_target_seconds(1), 4)

    def test_import_comfy_mp4s_archives_loose_files(self):
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CORTEX_VIDEO_ARCHIVE"] = str(Path(tmp) / "arch")
            comfy = Path(tmp) / "comfy"
            out = comfy / "output" / "video"
            out.mkdir(parents=True)
            (out / "MiniMax_H3_00001_.mp4").write_bytes(b"ftypfake" + b"\x00" * 80)
            imported = import_comfy_mp4s(comfy)
            self.assertEqual(len(imported), 1)
            vid = next(iter(imported))
            self.assertTrue(archive_mp4_path(vid).is_file())
            ghost = resolve_video_status(
                vid,
                local={"id": vid, "archived": True},
                history={},
                queue={"queue_running": [], "queue_pending": []},
            )
            self.assertEqual(ghost["status"], "completed")

    def test_t2v_turbo_graph(self):
        graph = t2v_api_prompt("a lantern over rain", turbo=True, duration_s=5)
        self.assertEqual(graph["unet"]["inputs"]["unet_name"], DIT)
        self.assertEqual(graph["clip"]["inputs"]["clip_name"], TEXT_ENCODER)
        self.assertEqual(graph["clip"]["inputs"]["type"], "minimax")
        self.assertEqual(graph["i2v"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(graph["i2v"]["inputs"]["length"], 124)
        self.assertEqual(graph["lora"]["inputs"]["lora_name"], TURBO_LORA)
        self.assertEqual(graph["sched"]["inputs"]["model"], ["lora", 0])
        self.assertEqual(graph["sched"]["inputs"]["steps"], 8)
        self.assertEqual(graph["save"]["class_type"], "SaveVideo")

    def test_t2v_wires_optional_keyframes(self):
        graph = t2v_api_prompt(
            "animate this still",
            first_image_name="start.jpg",
            last_image_name="end.jpg",
        )
        self.assertEqual(graph["first_img"]["class_type"], "LoadImage")
        self.assertEqual(graph["first_img"]["inputs"]["image"], "start.jpg")
        self.assertEqual(graph["i2v"]["inputs"]["first_frame"], ["first_img", 0])
        self.assertEqual(graph["last_img"]["inputs"]["image"], "end.jpg")
        self.assertEqual(graph["i2v"]["inputs"]["last_frame"], ["last_img", 0])

    def test_decode_data_url(self):
        png = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        )
        blob, mime = decode_image_payload(f"data:image/png;base64,{png}")
        self.assertEqual(mime, "image/png")
        self.assertGreater(len(blob), 16)
        self.assertIsNone(decode_image_payload(""))

    def test_t2v_non_turbo_skips_lora(self):
        graph = t2v_api_prompt("still scene", turbo=False, steps=20)
        self.assertNotIn("lora", graph)
        self.assertEqual(graph["sched"]["inputs"]["model"], ["unet", 0])
        self.assertEqual(graph["sched"]["inputs"]["steps"], 20)

    def test_example_recipe_loads(self):
        path = next(p for p in list_examples() if p.name == "minimax-h3-fl2va-comfyui.yaml")
        recipe = load_recipe(path)
        self.assertEqual(recipe.engine, "comfyui")
        self.assertEqual(recipe.model.served_name, "MiniMax-H3")
        self.assertEqual(recipe.model.repo, "Comfy-Org/MiniMax-H3")
        self.assertEqual(recipe.min_vram_mb, 22000)
        self.assertIn("int8", recipe.quant)
        blob = path.read_text(encoding="utf-8").lower()
        self.assertNotIn("v4", blob)
        self.assertNotRegex(blob, r"100\.64\.0\.")
