#!/usr/bin/env python3
"""
YOLO 2026 Detection Skill — Real-time object detection for SharpAI Aegis.

Communicates via JSON lines over stdin/stdout:
  stdin:  {"event": "frame", "frame_id": N, "camera_id": "...", "frame_path": "...", ...}
  stdout: {"event": "detections", "frame_id": N, "camera_id": "...", "objects": [...]}

On Apple Silicon (MPS), auto-converts to CoreML for ~2x faster inference via ANE.
Emits periodic performance statistics via "perf_stats" events.

Usage:
  python detect.py --config config.json
  python detect.py --model-size nano --confidence 0.5 --device auto
"""

import sys
import json
import argparse
import signal
import time
from pathlib import Path


# Model size → ultralytics model name mapping (YOLO26, released Jan 2026)
MODEL_SIZE_MAP = {
    "nano": "yolo26n",
    "small": "yolo26s",
    "medium": "yolo26m",
    "large": "yolo26l",
}

# How often to emit aggregate perf stats (every N frames)
PERF_STATS_INTERVAL = 50


# ───────────────────────────────────────────────────────────────────────────────
# Performance tracker — collects per-frame timings, emits aggregate stats
# ───────────────────────────────────────────────────────────────────────────────

class PerfTracker:
    """Tracks timing for each pipeline stage and emits periodic statistics."""

    def __init__(self, interval: int = PERF_STATS_INTERVAL):
        self.interval = interval
        self.frame_count = 0
        self.total_frames = 0
        self.error_count = 0

        # One-time timings (ms)
        self.model_load_ms = 0.0
        self.coreml_export_ms = 0.0

        # Per-frame accumulators (ms)
        self._timings: dict[str, list[float]] = {
            "file_read":    [],   # frame_path existence check + file I/O
            "inference":    [],   # model(frame_path, ...)
            "postprocess":  [],   # bbox extraction + filtering
            "emit":         [],   # JSON serialization + print
            "total":        [],   # end-to-end per frame
        }

    def record(self, stage: str, duration_ms: float):
        """Record a timing for a pipeline stage."""
        if stage in self._timings:
            self._timings[stage].append(duration_ms)

    def record_frame(self):
        """Increment frame counter and emit stats if interval reached."""
        self.frame_count += 1
        self.total_frames += 1
        if self.frame_count >= self.interval:
            self.emit_stats()
            self.frame_count = 0

    def emit_stats(self):
        """Emit aggregate statistics as a JSONL event."""
        stats = {
            "event": "perf_stats",
            "total_frames": self.total_frames,
            "window_size": len(self._timings["total"]) or 1,
            "errors": self.error_count,
            "model_load_ms": round(self.model_load_ms, 1),
            "timings_ms": {},
        }

        if self.coreml_export_ms > 0:
            stats["coreml_export_ms"] = round(self.coreml_export_ms, 1)

        for stage, values in self._timings.items():
            if not values:
                continue
            sorted_v = sorted(values)
            n = len(sorted_v)
            stats["timings_ms"][stage] = {
                "avg": round(sum(sorted_v) / n, 2),
                "min": round(sorted_v[0], 2),
                "max": round(sorted_v[-1], 2),
                "p50": round(sorted_v[n // 2], 2),
                "p95": round(sorted_v[int(n * 0.95)], 2),
                "p99": round(sorted_v[int(n * 0.99)], 2),
            }

        emit(stats)

        # Reset per-frame accumulators for next window
        for key in self._timings:
            self._timings[key].clear()

    def emit_final(self):
        """Emit remaining stats on shutdown."""
        if self._timings["total"]:
            self.emit_stats()


# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 2026 Detection Skill")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--model-size", type=str, default="nano",
                        choices=["nano", "small", "medium", "large"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--classes", type=str, default="person,car,dog,cat")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps", "rocm"])
    parser.add_argument("--fps", type=float, default=5)
    return parser.parse_args()


def load_config(args):
    """Load config from JSON file, CLI args, or AEGIS_SKILL_PARAMS env var."""
    import os

    # Priority 1: AEGIS_SKILL_PARAMS env var (set by Aegis skill-runtime-manager)
    env_params = os.environ.get("AEGIS_SKILL_PARAMS")
    if env_params:
        try:
            return json.loads(env_params)
        except json.JSONDecodeError:
            pass

    # Priority 2: Config file
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

    # Priority 3: CLI args
    return {
        "model_size": args.model_size,
        "confidence": args.confidence,
        "classes": args.classes.split(","),
        "device": args.device,
        "fps": args.fps,
    }


def select_device(preference: str) -> str:
    """Select the best available inference device."""
    if preference not in ("auto", ""):
        return preference
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def emit(event: dict):
    """Write a JSON line to stdout."""
    print(json.dumps(event), flush=True)


def log(msg: str):
    """Write a log message to stderr (visible in Aegis deploy console)."""
    print(f"[YOLO-2026] {msg}", file=sys.stderr, flush=True)


def try_coreml_export(model, model_name: str, perf: PerfTracker) -> "Path | None":
    """Export PyTorch model to CoreML. Returns path to .mlpackage or None."""
    coreml_path = Path(f"{model_name}.mlpackage")

    # Already exported
    if coreml_path.exists():
        log(f"CoreML model found: {coreml_path}")
        return coreml_path

    try:
        log(f"Exporting {model_name}.pt → CoreML (one-time, ~30s)...")
        t0 = time.perf_counter()
        exported = model.export(format="coreml", half=True, nms=False)
        perf.coreml_export_ms = (time.perf_counter() - t0) * 1000
        exported_path = Path(exported)
        if exported_path.exists():
            log(f"CoreML export complete: {exported_path} ({perf.coreml_export_ms:.0f}ms)")
            return exported_path
        log(f"CoreML export returned path {exported} but file not found")
    except Exception as e:
        log(f"CoreML export failed: {e}")

    return None


def load_model(model_name: str, device: str, use_coreml: bool, perf: PerfTracker):
    """Load YOLO model — CoreML on MPS if available, PyTorch otherwise."""
    from ultralytics import YOLO

    model_format = "pytorch"
    t0 = time.perf_counter()

    # Try CoreML on Apple Silicon
    if device == "mps" and use_coreml:
        pt_model = YOLO(f"{model_name}.pt")
        coreml_path = try_coreml_export(pt_model, model_name, perf)

        if coreml_path:
            try:
                model = YOLO(str(coreml_path))
                model_format = "coreml"
                perf.model_load_ms = (time.perf_counter() - t0) * 1000
                log(f"Loaded CoreML model ({coreml_path}) in {perf.model_load_ms:.0f}ms")
                return model, model_format
            except Exception as e:
                log(f"CoreML load failed, falling back to PyTorch MPS: {e}")

        # Fallback: use the already-loaded PyTorch model on MPS
        pt_model.to(device)
        perf.model_load_ms = (time.perf_counter() - t0) * 1000
        return pt_model, model_format

    # Non-CoreML path: standard PyTorch
    model = YOLO(f"{model_name}.pt")
    model.to(device)
    perf.model_load_ms = (time.perf_counter() - t0) * 1000
    return model, model_format


# ───────────────────────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = load_config(args)

    # Resolve config values
    model_size = config.get("model_size", "nano")
    device = select_device(config.get("device", "auto"))
    confidence = config.get("confidence", 0.5)
    fps = config.get("fps", 5)
    use_coreml = config.get("use_coreml", True)

    # Coerce use_coreml from string "true"/"false" if passed via env
    if isinstance(use_coreml, str):
        use_coreml = use_coreml.lower() in ("true", "1", "yes")

    # Map size to ultralytics model name
    model_name = MODEL_SIZE_MAP.get(model_size, "yolo26n")

    target_classes = config.get("classes", ["person", "car", "dog", "cat"])
    if isinstance(target_classes, str):
        target_classes = [c.strip() for c in target_classes.split(",")]

    # Performance tracker
    perf = PerfTracker(interval=PERF_STATS_INTERVAL)

    # Load YOLO model (with CoreML auto-conversion on MPS)
    try:
        model, model_format = load_model(model_name, device, use_coreml, perf)
        emit({
            "event": "ready",
            "model": f"yolo2026{model_size[0]}",
            "model_size": model_size,
            "device": device,
            "format": model_format,
            "classes": len(model.names),
            "fps": fps,
            "model_load_ms": round(perf.model_load_ms, 1),
            "available_sizes": list(MODEL_SIZE_MAP.keys()),
        })
    except Exception as e:
        emit({"event": "error", "message": f"Failed to load model: {e}", "retriable": False})
        sys.exit(1)

    # Graceful shutdown
    running = True
    def handle_signal(signum, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop: read frames from stdin, output detections to stdout
    for line in sys.stdin:
        if not running:
            break

        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        if msg.get("command") == "stop":
            break

        if msg.get("event") == "frame":
            t_frame_start = time.perf_counter()

            frame_path = msg.get("frame_path")
            frame_id = msg.get("frame_id")
            camera_id = msg.get("camera_id", "unknown")
            timestamp = msg.get("timestamp", "")

            # ── File check ──
            t0 = time.perf_counter()
            if not frame_path or not Path(frame_path).exists():
                emit({
                    "event": "error",
                    "frame_id": frame_id,
                    "message": f"Frame not found: {frame_path}",
                    "retriable": True,
                })
                perf.error_count += 1
                continue
            perf.record("file_read", (time.perf_counter() - t0) * 1000)

            # ── Inference ──
            try:
                t0 = time.perf_counter()
                results = model(frame_path, conf=confidence, verbose=False)
                perf.record("inference", (time.perf_counter() - t0) * 1000)

                # ── Postprocess ──
                t0 = time.perf_counter()
                objects = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        if cls_name in target_classes or not target_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            objects.append({
                                "class": cls_name,
                                "confidence": round(float(box.conf[0]), 3),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            })
                perf.record("postprocess", (time.perf_counter() - t0) * 1000)

                # ── Emit ──
                t0 = time.perf_counter()
                emit({
                    "event": "detections",
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "timestamp": timestamp,
                    "objects": objects,
                })
                perf.record("emit", (time.perf_counter() - t0) * 1000)

            except Exception as e:
                emit({
                    "event": "error",
                    "frame_id": frame_id,
                    "message": f"Inference error: {e}",
                    "retriable": True,
                })
                perf.error_count += 1
                continue

            # ── Total frame time ──
            perf.record("total", (time.perf_counter() - t_frame_start) * 1000)
            perf.record_frame()

    # Emit final stats on shutdown
    perf.emit_final()


if __name__ == "__main__":
    main()
