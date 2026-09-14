import csv
import json
import time

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class StageProfiler:
    """Wall-clock profiler with explicit CUDA synchronization.

    CUDA kernels are asynchronous. Synchronizing at each profiled CUDA boundary
    makes per-stage measurements truthful, at the cost of a small profiling
    overhead. The normal demo path is unchanged when profiling is disabled.
    """

    def __init__(self, enabled: bool = False, device: str = "cpu"):
        self.enabled = enabled
        self.device = str(device)
        self.records = defaultdict(list)
        self.metadata: Dict[str, Any] = {}
        self.started_at = time.perf_counter()

    @property
    def uses_cuda(self) -> bool:
        return (
            self.enabled
            and self.device.startswith("cuda")
            and torch.cuda.is_available()
        )

    def synchronize(self):
        if self.uses_cuda:
            torch.cuda.synchronize(torch.device(self.device))

    @contextmanager
    def measure(self, name: str, cuda: bool = False):
        if not self.enabled:
            yield
            return
        if cuda:
            self.synchronize()
        started_at = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                self.synchronize()
            self.records[name].append(time.perf_counter() - started_at)

    def add(self, name: str, seconds: float):
        if self.enabled:
            self.records[name].append(float(seconds))

    def set_metadata(self, **kwargs):
        if self.enabled:
            self.metadata.update(kwargs)

    def finish(self):
        if not self.enabled:
            return
        self.synchronize()
        self.metadata["end_to_end_seconds"] = time.perf_counter() - self.started_at

    def summary(self) -> Dict[str, Any]:
        end_to_end = float(self.metadata.get("end_to_end_seconds", 0.0))
        stages = {}
        for name in sorted(self.records):
            samples = self.records[name]
            total = float(sum(samples))
            stages[name] = {
                "calls": len(samples),
                "total_seconds": total,
                "mean_seconds": total / len(samples),
                "min_seconds": float(min(samples)),
                "max_seconds": float(max(samples)),
                "percent_end_to_end": (
                    total / end_to_end * 100.0 if end_to_end > 0 else None
                ),
            }
        return {
            "metadata": self.metadata,
            "stages": stages,
            "notes": [
                "CUDA stages use explicit synchronization for accurate boundaries.",
                "Stage totals are exclusive unless a stage name explicitly says total.",
                "Video decode combines compressed-file reads and CPU decoding because FFmpeg performs them as one streaming operation.",
                "SKEL skin and skeleton share one kinematic forward pass; their GPU generation is reported as a combined stage.",
                "Video encoding and file writing are combined because FFmpeg encodes and writes as one streaming operation.",
            ],
        }

    def report(self):
        if not self.enabled:
            return
        data = self.summary()
        total = data["metadata"]["end_to_end_seconds"]
        print("\n=== HSMR detailed stage profile ===")
        print(f"{'stage':52s} {'calls':>7s} {'total(s)':>11s} {'mean(ms)':>11s} {'end2end':>9s}")
        for name, stats in data["stages"].items():
            print(
                f"{name:52s} "
                f"{stats['calls']:7d} "
                f"{stats['total_seconds']:11.4f} "
                f"{stats['mean_seconds'] * 1000:11.3f} "
                f"{stats['percent_end_to_end']:8.2f}%"
            )
        print(f"{'END_TO_END':52s} {'1':>7s} {total:11.4f}")

    def dump(self, output_path: Union[str, Path]):
        if not self.enabled:
            return
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.summary()

        json_path = output_path.with_suffix(".json")
        csv_path = output_path.with_suffix(".csv")
        txt_path = output_path.with_suffix(".txt")

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)

        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "stage",
                "calls",
                "total_seconds",
                "mean_seconds",
                "min_seconds",
                "max_seconds",
                "percent_end_to_end",
            ])
            for name, stats in data["stages"].items():
                writer.writerow([
                    name,
                    stats["calls"],
                    stats["total_seconds"],
                    stats["mean_seconds"],
                    stats["min_seconds"],
                    stats["max_seconds"],
                    stats["percent_end_to_end"],
                ])

        with open(txt_path, "w", encoding="utf-8") as file:
            file.write("HSMR detailed stage profile\n")
            file.write(f"end_to_end_seconds: {data['metadata']['end_to_end_seconds']:.6f}\n\n")
            file.write(
                f"{'stage':52s} {'calls':>7s} {'total(s)':>11s} "
                f"{'mean(ms)':>11s} {'end2end':>9s}\n"
            )
            for name, stats in data["stages"].items():
                file.write(
                    f"{name:52s} "
                    f"{stats['calls']:7d} "
                    f"{stats['total_seconds']:11.4f} "
                    f"{stats['mean_seconds'] * 1000:11.3f} "
                    f"{stats['percent_end_to_end']:8.2f}%\n"
                )
            file.write("\nNotes:\n")
            for note in data["notes"]:
                file.write(f"- {note}\n")

        return json_path, csv_path, txt_path
