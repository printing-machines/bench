"""
PaintBench Logging

Handles logging of action traces, canvas snapshots, and benchmark results.
"""

import json
from datetime import datetime
from pathlib import Path

from PIL import Image

from .text_prompts import get_text_prompt
from .vlms.vlm import Action


class BenchmarkLogger:
    """Logger for a single benchmark run."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.snapshots_dir = self.run_dir / "snapshots"

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(exist_ok=True)

        # Action trace file
        self.actions_file = self.run_dir / "actions.jsonl"
        self.action_count = 0

        # Clear existing actions file
        self.actions_file.write_text("")

    def log_action(
        self,
        action: Action,
        canvas_image: Image.Image | None = None,
        timestamp: datetime | None = None,
        turn_number: int | None = None,
    ) -> None:
        """Log an action and optionally save canvas snapshot."""
        if timestamp is None:
            timestamp = datetime.now()

        self.action_count += 1

        # Build action record
        record = {
            "action_number": self.action_count,
            "timestamp": timestamp.isoformat(),
            "type": action.type,
            "params": action.params,
        }

        # Include turn number for multi-move mode
        if turn_number is not None:
            record["turn_number"] = turn_number

        # Include reasoning if present
        if action.reasoning:
            record["reasoning"] = action.reasoning

        # Include focus summary if present
        if action.focus_summary:
            record["focus_summary"] = action.focus_summary

        # Include notes content if present
        if action.notes_content is not None:
            record["notes_content"] = action.notes_content

        # Append to JSONL
        # Append to JSONL (flush immediately for real-time streaming)
        with open(self.actions_file, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

        # Save canvas snapshot
        if canvas_image is not None:
            snapshot_path = self.snapshots_dir / f"{self.action_count:04d}.png"
            canvas_image.save(snapshot_path)

    def save_target_image(self, target_image: Image.Image) -> None:
        """Save the target image for reference."""
        target_path = self.run_dir / "target.png"
        target_image.save(target_path)

    def save_input_image(self, input_image: Image.Image, step: int = 1) -> None:
        """Save an input image (e.g., initial canvas state for completion tasks)."""
        inputs_dir = self.run_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        input_path = inputs_dir / f"{step:04d}_input.png"
        input_image.save(input_path)

    def save_text_prompt(self, text_prompt_id: str) -> None:
        """Save the text prompt for reference."""
        text_prompt_path = self.run_dir / "target.txt"
        text_prompt_path.write_text(get_text_prompt(text_prompt_id))

    def save_final(self, canvas_image: Image.Image) -> None:
        """Save the final canvas image."""
        final_path = self.run_dir / "final.png"
        canvas_image.save(final_path)

    def save_run_config(
        self,
        task_config: dict,
        model: str,
        canvas_size: int,
        max_actions: int,
        show_grid: bool,
        save_snapshots: bool,
        compute_all_metrics: bool,
        log_reasoning: bool,
        enable_reset: bool,
        system_prompt: str,
        message_prompt_template: str,
        actions_per_turn: int = 1,
        focus_mode: bool = False,
        focus_window: int = 10,
        enable_notes: bool = False,
    ) -> None:
        """Save run configuration at the start of a benchmark run."""
        config = {
            **task_config,
            "model": model,
            "canvas_size": canvas_size,
            "max_actions": max_actions,
            "show_grid": show_grid,
            "save_snapshots": save_snapshots,
            "compute_all_metrics": compute_all_metrics,
            "log_reasoning": log_reasoning,
            "enable_reset": enable_reset,
            "system_prompt": system_prompt,
            "message_prompt_template": message_prompt_template,
            "actions_per_turn": actions_per_turn,
            "focus_mode": focus_mode,
            "focus_window": focus_window,
            "enable_notes": enable_notes,
            "timestamp": datetime.now().isoformat(),
        }

        config_path = self.run_dir / "run_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            f.flush()

    def save_result(
        self,
        metrics: dict,
        model: str,
        run_target: str,
        total_actions: int,
        stop_reason: str,
        canvas_size: int | None = None,
        actions_per_turn: int = 1,
    ) -> None:
        """Save final benchmark results."""
        result = {
            "model": model,
            "run_target": run_target,
            "total_actions": total_actions,
            "stop_reason": stop_reason,  # "stop_called", "budget_exhausted", "error"
            "metrics": metrics,
            "actions_per_turn": actions_per_turn,
            "timestamp": datetime.now().isoformat(),
        }
        if canvas_size is not None:
            result["canvas_size"] = canvas_size

        result_path = self.run_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()


def create_run_dir(
    base_dir: str | Path,
    model: str,
    benchmark_type: str,
    run_target: str,
    run_name: str | None = None,
) -> Path:
    """Create a unique run directory."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model}_{run_target}_{timestamp}"
        # Sanitize run name
        run_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_name)

    run_dir = base_dir / benchmark_type / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir
