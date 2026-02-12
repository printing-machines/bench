"""
VLM Interface for PaintBench

Handles image encoding, prompt formatting, and API calls to vision-language models.

Environment variables:
    OPENROUTER_API_KEY: API key for OpenRouter
    OPENROUTER_BASE_URL: Optional custom base URL for OpenRouter (defaults to https://openrouter.ai/api/v1)
"""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import dotenv_values, load_dotenv
from google.genai import types
from PIL import Image

from ..config import (
    CANVAS_SIZE as DEFAULT_CANVAS_SIZE,
)
from ..config import (
    DEFAULT_STROKE_WIDTH,
    HISTORY_WINDOW,
    MAX_ACTIONS,
    RunConfig,
)
from ..config import (
    TOOLS as DEFAULT_TOOLS,
)

if TYPE_CHECKING:
    from ..benchmark_task import BenchmarkTask


@dataclass
class Action:
    """Represents a parsed action with reasoning from VLM response."""

    type: str  # "draw_line", "draw_spline", "draw_polyline", "stop", "undo", "reset", "update_notes"
    params: dict  # Parameters for the action
    reasoning: str | None = None  # Model's reasoning/thinking
    focus_summary: str | None = None  # Short description of current focus
    notes_content: str | None = None  # For update_notes action (the entry to append)

    def __str__(self) -> str:
        if self.type == "draw_line":
            p = self.params
            rounded = p.get("rounded", True)
            return f"draw_line({p['x1']}, {p['y1']}, {p['x2']}, {p['y2']}, r={p['r']}, g={p['g']}, b={p['b']}, width={p['width']}, rounded={rounded})"
        elif self.type == "draw_spline":
            p = self.params
            points = p.get("points", [])
            points_str = ", ".join(f"({pt['x']},{pt['y']})" for pt in points)
            return f"draw_spline([{points_str}], r={p['r']}, g={p['g']}, b={p['b']}, width={p['width']})"
        elif self.type == "draw_polyline":
            p = self.params
            points = p.get("points", [])
            points_str = ", ".join(f"({pt['x']},{pt['y']})" for pt in points)
            rounded = p.get("rounded", True)
            return f"draw_polyline([{points_str}], r={p['r']}, g={p['g']}, b={p['b']}, width={p['width']}, rounded={rounded})"
        elif self.type == "draw_curve":
            p = self.params
            return (
                "draw_curve("
                f"{p['x1']}, {p['y1']}, {p['cx1']}, {p['cy1']}, "
                f"{p['cx2']}, {p['cy2']}, {p['x2']}, {p['y2']}, "
                f"r={p['r']}, g={p['g']}, b={p['b']}, width={p['width']})"
            )
        elif self.type == "notes_rejected":
            return f"[SYSTEM: {self.params.get('message', 'Notes update rejected')}]"
        return f"{self.type}()"


# Minimum size for images sent to VLMs (helps with spatial reasoning accuracy)
MIN_IMAGE_SIZE = 512


def encode_image_base64(image: Image.Image, min_size: int = MIN_IMAGE_SIZE) -> str:
    """Encode PIL Image to base64 string, upscaling small images for better VLM interpretation."""
    if image.width < min_size or image.height < min_size:
        scale = min_size / min(image.width, image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class VLMClient(ABC):
    """
    Client for calling VLM APIs.
    """

    def __init__(
        self,
        model: str,
        task: BenchmarkTask,
        tools: list[dict] | None = None,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        log_reasoning: bool = False,
        enable_reset: bool = True,
        enable_notes: bool = False,
    ):
        self.model = model
        self.task = task
        self.tools = tools if tools is not None else DEFAULT_TOOLS
        self.canvas_size = canvas_size
        self.log_reasoning = log_reasoning
        self.enable_reset = enable_reset
        self.enable_notes = enable_notes
        self._system_prompt = task.get_system_prompt(
            RunConfig(
                model=model,
                canvas_size=canvas_size,
                log_reasoning=log_reasoning,
                enable_reset=enable_reset,
                enable_notes=enable_notes,
            )
        )

        # Load .env from project root
        DOTENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
        load_dotenv(DOTENV_PATH)
        self.env_vars = dotenv_values(DOTENV_PATH)

    @abstractmethod
    def get_action(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        show_grid: bool = False,
        max_actions: int = 1,
    ) -> list[Action]:
        """Get next action(s) from VLM.

        Args:
            max_actions: Maximum number of actions to request (for multi-move mode).
                         When > 1, enables parallel tool calls.

        Returns:
            List of actions (1 to max_actions). Empty list on failure.
        """
        raise NotImplementedError("Subclasses must implement this method")

    async def get_action_async(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        show_grid: bool = False,
        max_actions: int = 1,
    ) -> list[Action]:
        """Async version of get_action. Override in subclasses for true async.

        Default implementation wraps sync method for backwards compatibility.
        """
        return self.get_action(
            target_image, canvas_image, history, action_count, show_grid, max_actions
        )

    @abstractmethod
    def parse_response(
        self,
        response: Any,
        verbose: bool = False,
    ) -> Action | None:
        """Parse response from VLM API."""
        raise NotImplementedError("Subclasses must implement this method")

    def convert_tools(self, tools: list[dict]) -> list[types.Tool]:
        """Optional: override to adapt tool schema per provider."""
        return [types.Tool(**tool) for tool in tools]

    def build_messages(
        self,
        target_image: Image.Image | None,
        canvas_image: Image.Image,
        history: list[Action],
        action_count: int,
        model: str,
        include_reasoning_summaries: bool = False,
        show_grid: bool = False,
    ) -> list[dict]:
        """Build message list for VLM API call."""
        target_b64 = encode_image_base64(target_image) if target_image else None
        canvas_b64 = encode_image_base64(canvas_image)

        history_text = self.format_history(history, include_reasoning=include_reasoning_summaries)

        remaining_actions = MAX_ACTIONS - action_count

        user_content = self.task.build_vlm_user_content(
            target_b64=target_b64,
            canvas_b64=canvas_b64,
            history_text=history_text,
            remaining=remaining_actions,
            current=action_count + 1,
            max_actions=MAX_ACTIONS,
        )

        # Convert to OpenAI Responses API format
        formatted_content = []
        for item in user_content:
            if item["type"] == "text":
                formatted_content.append({"type": "input_text", "text": item["text"]})
            elif item["type"] == "image_url":
                formatted_content.append(
                    {
                        "type": "input_image",
                        "detail": "high",
                        "image_url": item["image_url"]["url"],
                    }
                )

        return [
            {"type": "message", "role": "system", "content": self._system_prompt},
            {"type": "message", "role": "user", "content": formatted_content},
        ]

    def format_history(self, history: list[Action], include_reasoning: bool = False) -> str:
        """Format recent action history as text."""
        if not history:
            return "No actions taken yet."

        recent = history[-HISTORY_WINDOW:]
        start_idx = len(history) - len(recent) + 1

        lines = []
        for i, action in enumerate(recent):
            lines.append(f"Action {start_idx + i}: {action}")
            if include_reasoning and action.reasoning:
                lines.append(f"  Reasoning summary: {action.reasoning}")

        return "\n".join(lines)

    def format_history_with_focus(self, history: list[Action], window: int = 10) -> str:
        """Format recent actions with focus summaries only (no reasoning).

        This provides compact context by showing only the action and its
        one-sentence focus description, avoiding verbose reasoning traces.
        """
        if not history:
            return "No actions taken yet."

        recent = history[-window:]
        start_idx = len(history) - len(recent) + 1

        lines = ["Recent actions:"]
        for i, action in enumerate(recent):
            focus = action.focus_summary or "No focus provided"
            lines.append(f'- Action {start_idx + i}: {action} | Focus: "{focus}"')

        return "\n".join(lines)

    def create_action(
        self,
        name: str,
        args: dict,
        reasoning: str | None,
        focus: str | None = None,
    ) -> Action | None:
        # Extract focus from args if present (for focus mode)
        focus_summary = focus or args.pop("focus", None)

        if name == "draw_line":
            return Action(
                type="draw_line",
                params={
                    "x1": int(args.get("x1", 0)),
                    "y1": int(args.get("y1", 0)),
                    "x2": int(args.get("x2", 0)),
                    "y2": int(args.get("y2", 0)),
                    "r": int(args.get("r", 0)),
                    "g": int(args.get("g", 0)),
                    "b": int(args.get("b", 0)),
                    "width": DEFAULT_STROKE_WIDTH,
                    "rounded": bool(args.get("rounded", True)),
                },
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "stop":
            return Action(type="stop", params={}, reasoning=reasoning, focus_summary=focus_summary)
        elif name == "undo":
            return Action(type="undo", params={}, reasoning=reasoning, focus_summary=focus_summary)
        elif name == "reset":
            return Action(
                type="reset",
                params={},
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "undo_stroke":
            return Action(
                type="undo_stroke",
                params={},
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "draw_spline":
            raw_points = args.get("points", []) or []
            points = [
                {"x": int(point.get("x", 0)), "y": int(point.get("y", 0))}
                for point in raw_points
                if isinstance(point, dict)
            ]
            return Action(
                type="draw_spline",
                params={
                    "points": points,
                    "r": int(args.get("r", 0)),
                    "g": int(args.get("g", 0)),
                    "b": int(args.get("b", 0)),
                    "width": DEFAULT_STROKE_WIDTH,
                },
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "draw_curve":
            return Action(
                type="draw_curve",
                params={
                    "x1": int(args.get("x1", 0)),
                    "y1": int(args.get("y1", 0)),
                    "cx1": int(args.get("cx1", 0)),
                    "cy1": int(args.get("cy1", 0)),
                    "cx2": int(args.get("cx2", 0)),
                    "cy2": int(args.get("cy2", 0)),
                    "x2": int(args.get("x2", 0)),
                    "y2": int(args.get("y2", 0)),
                    "r": int(args.get("r", 0)),
                    "g": int(args.get("g", 0)),
                    "b": int(args.get("b", 0)),
                    "width": int(args.get("width", DEFAULT_STROKE_WIDTH)),
                },
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "draw_polyline":
            raw_points = args.get("points", []) or []
            points = [
                {"x": int(point.get("x", 0)), "y": int(point.get("y", 0))}
                for point in raw_points
                if isinstance(point, dict)
            ]
            return Action(
                type="draw_polyline",
                params={
                    "points": points,
                    "r": int(args.get("r", 0)),
                    "g": int(args.get("g", 0)),
                    "b": int(args.get("b", 0)),
                    "width": int(args.get("width", 3)),
                    "rounded": bool(args.get("rounded", True)),
                },
                reasoning=reasoning,
                focus_summary=focus_summary,
            )
        elif name == "update_notes":
            entry = args.get("entry", "")
            return Action(
                type="update_notes",
                params={},
                reasoning=reasoning,
                focus_summary=focus_summary,
                notes_content=entry,
            )

        return None

    def dotenv_get(self, key: str) -> str | None:
        val = self.env_vars.get(key)
        if val is None:
            return None
        val = str(val).strip()
        return val or None

    def dotenv_get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.dotenv_get(key)
        if raw is None:
            return default
        return raw in {"1", "true", "True", "yes", "YES", "on", "ON"}
