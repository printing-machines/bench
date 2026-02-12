"""
BenchmarkTask abstraction for different benchmark types.

Each task type owns its prompts, content building, evaluation, and target saving.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image

from .benchmark_logger import BenchmarkLogger
from .config import (
    RunConfig,
    ToolsConfig,
    get_canvas_doc,
    get_grid_overlay_doc,
    make_tools_docs,
    penalties_doc,
    stroke_permanence_doc,
)


class BenchmarkTask(ABC):
    """Abstract base class for benchmark tasks."""

    task_type: str
    run_target: str

    @abstractmethod
    def to_dict(self) -> dict:
        """Return task configuration as a dict for logging."""
        ...

    @abstractmethod
    def get_system_prompt(self, run_config: RunConfig) -> str:
        """Generate the system prompt for this task."""
        ...

    @abstractmethod
    def get_message_template(self) -> str:
        """Get the user message template for this task."""
        ...

    @abstractmethod
    def get_target_image(self) -> Image.Image | None:
        """Get the target image if applicable, None otherwise."""
        ...

    @abstractmethod
    def build_vlm_user_content(
        self,
        target_b64: str | None,
        canvas_b64: str,
        history_text: str,
        remaining: int,
        current: int,
        max_actions: int,
    ) -> list[dict]:
        """Build the user content for VLM API calls."""
        ...

    @abstractmethod
    def evaluate(self, final_image: Image.Image, compute_all: bool = True) -> dict:
        """Evaluate the final image against the task target."""
        ...

    @abstractmethod
    def save_target(self, logger: BenchmarkLogger) -> None:
        """Save the task target to the logger."""
        ...

    def get_initial_canvas_image(self) -> Image.Image | None:
        """Get the initial canvas image if task requires pre-loaded canvas, None for blank."""
        return None


class TargetImageTask(BenchmarkTask):
    """Task: Recreate a target image by drawing on canvas."""

    task_type = "target_image"

    def __init__(self, target_path: Path | str, canvas_size: int):
        self.target_path = Path(target_path)
        self.canvas_size = canvas_size
        self.run_target = self.target_path.stem

        self._target_image = _load_rgb_image(self.target_path, canvas_size)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "run_target": self.run_target,
            "target_path": str(self.target_path),
        }

    def get_target_image(self) -> Image.Image:
        return self._target_image

    def get_system_prompt(self, run_config: RunConfig) -> str:
        tool_docs = make_tools_docs(
            ToolsConfig(
                enable_reset=run_config.enable_reset,
                enable_notes=run_config.enable_notes,
                actions_per_turn=run_config.actions_per_turn,
            )
        )

        return f"""You are the best painter in the world. Your goal is to recreate the target image by drawing lines on a canvas. To accomplish this, you will be given the target image and your current canvas.

<ground_rules>
- Your goal is to recreate the target image as closely as possible.
- Your drawing does NOT need to be pixel perfect or exact, but your final drawing should capture all the details of the target image. For exmaple, if the target image is a chair with four legs, a seat, a backrest, and wheels, your final drawing should have all four legs, a seat, a backrest, and wheels.
- You do NOT need to match the target colors exactly in your final drawing.
- One way to draw complex strokes is to break them down into a series of smaller, easier strokes.
- Before emitting your next action, you MUST consider the state of the current canvas and your previous actions.
- Avoid repeating work.
</ground_rules>

<when_to_stop>
Call stop() when:
- The main shapes are drawn and in approximately the right positions. Ensure that all the details of the target image are captured.
- The colors are reasonably close (exact color matching is NOT required)
- The overall composition matches the target

Do NOT keep adjusting if:
- Small positional differences exist (a few pixels off is fine)
- Colors are slightly different shades (orange vs dark-orange is acceptable)
- Edges are not perfectly aligned

Minor imperfections are acceptable - perfection is NOT the goal. If you find yourself repeatedly redrawing the same shapes, STOP and accept the current result.
</when_to_stop>

{penalties_doc}

{stroke_permanence_doc}

{get_grid_overlay_doc(run_config.canvas_size)}

{get_canvas_doc(run_config.canvas_size)}

{tool_docs}"""

    def get_message_template(self) -> str:
        return """Target image:
[IMAGE: target]
Current canvas:
[IMAGE: current_canvas]
Recent actions:
{history}

Actions remaining: {remaining} (action {current} of {max_actions}).

Remember:
- Avoid repeating work.
- Before emitting your next action, you MUST consider the state of the current canvas and your previous actions.

What's your next action?"""

    def build_vlm_user_content(
        self,
        target_b64: str | None,
        canvas_b64: str,
        history_text: str,
        remaining: int,
        current: int,
        max_actions: int,
    ) -> list[dict]:
        return [
            {"type": "text", "text": "Target image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{target_b64}"},
            },
            {"type": "text", "text": "Current canvas:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{canvas_b64}"},
            },
            {
                "type": "text",
                "text": self.get_message_template().format(
                    history=history_text,
                    remaining=remaining,
                    current=current,
                    max_actions=max_actions,
                ),
            },
        ]

    def evaluate(self, final_image: Image.Image, compute_all: bool = True) -> dict:
        return _evaluate_image_metrics(final_image, self._target_image, compute_all)

    def save_target(self, logger: BenchmarkLogger) -> None:
        logger.save_target_image(self._target_image)


class TextPromptTask(BenchmarkTask):
    """Task: Draw an image based on a text description."""

    task_type = "text_prompt"

    def __init__(self, text_prompt_id: str):
        from .text_prompts import get_text_prompt

        self.text_prompt_id = text_prompt_id
        self.text_prompt = get_text_prompt(text_prompt_id)
        if not self.text_prompt:
            raise ValueError(f"Invalid text prompt ID: {text_prompt_id}")
        self.run_target = text_prompt_id

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "run_target": self.run_target,
            "text_prompt_id": self.text_prompt_id,
            "text_prompt": self.text_prompt,
        }

    def get_target_image(self) -> None:
        return None

    def get_system_prompt(self, run_config: RunConfig) -> str:
        tool_docs = make_tools_docs(
            ToolsConfig(
                enable_reset=run_config.enable_reset,
                enable_notes=run_config.enable_notes,
                actions_per_turn=run_config.actions_per_turn,
            )
        )

        return f"""You are a painting assistant. Your goal is to draw an image based on this text prompt: {self.text_prompt}.

<ground_rules>
- Your goal is to draw an image that is as close as possible to the text prompt.
- Your drawing does NOT need to be pixel perfect or exact.
- It is up to you to decide how to interpret the text prompt and draw an image that is as close as possible to the text prompt.
- It is OK to break down complex strokes into a series of smaller, easier strokes.
</ground_rules>

<when_to_stop>
Call stop() when:
- The image is drawn and in approximately the right positions
- The overall composition matches the text prompt

Do NOT keep adjusting if:
- Small positional differences exist (a few pixels off is fine)

AVOID resetting the canvas if your drawing already resembles the text prompt. Minor imperfections are acceptable - perfection is NOT the goal. If you find yourself repeatedly redrawing the same shapes, STOP and accept the current result.
</when_to_stop>

{penalties_doc}

{stroke_permanence_doc}

{get_grid_overlay_doc(run_config.canvas_size)}

{get_canvas_doc(run_config.canvas_size)}

{tool_docs}"""

    def get_message_template(self) -> str:
        return """Current canvas:
[IMAGE: current_canvas]
Recent actions:
{history}

Actions remaining: {remaining} (action {current} of {max_actions}).
What's your next action?"""

    def build_vlm_user_content(
        self,
        target_b64: str | None,
        canvas_b64: str,
        history_text: str,
        remaining: int,
        current: int,
        max_actions: int,
    ) -> list[dict]:
        message_text = (
            f"{_format_history_and_remaining(history_text, remaining, current, max_actions)}\n"
            "What's your next action?"
        )
        return _build_canvas_user_content(canvas_b64, message_text)

    def evaluate(self, final_image: Image.Image, compute_all: bool = True) -> dict:
        print("Skipping evaluation for text prompt benchmark, not implemented yet")
        return {}

    def save_target(self, logger: BenchmarkLogger) -> None:
        logger.save_text_prompt(self.text_prompt_id)


def create_task(
    task_type: str,
    target_path: Path | str | None = None,
    text_prompt_id: str | None = None,
    canvas_size: int = 1024,
) -> BenchmarkTask:
    """Factory function to create the appropriate task."""
    if task_type == "target_image":
        if not target_path:
            raise ValueError("target_path required for target_image task")
        return TargetImageTask(target_path, canvas_size)
    elif task_type == "text_prompt":
        if not text_prompt_id:
            raise ValueError("text_prompt_id required for text_prompt task")
        return TextPromptTask(text_prompt_id)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# Shared prompt helpers


def _load_rgb_image(path: Path, canvas_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if image.size != (canvas_size, canvas_size):
        image = image.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    return image


def _build_canvas_user_content(
    canvas_b64: str, message_text: str, canvas_label: str = "Current canvas:"
) -> list[dict]:
    return [
        {"type": "text", "text": canvas_label},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{canvas_b64}"},
        },
        {"type": "text", "text": message_text},
    ]


def _format_history_and_remaining(
    history_text: str, remaining: int, current: int, max_actions: int
) -> str:
    return (
        f"Recent actions:\n{history_text}\n\n"
        f"Actions remaining: {remaining} (action {current} of {max_actions})."
    )


def _evaluate_image_metrics(
    final_image: Image.Image, target_image: Image.Image, compute_all: bool
) -> dict:
    from .eval import (
        compute_clip_similarity,
        compute_lpips,
        compute_mse,
        compute_ssim,
    )

    metrics: dict[str, float | None] = {
        "mse": compute_mse(final_image, target_image),
        "ssim": compute_ssim(final_image, target_image),
    }

    if compute_all:
        try:
            metrics["clip_similarity"] = compute_clip_similarity(final_image, target_image)
        except Exception as e:
            print(f"Warning: Could not compute CLIP similarity: {e}")
            metrics["clip_similarity"] = None

        try:
            metrics["lpips"] = compute_lpips(final_image, target_image)
        except Exception as e:
            print(f"Warning: Could not compute LPIPS: {e}")
            metrics["lpips"] = None

    return metrics
