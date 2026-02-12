"""
PaintBench Configuration

Constants for canvas size, colors, stroke parameters, and benchmark settings.
"""

from enum import Enum

from pydantic import BaseModel

# Canvas dimensions (default, can be overridden at runtime)
CANVAS_SIZE = 1024

# Action budget
MAX_ACTIONS = 200

# History window size (number of recent actions shown to VLM)
HISTORY_WINDOW = 20

# Notes tool limits
MAX_NOTES_TOKENS = 20000  # Maximum tokens allowed in notes
MAX_NOTES_CHARS = 80000  # Approximate character limit (4 chars per token)

# Stroke width (fixed)
DEFAULT_STROKE_WIDTH = 2  # Fixed stroke width in pixels

MAX_WIDTH = 10


class ToolsConfig(BaseModel):
    enable_reset: bool
    enable_notes: bool
    actions_per_turn: int


class RunConfig(BaseModel):
    model: str
    canvas_size: int
    log_reasoning: bool = False
    enable_reset: bool = False
    actions_per_turn: int = 1
    focus_mode: bool = False
    focus_window: int = 10
    enable_notes: bool = False


def make_tools(canvas_size: int, enable_reset: bool = True) -> list[dict]:
    """Generate tool definitions with correct coordinate bounds for the given canvas size.

    Args:
        canvas_size: Size of the canvas in pixels
        enable_reset: If False, the reset tool is not included
    """
    max_coord = canvas_size - 1
    tools = [
        {
            "type": "function",
            "name": "draw_line",
            "description": f"Draw a line from (x1, y1) to (x2, y2) with the specified color.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {
                        "type": "integer",
                        "description": f"X coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y1": {
                        "type": "integer",
                        "description": f"Y coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "x2": {
                        "type": "integer",
                        "description": f"X coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y2": {
                        "type": "integer",
                        "description": f"Y coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "r": {
                        "type": "integer",
                        "description": "Red channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "width": {
                        "type": "integer",
                        "description": f"Stroke width in pixels (1-{MAX_WIDTH})",
                        "minimum": 1,
                        "maximum": MAX_WIDTH,
                    },
                    "rounded": {
                        "type": "boolean",
                        "description": "If true, line has round caps. If false, line has flat/butt caps.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on (e.g. 'Drawing left leaf outline')",
                    },
                },
                "required": [
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "r",
                    "g",
                    "b",
                    "width",
                    "rounded",
                    "focus",
                ],
            },
        },
        {
            "type": "function",
            "name": "stop",
            "description": "Stop painting and finalize your drawing. Call this when the canvas reasonably resembles the target - perfection is not required. If the main shapes and colors are approximately correct, call stop.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing why you're stopping (e.g. 'Drawing complete, all elements rendered')",
                    },
                },
                "required": ["focus"],
            },
        },
        {
            "type": "function",
            "name": "undo",
            "description": "Remove the last stroke you drew.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
        {
            "type": "function",
            "name": "draw_spline",
            "description": "Draw a smooth interpolating curve through 2-8 points. Unlike Bézier curves, the spline passes directly through every point you specify. The curve will be smooth at each point, with curvature determined by neighboring points.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "description": f"X coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                                "y": {
                                    "type": "integer",
                                    "description": f"Y coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "r": {
                        "type": "integer",
                        "description": "Red channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "width": {
                        "type": "integer",
                        "description": f"Stroke width in pixels (1-{MAX_WIDTH})",
                        "minimum": 1,
                        "maximum": MAX_WIDTH,
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on (e.g. 'Drawing leaf contour')",
                    },
                },
                "required": ["points", "r", "g", "b", "width", "focus", "rounded"],
            },
        },
        {
            "type": "function",
            "name": "draw_polyline",
            "description": "Draw straight segments through a list of points in order. Use this for piecewise linear paths.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "description": f"X coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                                "y": {
                                    "type": "integer",
                                    "description": f"Y coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "r": {
                        "type": "integer",
                        "description": "Red channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "width": {
                        "type": "integer",
                        "description": f"Stroke width in pixels (1-{MAX_WIDTH})",
                        "minimum": 1,
                        "maximum": MAX_WIDTH,
                    },
                    "rounded": {
                        "type": "boolean",
                        "description": "If true, line caps are round. If false, caps are flat.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on (e.g. 'Sketching outline')",
                    },
                },
                "required": ["points", "r", "g", "b", "width", "rounded", "focus"],
            },
        },
        {
            "type": "function",
            "name": "draw_curve",
            "description": f"Draw a cubic Bézier curve from (x1, y1) to (x2, y2) with control points (cx1, cy1) and (cx2, cy2). The curve starts at (x1, y1), is pulled toward (cx1, cy1), then toward (cx2, cy2), and ends at (x2, y2).",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {
                        "type": "integer",
                        "description": f"X coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y1": {
                        "type": "integer",
                        "description": f"Y coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "x2": {
                        "type": "integer",
                        "description": f"X coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y2": {
                        "type": "integer",
                        "description": f"Y coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cx1": {
                        "type": "integer",
                        "description": f"X coordinate of first control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cy1": {
                        "type": "integer",
                        "description": f"Y coordinate of first control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cx2": {
                        "type": "integer",
                        "description": f"X coordinate of second control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cy2": {
                        "type": "integer",
                        "description": f"Y coordinate of second control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "r": {
                        "type": "integer",
                        "description": "Red channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "g": {
                        "type": "integer",
                        "description": "Green channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "b": {
                        "type": "integer",
                        "description": "Blue channel (0-255)",
                        "minimum": 0,
                        "maximum": 255,
                    },
                    "width": {
                        "type": "integer",
                        "description": f"Stroke width in pixels (1-{MAX_WIDTH})",
                        "minimum": 1,
                        "maximum": MAX_WIDTH,
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on",
                    },
                },
                "required": [
                    "x1",
                    "y1",
                    "cx1",
                    "cy1",
                    "cx2",
                    "cy2",
                    "x2",
                    "y2",
                    "r",
                    "g",
                    "b",
                    "focus",
                ],
            },
        },
    ]

    if enable_reset:
        tools.append(
            {
                "type": "function",
                "name": "reset",
                "description": "Reset the canvas to blank white. Use sparingly - only if you made a major mistake. Do NOT reset just to make minor adjustments.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
            }
        )

    return tools


def make_notes_tool() -> dict:
    """Generate the update_notes tool definition.

    This is an optional tool that allows the model to maintain an append-only
    notes log for tracking progress, plans, and observations.
    """
    return {
        "type": "function",
        "name": "update_notes",
        "description": "Add an entry to your notes log. Entries are timestamped and persist until the log fills up (oldest entries removed first). Does NOT count as a drawing action.",
        "strict": True,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "entry": {
                    "type": "string",
                    "description": "Note entry to add",
                },
            },
            "required": ["entry"],
        },
    }


def make_fixed_style_tools(
    canvas_size: int,
    color: tuple[int, int, int] = (0, 0, 0),
    width: int = 2,
    enable_reset: bool = True,
) -> tuple[list[dict], dict]:
    """Generate simplified tool definitions with fixed color and width."""
    max_coord = canvas_size - 1
    fixed_params = {
        "r": color[0],
        "g": color[1],
        "b": color[2],
        "width": width,
        "rounded": True,
    }

    tools = [
        {
            "type": "function",
            "name": "draw_line",
            "description": f"Draw a black line (width={width}px) from (x1, y1) to (x2, y2).",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {
                        "type": "integer",
                        "description": f"X coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y1": {
                        "type": "integer",
                        "description": f"Y coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "x2": {
                        "type": "integer",
                        "description": f"X coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y2": {
                        "type": "integer",
                        "description": f"Y coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on",
                    },
                },
                "required": ["x1", "y1", "x2", "y2", "focus"],
            },
        },
        {
            "type": "function",
            "name": "draw_spline",
            "description": f"Draw a black smooth spline (width={width}px) through 2-8 points in order.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "description": f"X coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                                "y": {
                                    "type": "integer",
                                    "description": f"Y coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on",
                    },
                },
                "required": ["points", "focus"],
            },
        },
        {
            "type": "function",
            "name": "draw_polyline",
            "description": f"Draw a black polyline (width={width}px) through 2-12 points in order.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "points": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 12,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "description": f"X coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                                "y": {
                                    "type": "integer",
                                    "description": f"Y coordinate (0-{max_coord})",
                                    "minimum": 0,
                                    "maximum": max_coord,
                                },
                            },
                            "required": ["x", "y"],
                        },
                    },
                    "rounded": {
                        "type": "boolean",
                        "description": "If true, line caps are round. If false, caps are flat.",
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on",
                    },
                },
                "required": ["points", "rounded", "focus"],
            },
        },
        {
            "type": "function",
            "name": "draw_curve",
            "description": "Draw a cubic Bézier curve (width={width}px) from (x1, y1) to (x2, y2) with control points (cx1, cy1) and (cx2, cy2). The curve starts at (x1, y1), is pulled toward (cx1, cy1), then toward (cx2, cy2), and ends at (x2, y2).",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "x1": {
                        "type": "integer",
                        "description": f"X coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y1": {
                        "type": "integer",
                        "description": f"Y coordinate of start point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cx1": {
                        "type": "integer",
                        "description": f"X coordinate of first control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cy1": {
                        "type": "integer",
                        "description": f"Y coordinate of first control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cx2": {
                        "type": "integer",
                        "description": f"X coordinate of second control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "cy2": {
                        "type": "integer",
                        "description": f"Y coordinate of second control point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "x2": {
                        "type": "integer",
                        "description": f"X coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "y2": {
                        "type": "integer",
                        "description": f"Y coordinate of end point (0-{max_coord})",
                        "minimum": 0,
                        "maximum": max_coord,
                    },
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing what you're working on",
                    },
                },
                "required": ["x1", "y1", "cx1", "cy1", "cx2", "cy2", "x2", "y2", "focus"],
            },
        },
        {
            "type": "function",
            "name": "stop",
            "description": "Stop painting and finalize your drawing.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "One sentence describing why you're stopping",
                    },
                },
                "required": ["focus"],
            },
        },
        {
            "type": "function",
            "name": "undo",
            "description": "Remove the last stroke you drew.",
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            },
        },
    ]

    if enable_reset:
        tools.append(
            {
                "type": "function",
                "name": "reset",
                "description": "Reset the canvas to blank white.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                    "required": [],
                },
            }
        )

    return tools, fixed_params


# Default tool definitions (for backwards compatibility)
TOOLS = make_tools(CANVAS_SIZE)


class MODEL_TYPE(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    XAI = "xai"


MODEL_MAPPING = {
    "gpt-5.2": MODEL_TYPE.OPENAI,
    "gpt-5.2-2025-12-11": MODEL_TYPE.OPENAI,
    "gpt-5.1-2025-06-19": MODEL_TYPE.OPENAI,
    "gpt-5": MODEL_TYPE.OPENAI,
    "o3-2025-04-16": MODEL_TYPE.OPENAI,
    "gemini-3-pro-preview": MODEL_TYPE.GEMINI,
    "gemini-3-flash-preview": MODEL_TYPE.GEMINI,
    "claude-opus-4-6": MODEL_TYPE.ANTHROPIC,
    "claude-opus-4-5": MODEL_TYPE.ANTHROPIC,
    "claude-haiku-4-5": MODEL_TYPE.ANTHROPIC,
    "claude-sonnet-4-5": MODEL_TYPE.ANTHROPIC,
    "grok-4-1-fast-reasoning": MODEL_TYPE.XAI,
    "grok-4-fast-reasoning": MODEL_TYPE.XAI,
}

OPENROUTER_MODEL_MAPPING: dict[str, str] = {
    "gpt-5.2": "openai/gpt-5.2",
    "gpt-5.1": "openai/gpt-5.1",
    "gpt-5": "openai/gpt-5",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemini-3-pro-preview": "google/gemini-3-pro-preview",
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    "claude-opus-4-5": "anthropic/claude-opus-4.5",
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
    "grok-4-1-fast": "x-ai/grok-4.1-fast",
    "grok-4": "x-ai/grok-4",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "kimi-k2.5": "moonshotai/kimi-k2.5",
    "qwen3-vl-235b-a22b-thinking": "qwen/qwen3-vl-235b-a22b-thinking",
    "glm-4.6v": "z-ai/glm-4.6v",
}


def get_canvas_doc(canvas_size: int) -> str:
    """Generate canvas documentation section for system prompt."""
    max_coord = canvas_size - 1
    return f"""<canvas>
The canvas is {canvas_size}x{canvas_size} pixels. Coordinates range from 0 to {max_coord}.
- (0, 0) is the top-left corner
- ({max_coord}, 0) is the top-right corner  
- (0, {max_coord}) is the bottom-left corner
- ({max_coord}, {max_coord}) is the bottom-right corner
- ({canvas_size // 2}, {canvas_size // 2}) is the center of the canvas

Colors: RGB values (0-255 for each channel: r, g, b)
Stroke width: Fixed at {DEFAULT_STROKE_WIDTH} pixels

Common colors for reference:
- Black: r=0, g=0, b=0
- White: r=255, g=255, b=255
- Red: r=220, g=50, b=50
- Blue: r=50, g=100, b=200
- Green: r=50, g=180, b=80
- Yellow: r=240, g=200, b=50
</canvas>"""


def _get_multi_move_prompt_section(actions_per_turn: int) -> str:
    """Return prompt section explaining multi-move rules. Override for task-specific wording."""
    return f"""<multi_move_mode>
You can perform up to {actions_per_turn} drawing actions per turn before seeing the updated canvas.
- Plan your strokes carefully since you won't see intermediate results within a turn
- After your turn completes, you'll see the canvas with all your actions applied
- Use undo() to revert ALL actions from your previous turn (not just one stroke)
- stop() ends the benchmark immediately
- You may output fewer than {actions_per_turn} actions if you prefer
</multi_move_mode>"""


reset_doc = """<reset>
    Reset the canvas to blank white. Use sparingly - only if you made a major mistake. Do NOT reset just to make minor adjustments.
</reset>"""


notes_doc = """<notes_tool>
You have access to an update_notes tool that is an append only log that you can use to document your progress, plan, and observations.
Notes have a maximum length of {MAX_NOTES_TOKENS} tokens or {MAX_NOTES_CHARS} characters.
- Notes are automatically timestamped with turn number
- You can only ADD notes, not edit or delete previous ones
- Oldest notes are automatically removed when the log fills up
- Does NOT count as a drawing action

Your notes log is shown to you each turn in chronological order, like:
[Turn 1] <turn 1 notes>
[Turn 2] <turn 2 notes>
[Turn 3] <turn 3 notes>
</notes_tool>"""


penalties_doc = """<penalties>
Drawing with white or near-white colors (all RGB channels >= 250) is NOT allowed. The background is white, so this would act as an eraser which is prohibited.
- White stroke attempts cost 3 actions instead of 1 as a penalty
- The stroke will NOT be drawn
- Use the undo tool if you need to remove a previous stroke
</penalties>"""


stroke_permanence_doc = """<stroke_permanence>
Every stroke you draw is essentially permanent:
- The undo tool only removes your most recent stroke(s)
- You CANNOT erase or white-out previous strokes (white drawing is prohibited)
- Drawing over mistakes adds visual clutter — it does not fix them
- Plan carefully before drawing; imprecise early strokes compound into a messy result
- If your drawing becomes cluttered, use reset() to start fresh rather than layering more strokes
</stroke_permanence>"""


def get_grid_overlay_doc(canvas_size: int) -> str:
    return f"""<grid_overlay>
A gray grid can be overlaid on the images as a positioning aid. This grid divides the canvas into cells to help you estimate coordinates. 
The grid is NOT part of the target image - do NOT draw the grid lines. The grid is only there to help you position your strokes accurately.
Each cell is {canvas_size // 32}x{canvas_size // 32} pixels.
</grid_overlay>"""


def make_tools_docs(tools_config: ToolsConfig) -> str:
    multi_move_doc = (
        "\n\n" + _get_multi_move_prompt_section(tools_config.actions_per_turn)
        if tools_config.actions_per_turn > 1
        else ""
    )

    return f"""<tools>
<draw_line>
    Draw a line from (x1, y1) to (x2, y2) with the specified RGB color. Stroke width is fixed at {DEFAULT_STROKE_WIDTH} pixels. If rounded is true, the line has round caps. If rounded is false, the line has flat/butt caps.
</draw_line>
<draw_spline>
    Draw a smooth spline that passes through a list of points in order. Provide 4-8 points to define the path. Use this for smooth curves without specifying Bézier control points.
</draw_spline>
<draw_polyline>
    Draw straight segments through a list of points in order. Use this for piecewise linear approximations of a curve.
</draw_polyline>
<draw_curve>
    Draw a cubic Bézier curve from (x1, y1) to (x2, y2) with control points (cx1, cy1) and (cx2, cy2). The curve starts at (x1, y1), is pulled toward (cx1, cy1), then toward (cx2, cy2), and ends at (x2, y2). Stroke width is fixed at {DEFAULT_STROKE_WIDTH} pixels. Use this for smooth curved lines like arcs, waves, or organic shapes.
</draw_curve>
<stop>
    Stop painting and finalize your drawing. Call this when the canvas reasonably resembles the target - perfection is not required. If the main shapes and colors are approximately correct, call stop.
</stop>
<undo>
    Remove the last stroke you drew.
</undo>
{reset_doc if tools_config.enable_reset else ""}
{notes_doc if tools_config.enable_notes else ""}
{multi_move_doc}
</tools>"""
