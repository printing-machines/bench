# PaintBench

PaintBench is a benchmark for evaluating vision-language models (VLMs) on procedural painting tasks. Models are given access to drawing tools and must recreate target images or follow text prompts by making sequential drawing actions.

## Overview

PaintBench evaluates how well VLMs can:
- Understand visual targets and translate them into drawing actions
- Plan and execute multi-step drawing sequences
- Use spatial reasoning to place strokes accurately
- Manage drawing tools (lines, curves, splines) effectively

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Configuration

### Canvas Settings

- **Canvas Size**: Default `1024x1024` pixels (configurable via `canvas_size`)
- **Coordinate System**: (0, 0) is top-left, coordinates range from `0` to `canvas_size - 1`
- **Stroke Width**: Variable (1-10 pixels)

### Action Budget

- **Max Actions**: Default `200` actions per benchmark run
- **History Window**: Last `20` actions shown to the model
- **Actions Per Turn**: Configurable (default `1` for single-action mode, `>1` for multi-move mode)

### Notes Tool (Optional)

- **Max Tokens**: `20,000` tokens
- **Max Characters**: `~80,000` characters
- **Consecutive Limit**: Default `3` consecutive note updates before requiring a drawing action

## Available Tools

### Drawing Tools

1. **`draw_line`**: Draw a straight line from (x1, y1) to (x2, y2)
   - Variable stroke width (1-10 pixels)
   - RGB color (0-255 per channel)
   - Optional rounded/flat caps
   - Requires `focus` string describing current work

2. **`draw_spline`**: Draw a smooth interpolating curve through 2-8 points
   - Uses Catmull-Rom spline interpolation
   - Variable stroke width (1-10 pixels)
   - Passes directly through all specified points
   - Smooth curvature between points

3. **`draw_polyline`**: Draw straight segments through 2-12 points
   - Piecewise linear path
   - Variable stroke width (1-10 pixels)
   - Optional rounded/flat caps

4. **`draw_curve`**: Draw a cubic Bézier curve
   - Fixed stroke width: `3` pixels
   - Control points: (cx1, cy1) and (cx2, cy2)
   - Start point: (x1, y1), End point: (x2, y2)

### Control Tools

5. **`stop`**: Finalize the drawing and end the benchmark
   - Call when canvas reasonably resembles the target
   - Requires `focus` string explaining why stopping

6. **`undo`**: Remove the last stroke
   - In single-action mode: removes last stroke
   - In multi-move mode: removes entire previous turn

7. **`reset`** (optional): Reset canvas to blank white
   - Use sparingly - only for major mistakes
   - Can be disabled via `--no-reset` flag

8. **`update_notes`** (optional): Append to persistent notes log
   - Timestamped entries
   - Append-only (oldest removed when limit reached)
   - Does NOT count as a drawing action
   - Enabled via `--enable-notes` flag

## Task Types

### 1. Target Image (`target_image`)

Recreate a target image by drawing on a blank canvas.

```bash
paintbench \
  --benchmark-type target_image \
  --model gpt-5.2 \
  --target tasks/images/image_0.png
```

### 2. Text Prompt (`text_prompt`)

Follow a text description to create a drawing.

```bash
paintbench \
  --benchmark-type text_prompt \
  --model gpt-5.2 \
  --text-prompt-id text_prompt_1
```

Available prompts are defined in `tasks/text/text_prompts.py`.

## Running Benchmarks

### Single Run

```bash
paintbench \
  --benchmark-type target_image \
  --model gpt-5.2 \
  --target tasks/images/image_0.png \
  --output-dir benchmark/runs \
  --max-actions 200 \
  --canvas-size 1024
```

### Command-Line Options

**Required:**
- `--benchmark-type`: Task type (`target_image`, `text_prompt`)
- `--model`: Model name (e.g., `gpt-5.2`, `gemini-3-pro-preview`)

**Task-Specific:**
- `--target`: Path to target image (for `target_image`)
- `--text-prompt-id`: Prompt ID (for `text_prompt`)

**Configuration:**
- `--output-dir`: Output directory (default: `benchmark/runs`)
- `--max-actions`: Maximum actions allowed (default: `200`)
- `--canvas-size`: Canvas size in pixels (default: `1024`)
- `--grid`: Overlay 32x32 grid on images for spatial calibration
- `--no-snapshots`: Disable saving canvas snapshots after each action
- `--basic-metrics`: Only compute MSE/SSIM (skip CLIP/LPIPS)
- `--quiet`: Suppress progress output
- `--log-reasoning`: Log model reasoning traces to `actions.jsonl`
- `--no-reset`: Disable the reset tool
- `--run-name`: Explicit run directory name

**Multi-Move Mode:**
- `--actions-per-turn`: Actions per turn (default: `1` = single-action mode)
- `--focus-mode`: Use compact history with focus summaries instead of reasoning traces
- `--focus-window`: Number of recent action:focus pairs in history (default: `10`)

**Notes Tool:**
- `--enable-notes`: Enable the `update_notes` tool for persistent scratchpad

**Experimental:**
- `--fixed-style`: Use simplified tools with fixed color (black) and width (2px)

### Batch Mode

Run multiple benchmarks concurrently:

```bash
paintbench \
  --batch batch_config.json \
  --max-concurrent 5
```

Batch config format (`batch_config.json`):
```json
[
  {
    "task_type": "target_image",
    "model": "gpt-5.2",
    "target": "tasks/images/image_0.png",
    "canvas_size": 1024,
    "max_actions": 200
  },
  {
    "task_type": "text_prompt",
    "model": "gemini-3-pro-preview",
    "text_prompt_id": "text_prompt_1"
  }
]
```

## Supported Models

Models are mapped via OpenRouter. Supported models include:

- **OpenAI**: `gpt-5.2`, `gpt-5.1`, `gpt-5`, `o3-2025-04-16`
- **Google**: `gemini-3-pro-preview`, `gemini-3-flash-preview`
- **Anthropic**: `claude-opus-4-6`, `claude-opus-4-5`, `claude-haiku-4-5`, `claude-sonnet-4-5`
- **xAI**: `grok-4-1-fast-reasoning`, `grok-4-fast-reasoning`
- **Others**: `kimi-k2.5`, `qwen3-vl-235b-a22b-thinking`, `glm-4.6v`

Set `OPENROUTER_API_KEY` environment variable for API access.

## Output Structure

Each run creates a directory with:

```
run_dir/
├── target.png              # Target image
├── target_grid.png         # Grid-overlaid target (if --grid enabled)
├── final.png               # Final canvas state
├── config.json             # Run configuration
├── result.json             # Metrics and results
├── actions.jsonl           # Action log (one per line)
└── snapshots/              # Canvas state after each action (if enabled)
    ├── action_000.png
    ├── action_001.png
    └── ...
```

## Metrics

PaintBench computes multiple evaluation metrics:

- **MSE** (Mean Squared Error): Pixel-level difference
- **SSIM** (Structural Similarity Index): Perceptual similarity
- **CLIP Similarity**: Semantic similarity via CLIP embeddings
- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual distance

Use `--basic-metrics` to skip expensive CLIP/LPIPS computations.

## Constraints and Penalties

### White Drawing Prohibition

Drawing with white or near-white colors (RGB ≥ 250) is **not allowed**:
- Acts as an eraser (background is white)
- Penalty: `3` actions deducted
- Stroke is **not drawn**

### Stroke Permanence

- Strokes are essentially permanent
- `undo` only removes the most recent stroke(s)
- Drawing over mistakes adds visual clutter
- Use `reset()` for major mistakes, not minor adjustments

### Multi-Move Mode

When `actions_per_turn > 1`:
- Model can perform multiple actions before seeing results
- `undo()` reverts the **entire previous turn**
- Plan carefully - no intermediate feedback within a turn

## Focus Mode

When `--focus-mode` is enabled:
- History shows compact `action:focus` pairs instead of full reasoning traces
- Only last `--focus-window` actions shown (default: 10)
- Reduces token usage while preserving planning context

## Notes Tool

When `--enable-notes` is enabled:
- Model can maintain an append-only notes log
- Useful for planning and tracking progress
- Notes are shown to the model each turn
- Maximum `20,000` tokens or `~80,000` characters
- Oldest entries removed when limit reached
- Consecutive note updates limited (default: 3) to prevent infinite planning loops

## Grid Overlay

The `--grid` option overlays a 32x32 gray grid on images:
- Helps with spatial calibration and coordinate estimation
- Each cell is `canvas_size // 32` pixels
- Grid is **not** part of the target - do not draw grid lines
- Only shown to the VLM, not saved in final output

## Environment Variables

- `OPENROUTER_API_KEY`: Required for model API access

## License

See [LICENSE](LICENSE) file for details.
