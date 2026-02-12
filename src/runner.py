"""
PaintBench Runner

Main benchmark loop that orchestrates the VLM painting task.
"""

from pathlib import Path

import torch

from .benchmark_logger import BenchmarkLogger, create_run_dir
from .benchmark_task import BenchmarkTask, create_task
from .config import (
    CANVAS_SIZE as DEFAULT_CANVAS_SIZE,
)
from .config import (
    MAX_ACTIONS,
    OPENROUTER_MODEL_MAPPING,
    RunConfig,
    make_fixed_style_tools,
    make_tools,
)
from .renderer import CanvasState, canvas_to_image, draw_grid
from .vlms.openrouter import OpenRouterVLMClient
from .vlms.vlm import Action


def _should_reject_notes(history: list[Action], limit: int | None) -> bool:
    """Check if an update_notes action should be rejected due to consecutive limit.

    Returns True if the last `limit` actions were all update_notes, meaning we should
    reject the next one to prevent infinite planning loops.
    """
    if limit is None:
        return False
    if len(history) < limit:
        return False
    # Check if the last `limit` actions were all update_notes
    for action in history[-limit:]:
        if action.type != "update_notes":
            return False
    return True


def _clamp(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def _catmull_rom_to_beziers(
    points: list[dict], canvas_size: int
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]:
    if len(points) < 2:
        return []

    max_coord = canvas_size - 1
    pts = [{"x": p["x"], "y": p["y"]} for p in points]
    extended = [pts[0], *pts, pts[-1]]
    segments: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for i in range(1, len(extended) - 2):
        p0 = extended[i - 1]
        p1 = extended[i]
        p2 = extended[i + 1]
        p3 = extended[i + 2]

        c1x = p1["x"] + (p2["x"] - p0["x"]) / 6.0
        c1y = p1["y"] + (p2["y"] - p0["y"]) / 6.0
        c2x = p2["x"] - (p3["x"] - p1["x"]) / 6.0
        c2y = p2["y"] - (p3["y"] - p1["y"]) / 6.0

        segments.append(
            (
                (_clamp(p1["x"], 0, max_coord), _clamp(p1["y"], 0, max_coord)),
                (_clamp(c1x, 0, max_coord), _clamp(c1y, 0, max_coord)),
                (_clamp(c2x, 0, max_coord), _clamp(c2y, 0, max_coord)),
                (_clamp(p2["x"], 0, max_coord), _clamp(p2["y"], 0, max_coord)),
            )
        )

    return segments


def _is_stop_requested(run_dir: Path) -> bool:
    return (run_dir / "stop.requested").exists()


def run_benchmark(
    task: BenchmarkTask,
    model: str,
    tools: list[dict] | None = None,
    output_dir: str | Path = "benchmark/runs",
    max_actions: int = MAX_ACTIONS,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    show_grid: bool = False,
    save_snapshots: bool = True,
    compute_all_metrics: bool = True,
    verbose: bool = True,
    log_reasoning: bool = False,
    enable_reset: bool = True,
    run_name: str | None = None,
    actions_per_turn: int = 1,
    focus_mode: bool = False,
    focus_window: int = 10,
    enable_notes: bool = False,
    max_consecutive_notes: int | None = 3,
    fixed_style_params: dict | None = None,
) -> dict:
    """
    Run a single benchmark.

    Args:
        task: BenchmarkTask instance defining what to benchmark
        model: Model name (e.g., "gpt-5.2-2025-12-11")
        tools: Tool definitions (defaults to tools generated for canvas_size)
        output_dir: Directory for saving run outputs
        max_actions: Maximum number of actions allowed
        canvas_size: Canvas size in pixels
        show_grid: Overlay grid on images sent to VLM
        save_snapshots: Whether to save canvas after each stroke
        compute_all_metrics: Whether to compute expensive metrics (CLIP, LPIPS)
        verbose: Print progress
        log_reasoning: Log model reasoning traces
        enable_reset: If False, disable the reset tool
        run_name: Optional explicit run directory name
        actions_per_turn: Number of actions per turn (1 = single-action mode, >1 = multi-move)
        focus_mode: Use compact history with focus summaries instead of reasoning traces
        focus_window: Number of recent action:focus pairs to include in history (default 10)
        enable_notes: Enable the update_notes tool for persistent scratchpad
        max_consecutive_notes: Max consecutive update_notes before rejection (None = unlimited)
        fixed_style_params: If provided, these params (r, g, b, width, rounded) are injected
            into all draw_line and draw_curve actions. Used with simplified tools that don't
            expose color/width to the model.

    Returns:
        Dict with run results and metrics
    """
    if tools is None:
        tools = make_tools(canvas_size, enable_reset=enable_reset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canvas_state = CanvasState(device, canvas_size=canvas_size)

    # Initialize canvas with pre-loaded image if task provides one
    initial_canvas_image = task.get_initial_canvas_image()
    if initial_canvas_image is not None:
        canvas_state.set_initial_image(initial_canvas_image)

    # Create VLM client
    mapped_model = OPENROUTER_MODEL_MAPPING.get(model, model)
    vlm = OpenRouterVLMClient(
        task=task,
        tools=tools,
        run_config=RunConfig(
            model=mapped_model,
            canvas_size=canvas_size,
            log_reasoning=log_reasoning,
            enable_reset=enable_reset,
            focus_mode=focus_mode,
            focus_window=focus_window,
            enable_notes=enable_notes,
        ),
    )

    run_dir = create_run_dir(
        base_dir=output_dir,
        model=model,
        benchmark_type=task.task_type,
        run_target=task.run_target,
        run_name=run_name,
    )
    logger = BenchmarkLogger(run_dir)

    # Save target using task
    task.save_target(logger)

    # Get prompts from task
    system_prompt = task.get_system_prompt(
        RunConfig(
            model=model,
            canvas_size=canvas_size,
            log_reasoning=log_reasoning,
            enable_reset=enable_reset,
            focus_mode=focus_mode,
            focus_window=focus_window,
            enable_notes=enable_notes,
        )
    )
    message_prompt_template = task.get_message_template()

    logger.save_run_config(
        task_config=task.to_dict(),
        model=model,
        canvas_size=canvas_size,
        max_actions=max_actions,
        show_grid=show_grid,
        save_snapshots=save_snapshots,
        compute_all_metrics=compute_all_metrics,
        log_reasoning=log_reasoning,
        enable_reset=enable_reset,
        system_prompt=system_prompt,
        message_prompt_template=message_prompt_template,
        actions_per_turn=actions_per_turn,
        focus_mode=focus_mode,
        focus_window=focus_window,
        enable_notes=enable_notes,
    )

    # Save grid-overlaid target if applicable
    target_image = task.get_target_image()
    if show_grid and target_image:
        grid_target = draw_grid(target_image)
        grid_target.save(run_dir / "target_grid.png")

    if verbose:
        print(f"Starting {task.task_type} benchmark. Target: {task.run_target}")
        print(f"Model: {model}")
        print(f"Canvas size: {canvas_size}x{canvas_size}")
        if show_grid:
            cell_size = canvas_size // 32
            print(f"Grid: 32x32 ({cell_size}px cells)")
        if actions_per_turn > 1:
            print(f"Multi-move mode: {actions_per_turn} actions per turn")
        print(f"Output: {run_dir}")
        print("-" * 40)

    history: list[Action] = []
    action_count = 0
    turn_count = 0
    stop_reason = "budget_exhausted"
    multi_move = actions_per_turn > 1

    while action_count < max_actions:
        if _is_stop_requested(run_dir):
            stop_reason = "stop_requested"
            if verbose:
                print("Stop requested; ending run")
            break

        canvas_image = canvas_state.to_image()

        # Apply grid overlay if enabled (for VLM context only)
        vlm_target = vlm_canvas = None
        if show_grid:
            if target_image:
                vlm_target = draw_grid(target_image)
            vlm_canvas = draw_grid(canvas_image)
        else:
            if target_image:
                vlm_target = target_image
            vlm_canvas = canvas_image

        # Checkpoint before turn in multi-move mode
        if multi_move:
            canvas_state.checkpoint()

        # Calculate how many actions to request this turn
        remaining_budget = max_actions - action_count
        turn_size = min(actions_per_turn, remaining_budget)

        try:
            actions = vlm.get_action(
                vlm_target,
                vlm_canvas,
                history,
                action_count,
                show_grid=show_grid,
                max_actions=turn_size,
            )
        except Exception as e:
            if verbose:
                print(f"Error getting action: {e}")
            stop_reason = "error"
            break

        if not actions:
            if verbose:
                print("No action returned, stopping")
            stop_reason = "no_action"
            break

        turn_count += 1
        if verbose and multi_move:
            print(f"Turn {turn_count} ({len(actions)} action(s)):")

        # Execute all actions in this turn
        should_break = False
        for action in actions:
            # Inject fixed style params if provided (for controlled experiments)
            if fixed_style_params and action.type in (
                "draw_line",
                "draw_spline",
                "draw_polyline",
                "draw_curve",
            ):
                action.params.update(fixed_style_params)
            if _is_stop_requested(run_dir):
                stop_reason = "stop_requested"
                should_break = True
                break

            if verbose:
                prefix = "  " if multi_move else ""
                print(f"{prefix}Action {action_count + 1}: {action}")
                if action.reasoning:
                    reasoning = (
                        action.reasoning[:200] + "..."
                        if len(action.reasoning) > 200
                        else action.reasoning
                    )
                    print(f"{prefix}  Reasoning: {reasoning}")

            # Execute action
            if action.type == "stop":
                history.append(action)
                logger.log_action(
                    action,
                    canvas_image if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1
                stop_reason = "stop_called"
                should_break = True
                break

            elif action.type == "reset" and enable_reset:
                canvas_state.reset()
                history = []
                history.append(action)
                logger.log_action(
                    action,
                    canvas_state.to_image() if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1
                if verbose:
                    print("  -> Reset canvas")

            elif action.type == "undo":
                # In multi-move mode, undo reverts the entire previous turn
                if multi_move:
                    success = canvas_state.undo_turn()
                    if verbose:
                        print("  -> Undid previous turn" if success else "  -> Nothing to undo")
                else:
                    success = canvas_state.undo_stroke()
                    if verbose:
                        print("  -> Undid last stroke" if success else "  -> Nothing to undo")
                history.append(action)
                logger.log_action(
                    action,
                    canvas_state.to_image() if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1
                # In multi-move, undo ends the current turn
                if multi_move:
                    break

            elif action.type == "draw_line":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    if verbose:
                        print(
                            f"  -> REJECTED: white/near-white color not allowed ({penalty} action penalty)"
                        )
                else:
                    canvas_state.add_stroke(
                        p["x1"],
                        p["y1"],
                        p["x2"],
                        p["y2"],
                        p["r"],
                        p["g"],
                        p["b"],
                        p["width"],
                        p.get("rounded", True),
                    )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1

            elif action.type == "draw_spline":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    if verbose:
                        print(
                            f"  -> REJECTED: white/near-white color not allowed ({penalty} action penalty)"
                        )
                else:
                    points = p.get("points", [])
                    segments = _catmull_rom_to_beziers(points, canvas_size)
                    for start, c1, c2, end in segments:
                        canvas_state.add_curve(
                            start[0],
                            start[1],
                            c1[0],
                            c1[1],
                            c2[0],
                            c2[1],
                            end[0],
                            end[1],
                            p["r"],
                            p["g"],
                            p["b"],
                            p["width"],
                        )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1
                    if verbose:
                        print(
                            f"  -> Drew spline through {len(points)} points ({len(segments)} segments)"
                        )

            elif action.type == "draw_curve":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    if verbose:
                        print(
                            f"  -> REJECTED: white/near-white color not allowed ({penalty} action penalty)"
                        )
                else:
                    canvas_state.add_curve(
                        p["x1"],
                        p["y1"],
                        p["cx1"],
                        p["cy1"],
                        p["cx2"],
                        p["cy2"],
                        p["x2"],
                        p["y2"],
                        p["r"],
                        p["g"],
                        p["b"],
                        p["width"],
                    )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1
                    if verbose:
                        print("  -> Drew curve")

            elif action.type == "draw_polyline":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    if verbose:
                        print(
                            f"  -> REJECTED: white/near-white color not allowed ({penalty} action penalty)"
                        )
                else:
                    points = p.get("points", [])
                    for start, end in zip(points, points[1:], strict=False):
                        canvas_state.add_stroke(
                            start["x"],
                            start["y"],
                            end["x"],
                            end["y"],
                            p["r"],
                            p["g"],
                            p["b"],
                            p["width"],
                            p.get("rounded", True),
                        )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1
                    if verbose:
                        print(f"  -> Drew polyline with {len(points)} points")

            elif action.type == "update_notes":
                # Check for consecutive notes limit to prevent infinite planning loops
                if _should_reject_notes(history, max_consecutive_notes):
                    if verbose:
                        prefix = "  " if multi_move else ""
                        print(
                            f"{prefix}  -> Notes rejected: {max_consecutive_notes} consecutive updates without action"
                        )
                    # Add feedback to history so model knows why and what to do
                    rejection_msg = (
                        f"Notes update rejected: {max_consecutive_notes} consecutive note updates without drawing. "
                        "You must call draw_line(), draw_spline(), or draw_polyline() before updating notes again."
                    )
                    history.append(Action(type="notes_rejected", params={"message": rejection_msg}))
                    continue  # Skip this update_notes, process remaining actions in turn

                # Get the entry to append
                entry = action.notes_content or ""

                # Notes update doesn't count as an action, just updates state
                # Append to notes log (truncation is automatic)
                if hasattr(vlm, "append_notes_entry"):
                    vlm.append_notes_entry(turn_count, entry)
                history.append(action)
                logger.log_action(
                    action,
                    None,  # No snapshot for notes-only action
                    turn_number=turn_count if multi_move else None,
                )
                if verbose:
                    prefix = "  " if multi_move else ""
                    entry_preview = entry[:60]
                    if len(entry) > 60:
                        entry_preview += "..."
                    print(f"{prefix}  -> Notes entry added: {entry_preview}")

            else:
                if verbose:
                    print(f"Action not recognized: {action}")
                stop_reason = "unrecognized_action"
                action_count += 1
                should_break = True
                break

        if should_break:
            break

    final_image = canvas_to_image(canvas_state.get_canvas())
    logger.save_final(final_image)

    if verbose:
        print("-" * 40)
        print("Computing metrics...")

    # Use task for evaluation
    metrics = task.evaluate(final_image, compute_all=compute_all_metrics)

    logger.save_result(
        metrics=metrics,
        model=model,
        run_target=task.run_target,
        total_actions=action_count,
        stop_reason=stop_reason,
        canvas_size=canvas_size,
        actions_per_turn=actions_per_turn,
    )

    if verbose:
        print("Results:")
        print(f"  Total actions: {action_count}")
        print(f"  Stop reason: {stop_reason}")
        if metrics.get("mse") is not None:
            print(f"  MSE: {metrics['mse']:.6f}")
        if metrics.get("ssim") is not None:
            print(f"  SSIM: {metrics['ssim']:.4f}")
        if metrics.get("clip_similarity") is not None:
            print(f"  CLIP: {metrics['clip_similarity']:.4f}")
        if metrics.get("lpips") is not None:
            print(f"  LPIPS: {metrics['lpips']:.4f}")
        print(f"Output saved to: {run_dir}")

    return {
        "run_dir": str(run_dir),
        "model": model,
        "run_target": task.run_target,
        "canvas_size": canvas_size,
        "total_actions": action_count,
        "stop_reason": stop_reason,
        "metrics": metrics,
    }


async def run_benchmark_async(
    task: BenchmarkTask,
    model: str,
    tools: list[dict] | None = None,
    output_dir: str | Path = "benchmark/runs",
    max_actions: int = MAX_ACTIONS,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    show_grid: bool = False,
    save_snapshots: bool = True,
    compute_all_metrics: bool = True,
    verbose: bool = True,
    log_reasoning: bool = False,
    enable_reset: bool = True,
    run_name: str | None = None,
    actions_per_turn: int = 1,
    run_id: str | None = None,
    fixed_style_params: dict | None = None,
    focus_mode: bool = False,
    focus_window: int = 10,
    enable_notes: bool = False,
    max_consecutive_notes: int | None = 3,
) -> dict:
    """
    Async version of run_benchmark for concurrent execution.

    Args:
        Same as run_benchmark, plus:
        run_id: Optional identifier for logging in batch mode
        fixed_style_params: If provided, these params are injected into draw actions
        focus_mode: Use compact history with focus summaries instead of reasoning traces
        focus_window: Number of recent action:focus pairs to include in history (default 10)
        enable_notes: Enable the update_notes tool for persistent scratchpad
        max_consecutive_notes: Max consecutive update_notes before rejection (None = unlimited)

    Returns:
        Dict with run results and metrics
    """
    if tools is None:
        tools = make_tools(canvas_size, enable_reset=enable_reset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canvas_state = CanvasState(device, canvas_size=canvas_size)

    # Initialize canvas with pre-loaded image if task provides one
    initial_canvas_image = task.get_initial_canvas_image()
    if initial_canvas_image is not None:
        canvas_state.set_initial_image(initial_canvas_image)

    # Create VLM client
    mapped_model = OPENROUTER_MODEL_MAPPING.get(model, model)
    vlm = OpenRouterVLMClient(
        task=task,
        run_config=RunConfig(
            model=mapped_model,
            canvas_size=canvas_size,
            log_reasoning=log_reasoning,
            enable_reset=enable_reset,
            focus_mode=focus_mode,
            focus_window=focus_window,
            enable_notes=enable_notes,
        ),
        tools=tools,
    )

    run_dir = create_run_dir(
        base_dir=output_dir,
        model=model,
        benchmark_type=task.task_type,
        run_target=task.run_target,
        run_name=run_name,
    )
    logger = BenchmarkLogger(run_dir)

    # Save target using task
    task.save_target(logger)

    # Get prompts from task
    system_prompt = task.get_system_prompt(
        RunConfig(
            model=model,
            canvas_size=canvas_size,
            log_reasoning=log_reasoning,
            enable_reset=enable_reset,
            focus_mode=focus_mode,
            focus_window=focus_window,
            enable_notes=enable_notes,
        )
    )
    message_prompt_template = task.get_message_template()

    logger.save_run_config(
        task_config=task.to_dict(),
        model=model,
        canvas_size=canvas_size,
        max_actions=max_actions,
        show_grid=show_grid,
        save_snapshots=save_snapshots,
        compute_all_metrics=compute_all_metrics,
        log_reasoning=log_reasoning,
        enable_reset=enable_reset,
        system_prompt=system_prompt,
        message_prompt_template=message_prompt_template,
        actions_per_turn=actions_per_turn,
        focus_mode=focus_mode,
        focus_window=focus_window,
        enable_notes=enable_notes,
    )

    # Save grid-overlaid target if applicable
    target_image = task.get_target_image()
    if show_grid and target_image:
        grid_target = draw_grid(target_image)
        grid_target.save(run_dir / "target_grid.png")

    log_prefix = f"[{run_id}] " if run_id else ""
    if verbose:
        print(f"{log_prefix}Starting {task.task_type} benchmark. Target: {task.run_target}")
        print(f"{log_prefix}Model: {model}")
        print(f"{log_prefix}Output: {run_dir}")

    history: list[Action] = []
    action_count = 0
    turn_count = 0
    stop_reason = "budget_exhausted"
    multi_move = actions_per_turn > 1

    while action_count < max_actions:
        if _is_stop_requested(run_dir):
            stop_reason = "stop_requested"
            if verbose:
                print(f"{log_prefix}Stop requested; ending run")
            break

        canvas_image = canvas_state.to_image()

        # Apply grid overlay if enabled (for VLM context only)
        vlm_target = vlm_canvas = None
        if show_grid:
            if target_image:
                vlm_target = draw_grid(target_image)
            vlm_canvas = draw_grid(canvas_image)
        else:
            if target_image:
                vlm_target = target_image
            vlm_canvas = canvas_image

        # Checkpoint before turn in multi-move mode
        if multi_move:
            canvas_state.checkpoint()

        # Calculate how many actions to request this turn
        remaining_budget = max_actions - action_count
        turn_size = min(actions_per_turn, remaining_budget)

        try:
            # Use async version of get_action
            actions = await vlm.get_action_async(
                vlm_target,
                vlm_canvas,
                history,
                action_count,
                show_grid=show_grid,
                max_actions=turn_size,
            )
        except Exception as e:
            if verbose:
                print(f"{log_prefix}Error getting action: {e}")
            stop_reason = "error"
            break

        if not actions:
            if verbose:
                print(f"{log_prefix}No action returned, stopping")
            stop_reason = "no_action"
            break

        turn_count += 1

        # Execute all actions in this turn
        should_break = False
        for action in actions:
            # Inject fixed style params if provided (for controlled experiments)
            if fixed_style_params and action.type in (
                "draw_line",
                "draw_spline",
                "draw_polyline",
                "draw_curve",
            ):
                action.params.update(fixed_style_params)
            if _is_stop_requested(run_dir):
                stop_reason = "stop_requested"
                should_break = True
                break

            if verbose:
                print(f"{log_prefix}Action {action_count + 1}: {action}")

            # Execute action (same logic as sync version)
            if action.type == "stop":
                history.append(action)
                logger.log_action(
                    action,
                    canvas_image if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1
                stop_reason = "stop_called"
                should_break = True
                break

            elif action.type == "reset" and enable_reset:
                canvas_state.reset()
                history = []
                history.append(action)
                logger.log_action(
                    action,
                    canvas_state.to_image() if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1

            elif action.type == "undo":
                if multi_move:
                    canvas_state.undo_turn()
                else:
                    canvas_state.undo_stroke()
                history.append(action)
                logger.log_action(
                    action,
                    canvas_state.to_image() if save_snapshots else None,
                    turn_number=turn_count if multi_move else None,
                )
                action_count += 1
                if multi_move:
                    break

            elif action.type == "draw_line":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                else:
                    canvas_state.add_stroke(
                        p["x1"],
                        p["y1"],
                        p["x2"],
                        p["y2"],
                        p["r"],
                        p["g"],
                        p["b"],
                        p["width"],
                        p.get("rounded", True),
                    )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1

            elif action.type == "draw_spline":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                else:
                    points = p.get("points", [])
                    segments = _catmull_rom_to_beziers(points, canvas_size)
                    for start, c1, c2, end in segments:
                        canvas_state.add_curve(
                            start[0],
                            start[1],
                            c1[0],
                            c1[1],
                            c2[0],
                            c2[1],
                            end[0],
                            end[1],
                            p["r"],
                            p["g"],
                            p["b"],
                            p["width"],
                        )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1

            elif action.type == "draw_curve":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                else:
                    canvas_state.add_curve(
                        p["x1"],
                        p["y1"],
                        p["cx1"],
                        p["cy1"],
                        p["cx2"],
                        p["cy2"],
                        p["x2"],
                        p["y2"],
                        p["r"],
                        p["g"],
                        p["b"],
                        p["width"],
                    )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1

            elif action.type == "draw_polyline":
                p = action.params
                is_white = p["r"] >= 250 and p["g"] >= 250 and p["b"] >= 250

                if is_white:
                    penalty = 3
                    action_count += penalty
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                else:
                    points = p.get("points", [])
                    for start, end in zip(points, points[1:], strict=False):
                        canvas_state.add_stroke(
                            start["x"],
                            start["y"],
                            end["x"],
                            end["y"],
                            p["r"],
                            p["g"],
                            p["b"],
                            p["width"],
                            p.get("rounded", True),
                        )
                    history.append(action)
                    logger.log_action(
                        action,
                        canvas_state.to_image() if save_snapshots else None,
                        turn_number=turn_count if multi_move else None,
                    )
                    action_count += 1

            elif action.type == "update_notes":
                # Check for consecutive notes limit to prevent infinite planning loops
                if _should_reject_notes(history, max_consecutive_notes):
                    if verbose:
                        print(
                            f"{log_prefix}  -> Notes rejected: {max_consecutive_notes} consecutive updates without action"
                        )
                    # Add feedback to history so model knows why and what to do
                    rejection_msg = (
                        f"Notes update rejected: {max_consecutive_notes} consecutive note updates without drawing. "
                        "You must call draw_line() or draw_curve() before updating notes again."
                    )
                    history.append(Action(type="notes_rejected", params={"message": rejection_msg}))
                    continue  # Skip this update_notes, process remaining actions in turn

                # Get the entry to append
                entry = action.notes_content or ""

                # Notes update doesn't count as an action, just updates state
                # Append to notes log (truncation is automatic)
                if hasattr(vlm, "append_notes_entry"):
                    vlm.append_notes_entry(turn_count, entry)
                history.append(action)
                logger.log_action(
                    action,
                    None,  # No snapshot for notes-only action
                    turn_number=turn_count if multi_move else None,
                )
                if verbose:
                    entry_preview = entry[:60]
                    if len(entry) > 60:
                        entry_preview += "..."
                    print(f"{log_prefix}  -> Notes entry added: {entry_preview}")

            else:
                stop_reason = "unrecognized_action"
                action_count += 1
                should_break = True
                break

        if should_break:
            break

    final_image = canvas_to_image(canvas_state.get_canvas())
    logger.save_final(final_image)

    if verbose:
        print(f"{log_prefix}Computing metrics...")

    metrics = task.evaluate(final_image, compute_all=compute_all_metrics)

    logger.save_result(
        metrics=metrics,
        model=model,
        run_target=task.run_target,
        total_actions=action_count,
        stop_reason=stop_reason,
        canvas_size=canvas_size,
        actions_per_turn=actions_per_turn,
    )

    if verbose:
        print(f"{log_prefix}Completed: {action_count} actions, {stop_reason}")
        print(f"{log_prefix}Output saved to: {run_dir}")

    return {
        "run_dir": str(run_dir),
        "model": model,
        "run_target": task.run_target,
        "canvas_size": canvas_size,
        "total_actions": action_count,
        "stop_reason": stop_reason,
        "metrics": metrics,
    }


async def run_batch_async(
    configs: list[dict],
    max_concurrent: int = 5,
    verbose: bool = True,
) -> list[dict]:
    """
    Run multiple benchmarks concurrently using asyncio.

    Args:
        configs: List of benchmark configurations. Each dict should contain:
            - task_type: "target_image", "text_prompt"
            - model: Model name
            - target: Target image name (for target_image)
            - text_prompt_id: Prompt ID (for text_prompt)
            - And other optional parameters from run_benchmark
        max_concurrent: Maximum number of concurrent runs (semaphore limit)
        verbose: Print progress

    Returns:
        List of result dicts from each benchmark run
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(config: dict, run_id: str) -> dict:
        async with semaphore:
            try:
                canvas_size = config.get("canvas_size", DEFAULT_CANVAS_SIZE)
                enable_reset = config.get("enable_reset", True)

                # Create task from config
                task = create_task(
                    task_type=config.get("task_type", "target_image"),
                    target_path=config.get("target"),
                    text_prompt_id=config.get("text_prompt_id"),
                    canvas_size=canvas_size,
                )

                # Handle fixed-style mode
                tools = None
                fixed_style_params = None
                if config.get("fixed_style"):
                    tools, fixed_style_params = make_fixed_style_tools(
                        canvas_size=canvas_size,
                        color=(0, 0, 0),
                        width=2,
                        enable_reset=enable_reset,
                    )

                result = await run_benchmark_async(
                    task=task,
                    model=config["model"],
                    tools=tools,
                    output_dir=config.get("output_dir", "benchmark/runs"),
                    max_actions=config.get("max_actions", MAX_ACTIONS),
                    canvas_size=canvas_size,
                    show_grid=config.get("grid", False),
                    save_snapshots=config.get("save_snapshots", True),
                    compute_all_metrics=config.get("compute_all_metrics", True),
                    verbose=verbose,
                    log_reasoning=config.get("log_reasoning", False),
                    enable_reset=enable_reset,
                    run_name=config.get("run_name"),
                    actions_per_turn=config.get("actions_per_turn", 1),
                    run_id=run_id,
                    fixed_style_params=fixed_style_params,
                    focus_mode=config.get("focus_mode", False),
                    focus_window=config.get("focus_window", 10),
                    enable_notes=config.get("enable_notes", False),
                    max_consecutive_notes=config.get("max_consecutive_notes", 3),
                )
                return {"success": True, "run_id": run_id, **result}
            except Exception as e:
                if verbose:
                    print(f"[{run_id}] Error: {e}")
                return {"success": False, "run_id": run_id, "error": str(e)}

    # Create tasks with unique IDs
    tasks = [run_with_semaphore(config, f"run_{i + 1}") for i, config in enumerate(configs)]

    if verbose:
        print(f"Starting {len(configs)} benchmarks (max {max_concurrent} concurrent)...")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that weren't caught
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(
                {
                    "success": False,
                    "run_id": f"run_{i + 1}",
                    "error": str(result),
                }
            )
        else:
            processed_results.append(result)

    if verbose:
        successful = sum(1 for r in processed_results if r.get("success"))
        print(f"Completed: {successful}/{len(configs)} successful")

    return processed_results


def main():
    """CLI entry point."""
    import argparse
    import asyncio
    import json

    parser = argparse.ArgumentParser(
        description="Run PaintBench on a target image",
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        choices=["target_image", "text_prompt"],
        help="Benchmark type (required for single runs)",
    )
    parser.add_argument("--model", type=str, help="Model name (required for single runs)")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Path to target image (for target_image benchmark)",
    )
    parser.add_argument(
        "--text-prompt-id",
        type=str,
        default=None,
        help="Text prompt ID (for text_prompt benchmark)",
    )
    parser.add_argument("--output-dir", type=str, default="benchmark/runs", help="Output directory")
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS, help="Max actions")
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help=f"Canvas size in pixels (default: {DEFAULT_CANVAS_SIZE})",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Overlay 32x32 grid on images for VLM spatial calibration",
    )
    parser.add_argument("--no-snapshots", action="store_true", help="Do not save canvas snapshots")
    parser.add_argument("--basic-metrics", action="store_true", help="Only compute MSE/SSIM")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument(
        "--log-reasoning",
        action="store_true",
        help="Log model reasoning traces to actions.jsonl",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Disable the reset tool (prevents model from resetting the canvas)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Explicit run directory name (bypasses auto-generation)",
    )
    parser.add_argument(
        "--actions-per-turn",
        type=int,
        default=1,
        help="Actions per turn (1 = single-action mode, >1 = multi-move mode)",
    )
    parser.add_argument(
        "--focus-mode",
        action="store_true",
        help="Use compact history with focus summaries instead of reasoning traces",
    )
    parser.add_argument(
        "--focus-window",
        type=int,
        default=10,
        help="Number of recent action:focus pairs to include in history (default: 10)",
    )
    parser.add_argument(
        "--enable-notes",
        action="store_true",
        help="Enable the update_notes tool for persistent scratchpad/planning",
    )
    parser.add_argument(
        "--fixed-style",
        action="store_true",
        help="Use simplified tools with fixed color (black) and width (2px). "
        "Useful for controlled difficulty experiments.",
    )
    # Batch mode arguments
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to JSON file with batch configs, or inline JSON array",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent runs in batch mode (default: 5)",
    )

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        # Load configs from file or parse inline JSON
        if args.batch.startswith("["):
            configs = json.loads(args.batch)
        else:
            with open(args.batch) as f:
                configs = json.load(f)

        # Apply common options to all configs
        for config in configs:
            if "output_dir" not in config:
                config["output_dir"] = args.output_dir
            if "max_actions" not in config:
                config["max_actions"] = args.max_actions
            if "canvas_size" not in config:
                config["canvas_size"] = args.canvas_size
            if "grid" not in config:
                config["grid"] = args.grid
            if "save_snapshots" not in config:
                config["save_snapshots"] = not args.no_snapshots
            if "compute_all_metrics" not in config:
                config["compute_all_metrics"] = not args.basic_metrics
            if "log_reasoning" not in config:
                config["log_reasoning"] = args.log_reasoning
            if "enable_reset" not in config:
                config["enable_reset"] = not args.no_reset
            if "actions_per_turn" not in config:
                config["actions_per_turn"] = args.actions_per_turn
            if "focus_mode" not in config:
                config["focus_mode"] = args.focus_mode
            if "focus_window" not in config:
                config["focus_window"] = args.focus_window
            if "enable_notes" not in config:
                config["enable_notes"] = args.enable_notes

        results = asyncio.run(
            run_batch_async(
                configs,
                max_concurrent=args.max_concurrent,
                verbose=not args.quiet,
            )
        )

        # Print summary
        if not args.quiet:
            print("\n" + "=" * 40)
            print("Batch Summary:")
            for r in results:
                status = "OK" if r.get("success") else "FAILED"
                run_id = r.get("run_id", "?")
                if r.get("success"):
                    print(f"  [{status}] {run_id}: {r.get('run_target')} -> {r.get('run_dir')}")
                else:
                    print(f"  [{status}] {run_id}: {r.get('error')}")
        return

    # Single run mode - require benchmark-type and model
    if not args.benchmark_type:
        parser.error("--benchmark-type is required for single runs")
    if not args.model:
        parser.error("--model is required for single runs")

    task = create_task(
        task_type=args.benchmark_type,
        target_path=args.target,
        text_prompt_id=args.text_prompt_id,
        canvas_size=args.canvas_size,
    )

    # Handle fixed-style mode
    tools = None
    fixed_style_params = None
    if args.fixed_style:
        tools, fixed_style_params = make_fixed_style_tools(
            canvas_size=args.canvas_size,
            color=(0, 0, 0),
            width=2,
            enable_reset=not args.no_reset,
        )

    run_benchmark(
        task=task,
        model=args.model,
        tools=tools,
        output_dir=args.output_dir,
        max_actions=args.max_actions,
        canvas_size=args.canvas_size,
        show_grid=args.grid,
        save_snapshots=not args.no_snapshots,
        compute_all_metrics=not args.basic_metrics,
        verbose=not args.quiet,
        log_reasoning=args.log_reasoning,
        enable_reset=not args.no_reset,
        run_name=args.run_name,
        actions_per_turn=args.actions_per_turn,
        focus_mode=args.focus_mode,
        focus_window=args.focus_window,
        enable_notes=args.enable_notes,
        fixed_style_params=fixed_style_params,
    )


if __name__ == "__main__":
    main()
