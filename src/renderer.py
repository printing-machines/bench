"""
PaintBench Renderer

Wraps the SDF-based stroke renderer with discrete parameter handling
for the benchmark. Converts integer coordinates and color names to
the normalized tensor format expected by the underlying renderer.
"""

import numpy as np
import torch
from PIL import Image

from .config import CANVAS_SIZE as DEFAULT_CANVAS_SIZE

# =============================================================================
# Color Space Conversion
# =============================================================================


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)


# =============================================================================
# Pixel Grid Cache
# =============================================================================

_GRID_CACHE = {}


def _get_pixel_grid(H: int, W: int, device: torch.device) -> torch.Tensor:
    key = (H, W, device)
    if key not in _GRID_CACHE:
        y = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        x = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        _GRID_CACHE[key] = torch.stack([xx, yy], dim=-1)
    return _GRID_CACHE[key]


# =============================================================================
# SDF Rendering (Lines and Circles)
# =============================================================================


def _sdf_circle(
    grid: torch.Tensor,
    center: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed distance from each pixel to a circle.

    Args:
        grid: (H, W, 2) pixel coordinate grid
        center: (B, 2) center points
        radius: (B,) radii

    Returns:
        (B, H, W) distance tensor (negative inside, positive outside)
    """
    B = center.shape[0]

    P = grid.unsqueeze(0)  # (1, H, W, 2)
    C = center.view(B, 1, 1, 2)  # (B, 1, 1, 2)

    diff = P - C  # (B, H, W, 2)
    dist_from_center = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)  # (B, H, W)

    # Distance to circle edge (negative inside, positive outside)
    sdf = dist_from_center - radius.view(B, 1, 1)

    return sdf


def _sdf_segment(
    grid: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    rounded: bool = True,
) -> torch.Tensor:
    """
    Compute signed distance from each pixel to a line segment.

    Args:
        grid: (H, W, 2) pixel coordinate grid
        p1: (B, 2) start points
        p2: (B, 2) end points
        rounded: If True, use round caps (distance to closest point on segment).
                 If False, use flat/butt caps (only pixels within the rectangle).
    """
    B = p1.shape[0]
    _, _ = grid.shape[:2]

    P = grid.unsqueeze(0)
    A = p1.view(B, 1, 1, 2)
    B_pt = p2.view(B, 1, 1, 2)

    AB = B_pt - A
    AP = P - A

    AB_squared = (AB * AB).sum(dim=-1, keepdim=True) + 1e-8
    t = (AP * AB).sum(dim=-1, keepdim=True) / AB_squared
    t_clamped = torch.clamp(t, 0.0, 1.0)

    closest = A + t_clamped * AB
    diff = P - closest
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)

    if not rounded:
        # For flat/butt caps, add a large distance for points beyond the segment ends
        # This effectively clips the line to a rectangle instead of having round caps
        beyond_segment = (t.squeeze(-1) < 0.0) | (t.squeeze(-1) > 1.0)
        dist = torch.where(beyond_segment, dist + 1000.0, dist)

    return dist


def _cubic_bezier_sample(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_samples: int = 32,
) -> torch.Tensor:
    """
    Sample points along a cubic Bézier curve.

    Args:
        p0: (2,) start point
        p1: (2,) first control point
        p2: (2,) second control point
        p3: (2,) end point
        num_samples: Number of points to sample along the curve

    Returns:
        (num_samples, 2) tensor of sampled points
    """
    device = p0.device
    t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=torch.float32)
    t = t.view(-1, 1)  # (num_samples, 1)

    # Cubic Bézier: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    u = 1.0 - t
    points = (
        u**3 * p0.view(1, 2)
        + 3 * u**2 * t * p1.view(1, 2)
        + 3 * u * t**2 * p2.view(1, 2)
        + t**3 * p3.view(1, 2)
    )
    return points  # (num_samples, 2)


def _sdf_bezier_curve(
    grid: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    num_samples: int = 32,
) -> torch.Tensor:
    """
    Compute signed distance from each pixel to a cubic Bézier curve.

    Approximates the curve as a polyline of segments and takes the minimum
    distance to any segment (union of all segments).

    Args:
        grid: (H, W, 2) pixel coordinate grid
        p0: (2,) start point
        p1: (2,) first control point
        p2: (2,) second control point
        p3: (2,) end point
        num_samples: Number of points to sample (creates num_samples-1 segments)

    Returns:
        (1, H, W) distance tensor
    """
    # Sample points along the curve
    samples = _cubic_bezier_sample(p0, p1, p2, p3, num_samples)  # (N, 2)

    H, W = grid.shape[:2]
    device = grid.device

    # Compute distance to each segment and take minimum
    # Start with a large distance
    min_dist = torch.full((1, H, W), float("inf"), device=device, dtype=torch.float32)

    for i in range(num_samples - 1):
        seg_start = samples[i : i + 1, :]  # (1, 2)
        seg_end = samples[i + 1 : i + 2, :]  # (1, 2)

        # Compute SDF for this segment (always rounded for smooth curve)
        seg_dist = _sdf_segment(grid, seg_start, seg_end, rounded=True)  # (1, H, W)

        # Take minimum distance (union of segments)
        min_dist = torch.minimum(min_dist, seg_dist)

    return min_dist


def _dist_to_alpha(
    dist: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor:
    """Convert distance to alpha with hard edges and 1px anti-aliasing."""
    radius = width.view(-1, 1, 1) / 2.0

    # Hard edge with 1px anti-aliasing band
    # Inside radius: alpha = 1
    # Outside radius + 1px: alpha = 0
    # Transition zone: linear falloff
    aa_width = 1.0
    alpha = torch.clamp((radius + aa_width - dist) / aa_width, 0.0, 1.0)

    return alpha


def _blend(
    canvas: torch.Tensor,
    color: torch.Tensor,
    alpha: torch.Tensor,
    opacity: float = 1.0,
) -> torch.Tensor:
    """Blend a color onto the canvas with alpha and user-specified opacity.

    Args:
        canvas: (B, 3, H, W) destination canvas
        color: (B, 3) RGB color values in sRGB space (0-1)
        alpha: (B, H, W) per-pixel alpha from SDF (shape mask + anti-aliasing)
        opacity: User-specified opacity multiplier (0.0-1.0)
    """
    B, _, _, _ = canvas.shape

    # Combine SDF alpha with user opacity
    final_alpha = alpha.unsqueeze(1) * opacity
    src_rgb = color.view(B, 3, 1, 1)

    src_lin = srgb_to_linear(src_rgb)
    dst_lin = srgb_to_linear(canvas)

    out_lin = src_lin * final_alpha + dst_lin * (1.0 - final_alpha)
    out_srgb = linear_to_srgb(out_lin)

    return out_srgb.clamp(0.0, 1.0)


def _render_line_internal(
    canvas: torch.Tensor,
    p1_px: torch.Tensor,
    p2_px: torch.Tensor,
    width_px: torch.Tensor,
    color: torch.Tensor,
    rounded: bool = True,
    opacity: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal render with pixel coordinates and hard edges.

    Args:
        canvas: (1, 3, H, W) current canvas
        p1_px, p2_px: (1, 2) start/end points in pixel coordinates
        width_px: (1, 1) stroke width
        color: (1, 3) RGB color in sRGB space (0-1)
        rounded: Whether to use round caps
        opacity: User-specified opacity (0.0-1.0)
    """
    device = canvas.device
    _, _, H, W = canvas.shape

    width_px = torch.clamp(width_px, min=1.0)
    grid = _get_pixel_grid(H, W, device)

    dist = _sdf_segment(grid, p1_px, p2_px, rounded=rounded)
    alpha = _dist_to_alpha(dist, width_px)
    result = _blend(canvas, color, alpha, opacity=opacity)

    return result, alpha


# =============================================================================
# Public API for Benchmark
# =============================================================================


def create_canvas(
    device: torch.device | None = None, size: int = DEFAULT_CANVAS_SIZE
) -> torch.Tensor:
    """Create a blank white canvas of the specified size."""
    if device is None:
        device = torch.device("cpu")
    return torch.ones(1, 3, size, size, device=device)


def draw_line(
    canvas: torch.Tensor,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    r: int,
    g: int,
    b: int,
    width: int,
    rounded: bool = True,
    opacity: float = 1.0,
) -> torch.Tensor:
    """
    Draw a line on the canvas with discrete parameters.

    Args:
        canvas: (1, 3, H, W) current canvas
        x1, y1: Start point (0 to canvas_size-1)
        x2, y2: End point (0 to canvas_size-1)
        r, g, b: RGB color values (0-255 each)
        width: Stroke width in pixels (1-20)
        rounded: If True, use round caps. If False, use flat/butt caps.
        opacity: Opacity value (always 1.0, kept for internal compatibility)

    Returns:
        Updated canvas tensor
    """
    device = canvas.device
    _, _, H, W = canvas.shape
    canvas_size = H  # Assuming square canvas

    # Clamp coordinates to canvas bounds
    x1 = max(0, min(canvas_size - 1, int(x1)))
    y1 = max(0, min(canvas_size - 1, int(y1)))
    x2 = max(0, min(canvas_size - 1, int(x2)))
    y2 = max(0, min(canvas_size - 1, int(y2)))

    # Clamp width (use default, max 20 for lines)
    width = max(1, min(20, int(width)))

    # Clamp RGB values to 0-255 and convert to 0-1
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    color_tensor = torch.tensor(
        [[r / 255.0, g / 255.0, b / 255.0]],
        device=device,
        dtype=torch.float32,
    )

    # Clamp opacity to 0-1
    opacity = max(0.0, min(1.0, float(opacity)))

    # Convert to pixel coordinates (add 0.5 for center of pixel)
    p1_px = torch.tensor([[x1 + 0.5, y1 + 0.5]], device=device, dtype=torch.float32)
    p2_px = torch.tensor([[x2 + 0.5, y2 + 0.5]], device=device, dtype=torch.float32)
    width_px = torch.tensor([[float(width)]], device=device, dtype=torch.float32)

    result, _ = _render_line_internal(
        canvas, p1_px, p2_px, width_px, color_tensor, rounded=rounded, opacity=opacity
    )
    return result


def draw_curve(
    canvas: torch.Tensor,
    x1: int,
    y1: int,
    cx1: int,
    cy1: int,
    cx2: int,
    cy2: int,
    x2: int,
    y2: int,
    r: int,
    g: int,
    b: int,
    width: int,
    opacity: float = 1.0,
) -> torch.Tensor:
    """
    Draw a cubic Bézier curve on the canvas.

    Args:
        canvas: (1, 3, H, W) current canvas
        x1, y1: Start point (0 to canvas_size-1)
        cx1, cy1: First control point (0 to canvas_size-1)
        cx2, cy2: Second control point (0 to canvas_size-1)
        x2, y2: End point (0 to canvas_size-1)
        r, g, b: RGB color values (0-255 each)
        width: Stroke width in pixels (1-20)
        opacity: Opacity value (always 1.0, kept for internal compatibility)

    Returns:
        Updated canvas tensor
    """
    device = canvas.device
    _, _, H, W = canvas.shape
    canvas_size = H  # Assuming square canvas

    # Clamp coordinates to canvas bounds
    x1 = max(0, min(canvas_size - 1, int(x1)))
    y1 = max(0, min(canvas_size - 1, int(y1)))
    cx1 = max(0, min(canvas_size - 1, int(cx1)))
    cy1 = max(0, min(canvas_size - 1, int(cy1)))
    cx2 = max(0, min(canvas_size - 1, int(cx2)))
    cy2 = max(0, min(canvas_size - 1, int(cy2)))
    x2 = max(0, min(canvas_size - 1, int(x2)))
    y2 = max(0, min(canvas_size - 1, int(y2)))

    # Clamp width (use default, max 20 for curves)
    width = max(1, min(20, int(width)))

    # Clamp RGB values to 0-255 and convert to 0-1
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    color_tensor = torch.tensor(
        [[r / 255.0, g / 255.0, b / 255.0]],
        device=device,
        dtype=torch.float32,
    )

    # Clamp opacity to 0-1
    opacity = max(0.0, min(1.0, float(opacity)))

    # Convert to pixel coordinates (add 0.5 for center of pixel)
    p0 = torch.tensor([x1 + 0.5, y1 + 0.5], device=device, dtype=torch.float32)
    p1 = torch.tensor([cx1 + 0.5, cy1 + 0.5], device=device, dtype=torch.float32)
    p2 = torch.tensor([cx2 + 0.5, cy2 + 0.5], device=device, dtype=torch.float32)
    p3 = torch.tensor([x2 + 0.5, y2 + 0.5], device=device, dtype=torch.float32)
    width_px = torch.tensor([float(width)], device=device, dtype=torch.float32)

    # Get pixel grid and compute SDF for the Bézier curve
    grid = _get_pixel_grid(H, W, device)
    dist = _sdf_bezier_curve(grid, p0, p1, p2, p3)  # (1, H, W)

    # Convert distance to alpha and blend
    alpha = _dist_to_alpha(dist, width_px)
    result = _blend(canvas, color_tensor, alpha, opacity=opacity)

    return result


def canvas_to_image(canvas: torch.Tensor) -> Image.Image:
    """Convert canvas tensor to PIL Image."""
    # (1, 3, H, W) -> (H, W, 3)
    img = canvas[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def image_to_tensor(
    image: Image.Image,
    device: torch.device | None = None,
    size: int = DEFAULT_CANVAS_SIZE,
) -> torch.Tensor:
    """Convert PIL Image to canvas tensor, resizing to the specified size if needed."""
    if device is None:
        device = torch.device("cpu")

    # Resize if needed
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # (H, W, 3) -> (1, 3, H, W)
    img = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


# =============================================================================
# Grid Overlay for VLM Calibration
# =============================================================================


def draw_grid(
    image: Image.Image,
    cells: int = 32,
    color: tuple[int, int, int, int] = (128, 128, 128, 100),
) -> Image.Image:
    """
    Draw a grid overlay on an image to help VLMs with spatial calibration.

    Args:
        image: Input PIL Image
        cells: Number of grid cells (default 32 for a 32x32 grid)
        color: RGBA color for grid lines (default semi-transparent gray)

    Returns:
        New image with grid overlay (original is not modified)
    """
    from PIL import ImageDraw

    # Convert to RGBA if needed to support transparency
    if image.mode != "RGBA":
        result = image.convert("RGBA")
    else:
        result = image.copy()

    width, height = result.size
    cell_width = width / cells
    cell_height = height / cells

    # Create a transparent overlay for the grid
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw vertical lines
    for i in range(1, cells):
        x = int(i * cell_width)
        draw.line([(x, 0), (x, height - 1)], fill=color, width=1)

    # Draw horizontal lines
    for i in range(1, cells):
        y = int(i * cell_height)
        draw.line([(0, y), (width - 1, y)], fill=color, width=1)

    # Composite the grid overlay onto the image
    result = Image.alpha_composite(result, overlay)

    # Convert back to RGB for compatibility
    return result.convert("RGB")


# =============================================================================
# Canvas State
# =============================================================================


class CanvasState:
    """Stateful canvas with undo support for V2 workflow."""

    def __init__(self, device: torch.device | None = None, canvas_size: int = DEFAULT_CANVAS_SIZE):
        if device is None:
            device = torch.device("cpu")
        self.device = device
        self.canvas_size = canvas_size
        self.strokes: list[dict] = []
        self._canvas = create_canvas(device, size=canvas_size)
        self._turn_checkpoint: list[dict] | None = None
        self._initial_image: torch.Tensor | None = None

    def set_initial_image(self, image: Image.Image) -> None:
        """Initialize canvas with an existing image instead of blank white.

        This is used for tasks like image completion where the canvas
        starts with a pre-loaded image.
        """
        self._initial_image = image_to_tensor(image, self.device, self.canvas_size)
        self._canvas = self._initial_image.clone()

    def add_stroke(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        r: int,
        g: int,
        b: int,
        width: int,
        rounded: bool = True,
    ):
        """Add a stroke with RGB color."""
        self.strokes.append(
            dict(
                type="line",
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                r=r,
                g=g,
                b=b,
                width=width,
                rounded=rounded,
            )
        )
        self._canvas = draw_line(self._canvas, x1, y1, x2, y2, r, g, b, width, rounded=rounded)

    def add_curve(
        self,
        x1: int,
        y1: int,
        cx1: int,
        cy1: int,
        cx2: int,
        cy2: int,
        x2: int,
        y2: int,
        r: int,
        g: int,
        b: int,
        width: int,
    ):
        """Add a cubic Bézier curve with RGB color."""
        self.strokes.append(
            dict(
                type="curve",
                x1=x1,
                y1=y1,
                cx1=cx1,
                cy1=cy1,
                cx2=cx2,
                cy2=cy2,
                x2=x2,
                y2=y2,
                r=r,
                g=g,
                b=b,
                width=width,
            )
        )
        self._canvas = draw_curve(self._canvas, x1, y1, cx1, cy1, cx2, cy2, x2, y2, r, g, b, width)

    def undo_stroke(self) -> bool:
        if not self.strokes:
            return False
        self.strokes.pop()
        self._rebuild_canvas()
        return True

    def reset(self):
        self.strokes = []
        if self._initial_image is not None:
            self._canvas = self._initial_image.clone()
        else:
            self._canvas = create_canvas(self.device, size=self.canvas_size)
        self._turn_checkpoint = None

    def checkpoint(self) -> None:
        """Save current state before a turn (for multi-move mode)."""
        self._turn_checkpoint = list(self.strokes)

    def undo_turn(self) -> bool:
        """Revert to checkpoint, undoing all actions from the previous turn."""
        if self._turn_checkpoint is not None:
            self.strokes = self._turn_checkpoint
            self._turn_checkpoint = None
            self._rebuild_canvas()
            return True
        return False

    def _rebuild_canvas(self):
        if self._initial_image is not None:
            self._canvas = self._initial_image.clone()
        else:
            self._canvas = create_canvas(self.device, size=self.canvas_size)
        for s in self.strokes:
            stroke_type = s.get("type", "line")  # Default to line for backwards compatibility
            if stroke_type == "line":
                self._canvas = draw_line(
                    self._canvas,
                    s["x1"],
                    s["y1"],
                    s["x2"],
                    s["y2"],
                    s["r"],
                    s["g"],
                    s["b"],
                    s["width"],
                    s.get("rounded", True),
                )
            elif stroke_type == "curve":
                self._canvas = draw_curve(
                    self._canvas,
                    s["x1"],
                    s["y1"],
                    s["cx1"],
                    s["cy1"],
                    s["cx2"],
                    s["cy2"],
                    s["x2"],
                    s["y2"],
                    s["r"],
                    s["g"],
                    s["b"],
                    s["width"],
                )

    def get_canvas(self) -> torch.Tensor:
        return self._canvas.clone()

    def render_display(self) -> torch.Tensor:
        return self._canvas.clone()

    def to_image(self) -> Image.Image:
        return canvas_to_image(self.render_display())
