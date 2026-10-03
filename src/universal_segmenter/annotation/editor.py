"""Native-resolution Tk mask editor ported from the reconstruction project."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

def boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))


def create_overlay(rgb: np.ndarray, mask: np.ndarray,
                   proposal: np.ndarray | None = None) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    result[mask] = 0.86 * result[mask] + 0.14 * np.asarray((25, 220, 45))
    if proposal is not None:
        proposal_edge = ndimage.binary_dilation(boundary(proposal), iterations=1)
        result[proposal_edge] = (255, 40, 210)
    current_edge = ndimage.binary_dilation(boundary(mask), iterations=1)
    result[current_edge] = (20, 255, 35)
    return np.clip(result, 0, 255).astype(np.uint8)


def regularize_paper_mask(mask: np.ndarray, closing_iterations: int = 1,
                          keep_largest: bool = False) -> np.ndarray:
    """Close tiny contour gaps and fill the resulting paper region efficiently.

    Work is limited to the foreground bounding box. This gives the useful
    closed-loop behavior without repeatedly processing the full native scan
    while the pointer is moving.
    """
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("regularize_paper_mask expects a 2-D mask")
    if closing_iterations < 0:
        raise ValueError("closing_iterations cannot be negative")
    ys, xs = np.nonzero(value)
    if not xs.size:
        return value.copy()

    padding = max(2, closing_iterations + 1)
    top = max(0, int(ys.min()) - padding)
    bottom = min(value.shape[0], int(ys.max()) + padding + 1)
    left = max(0, int(xs.min()) - padding)
    right = min(value.shape[1], int(xs.max()) + padding + 1)
    region = value[top:bottom, left:right].copy()
    structure = np.ones((3, 3), dtype=bool)
    if closing_iterations:
        region = ndimage.binary_closing(
            region, structure=structure, iterations=closing_iterations)
    if keep_largest:
        labels, count = ndimage.label(region, structure=structure)
        if count:
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            region = labels == int(np.argmax(areas))
    region = ndimage.binary_fill_holes(region).astype(bool)
    result = np.zeros_like(value)
    result[top:bottom, left:right] = region
    return result


class MaskEditor:
    """Correct an initial proposal without resampling the saved mask."""

    def __init__(self, rgb: np.ndarray, mask: np.ndarray,
                 proposal: np.ndarray, title: str,
                 display_max_side: int = 1150,
                 initial_brush_radius: int = 12,
                 auto_regularize: bool = True,
                 regularize_closing_iterations: int = 1,
                 proposal_key: str = "proposal",
                 proposal_name: str = "Proposal",
                 proposal_choices: Sequence[tuple[str, str]] | None = None,
                 proposal_loader: Callable[[str], np.ndarray] | None = None,
                 background_status: Callable[[], str] | None = None,
                 dataset_split: str = "train",
                 dataset_split_choices: Sequence[tuple[str, str]] | None = None,
                 ) -> None:
        if rgb.shape[:2] != mask.shape or mask.shape != proposal.shape:
            raise ValueError("RGB, current mask, and proposal dimensions must match")
        self.rgb = rgb
        self.mask = mask.copy()
        self.proposal = proposal.copy()
        self.selected_proposal_key = proposal_key
        self.selected_proposal_name = proposal_name
        self.proposal_choices = list(
            proposal_choices or [(proposal_key, proposal_name)])
        self.proposal_loader = proposal_loader
        self.background_status = background_status
        self.selected_dataset_split = dataset_split
        self.dataset_split_choices = list(dataset_split_choices or [
            ("train", "Train"),
            ("validation", "Validation"),
            ("calibration", "Calibration"),
            ("holdout", "Final test"),
        ])
        if dataset_split not in dict(self.dataset_split_choices):
            raise ValueError(f"unknown initial dataset split: {dataset_split}")
        self.title = title
        self.display_max_side = display_max_side
        self.radius = initial_brush_radius
        self.auto_regularize = auto_regularize
        self.regularize_closing_iterations = regularize_closing_iterations
        self.last_regularize_ms = 0.0
        self.show_proposal = True
        self.show_fill = True
        self.undo: list[np.ndarray] = []
        self.paint_value = True
        self.base_scale = self.zoom = self.scale = 1.0
        self.view_width = rgb.shape[1]
        self.view_height = rgb.shape[0]
        self.offset_x = self.offset_y = 0.0
        self.last_paint_point: tuple[float, float] | None = None
        self.pan_origin: tuple[int, int, float, float] | None = None
        self.action = "quit"
        self.root = self.canvas = self.canvas_image = None
        self.brush_outline = self.photo = self.status = self.redraw_job = None
        self.proposal_boundary = ndimage.binary_dilation(boundary(self.proposal), iterations=1)
        self.mask_fraction = float(self.mask.mean())
        self.mask_fraction_dirty = False
        self._view_cache_key = None
        self._base_view: np.ndarray | None = None
        self._proposal_display: np.ndarray | None = None
        self._mask_display: np.ndarray | None = None

    def _status_text(self) -> str:
        background = ""
        if self.background_status is not None:
            try:
                background = "\nCache: " + self.background_status()
            except Exception as error:
                background = f"\nCache status unavailable: {error}"
        return (
            f"Guess: {self.selected_proposal_name}    "
            f"Dataset: {dict(self.dataset_split_choices)[self.selected_dataset_split]}    "
            f"Zoom: {self.zoom:.2f}x    Brush: {self.radius}px    "
            f"Mask area: {self.mask_fraction:.1%}    "
            f"Auto-close: {'ON' if self.auto_regularize else 'OFF'}"
            + (f" ({self.last_regularize_ms:.0f} ms)    "
               if self.auto_regularize and self.last_regularize_ms else "    ")
            + "Green = corrected edge    Magenta = selected guess edge"
            + background
        )

    def _select_dataset_split(self, split: str) -> None:
        if split not in dict(self.dataset_split_choices):
            return
        self.selected_dataset_split = split
        if self.status is not None:
            self.status.configure(text=self._status_text())

    def _refresh_status(self) -> None:
        if self.root is None:
            return
        self.status.configure(text=self._status_text())
        self.root.after(750, self._refresh_status)

    def _replace_with_proposal(self, key: str) -> None:
        labels = dict(self.proposal_choices)
        if key not in labels:
            return
        if self.proposal_loader is None:
            value = self.proposal.copy()
        else:
            if self.status is not None:
                self.status.configure(text=f"Loading {labels[key]} guess...")
            if self.root is not None:
                self.root.update_idletasks()
            try:
                value = np.asarray(self.proposal_loader(key), dtype=bool)
            except Exception as error:
                if self.root is not None:
                    from tkinter import messagebox
                    messagebox.showerror(
                        "Could not load guess", str(error), parent=self.root)
                if self.status is not None:
                    self.status.configure(text=self._status_text())
                return
        if value.shape != self.mask.shape:
            raise ValueError(
                f"replacement proposal shape {value.shape} != {self.mask.shape}")
        self._push_undo()
        self.selected_proposal_key = key
        self.selected_proposal_name = labels[key]
        self.proposal = value.copy()
        self.proposal_boundary = ndimage.binary_dilation(
            boundary(self.proposal), iterations=1)
        self.mask = self.proposal.copy()
        self._view_cache_key = None
        self._base_view = None
        self._proposal_display = None
        self._mask_changed()
        self._finish_mask_change()
        self._schedule_redraw(immediate=True)

    def _mask_changed(self) -> None:
        self._mask_display = None
        self.mask_fraction_dirty = True

    def _finish_mask_change(self) -> None:
        if self.mask_fraction_dirty:
            self.mask_fraction = float(self.mask.mean())
            self.mask_fraction_dirty = False

    def _push_undo(self) -> None:
        self.undo.append(self.mask.copy())
        if len(self.undo) > 30:
            self.undo.pop(0)

    def _event_to_source(self, x: int, y: int) -> tuple[float, float]:
        return self.offset_x + x / self.scale, self.offset_y + y / self.scale

    def _paint_source(self, x: float, y: float) -> None:
        radius = max(1.0, self.radius / self.scale)
        left, right = max(0, int(x - radius)), min(self.mask.shape[1], int(x + radius) + 2)
        top, bottom = max(0, int(y - radius)), min(self.mask.shape[0], int(y + radius) + 2)
        if left >= right or top >= bottom:
            return
        yy, xx = np.ogrid[top:bottom, left:right]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        self.mask[top:bottom, left:right][disk] = self.paint_value
        self._mask_changed()

    def _paint_to(self, x: float, y: float) -> None:
        if self.last_paint_point is None:
            self._paint_source(x, y)
        else:
            last_x, last_y = self.last_paint_point
            distance = float(np.hypot(x - last_x, y - last_y))
            radius = max(1.0, self.radius / self.scale)
            steps = max(1, int(np.ceil(distance / max(0.75, radius * 0.45))))
            for step in range(1, steps + 1):
                fraction = step / steps
                self._paint_source(last_x + (x - last_x) * fraction,
                                   last_y + (y - last_y) * fraction)
        self.last_paint_point = (x, y)

    def _begin_stroke(self, event, value: bool) -> None:
        self._push_undo()
        self.paint_value = value
        self.last_paint_point = None
        self._paint_to(*self._event_to_source(event.x, event.y))
        self._schedule_redraw()

    def _continue_stroke(self, event, value: bool) -> None:
        self.paint_value = value
        self._paint_to(*self._event_to_source(event.x, event.y))
        self._schedule_redraw()

    def _end_stroke(self, _event=None) -> None:
        self.last_paint_point = None
        if self.auto_regularize and self.mask_fraction_dirty:
            started = perf_counter()
            self.mask = regularize_paper_mask(
                self.mask,
                closing_iterations=self.regularize_closing_iterations,
                keep_largest=False,
            )
            self.last_regularize_ms = (perf_counter() - started) * 1000.0
            self._mask_changed()
        self._finish_mask_change()
        self._schedule_redraw(immediate=True)

    def _clamp_view(self) -> None:
        maximum_x = max(0.0, self.mask.shape[1] - self.view_width / self.scale)
        maximum_y = max(0.0, self.mask.shape[0] - self.view_height / self.scale)
        self.offset_x = float(np.clip(self.offset_x, 0.0, maximum_x))
        self.offset_y = float(np.clip(self.offset_y, 0.0, maximum_y))

    def _display_view(self) -> np.ndarray:
        visible_width = self.view_width / self.scale
        visible_height = self.view_height / self.scale
        left, top = int(np.floor(self.offset_x)), int(np.floor(self.offset_y))
        right = min(self.rgb.shape[1], int(np.ceil(self.offset_x + visible_width)))
        bottom = min(self.rgb.shape[0], int(np.ceil(self.offset_y + visible_height)))
        left, top = max(0, left), max(0, top)
        key = (left, top, right, bottom, self.view_width, self.view_height)
        if key != self._view_cache_key:
            image_crop = Image.fromarray(self.rgb[top:bottom, left:right]).resize(
                (self.view_width, self.view_height), Image.Resampling.BILINEAR)
            proposal_crop = Image.fromarray(
                self.proposal_boundary[top:bottom, left:right].astype(np.uint8) * 255
            ).resize((self.view_width, self.view_height), Image.Resampling.NEAREST)
            self._base_view = np.array(image_crop, dtype=np.uint8, copy=True)
            self._proposal_display = np.asarray(proposal_crop) >= 128
            self._mask_display = None
            self._view_cache_key = key
        if self._mask_display is None:
            mask_crop = Image.fromarray(
                self.mask[top:bottom, left:right].astype(np.uint8) * 255).resize(
                    (self.view_width, self.view_height), Image.Resampling.NEAREST)
            self._mask_display = np.asarray(mask_crop) >= 128
        view = self._base_view.copy()
        display_mask = self._mask_display
        if self.show_fill:
            blended = view[display_mask].astype(np.float32)
            view[display_mask] = np.clip(
                0.86 * blended + 0.14 * np.asarray((25, 220, 45)), 0, 255).astype(np.uint8)
        if self.show_proposal:
            view[self._proposal_display] = (255, 40, 210)
        view[ndimage.binary_dilation(boundary(display_mask), iterations=1)] = (20, 255, 35)
        return view

    def _schedule_redraw(self, immediate: bool = False) -> None:
        if self.root is not None and self.redraw_job is None:
            self.redraw_job = self.root.after(0 if immediate else 30, self._redraw)

    def _redraw(self) -> None:
        from PIL import ImageTk
        self.redraw_job = None
        self.photo = ImageTk.PhotoImage(Image.fromarray(self._display_view()))
        self.canvas.itemconfigure(self.canvas_image, image=self.photo)
        self.status.configure(text=self._status_text())

    def _cursor(self, event) -> None:
        if self.brush_outline is not None:
            radius = self.radius
            self.canvas.coords(self.brush_outline, event.x - radius, event.y - radius,
                               event.x + radius, event.y + radius)

    def _zoom_at(self, x: int, y: int, direction: int) -> None:
        source_x, source_y = self._event_to_source(x, y)
        new_zoom = float(np.clip(self.zoom * (1.25 if direction > 0 else 0.8), 1.0, 12.0))
        if new_zoom == self.zoom:
            return
        self.zoom, self.scale = new_zoom, self.base_scale * new_zoom
        self.offset_x = source_x - x / self.scale
        self.offset_y = source_y - y / self.scale
        self._clamp_view()
        self._schedule_redraw(immediate=True)

    def _mouse_wheel(self, event) -> str:
        self._zoom_at(event.x, event.y, 1 if event.delta > 0 else -1)
        return "break"

    def _begin_pan(self, event) -> None:
        self.pan_origin = event.x, event.y, self.offset_x, self.offset_y

    def _continue_pan(self, event) -> None:
        if self.pan_origin is None:
            return
        start_x, start_y, offset_x, offset_y = self.pan_origin
        self.offset_x = offset_x - (event.x - start_x) / self.scale
        self.offset_y = offset_y - (event.y - start_y) / self.scale
        self._clamp_view()
        self._schedule_redraw()

    def _reset_view(self) -> None:
        self.zoom, self.scale = 1.0, self.base_scale
        self.offset_x = self.offset_y = 0.0
        self._clamp_view()
        self._schedule_redraw(immediate=True)

    def _pan_key(self, horizontal: float, vertical: float) -> None:
        self.offset_x += horizontal * self.view_width / self.scale
        self.offset_y += vertical * self.view_height / self.scale
        self._clamp_view()
        self._schedule_redraw(immediate=True)

    def _keep_largest(self) -> None:
        self._push_undo()
        labels, count = ndimage.label(self.mask)
        if count:
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            self.mask = labels == int(np.argmax(areas))
        self._mask_changed()
        self._finish_mask_change()
        self._schedule_redraw()

    def _fill_holes(self) -> None:
        self._push_undo()
        self.mask = ndimage.binary_fill_holes(self.mask).astype(bool)
        self._mask_changed()
        self._finish_mask_change()
        self._schedule_redraw()

    def _finish(self, action: str) -> None:
        if action == "save" and self.auto_regularize:
            self.mask = regularize_paper_mask(
                self.mask,
                closing_iterations=self.regularize_closing_iterations,
                keep_largest=True,
            )
            self._mask_changed()
            self._finish_mask_change()
        self.action = action
        if self.root is not None:
            self.root.quit()

    def _key(self, event) -> str:
        key = event.keysym.lower()
        if key in ("s", "return"):
            self._finish("save")
        elif key in ("q", "escape"):
            self._finish("quit")
        elif key in ("u", "z") and self.undo:
            self.mask = self.undo.pop()
            self._mask_changed(); self._finish_mask_change()
            self._schedule_redraw()
        elif key == "r":
            self._push_undo(); self.mask = self.proposal.copy()
            self._mask_changed(); self._finish_mask_change(); self._schedule_redraw()
        elif key in {str(index) for index in range(1, 10)}:
            index = int(key) - 1
            if index < len(self.proposal_choices):
                self._replace_with_proposal(self.proposal_choices[index][0])
        elif key == "k":
            self._keep_largest()
        elif key == "f":
            self._fill_holes()
        elif key == "a":
            self.auto_regularize = not self.auto_regularize
            self._schedule_redraw(immediate=True)
        elif key == "m":
            self.show_proposal = not self.show_proposal; self._schedule_redraw()
        elif key == "o":
            self.show_fill = not self.show_fill; self._schedule_redraw()
        elif key == "t":
            self._select_dataset_split("train")
        elif key == "v":
            self._select_dataset_split("validation")
        elif key == "c":
            self._select_dataset_split("calibration")
        elif key == "h":
            self._select_dataset_split("holdout")
        elif key in ("0", "home"):
            self._reset_view()
        elif key in ("left", "right", "up", "down"):
            moves = {"left": (-0.15, 0), "right": (0.15, 0),
                     "up": (0, -0.15), "down": (0, 0.15)}
            self._pan_key(*moves[key])
        elif key in ("bracketleft", "minus"):
            self.radius = max(1, self.radius - 2); self._schedule_redraw()
        elif key in ("bracketright", "plus", "equal"):
            self.radius = min(100, self.radius + 2); self._schedule_redraw()
        return "break"

    def run(self) -> tuple[str, np.ndarray]:
        try:
            import tkinter as tk
            from tkinter import ttk
            from PIL import ImageTk  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "The annotation editor needs Tkinter and Pillow ImageTk. "
                "On Windows, run `python -m tkinter` to verify the installation.") from error

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._finish("quit"))
        height, width = self.rgb.shape[:2]
        available_width = max(320, self.root.winfo_screenwidth() - 80)
        available_height = max(240, self.root.winfo_screenheight() - 260)
        self.base_scale = min(1.0, self.display_max_side / max(height, width),
                              available_width / width, available_height / height)
        self.zoom = 1.0
        self.scale = self.base_scale
        self.view_width = max(1, round(width * self.base_scale))
        self.view_height = max(1, round(height * self.base_scale))

        ttk.Label(
            self.root,
            text=("LEFT drag: add paper     RIGHT drag: erase background     "
                  "S/Enter: save and next     Q/Esc: quit safely\n"
                  "WHEEL: zoom at cursor     MIDDLE drag: pan     0: fit view     "
                  "U/Z: undo     R: reset to selected guess     "
                  "R: reset to the SegFormer guess\n"
                  "K: keep largest     F: fill holes     [ ]: brush size     "
                  "A: auto-close on/off     M: proposal edge     O: fill\n"
                  "T: train     V: validation     C: calibration     "
                  "H: final test"),
            justify="left", padding=(8, 7)).pack(fill="x")
        if len(self.proposal_choices) > 1:
            proposal_bar = ttk.Frame(self.root, padding=(8, 2))
            proposal_bar.pack(fill="x")
            ttk.Label(proposal_bar, text="Replace mask with:").pack(side="left")
            for index, (key, label) in enumerate(self.proposal_choices, start=1):
                ttk.Button(
                    proposal_bar,
                    text=f"{index}: {label}",
                    command=lambda selected=key: self._replace_with_proposal(selected),
                ).pack(side="left", padx=(7, 0))
        split_bar = ttk.Frame(self.root, padding=(8, 2))
        split_bar.pack(fill="x")
        ttk.Label(split_bar, text="Use this physical piece for:").pack(
            side="left")
        for key, label in self.dataset_split_choices:
            ttk.Button(
                split_bar,
                text=f"{label} ({key[0].upper()})",
                command=lambda selected=key: self._select_dataset_split(selected),
            ).pack(side="left", padx=(7, 0))
        self.status = ttk.Label(self.root, text="", padding=(8, 3))
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(
            self.root, width=self.view_width, height=self.view_height,
            highlightthickness=0, cursor="crosshair", background="#222222")
        self.canvas.pack()
        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")
        self.brush_outline = self.canvas.create_oval(
            -20, -20, -10, -10, outline="#ffff00", width=1)
        self.canvas.bind("<Button-1>", lambda event: self._begin_stroke(event, True))
        self.canvas.bind("<B1-Motion>", lambda event: self._continue_stroke(event, True))
        self.canvas.bind("<ButtonRelease-1>", self._end_stroke)
        self.canvas.bind("<Button-3>", lambda event: self._begin_stroke(event, False))
        self.canvas.bind("<B3-Motion>", lambda event: self._continue_stroke(event, False))
        self.canvas.bind("<ButtonRelease-3>", self._end_stroke)
        self.canvas.bind("<Button-2>", self._begin_pan)
        self.canvas.bind("<B2-Motion>", self._continue_pan)
        self.canvas.bind("<ButtonRelease-2>", lambda _event: setattr(self, "pan_origin", None))
        self.canvas.bind("<Motion>", self._cursor, add="+")
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event.x, event.y, 1))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event.x, event.y, -1))
        self.root.bind("<KeyPress>", self._key)
        self._redraw()
        self.root.after(750, self._refresh_status)
        self.root.update_idletasks()
        self.root.focus_force()
        try:
            self.root.mainloop()
        finally:
            self.root.destroy()
            self.root = None
        return self.action, self.mask
