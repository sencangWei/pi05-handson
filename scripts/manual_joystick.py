"""Small hold-to-move screen-plane joystick; callbacks own all simulation work."""
import math
import sys
import tkinter as tk
from tkinter import ttk


def translation_vector(dx, dy, radius=55):
    """Map canvas offsets to unit-circle XY: right positive X, up positive Y."""
    if not all(math.isfinite(value) for value in (dx, dy, radius)) or radius <= 0:
        raise ValueError("Joystick offsets must be finite and radius positive")
    length = math.hypot(dx, dy)
    if length <= radius * .05:
        return 0., 0.
    scale = max(radius, length)
    return dx / scale, -dy / scale


class TranslationJoystick(ttk.Frame):
    """on_move(x,y): True continues at 20 Hz; False or an exception stops.

    Starts disabled. The owner must call set_enabled(False) when leaving its
    control mode. Unmapping, focus loss, Escape, release and destruction stop it.
    """
    def __init__(self, parent, on_move, on_error=None):
        super().__init__(parent)
        self.on_move, self.on_error = on_move, on_error
        self._enabled = False
        self._held = False
        self._after_id = None
        self._vector = (0., 0.)
        self.canvas = tk.Canvas(self, width=150, height=150, highlightthickness=0, takefocus=True)
        self.canvas.pack()
        self.canvas.create_oval(20, 20, 130, 130, outline="#777777", width=2)
        self.canvas.create_line(20, 75, 130, 75, fill="#bbbbbb")
        self.canvas.create_line(75, 20, 75, 130, fill="#bbbbbb")
        self._thumb = self.canvas.create_oval(65, 65, 85, 85, fill="#999999", outline="")
        ttk.Label(self, text="按住拖动平移，松手即停；Z仍用上下按钮", wraplength=210).pack()
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self.stop)
        self.canvas.bind("<Escape>", self.stop)
        self.canvas.bind("<FocusOut>", self.stop)
        self.bind("<Unmap>", self.stop)
        self.bind("<Destroy>", self.stop)

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if not self._enabled:
            self.stop()
        self.canvas.itemconfigure(self._thumb, fill="#3878bd" if self._enabled else "#999999")

    def _press(self, event):
        if not self._enabled:
            return
        self.stop()
        try:
            self.canvas.focus_set()
            self.canvas.grab_set()
            self._held = True
            self._motion(event)
        except Exception as error:
            self._fail(error)

    def _motion(self, event):
        if not (self._enabled and self._held):
            return
        self._vector = translation_vector(event.x - 75, event.y - 75)
        x, y = self._vector
        self.canvas.coords(self._thumb, 65 + 55*x, 65 - 55*y, 85 + 55*x, 85 - 55*y)
        if x == 0 and y == 0:
            self._cancel_tick()
        else:
            self._schedule()

    def _schedule(self):
        if self._enabled and self._held and self._vector != (0., 0.) and self._after_id is None:
            self._after_id = self.after(50, self._tick)

    def _tick(self):
        self._after_id = None
        if not (self._enabled and self._held) or self._vector == (0., 0.):
            return
        try:
            if not self.on_move(*self._vector):
                self.stop()
                return
        except Exception as error:
            self._fail(error)
            return
        self._schedule()

    def _cancel_tick(self):
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None

    def stop(self, _event=None):
        self._held = False
        self._vector = (0., 0.)
        self._cancel_tick()
        try:
            # Native ttk popdowns have Tcl paths but no Python Widget wrapper.
            current = self.canvas.tk.call("grab", "current", str(self.canvas))
            if str(current) == str(self.canvas):
                self.canvas.grab_release()
            self.canvas.coords(self._thumb, 65, 65, 85, 85)
        except tk.TclError:
            pass  # Destruction may already have removed the canvas.

    def _fail(self, error):
        self.stop()
        if self.on_error is not None:
            try:
                self.on_error(error)
                return
            except Exception as reporting_error:
                print(f"Joystick error handler failed: {reporting_error}", file=sys.stderr)
        print(f"Joystick stopped: {error}", file=sys.stderr)

    def destroy(self):
        self.stop()
        super().destroy()
