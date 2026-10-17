"""MuJoCo's native viewer, with explicit pause and single-step controls."""
import time


class LivePreview:
    def __init__(self, env):
        import mujoco.viewer

        self.paused = True
        self.single_step = False
        self.quit_requested = False
        self.step_started = time.monotonic()
        self.control_dt = env.env.control_timestep
        self.handle = mujoco.viewer.launch_passive(
            env.sim.model._model, env.sim.data._data, key_callback=self.on_key,
        )
        with self.handle.lock():
            self.handle.cam.lookat[:] = [0, 0, 1.0]
            self.handle.cam.distance = 1.6
            self.handle.cam.azimuth = 130
            self.handle.cam.elevation = -30
        self.handle.sync()
        print("MuJoCo window ready: SPACE run/pause; N one control step; Q finish. Starts paused.", flush=True)

    def on_key(self, key):
        if key == 32:
            self.paused = not self.paused
        elif key == ord("N"):
            self.paused = True
            self.single_step = True
        elif key == ord("Q"):
            self.quit_requested = True

    def is_running(self):
        return not self.quit_requested and self.handle.is_running()

    def lock(self):
        return self.handle.lock()

    def before_step(self):
        while self.is_running():
            self.handle.sync()
            if not self.paused or self.single_step:
                self.single_step = False
                self.step_started = time.monotonic()
                return True
            time.sleep(0.02)
        return False

    def after_step(self):
        self.handle.sync()
        time.sleep(max(0, self.control_dt - (time.monotonic() - self.step_started)))

    def hold_final(self, outcome):
        print(f"Trial finished: {outcome}. Inspect the scene; close the window or press Q to save the result.", flush=True)
        while self.is_running():
            self.handle.sync()
            time.sleep(0.02)

    def close(self):
        self.handle.close()
