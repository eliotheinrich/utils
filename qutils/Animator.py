from qutils.qutils_bindings import QUTILS_BUILT_WITH_GLFW
from time import time, sleep

if QUTILS_BUILT_WITH_GLFW:
    from qutils.qutils_bindings import GLFWAnimator
    class SimulatorAnimator:
        def __init__(self, simulator, background_color=None, fps=60, steps_per_frame=1):
            self.simulator = simulator
            self.fps = fps
            self.steps_per_frame = steps_per_frame
            if background_color is None:
                self.background_color = [0.0, 0.0, 0.0, 0.0]
            else:
                self.background_color = background_color

            self.animator = GLFWAnimator(*self.background_color)

        def start(self, width, height):
            self.animator.start(width, height)

            while True:
                t1 = time()
                if not self.animator.is_paused():
                    self.simulator.timesteps(self.steps_per_frame)
                data, n, m = self.simulator.get_texture()
                frame_data = self.animator.new_frame(data, n, m)
                keys = frame_data.keys
                for key in keys:
                    self.simulator.key_callback(key)
                t2 = time()

                dt = t2 - t1
                target_dt = 1.0/self.fps

                if dt < target_dt:
                    sleep(target_dt - dt)

                if frame_data.status_code == 0:
                    break
