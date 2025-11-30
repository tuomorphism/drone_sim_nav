from manim import *
import numpy as np


class DroneFlightScene(ThreeDScene):
    data_file: str = "default_data_file.npz"

    def construct(self):
        # ---- Load simulation data ----
        data = np.load(self.data_file)
        traj = data["trajectory"]            # (N, 3)
        vel = data["velocity"]               # (N, 3)
        timesteps = data["timesteps"]        # (N,)
        obstacles = data.get("obstacles", np.zeros((0, 4)))
        goal_traj = data["goal_traj"]        # (N, 3)

        N = traj.shape[0]
        t_start, t_end = float(timesteps[0]), float(timesteps[-1])
        t_tracker = ValueTracker(t_start)

        def sample_array_time(arr, t: float):
            t = np.clip(t, t_start, t_end)
            i = np.searchsorted(timesteps, t) - 1
            i = int(np.clip(i, 0, N - 2))

            t0, t1 = timesteps[i], timesteps[i + 1]
            if t1 == t0:
                frac = 0.0
            else:
                frac = float((t - t0) / (t1 - t0))

            return (1.0 - frac) * arr[i] + frac * arr[i + 1]

        vel_norm = np.linalg.norm(vel, axis=1)
        max_vel = np.max(vel_norm) if np.max(vel_norm) > 1e-6 else 1.0
        vel_scale = 0.5 / max_vel

        x_min, x_max = traj[:, 0].min() - 0.5, traj[:, 0].max() + 0.5
        y_min, y_max = traj[:, 1].min() - 0.5, traj[:, 1].max() + 0.5
        z_min, z_max = traj[:, 2].min() - 0.2, traj[:, 2].max() + 0.5
        axes = ThreeDAxes(
            x_range=[x_min, x_max, 0.5],
            y_range=[y_min, y_max, 0.5],
            z_range=[z_min, z_max, 0.5],
            x_length=7,
            y_length=7,
            z_length=4,
        )

        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES, zoom = 0.9, frame_center=[0, 0, 0.5])
        self.add(axes)

        path = VMobject()
        path.set_points_as_corners([axes.c2p(*p) for p in traj])
        path.set_stroke(color=BLUE, width=3)
        self.add(path)

        def drone_mobject():
            t = t_tracker.get_value()
            pos = sample_array_time(traj, t)
            s = Sphere(radius=0.05, color=YELLOW)
            s.move_to(axes.c2p(*pos))
            return s

        drone = always_redraw(drone_mobject)



        # ---- Goal ----
        def goal_mobject():
            t = t_tracker.get_value()
            g = sample_array_time(goal_traj, t)
            m = Sphere(radius=0.07, color=RED)
            m.set_color(GREEN)
            m.move_to(axes.c2p(*g))
            return m

        goal_marker = always_redraw(goal_mobject)

        # ---- Obstacles ----
        z_low, z_high = z_min, z_max
        obstacle_mobs = []
        for x1, x2, y1, y2 in obstacles:
            width = x2 - x1
            depth = y2 - y1
            height = z_high - z_low

            center_world = np.array([
                0.5 * (x1 + x2),
                0.5 * (y1 + y2),
                0.5 * (z_low + z_high),
            ])

            box = Prism(
                dimensions=(width, depth, height),
                fill_color=GREEN,
                fill_opacity=0.3,
                stroke_width=0.5,
            )
            box.move_to(axes.c2p(*center_world))
            obstacle_mobs.append(box)

        self.add(drone, goal_marker, *obstacle_mobs)
        self.begin_ambient_camera_rotation(rate=0.1)

        self.play(
            t_tracker.animate.set_value(t_end),
            run_time=t_end - t_start,
            rate_func=linear,
        )
