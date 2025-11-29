from manim import *
import numpy as np

DATA_FILE = "./assets/drone_sim_data.npz"


class DroneFlightScene(ThreeDScene):
    def construct(self):
        # ---- Load simulation data ----
        data = np.load(DATA_FILE)
        traj = data["trajectory"]            # (N, 3)
        vel = data["velocity"]               # (N, 3)
        timesteps = data["timesteps"]        # (N,)
        obstacles = data.get("obstacles", np.zeros((0, 4)))

        # Moving goal if present, otherwise use a static one
        if "goal_traj" in data:
            goal_traj = data["goal_traj"]    # (N, 3)
        else:
            # fallback: constant goal at first sample
            goal_traj = np.repeat(traj[-1][None, :], len(traj), axis=0)

        N = traj.shape[0]

        # ---- Helpers: interpolate arrays by alpha in [0,1] ----
        def sample_array(arr, alpha):
            """
            arr: shape (N, 3)
            alpha in [0, 1]
            """
            t = np.clip(alpha, 0.0, 1.0) * (N - 1)
            i = int(np.floor(t))
            if i >= N - 1:
                return arr[-1]
            frac = t - i
            return (1.0 - frac) * arr[i] + frac * arr[i + 1]

        # Precompute a nice scale for the velocity arrow
        vel_norm = np.linalg.norm(vel, axis=1)
        max_vel = np.max(vel_norm) if np.max(vel_norm) > 1e-6 else 1.0
        vel_scale = 0.5 / max_vel   # tweak this if arrow is too small/large

        # ---- Axes & camera ----
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

        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)

        # ---- Path curve ----
        path = VMobject()
        path.set_points_as_corners([axes.c2p(*p) for p in traj])
        path.set_stroke(color=BLUE, width=3)
        self.add(path)

        # ---- Time parameter ----
        alpha = ValueTracker(timesteps[0])

        # ---- Drone (moving point) ----
        def drone_mobject():
            a = alpha.get_value()
            pos = sample_array(traj, a)
            s = Sphere(radius=0.05, color=YELLOW)
            s.move_to(axes.c2p(*pos))
            return s

        drone = always_redraw(drone_mobject)

        # ---- Velocity arrow ----
        def vel_arrow_mobject():
            a = alpha.get_value()
            pos = sample_array(traj, a)
            v = sample_array(vel, a)
            v_scaled = v * vel_scale

            start = axes.c2p(*pos)
            end = axes.c2p(*(pos + v_scaled))
            # buff=0 matches the drone surface more or less, you can tweak
            return Arrow3D(
                start=start,
                end=end,
                thickness=0.015,
                max_tip_length_to_length_ratio=0.2,
                max_stroke_width_to_length_ratio=4,
                color=ORANGE,
            )

        vel_arrow = always_redraw(vel_arrow_mobject)

        # ---- Goal marker (moving or static) ----
        def goal_mobject():
            a = alpha.get_value()
            g = sample_array(goal_traj, a)
            m = Sphere(radius=0.07, color=RED)
            m.move_to(axes.c2p(*g))
            return m

        goal_marker = always_redraw(goal_mobject)

        # ---- Obstacles as tall boxes ----
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

        # ---- Add everything ----
        self.add(drone, vel_arrow, goal_marker, *obstacle_mobs)

        # Optional camera motion
        self.begin_ambient_camera_rotation(rate=0.1)

        # ---- Animate alpha from 0 -> 1 (full flight) ----
        self.play(
            alpha.animate.set_value(timesteps[-1]),
            run_time=8,
            rate_func=linear,
        )

        self.wait(1)
