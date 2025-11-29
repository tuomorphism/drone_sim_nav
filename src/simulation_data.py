from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class DroneSimData:
    """
    Canonical save format for the drone → Manim pipeline.

    Shapes:
        trajectory: (N, 3)   drone positions
        velocity:   (N, 3)   drone velocities
        timesteps:  (N,)     simulation times
        goal_traj:  (N, 3)   goal positions (can be static or moving)
        obstacles:  (M, 4)   rows [x_min, x_max, y_min, y_max]
    """
    trajectory: np.ndarray
    velocity: np.ndarray
    timesteps: np.ndarray
    goal_traj: np.ndarray
    obstacles: np.ndarray

    @property
    def n_steps(self) -> int:
        return self.trajectory.shape[0]

    @property
    def duration(self) -> float:
        return float(self.timesteps[-1] - self.timesteps[0])

    def to_npz(self, path: str) -> None:
        """
        Save the simulation data to an .npz file.
        """
        np.savez(
            path,
            trajectory=self.trajectory,
            velocity=self.velocity,
            timesteps=self.timesteps,
            goal_traj=self.goal_traj,
            obstacles=self.obstacles,
        )

    @classmethod
    def from_npz(cls, path: str) -> "DroneSimData":
        """
        Load the simulation data from an .npz file.

        Falls back to:
          - static goal at final trajectory position if goal_traj missing
          - empty obstacle array if obstacles missing
        """
        data = np.load(path)

        traj = data["trajectory"]
        vel = data["velocity"]
        t = data["timesteps"]

        # goal trajectory: optional, default = constant at last trajectory point
        if "goal_traj" in data.files:
            goal_traj = data["goal_traj"]
        else:
            goal_traj = np.repeat(traj[-1][None, :], traj.shape[0], axis=0)

        # obstacles: optional, default = no obstacles
        if "obstacles" in data.files:
            obstacles = data["obstacles"]
        else:
            obstacles = np.zeros((0, 4))

        return cls(
            trajectory=traj,
            velocity=vel,
            timesteps=t,
            goal_traj=goal_traj,
            obstacles=obstacles,
        )
