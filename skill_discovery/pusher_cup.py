"""A vectorized, non-physical Pusher-Cup environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TRAJECTORY_CLASSES = ("no_contact", "contact_without_inside", "ball_inside")


@dataclass(frozen=True)
class PusherCupConfig:
    """Configuration for a deterministic 2D contact-transfer world."""

    num_envs: int = 1024
    episode_length: int = 48
    world_low: float = -1.0
    world_high: float = 1.0
    action_scale: float = 0.055
    cup_center: tuple[float, float] = (0.35, 0.0)
    cup_axes: tuple[float, float] = (0.14, 0.16)
    pusher_start: tuple[float, float] = (-0.55, 0.0)
    ball_start: tuple[float, float] = (-0.25, 0.0)
    pusher_radius: float = 0.045
    ball_radius: float = 0.045
    contact_margin: float = 0.005
    transfer_ratio: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_envs <= 0 or self.episode_length <= 0:
            raise ValueError("num_envs and episode_length must be positive")
        if self.world_low >= self.world_high:
            raise ValueError("world_low must be smaller than world_high")
        if self.action_scale <= 0:
            raise ValueError("action_scale must be positive")
        if min(self.cup_axes) <= 0:
            raise ValueError("cup axes must be positive")
        if min(self.pusher_radius, self.ball_radius) <= 0:
            raise ValueError("object radii must be positive")
        if self.contact_margin < 0 or self.transfer_ratio <= 0:
            raise ValueError("contact margin must be non-negative and transfer ratio positive")


class PusherCupEnv:
    """Move a pusher directly; move the ball only through contact transfer.

    This environment intentionally does not approximate rigid-body physics. A
    pusher displacement transfers to the ball when the pusher reaches the ball
    while moving toward it. The rule is deterministic, vectorized, and easy to
    audit visually.
    """

    def __init__(self, config: PusherCupConfig | None = None):
        self.config = config or PusherCupConfig()
        self.rng = np.random.default_rng(self.config.seed)
        shape = (self.config.num_envs, 2)
        self.pusher_positions = np.empty(shape, dtype=np.float32)
        self.ball_positions = np.empty(shape, dtype=np.float32)
        self.cup_centers = np.broadcast_to(
            np.asarray(self.config.cup_center, dtype=np.float32), shape
        ).copy()
        self.cup_axes = np.broadcast_to(
            np.asarray(self.config.cup_axes, dtype=np.float32), shape
        ).copy()
        self.step_counts = np.zeros(self.config.num_envs, dtype=np.int32)
        self.ever_contact = np.zeros(self.config.num_envs, dtype=bool)
        self.current_contact = np.zeros(self.config.num_envs, dtype=bool)
        self.ball_path_length = np.zeros(self.config.num_envs, dtype=np.float32)
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        pusher_positions: np.ndarray | None = None,
        ball_positions: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pusher_positions = self._position_array(
            pusher_positions,
            self.config.pusher_start,
            "pusher_positions",
        )
        self.ball_positions = self._position_array(
            ball_positions,
            self.config.ball_start,
            "ball_positions",
        )
        self.step_counts.fill(0)
        self.ever_contact.fill(False)
        self.current_contact.fill(False)
        self.ball_path_length.fill(0.0)
        return self.observe()

    def set_layout(
        self,
        *,
        centers: np.ndarray | tuple[float, float] | None = None,
        axes: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        if centers is not None:
            self.cup_centers = self._layout_array(centers, "centers")
        if axes is not None:
            next_axes = self._layout_array(axes, "axes")
            if np.any(next_axes <= 0):
                raise ValueError("all cup axes must be positive")
            self.cup_axes = next_axes

    def sample_layouts(
        self,
        *,
        center_low: tuple[float, float] = (0.15, -0.30),
        center_high: tuple[float, float] = (0.55, 0.30),
        axes_low: tuple[float, float] = (0.10, 0.10),
        axes_high: tuple[float, float] = (0.22, 0.22),
    ) -> None:
        centers = self.rng.uniform(center_low, center_high, size=self.cup_centers.shape)
        axes = self.rng.uniform(axes_low, axes_high, size=self.cup_axes.shape)
        self.set_layout(centers=centers, axes=axes)

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        values = np.asarray(actions, dtype=np.float32)
        if values.shape != self.pusher_positions.shape:
            raise ValueError(
                f"actions must have shape {self.pusher_positions.shape}, got {values.shape}"
            )

        displacement = self.config.action_scale * np.clip(values, -1.0, 1.0)
        old_pusher = self.pusher_positions.copy()
        old_ball = self.ball_positions.copy()
        next_pusher = np.clip(
            old_pusher + displacement,
            self.config.world_low,
            self.config.world_high,
        )

        relative = old_ball - old_pusher
        moving_toward_ball = np.sum(relative * displacement, axis=-1) > 1.0e-8
        contact_distance = self.config.pusher_radius + self.config.ball_radius + self.config.contact_margin
        reaches_ball = np.linalg.norm(old_ball - next_pusher, axis=-1) <= contact_distance
        contact = moving_toward_ball & reaches_ball

        ball_displacement = np.zeros_like(displacement)
        ball_displacement[contact] = self.config.transfer_ratio * displacement[contact]
        next_ball = np.clip(
            old_ball + ball_displacement,
            self.config.world_low,
            self.config.world_high,
        )

        self.pusher_positions = next_pusher
        self.ball_positions = next_ball
        actual_ball_displacement = next_ball - old_ball
        self.ball_path_length += np.linalg.norm(actual_ball_displacement, axis=-1)
        self.current_contact = contact
        self.ever_contact |= contact
        self.step_counts += 1
        classes = self.trajectory_classes()
        done = self.step_counts >= self.config.episode_length
        return self.observe(), classes, done.copy()

    def trajectory_classes(self) -> np.ndarray:
        classes = np.zeros(self.config.num_envs, dtype=np.int64)
        moved_by_contact = self.ever_contact & (self.ball_path_length > 1.0e-6)
        classes[moved_by_contact] = 1
        classes[self._ball_inside()] = 2
        return classes

    def observe(self) -> dict[str, np.ndarray]:
        ball_inside = self._ball_inside()
        relative = self.ball_positions - self.pusher_positions
        relation_features = np.stack(
            [
                self.current_contact.astype(np.float32),
                self.ever_contact.astype(np.float32),
                ball_inside.astype(np.float32),
                self.ball_path_length,
            ],
            axis=-1,
        )
        return {
            "pusher_position": self.pusher_positions.copy(),
            "ball_position": self.ball_positions.copy(),
            "cup_center": self.cup_centers.copy(),
            "cup_axes": self.cup_axes.copy(),
            "pusher_to_ball": relative,
            "relation_features": relation_features,
            "trajectory_class": self.trajectory_classes(),
        }

    def render(self, env_ids: np.ndarray | list[int] | None = None, *, size: int = 128) -> np.ndarray:
        if size < 24:
            raise ValueError("render size must be at least 24")
        if env_ids is None:
            ids = np.arange(self.config.num_envs, dtype=np.int64)
        else:
            ids = np.asarray(env_ids, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.config.num_envs):
            raise IndexError("env_ids contains an out-of-range environment")

        frames = np.full((len(ids), size, size, 3), 248, dtype=np.uint8)
        axis = np.linspace(self.config.world_low, self.config.world_high, size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(axis, axis[::-1])
        for frame, env_id in zip(frames, ids):
            cup_center = self.cup_centers[env_id]
            cup_axes = self.cup_axes[env_id]
            cup_rho = np.sqrt(
                ((grid_x - cup_center[0]) / cup_axes[0]) ** 2
                + ((grid_y - cup_center[1]) / cup_axes[1]) ** 2
            )
            frame[cup_rho <= 1.0] = (205, 232, 238)
            frame[np.abs(cup_rho - 1.0) <= 0.06] = (41, 98, 112)

            pusher = self.pusher_positions[env_id]
            pusher_mask = (
                (grid_x - pusher[0]) ** 2 + (grid_y - pusher[1]) ** 2
                <= self.config.pusher_radius**2
            )
            pusher_color = (37, 139, 96) if self.current_contact[env_id] else (222, 132, 49)
            frame[pusher_mask] = pusher_color

            ball = self.ball_positions[env_id]
            ball_mask = (
                (grid_x - ball[0]) ** 2 + (grid_y - ball[1]) ** 2
                <= self.config.ball_radius**2
            )
            frame[ball_mask] = (218, 67, 56)
        return frames

    def _ball_inside(self) -> np.ndarray:
        normalized = (self.ball_positions - self.cup_centers) / self.cup_axes
        return np.linalg.norm(normalized, axis=-1) <= 1.0

    def _position_array(
        self,
        values: np.ndarray | None,
        default: tuple[float, float],
        name: str,
    ) -> np.ndarray:
        if values is None:
            array = np.broadcast_to(
                np.asarray(default, dtype=np.float32),
                self.pusher_positions.shape,
            )
        else:
            array = np.asarray(values, dtype=np.float32)
            if array.shape != self.pusher_positions.shape:
                raise ValueError(
                    f"{name} must have shape {self.pusher_positions.shape}, got {array.shape}"
                )
        return np.clip(array, self.config.world_low, self.config.world_high).copy()

    def _layout_array(self, values: np.ndarray | tuple[float, float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape == (2,):
            return np.broadcast_to(array, self.pusher_positions.shape).copy()
        if array.shape != self.pusher_positions.shape:
            raise ValueError(
                f"{name} must have shape (2,) or {self.pusher_positions.shape}, got {array.shape}"
            )
        return array.copy()
