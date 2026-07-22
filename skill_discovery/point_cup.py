"""A tiny non-physical shape world for probing semantic distance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SEMANTIC_LABELS = ("outside", "boundary", "entering", "inside", "leaving")
LABEL_TO_ID = {name: index for index, name in enumerate(SEMANTIC_LABELS)}


@dataclass(frozen=True)
class ShapeWorldConfig:
    """Configuration for a vectorized Point-Cup shape world."""

    num_envs: int = 1024
    episode_length: int = 64
    world_low: float = -1.0
    world_high: float = 1.0
    action_scale: float = 0.08
    cup_center: tuple[float, float] = (0.0, 0.0)
    cup_axes: tuple[float, float] = (0.14, 0.14)
    boundary_width: float = 0.025
    ball_radius: float = 0.035
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.world_low >= self.world_high:
            raise ValueError("world_low must be smaller than world_high")
        if self.action_scale <= 0:
            raise ValueError("action_scale must be positive")
        if min(self.cup_axes) <= 0:
            raise ValueError("cup axes must be positive")
        if self.boundary_width < 0:
            raise ValueError("boundary_width must be non-negative")
        if self.ball_radius <= 0:
            raise ValueError("ball_radius must be positive")


class PointCupEnv:
    """Vectorized direct-control environment with relational semantic labels.

    The environment deliberately has no physics. An action moves the ball, the
    resulting position is clipped to the world, and relations are recomputed
    from deterministic geometry. Cup layouts can differ per environment.
    """

    def __init__(self, config: ShapeWorldConfig | None = None):
        self.config = config or ShapeWorldConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.positions = np.zeros((self.config.num_envs, 2), dtype=np.float32)
        self.cup_centers = np.broadcast_to(
            np.asarray(self.config.cup_center, dtype=np.float32),
            (self.config.num_envs, 2),
        ).copy()
        self.cup_axes = np.broadcast_to(
            np.asarray(self.config.cup_axes, dtype=np.float32),
            (self.config.num_envs, 2),
        ).copy()
        self.step_counts = np.zeros(self.config.num_envs, dtype=np.int32)
        self.previous_inside = np.zeros(self.config.num_envs, dtype=bool)
        self.reset()

    def reset(
        self,
        *,
        seed: int | None = None,
        positions: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset all environments and return a fresh observation."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if positions is None:
            self.positions = self.rng.uniform(
                self.config.world_low,
                self.config.world_high,
                size=(self.config.num_envs, 2),
            ).astype(np.float32)
        else:
            values = np.asarray(positions, dtype=np.float32)
            if values.shape != self.positions.shape:
                raise ValueError(f"positions must have shape {self.positions.shape}, got {values.shape}")
            self.positions = np.clip(values, self.config.world_low, self.config.world_high)
        self.step_counts.fill(0)
        self.previous_inside = self._inside(self.positions)
        return self.observe()

    def set_layout(
        self,
        *,
        centers: np.ndarray | tuple[float, float] | None = None,
        axes: np.ndarray | tuple[float, float] | None = None,
    ) -> None:
        """Set one shared or one per-environment elliptical cup layout."""

        if centers is not None:
            self.cup_centers = self._layout_array(centers, "centers")
        if axes is not None:
            next_axes = self._layout_array(axes, "axes")
            if np.any(next_axes <= 0):
                raise ValueError("all cup axes must be positive")
            self.cup_axes = next_axes
        self.previous_inside = self._inside(self.positions)

    def sample_layouts(
        self,
        *,
        center_low: tuple[float, float] = (-0.35, -0.35),
        center_high: tuple[float, float] = (0.35, 0.35),
        axes_low: tuple[float, float] = (0.09, 0.09),
        axes_high: tuple[float, float] = (0.22, 0.22),
    ) -> None:
        """Sample per-environment layouts for later deformation tests."""

        centers = self.rng.uniform(center_low, center_high, size=self.cup_centers.shape)
        axes = self.rng.uniform(axes_low, axes_high, size=self.cup_axes.shape)
        self.set_layout(centers=centers, axes=axes)

    def step(self, actions: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Apply direct ball displacements and return observation, labels, done."""

        values = np.asarray(actions, dtype=np.float32)
        if values.shape != self.positions.shape:
            raise ValueError(f"actions must have shape {self.positions.shape}, got {values.shape}")

        before_inside = self._inside(self.positions)
        bounded_actions = np.clip(values, -1.0, 1.0)
        self.positions = np.clip(
            self.positions + self.config.action_scale * bounded_actions,
            self.config.world_low,
            self.config.world_high,
        )
        after_inside = self._inside(self.positions)
        labels = self._labels(before_inside, after_inside)
        self.previous_inside = after_inside
        self.step_counts += 1
        done = self.step_counts >= self.config.episode_length
        return self.observe(labels=labels), labels, done.copy()

    def observe(self, *, labels: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Return raw state, layout, relation features, and semantic labels."""

        rho = self._elliptical_radius(self.positions)
        inside = rho <= 1.0
        if labels is None:
            labels = np.full(self.config.num_envs, LABEL_TO_ID["outside"], dtype=np.int64)
            labels[self._boundary(rho)] = LABEL_TO_ID["boundary"]
            labels[inside] = LABEL_TO_ID["inside"]
        relation_features = np.stack(
            [inside.astype(np.float32), np.clip(rho - 1.0, -1.0, 4.0).astype(np.float32)],
            axis=-1,
        )
        return {
            "state": self.positions.copy(),
            "cup_center": self.cup_centers.copy(),
            "cup_axes": self.cup_axes.copy(),
            "relation_features": relation_features,
            "semantic_label": labels.copy(),
        }

    def render(self, env_ids: np.ndarray | list[int] | None = None, *, size: int = 96) -> np.ndarray:
        """Rasterize selected graph states to RGB without a rendering engine."""

        if size < 16:
            raise ValueError("render size must be at least 16")
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
            center = self.cup_centers[env_id]
            axes = self.cup_axes[env_id]
            cup_rho = np.sqrt(
                ((grid_x - center[0]) / axes[0]) ** 2 + ((grid_y - center[1]) / axes[1]) ** 2
            )
            frame[cup_rho <= 1.0] = (205, 232, 238)
            frame[np.abs(cup_rho - 1.0) <= 0.08] = (41, 98, 112)

            ball = self.positions[env_id]
            ball_mask = (grid_x - ball[0]) ** 2 + (grid_y - ball[1]) ** 2 <= self.config.ball_radius**2
            frame[ball_mask] = (218, 67, 56)

        return frames

    def relation_names(self, labels: np.ndarray) -> np.ndarray:
        """Convert integer semantic labels to stable string names."""

        values = np.asarray(labels)
        if np.any(values < 0) or np.any(values >= len(SEMANTIC_LABELS)):
            raise ValueError("labels contains an unknown semantic id")
        return np.asarray(SEMANTIC_LABELS, dtype=object)[values]

    def _layout_array(self, values: np.ndarray | tuple[float, float], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape == (2,):
            return np.broadcast_to(array, self.positions.shape).copy()
        if array.shape != self.positions.shape:
            raise ValueError(f"{name} must have shape (2,) or {self.positions.shape}, got {array.shape}")
        return array.copy()

    def _elliptical_radius(self, positions: np.ndarray) -> np.ndarray:
        normalized = (positions - self.cup_centers) / self.cup_axes
        return np.linalg.norm(normalized, axis=-1)

    def _inside(self, positions: np.ndarray) -> np.ndarray:
        return self._elliptical_radius(positions) <= 1.0

    def _boundary(self, rho: np.ndarray) -> np.ndarray:
        normalized_width = self.config.boundary_width / np.min(self.cup_axes, axis=-1)
        return np.abs(rho - 1.0) <= normalized_width

    def _labels(self, before_inside: np.ndarray, after_inside: np.ndarray) -> np.ndarray:
        rho = self._elliptical_radius(self.positions)
        labels = np.full(self.config.num_envs, LABEL_TO_ID["outside"], dtype=np.int64)
        labels[self._boundary(rho)] = LABEL_TO_ID["boundary"]
        labels[after_inside] = LABEL_TO_ID["inside"]
        labels[~before_inside & after_inside] = LABEL_TO_ID["entering"]
        labels[before_inside & ~after_inside] = LABEL_TO_ID["leaving"]
        return labels
