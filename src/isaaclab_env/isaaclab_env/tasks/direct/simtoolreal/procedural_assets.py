"""Procedural object generation for the Isaac Lab SimToolReal direct task."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ProceduralObjectVariant:
    """Metadata for one fixed object asset variant."""

    urdf_path: Path
    usd_path: Path
    object_scale: tuple[float, float, float]
    object_type: str
    handle_scale: tuple[float, ...]
    head_scale: tuple[float, ...] | None
    handle_density: float
    head_density: float | None
    mass: float
    center_of_mass: tuple[float, float, float]
    diagonal_inertia: tuple[float, float, float]


def _usd_tuple(values: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(value):.9g}" for value in values) + ")"


def _shape_usda(
    *,
    name: str,
    scale: tuple[float, ...],
    xyz: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool,
) -> str:
    api = ' (\n            prepend apiSchemas = ["PhysicsCollisionAPI"]\n        )' if collision else ""
    visibility = '            token visibility = "invisible"\n' if collision else ""
    color_attr = "" if collision else f"            color3f[] primvars:displayColor = [{_usd_tuple(color)}]\n"
    if len(scale) == 3:
        body = f"""        def Cube "{name}"{api}
        {{
{visibility}{color_attr}            double size = 1
            double3 xformOp:translate = {_usd_tuple(xyz)}
            double3 xformOp:scale = {_usd_tuple(scale)}
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
"""
    elif len(scale) == 2:
        height, diameter = scale
        axis = "X" if name.startswith("handle") else "Y"
        body = f"""        def Cylinder "{name}"{api}
        {{
{visibility}{color_attr}            uniform token axis = "{axis}"
            double height = {float(height):.9g}
            double radius = {float(diameter) / 2.0:.9g}
            double3 xformOp:translate = {_usd_tuple(xyz)}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
"""
    else:
        raise ValueError(f"Invalid shape scale with {len(scale)} elements: {scale}")
    return body


def _shape_mass_and_inertia(scale: tuple[float, ...], density: float) -> tuple[float, float, float, float]:
    """Match the Isaac Gym procedural URDF mass/inertia approximation."""
    if len(scale) == 3:
        lx, ly, lz = scale
        volume = lx * ly * lz
        mass = volume * density
        ixx = (1.0 / 12.0) * mass * (ly**2 + lz**2)
        iyy = (1.0 / 12.0) * mass * (lx**2 + lz**2)
        izz = (1.0 / 12.0) * mass * (lx**2 + ly**2)
    elif len(scale) == 2:
        height, diameter = scale
        radius = diameter / 2.0
        cylinder_mass = density * math.pi * radius**2 * height
        hemisphere_mass = density * (2.0 / 3.0) * math.pi * radius**3
        mass = cylinder_mass + 2.0 * hemisphere_mass

        cylinder_axis = 0.5 * cylinder_mass * radius**2
        cylinder_perp = (1.0 / 12.0) * cylinder_mass * (3.0 * radius**2 + height**2)
        hemisphere_axis = (2.0 / 5.0) * hemisphere_mass * radius**2
        hemisphere_perp = (83.0 / 320.0) * hemisphere_mass * radius**2
        hemisphere_com_offset = (height / 2.0) + (3.0 * radius / 8.0)

        izz = cylinder_axis + 2.0 * hemisphere_axis
        ixx = cylinder_perp + 2.0 * (hemisphere_perp + hemisphere_mass * hemisphere_com_offset**2)
        iyy = ixx
    else:
        raise ValueError(f"Invalid shape scale with {len(scale)} elements: {scale}")
    return mass, ixx, iyy, izz


def _handle_head_mass_properties(
    *,
    handle_scale: tuple[float, ...],
    head_scale: tuple[float, ...] | None,
    handle_density: float,
    head_density: float | None,
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    handle_mass, handle_ixx, handle_iyy, handle_izz = _shape_mass_and_inertia(handle_scale, handle_density)
    if len(handle_scale) == 2:
        handle_ixx, handle_iyy, handle_izz = handle_izz, handle_iyy, handle_ixx
    if head_scale is None:
        return handle_mass, (0.0, 0.0, 0.0), (handle_ixx, handle_iyy, handle_izz)
    if head_density is None:
        raise ValueError("head_density must be set when head_scale is set")

    head_mass, head_ixx, head_iyy, head_izz = _shape_mass_and_inertia(head_scale, head_density)
    if len(head_scale) == 3:
        x_offset = handle_scale[0] / 2.0 + head_scale[0] / 2.0
    elif len(head_scale) == 2:
        x_offset = handle_scale[0] / 2.0 + head_scale[1] / 2.0
        head_ixx, head_iyy, head_izz = head_ixx, head_izz, head_iyy
    else:
        raise ValueError(f"Invalid head scale with {len(head_scale)} elements: {head_scale}")

    total_mass = handle_mass + head_mass
    com_x = head_mass * x_offset / total_mass
    handle_dx = -com_x
    head_dx = x_offset - com_x
    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * handle_dx**2) + (head_iyy + head_mass * head_dx**2)
    izz = (handle_izz + handle_mass * handle_dx**2) + (head_izz + head_mass * head_dx**2)
    return total_mass, (com_x, 0.0, 0.0), (ixx, iyy, izz)


def _write_handle_head_usda(
    *,
    filepath: Path,
    handle_scale: tuple[float, ...],
    head_scale: tuple[float, ...] | None,
    mass: float,
    center_of_mass: tuple[float, float, float],
    diagonal_inertia: tuple[float, float, float],
) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    shapes: list[tuple[str, tuple[float, ...], tuple[float, float, float], tuple[float, float, float]]] = [
        ("handle", handle_scale, (0.0, 0.0, 0.0), (0.55, 0.27, 0.07)),
    ]
    if head_scale is not None:
        if len(head_scale) == 3:
            x_offset = handle_scale[0] / 2.0 + head_scale[0] / 2.0
        elif len(head_scale) == 2:
            x_offset = handle_scale[0] / 2.0 + head_scale[1] / 2.0
        else:
            raise ValueError(f"Invalid head scale with {len(head_scale)} elements: {head_scale}")
        shapes.append(("head", head_scale, (x_offset, 0.0, 0.0), (0.5, 0.5, 0.5)))

    visual_shapes = "".join(
        _shape_usda(name=f"{name}_visual", scale=scale, xyz=xyz, color=color, collision=False)
        for name, scale, xyz, color in shapes
    )
    collision_shapes = "".join(
        _shape_usda(name=f"{name}_collision", scale=scale, xyz=xyz, color=color, collision=True)
        for name, scale, xyz, color in shapes
    )
    filepath.write_text(
        f"""#usda 1.0
(
    defaultPrim = "handle_head"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "handle_head" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    point3f physics:centerOfMass = {_usd_tuple(center_of_mass)}
    float3 physics:diagonalInertia = {_usd_tuple(diagonal_inertia)}
    float physics:mass = {float(mass):.9g}

    def Xform "visuals"
    {{
{visual_shapes}    }}

    def Xform "collisions"
    {{
{collision_shapes}    }}
}}
""",
        encoding="utf-8",
    )


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_three_axis_scale(scale: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in scale)
    if len(values) == 3:
        return values
    if len(values) == 2:
        return (values[0], values[1], values[1])
    raise ValueError(f"Invalid object scale with {len(values)} elements: {values}")


def generate_handle_head_variants(
    *,
    repo_root: Path,
    generated_root: Path,
    handle_head_types: Iterable[str],
    num_objects_per_distribution: int,
    object_base_size: float,
    seed: int = 42,
    max_variants: int | None = None,
) -> list[ProceduralObjectVariant]:
    """Generate the same handle/head URDF family used by the Isaac Gym task.

    The old Isaac Gym task generated 100 objects per size distribution, shuffled the
    final list, then assigned object ``env_id % num_objects``.  This function keeps
    that ordering rule and returns both the generated URDF path and the normalized
    object scale needed by observations/keypoints.
    """

    if num_objects_per_distribution <= 0:
        raise ValueError("num_objects_per_distribution must be positive")
    if object_base_size <= 0:
        raise ValueError("object_base_size must be positive")

    simtoolreal_dir = repo_root / "isaacgymenvs" / "tasks" / "simtoolreal"
    generate_objects = _load_module("simtoolreal_generate_objects", simtoolreal_dir / "generate_objects.py")
    object_sizes = _load_module("simtoolreal_object_size_distributions", simtoolreal_dir / "object_size_distributions.py")

    generated_root.mkdir(parents=True, exist_ok=True)
    for old_urdf in generated_root.glob("*.urdf"):
        old_urdf.unlink()
    usd_root = generated_root.parent.parent / "usd" / "manual_objects"
    usd_root.mkdir(parents=True, exist_ok=True)
    for old_usd in usd_root.glob("*.usda"):
        old_usd.unlink()

    requested_types = set(handle_head_types)
    distributions = [obj for obj in object_sizes.OBJECT_SIZE_DISTRIBUTIONS if obj.type in requested_types]
    if not distributions:
        raise ValueError(f"No procedural object distributions selected from {sorted(requested_types)}")

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        variants: list[ProceduralObjectVariant] = []
        for distribution in distributions:
            handle_densities = distribution.sample_handle_densities(num_objects_per_distribution)
            head_densities = distribution.sample_head_densities(num_objects_per_distribution)
            handle_scales = distribution.sample_handle_scales(num_objects_per_distribution)
            head_scales = distribution.sample_head_scales(num_objects_per_distribution)

            for index in range(num_objects_per_distribution):
                handle_scale = tuple(float(x) for x in handle_scales[index])
                head_scale = None if head_scales is None else tuple(float(x) for x in head_scales[index])
                handle_density = float(handle_densities[index])
                head_density = None if head_densities is None else float(head_densities[index])
                filename = (
                    f"{index:03d}_{distribution.type}_handle_head_"
                    f"{handle_scale}_{head_scale}_{handle_density}_{head_density}".replace(".", "-")
                    + ".urdf"
                )
                urdf_path = generated_root / filename
                usd_path = usd_root / filename.replace(".urdf", ".usda")
                generate_objects.generate_handle_head_urdf(
                    filepath=urdf_path,
                    handle_scale=handle_scale,
                    head_scale=head_scale,
                    handle_density=handle_density,
                    head_density=head_density,
                )
                mass, center_of_mass, diagonal_inertia = _handle_head_mass_properties(
                    handle_scale=handle_scale,
                    head_scale=head_scale,
                    handle_density=handle_density,
                    head_density=head_density,
                )
                _write_handle_head_usda(
                    filepath=usd_path,
                    handle_scale=handle_scale,
                    head_scale=head_scale,
                    mass=mass,
                    center_of_mass=center_of_mass,
                    diagonal_inertia=diagonal_inertia,
                )
                metric_scale = _to_three_axis_scale(handle_scale)
                object_scale = tuple(value / object_base_size for value in metric_scale)
                variants.append(
                    ProceduralObjectVariant(
                        urdf_path=urdf_path,
                        usd_path=usd_path,
                        object_scale=object_scale,
                        object_type=distribution.type,
                        handle_scale=handle_scale,
                        head_scale=head_scale,
                        handle_density=handle_density,
                        head_density=head_density,
                        mass=mass,
                        center_of_mass=center_of_mass,
                        diagonal_inertia=diagonal_inertia,
                    )
                )

        indices = list(range(len(variants)))
        np.random.shuffle(indices)
        variants = [variants[index] for index in indices]
    finally:
        np.random.set_state(rng_state)

    if max_variants is not None and max_variants > 0:
        variants = variants[:max_variants]
    if not variants:
        raise RuntimeError("Procedural object generation produced no variants")
    return variants
