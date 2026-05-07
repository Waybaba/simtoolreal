"""Isaac Lab DirectRLEnv port of the SimToolReal Kuka+SharpA task."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, random_orientation, sample_log_uniform

from .procedural_assets import ProceduralObjectVariant, generate_handle_head_variants


REPO_ROOT = Path(__file__).resolve().parents[6]
ASSETS_ROOT = REPO_ROOT / "assets"
GENERATED_ROOT = REPO_ROOT / "outputs" / "isaaclab_generated_assets" / "simtoolreal_direct"

ARM_JOINT_PATTERN = "iiwa14_joint_[1-7]"
LEFT_SHARPA_FINGERTIPS = [
    "left_index_DP",
    "left_middle_DP",
    "left_ring_DP",
    "left_thumb_DP",
    "left_pinky_DP",
]
FINGERTIP_OFFSETS = (
    (0.02, 0.002, 0.0),
    (0.02, 0.002, 0.0),
    (0.02, 0.002, 0.0),
    (0.02, 0.002, 0.0),
    (0.02, 0.002, 0.0),
)
ISAACGYM_ACTION_JOINT_NAMES = (
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
)
KUKA_STIFFNESSES = (600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0)
KUKA_DAMPINGS = (
    27.027026473513512,
    27.027026473513512,
    24.672186769721083,
    22.067474708266914,
    9.752538131173853,
    9.147747263670984,
    9.147747263670984,
)
HAND_STIFFNESSES = (
    6.95,
    13.2,
    4.76,
    6.62,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    4.76,
    6.62,
    0.9,
    0.9,
    1.38,
    4.76,
    6.62,
    0.9,
    0.9,
)
HAND_DAMPINGS = (
    0.28676845,
    0.40845109,
    0.20394083,
    0.24044435,
    0.04190723,
    0.20859232,
    0.24595532,
    0.04243185,
    0.03504461,
    0.2085923,
    0.24595532,
    0.04243185,
    0.03504461,
    0.20859226,
    0.24595528,
    0.04243183,
    0.0350446,
    0.02782345,
    0.20859229,
    0.24595528,
    0.04243183,
    0.0350446,
)
HAND_ARMATURES = (
    0.0032,
    0.0032,
    0.00265,
    0.00265,
    0.0006,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
    0.00012,
    0.00265,
    0.00265,
    0.0006,
    0.00042,
)
HAND_FRICTIONS = (
    0.132,
    0.132,
    0.07456,
    0.07456,
    0.01276,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
    0.012,
    0.07456,
    0.07456,
    0.01276,
    0.00378738,
)
OBJECT_KEYPOINT_SIGNS = (
    (1.0, 1.0, 1.0),
    (1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, -1.0, -1.0),
)

KUKA_STIFFNESS_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[:7], KUKA_STIFFNESSES))
KUKA_DAMPING_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[:7], KUKA_DAMPINGS))
HAND_STIFFNESS_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[7:], HAND_STIFFNESSES))
HAND_DAMPING_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[7:], HAND_DAMPINGS))
HAND_ARMATURE_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[7:], HAND_ARMATURES))
HAND_FRICTION_BY_JOINT = dict(zip(ISAACGYM_ACTION_JOINT_NAMES[7:], HAND_FRICTIONS))
assert len(ISAACGYM_ACTION_JOINT_NAMES) == 29
assert len(HAND_STIFFNESS_BY_JOINT) == len(HAND_DAMPING_BY_JOINT) == 22


def _unscale(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (value - lower) / torch.clamp(upper - lower, min=1.0e-6) - 1.0


def _scale(value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + 1.0) * (upper - lower) + lower


def _variant_asset_cfg(
    variant: ProceduralObjectVariant,
    *,
    goal: bool,
) -> sim_utils.UsdFileCfg:
    visual_material = None
    if goal:
        visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1), roughness=0.65, metallic=0.0)
    return sim_utils.UsdFileCfg(usd_path=str(variant.usd_path), visual_material=visual_material)


@configclass
class SimToolRealDirectEnvCfg(DirectRLEnvCfg):
    """Configuration for the Isaac Lab direct SimToolReal task."""

    decimation = 1
    episode_length_s = 10.0
    action_space = 29
    observation_space = 140
    state_space = 162

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            solver_type=1,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.2,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ASSETS_ROOT / "urdf" / "kuka_sharpa_description" / "iiwa14_left_sharpa_adjusted_restricted.urdf"),
            usd_dir=str(GENERATED_ROOT / "usd" / "robot"),
            usd_file_name="iiwa14_left_sharpa_adjusted_restricted.usd",
            fix_base=True,
            merge_fixed_joints=True,
            convert_mimic_joints_to_normal_joints=False,
            self_collision=False,
            make_instanceable=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.8, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "iiwa14_joint_1": -1.571,
                "iiwa14_joint_2": 1.571,
                "iiwa14_joint_3": 0.0,
                "iiwa14_joint_4": 1.376,
                "iiwa14_joint_5": 0.0,
                "iiwa14_joint_6": 1.485,
                "iiwa14_joint_7": 1.308,
            },
        ),
        actuators={
            "kuka_arm": ImplicitActuatorCfg(
                joint_names_expr=[ARM_JOINT_PATTERN],
                effort_limit_sim=300.0,
                stiffness=KUKA_STIFFNESS_BY_JOINT,
                damping=KUKA_DAMPING_BY_JOINT,
            ),
            "left_sharpa_hand": ImplicitActuatorCfg(
                joint_names_expr=["left_.*"],
                effort_limit_sim=40.0,
                stiffness=HAND_STIFFNESS_BY_JOINT,
                damping=HAND_DAMPING_BY_JOINT,
                armature=HAND_ARMATURE_BY_JOINT,
                friction=HAND_FRICTION_BY_JOINT,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.38), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.475, 0.4, 0.3),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=500.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.56, 0.35), roughness=0.8, metallic=0.0),
        ),
    )
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.02, 0.02),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
    )
    goal_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/GoalObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.02, 0.02),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1), roughness=0.65, metallic=0.0),
        ),
    )

    handle_head_types = ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush")
    procedural_objects_per_distribution = 100
    max_object_variants = 0
    procedural_seed = 42

    reset_position_noise_x = 0.1
    reset_position_noise_y = 0.1
    reset_position_noise_z = 0.02
    reset_dof_pos_noise_fingers = 0.1
    reset_dof_pos_noise_arm = 0.1
    reset_dof_vel_noise = 0.5
    randomize_object_rotation = True
    table_reset_z = 0.38
    table_reset_z_range = 0.01
    table_object_z_offset = 0.25

    dof_speed_scale = 1.5
    use_relative_control = False
    arm_moving_average = 0.1
    hand_moving_average = 0.1
    clamp_abs_observations = 10.0

    force_scale = 20.0
    force_prob_range = (0.001, 0.1)
    torque_scale = 2.0
    torque_prob_range = (0.001, 0.1)
    force_only_when_lifted = True
    torque_only_when_lifted = True

    lifting_rew_scale = 20.0
    lifting_bonus = 300.0
    lifting_bonus_threshold = 0.15
    keypoint_rew_scale = 200.0
    distance_delta_rew_scale = 50.0
    reach_goal_bonus = 1000.0
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003
    fall_distance = 0.24
    object_lin_vel_penalty_scale = 0.0
    object_ang_vel_penalty_scale = 0.0
    success_tolerance = 0.075
    success_steps = 10
    max_consecutive_successes = 50
    force_consecutive_near_goal_steps = False

    keypoint_scale = 1.5
    object_base_size = 0.04
    fixed_size_keypoint_reward = True
    fixed_size = (0.141, 0.03025, 0.0271)
    object_scale_noise_multiplier_range = (1.0, 1.0)

    target_volume_region_scale = 1.0
    target_volume_mins = (-0.35, -0.2, 0.6)
    target_volume_maxs = (0.35, 0.2, 0.95)
    goal_sampling_type = "delta"
    delta_goal_distance = 0.1
    delta_rotation_degrees = 90.0

    debug_visual_ground = False
    debug_ground_size = (12.0, 12.0, 0.02)
    debug_ground_z = -0.015
    debug_ground_color = (0.16, 0.17, 0.18)
    dome_light_intensity = 1800.0
    dome_light_color = (0.75, 0.75, 0.75)
    key_light_intensity = 0.0
    key_light_color = (1.0, 0.94, 0.82)
    key_light_angle = 2.0

    use_action_delay = True
    action_delay_max = 3
    use_object_state_delay_noise = True
    object_state_delay_max = 10
    object_state_xyz_noise_std = 0.01
    object_state_rotation_noise_degrees = 5.0
    joint_velocity_obs_noise_std = 0.1

    def __post_init__(self) -> None:
        max_variants = self.max_object_variants if self.max_object_variants > 0 else None
        self.object_variants = generate_handle_head_variants(
            repo_root=REPO_ROOT,
            generated_root=GENERATED_ROOT / "urdf" / "handle_head_primitives",
            handle_head_types=self.handle_head_types,
            num_objects_per_distribution=self.procedural_objects_per_distribution,
            object_base_size=self.object_base_size,
            seed=self.procedural_seed,
            max_variants=max_variants,
        )
        object_assets = [_variant_asset_cfg(variant, goal=False) for variant in self.object_variants]
        goal_assets = [_variant_asset_cfg(variant, goal=True) for variant in self.object_variants]

        self.object_cfg.spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=object_assets,
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                linear_damping=0.01,
                angular_damping=0.01,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        self.goal_object_cfg.spawn = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=goal_assets,
            random_choice=False,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )


@configclass
class SimToolRealDirectDebugEnvCfg(SimToolRealDirectEnvCfg):
    """Small variant for fast Isaac Lab bring-up."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 4
        self.scene.env_spacing = 1.6
        self.procedural_objects_per_distribution = 1
        self.max_object_variants = 6
        self.debug_visual_ground = True
        self.dome_light_intensity = 400.0
        self.dome_light_color = (0.24, 0.27, 0.30)
        self.key_light_intensity = 2600.0
        super().__post_init__()


@configclass
class SimToolRealDirectPlayEnvCfg(SimToolRealDirectDebugEnvCfg):
    """Single-env visual/debug variant."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0


class SimToolRealDirectEnv(DirectRLEnv):
    """DirectRLEnv implementation of SimToolReal with fixed per-env object variants."""

    cfg: SimToolRealDirectEnvCfg

    def __init__(self, cfg: SimToolRealDirectEnvCfg, render_mode: str | None = None, **kwargs):
        self._obs_keys = (
            "joint_pos",
            "joint_vel",
            "prev_action_targets",
            "palm_pos",
            "palm_rot",
            "object_rot",
            "fingertip_pos_rel_palm",
            "keypoints_rel_palm",
            "keypoints_rel_goal",
            "object_scales",
        )
        self._state_keys = (
            "joint_pos",
            "joint_vel",
            "prev_action_targets",
            "palm_pos",
            "palm_rot",
            "palm_vel",
            "object_rot",
            "object_vel",
            "fingertip_pos_rel_palm",
            "keypoints_rel_palm",
            "keypoints_rel_goal",
            "object_scales",
            "closest_keypoint_max_dist",
            "closest_fingertip_dist",
            "lifted_object",
            "progress",
            "successes",
            "reward",
        )
        super().__init__(cfg, render_mode, **kwargs)

        self.control_dt = self.cfg.sim.dt * self.cfg.decimation
        joint_name_to_id = {name: index for index, name in enumerate(self.robot.joint_names)}
        missing_joints = [name for name in ISAACGYM_ACTION_JOINT_NAMES if name not in joint_name_to_id]
        if missing_joints:
            raise RuntimeError(f"Missing expected SimToolReal joints in Isaac Lab articulation: {missing_joints}")
        self.actuated_joint_ids = [joint_name_to_id[name] for name in ISAACGYM_ACTION_JOINT_NAMES]
        self.arm_joint_ids = self.actuated_joint_ids[:7]
        self.hand_joint_ids = self.actuated_joint_ids[7:]
        if len(self.actuated_joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                f"Expected {self.cfg.action_space} robot actions, found {len(self.actuated_joint_ids)} joints: "
                f"{self.robot.joint_names}"
            )

        self.palm_body_id = self.robot.body_names.index("iiwa14_link_7")
        self.fingertip_body_ids = [self.robot.body_names.index(name) for name in LEFT_SHARPA_FINGERTIPS]
        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].clone()
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = torch.zeros_like(self.default_joint_pos)
        self.joint_targets = self.default_joint_pos.clone()
        self.prev_targets = self.default_joint_pos[:, self.actuated_joint_ids].clone()
        self.cur_targets = self.prev_targets.clone()
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        num_variants = len(self.cfg.object_variants)
        self.object_variant_ids = torch.arange(self.num_envs, device=self.device) % num_variants
        self.object_scales = torch.tensor(
            [variant.object_scale for variant in self.cfg.object_variants],
            dtype=torch.float32,
            device=self.device,
        )[self.object_variant_ids]
        self.keypoint_signs = torch.tensor(OBJECT_KEYPOINT_SIGNS, dtype=torch.float32, device=self.device)
        self.fingertip_offsets = torch.tensor(FINGERTIP_OFFSETS, dtype=torch.float32, device=self.device)
        self.object_keypoint_offsets = self._make_object_keypoint_offsets(self.object_scales)
        fixed_scales = torch.tensor(self.cfg.fixed_size, dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.object_keypoint_offsets_fixed_size = self._make_object_keypoint_offsets(fixed_scales, already_metric=True)

        self.object_scale_noise_multiplier = torch.ones((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.goal_states = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.object_init_state = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.table_init_state = torch.zeros((self.num_envs, 13), dtype=torch.float32, device=self.device)
        self.goal_displacement = torch.tensor((-0.35, -0.06, 0.08), dtype=torch.float32, device=self.device)

        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.closest_keypoint_max_dist_fixed_size = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.closest_fingertip_dist = -torch.ones((self.num_envs, len(self.fingertip_body_ids)), dtype=torch.float32, device=self.device)
        self.furthest_hand_dist = -torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)
        self.prev_episode_closest_keypoint_max_dist = 1000.0 * torch.ones_like(self.successes)
        self.total_episode_closest_keypoint_max_dist = torch.zeros_like(self.successes)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros_like(self.successes)

        self.action_queue = torch.zeros(
            (self.num_envs, self.cfg.action_delay_max + 1, self.cfg.action_space), dtype=torch.float32, device=self.device
        )
        self.object_state_queue = torch.zeros(
            (self.num_envs, self.cfg.object_state_delay_max + 1, 13), dtype=torch.float32, device=self.device
        )
        self.random_force_prob = sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1], self.num_envs, self.device
        )
        self.random_torque_prob = sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1], self.num_envs, self.device
        )
        self.last_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.target_volume_min = torch.tensor(self.cfg.target_volume_mins, dtype=torch.float32, device=self.device)
        self.target_volume_max = torch.tensor(self.cfg.target_volume_maxs, dtype=torch.float32, device=self.device)
        center = 0.5 * (self.target_volume_min + self.target_volume_max)
        half_extent = 0.5 * (self.target_volume_max - self.target_volume_min) * self.cfg.target_volume_region_scale
        self.target_volume_min = center - half_extent
        self.target_volume_max = center + half_extent

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal_object = RigidObject(self.cfg.goal_object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_object"] = self.goal_object
        self.scene.rigid_objects["table"] = self.table

        if self.cfg.debug_visual_ground:
            ground_cfg = sim_utils.CuboidCfg(
                size=self.cfg.debug_ground_size,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=self.cfg.debug_ground_color,
                    roughness=0.9,
                    metallic=0.0,
                ),
            )
            ground_cfg.func("/World/DebugGround", ground_cfg, translation=(0.0, 0.0, self.cfg.debug_ground_z))

        dome_light_cfg = sim_utils.DomeLightCfg(intensity=self.cfg.dome_light_intensity, color=self.cfg.dome_light_color)
        dome_light_cfg.func("/World/AmbientLight", dome_light_cfg)
        if self.cfg.key_light_intensity > 0.0:
            key_light_cfg = sim_utils.DistantLightCfg(
                intensity=self.cfg.key_light_intensity,
                color=self.cfg.key_light_color,
                angle=self.cfg.key_light_angle,
            )
            key_light_cfg.func("/World/KeyLight", key_light_cfg)

    def _make_object_keypoint_offsets(self, scales: torch.Tensor, *, already_metric: bool = False) -> torch.Tensor:
        if already_metric:
            metric = scales
        else:
            metric = scales * self.cfg.object_base_size
        return self.keypoint_signs.unsqueeze(0) * metric.unsqueeze(1) * (self.cfg.keypoint_scale / 2.0)

    def _update_queue(self, queue: torch.Tensor, current_values: torch.Tensor) -> torch.Tensor:
        queue[:, 1:] = queue[:, :-1].clone()
        queue[:, 0] = current_values
        return queue

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = torch.clamp(actions, -1.0, 1.0)
        self.action_queue = self._update_queue(self.action_queue, actions)
        if self.cfg.use_action_delay:
            delay = torch.randint(0, self.action_queue.shape[1], (self.num_envs,), device=self.device)
            actions = self.action_queue[torch.arange(self.num_envs, device=self.device), delay].clone()
        self.actions = actions

    def _apply_action(self) -> None:
        joint_pos = self.robot.data.joint_pos[:, self.actuated_joint_ids]
        lower = self.joint_lower_limits[self.actuated_joint_ids]
        upper = self.joint_upper_limits[self.actuated_joint_ids]

        if self.cfg.use_relative_control:
            arm_targets = joint_pos[:, :7] + self.cfg.dof_speed_scale * self.control_dt * self.actions[:, :7]
        else:
            arm_targets = self.prev_targets[:, :7] + self.cfg.dof_speed_scale * self.control_dt * self.actions[:, :7]
        arm_targets = torch.clamp(arm_targets, lower[:7], upper[:7])
        arm_targets = self.cfg.arm_moving_average * arm_targets + (1.0 - self.cfg.arm_moving_average) * self.prev_targets[:, :7]

        hand_targets = _scale(self.actions[:, 7:], lower[7:], upper[7:])
        hand_targets = self.cfg.hand_moving_average * hand_targets + (1.0 - self.cfg.hand_moving_average) * self.prev_targets[:, 7:]
        hand_targets = torch.clamp(hand_targets, lower[7:], upper[7:])

        self.cur_targets = torch.cat((arm_targets, hand_targets), dim=-1)
        self.prev_targets.copy_(self.cur_targets)
        self.joint_targets[:, self.actuated_joint_ids] = self.cur_targets
        self.robot.set_joint_position_target(self.joint_targets)
        self._apply_random_wrenches()

    def _apply_random_wrenches(self) -> None:
        forces = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        torques = torch.zeros_like(forces)
        if self.cfg.force_scale > 0:
            force_mask = torch.rand(self.num_envs, device=self.device) < self.random_force_prob
            if self.cfg.force_only_when_lifted:
                force_mask &= self.lifted_object
            forces[force_mask, 0] = torch.randn((int(force_mask.sum()), 3), device=self.device) * self.cfg.force_scale * 0.1
        if self.cfg.torque_scale > 0:
            torque_mask = torch.rand(self.num_envs, device=self.device) < self.random_torque_prob
            if self.cfg.torque_only_when_lifted:
                torque_mask &= self.lifted_object
            torques[torque_mask, 0] = torch.randn((int(torque_mask.sum()), 3), device=self.device) * self.cfg.torque_scale * 0.1
        self.object.set_external_force_and_torque(forces, torques, is_global=True)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs_dict = self._build_obs_dict()
        policy = torch.cat([obs_dict[key].reshape(self.num_envs, -1) for key in self._obs_keys], dim=-1)
        critic = torch.cat([obs_dict[key].reshape(self.num_envs, -1) for key in self._state_keys], dim=-1)
        if self.cfg.clamp_abs_observations > 0:
            policy = torch.clamp(policy, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
            critic = torch.clamp(critic, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
        return {"policy": policy, "critic": critic}

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        lower = self.joint_lower_limits[self.actuated_joint_ids]
        upper = self.joint_upper_limits[self.actuated_joint_ids]
        joint_pos = _unscale(self.robot.data.joint_pos[:, self.actuated_joint_ids], lower, upper)
        joint_vel = self.robot.data.joint_vel[:, self.actuated_joint_ids]
        if self.cfg.joint_velocity_obs_noise_std > 0:
            joint_vel = joint_vel + torch.randn_like(joint_vel) * self.cfg.joint_velocity_obs_noise_std
        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "prev_action_targets": self.prev_targets.clone(),
            "palm_pos": self.palm_center_pos,
            "palm_rot": self.palm_quat,
            "palm_vel": self.palm_vel,
            "object_rot": self.observed_object_rot,
            "object_vel": self.object_vel,
            "fingertip_pos_rel_palm": self.fingertip_pos_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_palm": self.keypoints_rel_palm.reshape(self.num_envs, -1),
            "keypoints_rel_goal": self.keypoints_rel_goal.reshape(self.num_envs, -1),
            "object_scales": self.object_scales * self.object_scale_noise_multiplier,
            "closest_keypoint_max_dist": self.closest_keypoint_max_dist_for_obs.unsqueeze(-1),
            "closest_fingertip_dist": self.closest_fingertip_dist,
            "lifted_object": self.lifted_object.float().unsqueeze(-1),
            "progress": torch.log(self.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1),
            "successes": torch.log(self.successes + 1.0).unsqueeze(-1),
            "reward": 0.01 * self.last_reward.unsqueeze(-1),
        }

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew = self._keypoint_reward(lifted_object)

        keypoint_success_tolerance = self.cfg.success_tolerance * self.cfg.keypoint_scale
        near_goal = self.keypoints_max_dist_for_reward <= keypoint_success_tolerance
        if self.cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.long()) * near_goal.long()
        else:
            self.near_goal_steps += near_goal.long()
        is_success = self.near_goal_steps >= self.cfg.success_steps
        self.successes += is_success.float()

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_vel[:, :3]), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_vel[:, 3:]), dim=-1)
        kuka_actions_penalty = -torch.sum(torch.abs(self.robot.data.joint_vel[:, self.actuated_joint_ids[:7]]), dim=-1)
        hand_actions_penalty = -torch.sum(torch.abs(self.robot.data.joint_vel[:, self.actuated_joint_ids[7:]]), dim=-1)

        bonus_rew = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)
        if self.cfg.force_consecutive_near_goal_steps:
            bonus_rew = is_success.float() * self.cfg.reach_goal_bonus

        reward = (
            fingertip_delta_rew * self.cfg.distance_delta_rew_scale
            + hand_delta_penalty * self.cfg.distance_delta_rew_scale * 0.0
            + lifting_rew * self.cfg.lifting_rew_scale
            + lift_bonus_rew
            + keypoint_rew * self.cfg.keypoint_rew_scale
            + kuka_actions_penalty * self.cfg.kuka_actions_penalty_scale
            + hand_actions_penalty * self.cfg.hand_actions_penalty_scale
            + bonus_rew
            + object_lin_vel_penalty * self.cfg.object_lin_vel_penalty_scale
            + object_ang_vel_penalty * self.cfg.object_ang_vel_penalty_scale
        )
        self.last_reward = reward
        self.extras.setdefault("log", {})
        self.extras["log"]["success_rate"] = is_success.float().mean()
        self.extras["log"]["closest_keypoint_max_dist"] = self.keypoints_max_dist_for_reward.mean()
        self.extras["log"]["object_height"] = self.object_pos[:, 2].mean()
        self.extras["log"]["lifted_rate"] = self.lifted_object.float().mean()
        return reward

    def _distance_delta_rewards(self, lifted_object: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fingertip_deltas = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)
        hand_deltas = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])
        fingertip_delta_rew = torch.clip(fingertip_deltas, 0.0, 10.0).sum(dim=-1) * (~lifted_object)
        hand_delta_penalty = torch.clip(hand_deltas, -10.0, 0.0) * (~lifted_object) * len(self.fingertip_body_ids)
        return fingertip_delta_rew, hand_delta_penalty

    def _lifting_reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0.0, 0.5)
        lifted_object = (z_lift > self.cfg.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted_object & (~self.lifted_object)
        lift_bonus_rew = self.cfg.lifting_bonus * just_lifted.float()
        lifting_rew *= (~lifted_object).float()
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: torch.Tensor) -> torch.Tensor:
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        max_keypoint_deltas_fixed_size = self.closest_keypoint_max_dist_fixed_size - self.keypoints_max_dist_fixed_size
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        self.closest_keypoint_max_dist_fixed_size = torch.minimum(
            self.closest_keypoint_max_dist_fixed_size, self.keypoints_max_dist_fixed_size
        )
        keypoint_rew = torch.clip(max_keypoint_deltas, 0.0, 100.0) * lifted_object.float()
        keypoint_rew_fixed = torch.clip(max_keypoint_deltas_fixed_size, 0.0, 100.0) * lifted_object.float()
        return keypoint_rew_fixed if self.cfg.fixed_size_keypoint_reward else keypoint_rew

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        object_z_low = self.object_pos[:, 2] < 0.1
        max_success = self.successes >= self.cfg.max_consecutive_successes
        hand_far = self.curr_fingertip_distances.max(dim=-1).values > 1.5
        terminated = object_z_low | max_success | hand_far
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        self.prev_episode_successes[env_ids] = self.successes[env_ids]
        self.successes[env_ids] = 0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist_fixed_size[env_ids] = -1.0
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.action_queue[env_ids] = 0.0
        self.object_state_queue[env_ids] = 0.0
        self.random_force_prob[env_ids] = sample_log_uniform(
            self.cfg.force_prob_range[0], self.cfg.force_prob_range[1], len(env_ids), self.device
        )
        self.random_torque_prob[env_ids] = sample_log_uniform(
            self.cfg.torque_prob_range[0], self.cfg.torque_prob_range[1], len(env_ids), self.device
        )

        self._reset_table_and_object(env_ids)
        self._reset_goal(env_ids, is_first_goal=True)
        self._reset_robot(env_ids)

    def _reset_table_and_object(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        table_z = self.cfg.table_reset_z + (
            torch.rand((count, 1), device=self.device) * 2.0 - 1.0
        ) * self.cfg.table_reset_z_range
        table_state = self.table.data.default_root_state[env_ids].clone()
        object_state = self.object.data.default_root_state[env_ids].clone()
        table_state[:, :3] = self.scene.env_origins[env_ids]
        table_state[:, 2:3] += table_z
        object_state[:, :3] = self.scene.env_origins[env_ids]
        object_state[:, 2:3] += table_z + self.cfg.table_object_z_offset

        pos_noise = torch.rand((count, 3), device=self.device) * 2.0 - 1.0
        object_state[:, 0] += self.cfg.reset_position_noise_x * pos_noise[:, 0]
        object_state[:, 1] += self.cfg.reset_position_noise_y * pos_noise[:, 1]
        object_state[:, 2] += self.cfg.reset_position_noise_z * pos_noise[:, 2]
        if self.cfg.randomize_object_rotation:
            object_state[:, 3:7] = random_orientation(count, self.device)
        object_state[:, 7:13] = 0.0
        table_state[:, 7:13] = 0.0

        self.table.write_root_state_to_sim(table_state, env_ids=env_ids)
        self.object.write_root_state_to_sim(object_state, env_ids=env_ids)

        local_object_state = object_state.clone()
        local_table_state = table_state.clone()
        local_object_state[:, :3] -= self.scene.env_origins[env_ids]
        local_table_state[:, :3] -= self.scene.env_origins[env_ids]
        self.object_init_state[env_ids] = local_object_state
        self.table_init_state[env_ids] = local_table_state
        noise_min, noise_max = self.cfg.object_scale_noise_multiplier_range
        self.object_scale_noise_multiplier[env_ids] = noise_min + torch.rand((count, 3), device=self.device) * (noise_max - noise_min)

    def _reset_goal(self, env_ids: torch.Tensor, *, is_first_goal: bool) -> None:
        count = len(env_ids)
        goal_state = self.goal_object.data.default_root_state[env_ids].clone()
        if (not is_first_goal) and self.cfg.goal_sampling_type == "delta":
            local_goal = self.goal_states[env_ids].clone()
            local_goal[:, :3] += (torch.rand((count, 3), device=self.device) * 2.0 - 1.0) * self.cfg.delta_goal_distance
            local_goal[:, :3] = torch.maximum(torch.minimum(local_goal[:, :3], self.target_volume_max), self.target_volume_min)
            local_goal[:, 3:7] = self._sample_delta_quat(local_goal[:, 3:7], self.cfg.delta_rotation_degrees)
        else:
            local_goal = torch.zeros((count, 13), dtype=torch.float32, device=self.device)
            local_goal[:, :3] = self.target_volume_min + torch.rand((count, 3), device=self.device) * (
                self.target_volume_max - self.target_volume_min
            )
            local_goal[:, 3:7] = random_orientation(count, self.device)
            min_z = self.object_init_state[env_ids, 2] - 0.05 + self.cfg.lifting_bonus_threshold
            local_goal[:, 2] = torch.maximum(local_goal[:, 2], min_z)
        goal_state[:, :3] = local_goal[:, :3] + self.scene.env_origins[env_ids]
        goal_state[:, 3:7] = local_goal[:, 3:7]
        goal_state[:, 7:13] = 0.0
        self.goal_object.write_root_state_to_sim(goal_state, env_ids=env_ids)
        self.goal_states[env_ids] = local_goal

    def _sample_delta_quat(self, quat: torch.Tensor, delta_degrees: float) -> torch.Tensor:
        if delta_degrees <= 0.0:
            return quat
        axis = torch.randn((quat.shape[0], 3), dtype=torch.float32, device=self.device)
        axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
        angle = (torch.rand(quat.shape[0], device=self.device) * 2.0 - 1.0) * torch.deg2rad(
            torch.tensor(float(delta_degrees), device=self.device)
        )
        return quat_mul(quat_from_angle_axis(angle, axis), quat)

    def _reset_robot(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        joint_pos = self.default_joint_pos[env_ids].clone()
        joint_vel = self.default_joint_vel[env_ids].clone()
        lower = self.joint_lower_limits[self.actuated_joint_ids]
        upper = self.joint_upper_limits[self.actuated_joint_ids]
        default = joint_pos[:, self.actuated_joint_ids]
        rand = torch.rand((count, len(self.actuated_joint_ids)), device=self.device)
        delta_min = lower - default
        delta_max = upper - default
        coeff = torch.ones_like(default)
        coeff[:, :7] = self.cfg.reset_dof_pos_noise_arm
        coeff[:, 7:] = self.cfg.reset_dof_pos_noise_fingers
        reset_pos = default + coeff * (delta_min + (delta_max - delta_min) * rand)
        reset_pos = torch.clamp(reset_pos, lower, upper)
        joint_pos[:, self.actuated_joint_ids] = reset_pos
        joint_vel[:, self.actuated_joint_ids] = (
            torch.rand((count, len(self.actuated_joint_ids)), device=self.device) * 2.0 - 1.0
        ) * self.cfg.reset_dof_vel_noise
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.joint_targets[env_ids] = joint_pos
        self.prev_targets[env_ids] = reset_pos
        self.cur_targets[env_ids] = reset_pos
        self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    def _compute_intermediate_values(self) -> None:
        env_origins = self.scene.env_origins
        object_state = self.object.data.root_state_w.clone()
        goal_state = self.goal_object.data.root_state_w.clone()
        object_state[:, :3] -= env_origins
        goal_state[:, :3] -= env_origins
        self.object_state_queue = self._update_queue(self.object_state_queue, object_state)
        observed_object_state = object_state.clone()
        if self.cfg.use_object_state_delay_noise:
            delay = torch.randint(0, self.object_state_queue.shape[1], (self.num_envs,), device=self.device)
            observed_object_state = self.object_state_queue[torch.arange(self.num_envs, device=self.device), delay].clone()
            observed_object_state[:, :3] += torch.randn_like(observed_object_state[:, :3]) * self.cfg.object_state_xyz_noise_std
            observed_object_state[:, 3:7] = self._sample_delta_quat(
                observed_object_state[:, 3:7], self.cfg.object_state_rotation_noise_degrees
            )

        self.object_pos = object_state[:, :3]
        self.object_rot = object_state[:, 3:7]
        self.object_vel = object_state[:, 7:13]
        self.observed_object_rot = observed_object_state[:, 3:7]
        self.goal_pos = goal_state[:, :3]
        self.goal_rot = goal_state[:, 3:7]

        body_state = self.robot.data.body_state_w.clone()
        body_state[:, :, :3] -= env_origins.unsqueeze(1)
        palm_state = body_state[:, self.palm_body_id]
        self.palm_quat = palm_state[:, 3:7]
        self.palm_vel = palm_state[:, 7:13]
        palm_offset = torch.tensor((0.0, -0.02, 0.16), dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        self.palm_center_pos = palm_state[:, :3] + quat_apply(self.palm_quat, palm_offset)

        fingertip_state = body_state[:, self.fingertip_body_ids]
        fingertip_pos = fingertip_state[:, :, :3]
        fingertip_quat = fingertip_state[:, :, 3:7]
        fingertip_pos_offset = fingertip_pos + quat_apply(
            fingertip_quat.reshape(-1, 4),
            self.fingertip_offsets.unsqueeze(0).repeat(self.num_envs, 1, 1).reshape(-1, 3),
        ).reshape(self.num_envs, len(self.fingertip_body_ids), 3)

        self.fingertip_pos_rel_object = fingertip_pos_offset - self.object_pos.unsqueeze(1)
        self.curr_fingertip_distances = torch.linalg.norm(self.fingertip_pos_rel_object, dim=-1)
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )
        self.fingertip_pos_rel_palm = fingertip_pos_offset - self.palm_center_pos.unsqueeze(1)

        noisy_offsets = self.object_keypoint_offsets * self.object_scale_noise_multiplier.unsqueeze(1)
        self.obj_keypoint_pos = self.object_pos.unsqueeze(1) + quat_apply(
            self.object_rot.unsqueeze(1).repeat(1, self.keypoint_signs.shape[0], 1).reshape(-1, 4),
            noisy_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, self.keypoint_signs.shape[0], 3)
        self.goal_keypoint_pos = self.goal_pos.unsqueeze(1) + quat_apply(
            self.goal_rot.unsqueeze(1).repeat(1, self.keypoint_signs.shape[0], 1).reshape(-1, 4),
            noisy_offsets.reshape(-1, 3),
        ).reshape(self.num_envs, self.keypoint_signs.shape[0], 3)
        self.obj_keypoint_pos_fixed_size = self.object_pos.unsqueeze(1) + quat_apply(
            self.object_rot.unsqueeze(1).repeat(1, self.keypoint_signs.shape[0], 1).reshape(-1, 4),
            self.object_keypoint_offsets_fixed_size.reshape(-1, 3),
        ).reshape(self.num_envs, self.keypoint_signs.shape[0], 3)
        self.goal_keypoint_pos_fixed_size = self.goal_pos.unsqueeze(1) + quat_apply(
            self.goal_rot.unsqueeze(1).repeat(1, self.keypoint_signs.shape[0], 1).reshape(-1, 4),
            self.object_keypoint_offsets_fixed_size.reshape(-1, 3),
        ).reshape(self.num_envs, self.keypoint_signs.shape[0], 3)

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        self.keypoints_rel_palm = self.obj_keypoint_pos - self.palm_center_pos.unsqueeze(1)
        self.keypoints_max_dist = torch.linalg.norm(self.keypoints_rel_goal, dim=-1).max(dim=-1).values
        self.keypoints_max_dist_fixed_size = torch.linalg.norm(
            self.obj_keypoint_pos_fixed_size - self.goal_keypoint_pos_fixed_size, dim=-1
        ).max(dim=-1).values
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )
        self.closest_keypoint_max_dist_fixed_size = torch.where(
            self.closest_keypoint_max_dist_fixed_size < 0.0,
            self.keypoints_max_dist_fixed_size,
            self.closest_keypoint_max_dist_fixed_size,
        )
        if self.cfg.fixed_size_keypoint_reward:
            self.keypoints_max_dist_for_reward = self.keypoints_max_dist_fixed_size
            self.closest_keypoint_max_dist_for_obs = self.closest_keypoint_max_dist_fixed_size
        else:
            self.keypoints_max_dist_for_reward = self.keypoints_max_dist
            self.closest_keypoint_max_dist_for_obs = self.closest_keypoint_max_dist
