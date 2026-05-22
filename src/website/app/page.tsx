"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type WheelEvent,
} from "react";

type NodeKind = "topic" | "decision" | "evidence" | "next";

type CanvasNode = {
  id: string;
  step: string;
  kind: NodeKind;
  status: string;
  label: string;
  title: string;
  summary: string;
  depth: number;
  x: number;
  y: number;
  width: number;
  detail: {
    title: string;
    body: string;
    items: string[];
  };
  branchTitle: string;
  branches: string[];
  experimentIds?: string[];
};

type VideoDirectoryConfig = {
  title: string;
  path: string;
  caption: string;
  defaultStep?: number;
  minSizeBytes?: number;
};

type ExperimentNote = {
  id: string;
  title: string;
  date: string;
  status: string;
  objective: string;
  result: string;
  conclusion: string;
  nextStep: string;
  runName: string;
  tags: string[];
  links: { label: string; href: string }[];
  facts: [string, string][];
  videoDir?: VideoDirectoryConfig;
  extraVideoDirs?: VideoDirectoryConfig[];
  videoMissingReason?: string;
  curveDescription: string;
  metricLabel: string;
  metricSuffix: string;
  yDomain: [number, number];
  curve: { step: string; value: number; syncStep?: number }[];
  extraCurves?: {
    curveDescription: string;
    metricLabel: string;
    metricSuffix: string;
    yDomain: [number, number];
    curve: { step: string; value: number; syncStep?: number }[];
  }[];
  noteSections: [string, string][];
};

type VideoRecord = {
  label: string;
  path: string;
  step: number;
  sizeBytes: number;
  modifiedMs: number;
};

const defaultView = { x: 390, y: 36, scale: 0.74 };
const storageKey = "simtoolreal-next-canvas-view-v5";

const nodes: CanvasNode[] = [
  {
    id: "scope",
    step: "1",
    kind: "topic",
    status: "Current",
    label: "Goal",
    title: "RL controller to real",
    summary: "Train the manipulation skill first, then use it for a higher-level real-robot controller.",
    depth: 0,
    x: 24,
    y: 16,
    width: 420,
    detail: {
      title: "High-level RL controller path",
      body:
        "The project goal is to get a reliable manipulation skill first, then use that skill inside a higher-level controller and eventually transfer the behavior toward the real robot.",
      items: [
        "Reproduce the original IsaacGym baseline so there is a behavior anchor.",
        "Port the task to IsaacLab and make the simple setting train first.",
        "Add randomization and harder variants only after the simple path is stable.",
      ],
    },
    branchTitle: "Main tracks",
    branches: [
      "Reproduce IsaacGym behavior.",
      "Migrate the task to IsaacLab.",
      "Use controlled experiments before reintroducing full procedural difficulty.",
    ],
  },
  {
    id: "baseline",
    step: "1.1",
    kind: "decision",
    status: "Done",
    label: "Reproduce",
    title: "Baseline",
    summary: "Reproduce the original baseline before changing the task or migrating it.",
    depth: 1,
    x: 128,
    y: 128,
    width: 620,
    detail: {
      title: "Reproduce baseline",
      body:
        "The first branch is about reproducing the original baseline. IsaacGym is the reference implementation used for this reproduction, not the main point of the node itself.",
      items: [
        "Full-object reproduction is useful as a reference but not fast enough for daily debugging.",
        "Marker/pen-like thin objects were tested as a simple-object path.",
        "Hammer became the main simple-object baseline.",
      ],
    },
    branchTitle: "Reproduce ladder",
    branches: [
      "Full procedural run.",
      "Single marker comparison.",
      "Single hammer baseline.",
    ],
  },
  {
    id: "full-set",
    step: "1.1.1",
    kind: "evidence",
    status: "Done",
    label: "Full",
    title: "Original object set",
    summary: "The broad baseline starts learning late and stays low, so it is a reference, not the main debug loop.",
    depth: 2,
    x: 232,
    y: 248,
    width: 610,
    detail: {
      title: "Full-object IsaacGym reproduction",
      body:
        "This run checks the original broad setting. It is important historically, but it is too slow and noisy to use as the first IsaacLab target.",
      items: [
        "Large env count on the RTX 3090.",
        "Success starts only after billions of frames.",
        "Use this as a reference curve, not as the first debug target.",
      ],
    },
    branchTitle: "Takeaway",
    branches: [
      "The full setting is hard to debug directly.",
      "Move to single-object tasks before migrating to IsaacLab.",
    ],
    experimentIds: ["isaacgym-3090-repro"],
  },
  {
    id: "marker",
    step: "1.1.2",
    kind: "evidence",
    status: "Checked",
    label: "Marker",
    title: "Thin-object trial",
    summary: "A marker-only path was tested; hammer became the cleaner baseline for the next step.",
    depth: 2,
    x: 232,
    y: 386,
    width: 610,
    detail: {
      title: "Marker single-object comparison",
      body:
        "The marker-only run is a useful comparison because it removes category variance but keeps a thin object geometry. It was not kept as the main simplified baseline.",
      items: [
        "Single marker object.",
        "Size/density noise disabled.",
        "Useful comparison, but hammer is the cleaner debug target.",
      ],
    },
    branchTitle: "Takeaway",
    branches: [
      "Object simplification alone is not enough; object geometry still matters.",
      "Keep the marker run as evidence, but do not make it the first IsaacLab target.",
    ],
    experimentIds: ["isaacgym-simple-marker-3090"],
  },
  {
    id: "hammer",
    step: "1.1.3",
    kind: "evidence",
    status: "Done",
    label: "Hammer",
    title: "Single-object baseline",
    summary: "One fixed hammer removes object variance and gives the cleanest IsaacGym target.",
    depth: 2,
    x: 232,
    y: 524,
    width: 610,
    detail: {
      title: "Hammer single-object baseline",
      body:
        "This became the intended simple baseline: one hammer, fixed object parameters, and delay/noise disabled while debugging the training stack.",
      items: [
        "Fixed hammer object.",
        "No size or density randomization.",
        "Delay/noise disabled for the debug branch.",
      ],
    },
    branchTitle: "Simplification rule",
    branches: [
      "Use one fixed hammer object.",
      "Remove size/density randomization.",
      "Re-enable delay/noise later only after parity is clear.",
    ],
    experimentIds: ["isaacgym-simple-hammer-3090"],
  },
  {
    id: "isaaclab",
    step: "1.2",
    kind: "decision",
    status: "Active",
    label: "IsaacLab",
    title: "Direct migration",
    summary: "Port the environment and train the same one-hammer task before adding variants.",
    depth: 1,
    x: 128,
    y: 630,
    width: 620,
    detail: {
      title: "IsaacLab migration branch",
      body:
        "The second branch ports the task to IsaacLab. This should stay comparable to the IsaacGym hammer baseline before the project adds more object variants.",
      items: [
        "Scene: Kuka plus SharpA, table, object, goal marker, lights, camera.",
        "Trainer: CleanRL-style PPO path for the DirectRLEnv.",
        "Evidence: W&B metrics, videos, and compact trajectory logs.",
      ],
    },
    branchTitle: "Migration steps",
    branches: [
      "Direct environment port.",
      "Single-hammer IsaacLab training.",
      "Metrics, video, and replay logs.",
      "Checkpoint evaluation and shaped trajectories.",
    ],
  },
  {
    id: "direct-env",
    step: "1.2.1",
    kind: "decision",
    status: "Built",
    label: "Env",
    title: "DirectRLEnv port",
    summary: "Scene and task API exist; now training behavior is the important comparison.",
    depth: 2,
    x: 232,
    y: 738,
    width: 610,
    detail: {
      title: "DirectRLEnv implementation",
      body:
        "The IsaacLab task uses a DirectRLEnv path rather than ManagerBased. The goal is to keep the old IsaacGym task available while the new scene, reset, reward, and trainer are validated.",
      items: [
        "Kuka plus SharpA scene.",
        "Object and goal reset logic.",
        "Reward/reset parity is still the training-critical area.",
      ],
    },
    branchTitle: "Implementation state",
    branches: [
      "Direct env is the right implementation shape.",
      "Training parity is the next hard check.",
    ],
  },
  {
    id: "isaaclab-hammer",
    step: "1.2.2",
    kind: "evidence",
    status: "Evidence",
    label: "Hammer",
    title: "First one-hammer run",
    summary: "The first IsaacLab simple-hammer run showed the task could train, but the metric story was still unclear.",
    depth: 2,
    x: 232,
    y: 856,
    width: 610,
    detail: {
      title: "First IsaacLab one-hammer training run",
      body:
        "This is the first strong IsaacLab one-hammer training result. It proved the simplified task could work, but it mainly tracked average successes, so the next step was to add clearer episode-level metrics.",
      items: [
        "Single hammer object.",
        "Runs on the RTX 5080 Docker path.",
        "This run came before the richer last-episode logging pass.",
      ],
    },
    branchTitle: "Takeaway",
    branches: [
      "The simple IsaacLab task can train.",
      "Average success count alone was not enough to explain behavior.",
      "Next step: add episode-level metrics and trajectory logs.",
    ],
    experimentIds: ["isaaclab-simple-hammer-oneobj-5080"],
  },
  {
    id: "replay-logs",
    step: "1.2.3",
    kind: "evidence",
    status: "Done",
    label: "Metrics",
    title: "Add episode logging",
    summary: "After the first hammer run, add last-episode success curves and trajectory logs to track the experiment process.",
    depth: 2,
    x: 232,
    y: 974,
    width: 610,
    detail: {
      title: "Episode metrics and trajectory logging",
      body:
        "This came after the first hammer run. The goal was to make the experiment easier to reason about: record last-episode success curves, keep videos, and save compact trajectory logs for replay.",
      items: [
        "Add episode/success_rate_any.",
        "Add episode/success_count_mean.",
        "Save selected-env trajectory logs for later 3D replay.",
      ],
    },
    branchTitle: "Why this step exists",
    branches: [
      "The first hammer run worked but was hard to interpret from one success metric.",
      "Last-episode metrics show whether all envs are actually succeeding.",
      "Trajectory logs preserve the process so the visualizer can replay what happened.",
    ],
    experimentIds: ["isaaclab-richer-log-5080"],
  },
  {
    id: "checkpoint-eval",
    step: "1.2.4",
    kind: "evidence",
    status: "Evidence",
    label: "Eval",
    title: "Checkpoint rollout",
    summary: "Load the trained checkpoint, test the normal reset distribution, then force shaped goal trajectories and render evidence.",
    depth: 2,
    x: 232,
    y: 1092,
    width: 610,
    detail: {
      title: "Evaluate checkpoint behavior",
      body:
        "This node checks whether a saved IsaacLab checkpoint can be loaded and used as a policy, then probes whether it can follow structured goal paths that still stay inside the training target volume.",
      items: [
        "Load best_episode_success.pt from the richer-log hammer run.",
        "First test the normal fixed target setting as a sanity check.",
        "Then test line, circle, and strike-shaped moving goals with rendered rollout videos.",
      ],
    },
    branchTitle: "Eval ladder",
    branches: [
      "Checkpoint loads and produces a valid rollout.",
      "Fixed target eval reaches high success under the comparable simple-hammer setting.",
      "Shaped trajectories probe whether the policy tracks moving goals rather than only replaying one pose.",
    ],
    experimentIds: ["isaaclab-checkpoint-eval-5080"],
  },
  {
    id: "next",
    step: "1.3",
    kind: "next",
    status: "Next",
    label: "Ladder",
    title: "Training ladder",
    summary: "One hammer first, then variants, then the full procedural object set.",
    depth: 1,
    x: 128,
    y: 1240,
    width: 620,
    detail: {
      title: "Controlled next training path",
      body:
        "The safe path is to stabilize the simplest comparable task first. Only then should the project add object variants, delay/noise, and the full procedural distribution back.",
      items: [
        "One fixed hammer.",
        "Single-category variants.",
        "Full procedural object set.",
      ],
    },
    branchTitle: "Order",
    branches: [
      "One fixed hammer.",
      "Add variants inside one category.",
      "Restore full procedural objects and randomization.",
    ],
  },
];

const experiments: ExperimentNote[] = [
  {
    id: "isaaclab-checkpoint-eval-5080",
    title: "IsaacLab checkpoint evaluation",
    date: "2026-05-16",
    status: "checkpoint rollout",
    objective:
      "Evaluate whether the trained IsaacLab hammer checkpoint can be loaded, produce high success under the comparable target setting, and follow simple shaped goal trajectories.",
    result:
      "best_episode_success.pt loads successfully. The latest diagnostic run uses a 5cm gate and logs per-frame render-env metrics plus per-env threshold summaries.",
    conclusion:
      "The old high success number is not visual success. Reward-tolerance any-success is 100%, but the 5cm gated sequence completion is only 3.33%, and the rendered video env completes 0 targets.",
    nextStep:
      "Use visual_success_rate_any and render_env_visual_success_frame_rate when judging videos. Keep the wider train-tolerance metric only as reward/debug context.",
    runName: "best_episode_success checkpoint eval",
    tags: ["isaaclab", "eval", "checkpoint", "trajectory", "5080"],
    links: [],
    facts: [
      ["checkpoint", "best_episode_success.pt"],
      ["ckpt update", "14,653"],
      ["ckpt step", "1.920B"],
      ["peak log any", "79.89% at update 16,609"],
      ["eval envs", "30"],
      ["eval length", "900 steps"],
      ["gpu", "RTX 5080 only"],
      ["visual tol", "5cm + lifted"],
      ["train tol", "11.25cm + lifted"],
      ["visual/gate any", "56.67%"],
      ["reward-tol any", "100%"],
      ["video env visual", "0.22% frames"],
      ["video env reward", "13.89% frames"],
      ["sequence complete", "3.33%"],
      ["video targets", "0 / 4"],
      ["render best dist", "0.0487"],
    ],
    videoDir: {
      title: "5cm-gated strike diagnostic",
      path: "isaaclab_eval/eval_best_episode_gated_strike_5cm_gate_diag_30env_900step_5080",
      caption: "Latest eval with 5cm gate, multi-threshold diagnostics, render_env_trace.csv, and env_summary.csv.",
      minSizeBytes: 100000,
    },
    extraVideoDirs: [
      {
        title: "Line goal trajectory",
        path: "isaaclab_eval/eval_best_episode_line_30env_600step_5080",
        caption: "Goal moves along one axis inside the training target volume. Success rate any: 93.33%.",
        minSizeBytes: 100000,
      },
      {
        title: "Circle goal trajectory",
        path: "isaaclab_eval/eval_best_episode_circle_30env_600step_5080",
        caption: "Goal follows a small circle inside the target volume. Success rate any: 96.67%.",
        minSizeBytes: 100000,
      },
      {
        title: "Strike-shaped trajectory",
        path: "isaaclab_eval/eval_best_episode_strike_30env_600step_5080",
        caption: "Goal moves through a short hammer-like stroke with pitch change. Success rate any: 96.67%; visible env best keypoint distance: 0.0680.",
        minSizeBytes: 100000,
      },
    ],
    curveDescription:
      "Latest gated strike eval. Visual success is stricter: max keypoint distance <= 5cm and object lifted. Train tolerance uses the reward threshold, 11.25cm.",
    metricLabel: "checkpoint eval metric",
    metricSuffix: "%",
    yDomain: [0, 100],
    curve: [
      { step: "visual any", value: 56.67 },
      { step: "video visual", value: 0.22 },
      { step: "reward any", value: 100.0 },
      { step: "seq complete", value: 3.33 },
    ],
    extraCurves: [
      {
        curveDescription:
          "Rendered env frame rates. These are the numbers to compare against the visible video.",
        metricLabel: "render-env frame rate",
        metricSuffix: "%",
        yDomain: [0, 25],
        curve: [
          { step: "visual", value: 0.22 },
          { step: "reward tol", value: 13.89 },
          { step: "lifted", value: 21.11 },
        ],
      },
      {
        curveDescription:
          "Best keypoint distance for the rendered env. Lower is better. The visual-success threshold is 0.05m; this run barely crosses it for two frames, but not long enough to complete a target.",
        metricLabel: "render-env best distance",
        metricSuffix: "",
        yDomain: [0, 0.12],
        curve: [
          { step: "visual tol", value: 0.05 },
          { step: "best dist", value: 0.0487 },
          { step: "train tol", value: 0.1125 },
        ],
      },
    ],
    noteSections: [
      ["Motivation", "Training videos show behavior during learning, but checkpoint eval answers a different question: can a saved policy be loaded later and reproduce useful behavior under controlled goals?"],
      ["Problem found", "The old aggregate success was too optimistic for video reading. It counted whether an env ever got within the training reward tolerance. That tolerance is 11.25cm, so the object can look visibly off-target and still count."],
      ["Latest result", "With a 5cm gate, visual/gate any-success is 56.67%, reward-tolerance any-success remains 100%, but full sequence completion is only 3.33%. The rendered env completes 0 targets."],
      ["Trace evidence", "render_env_trace.csv shows the visible env only has two 5cm-success frames, at steps 898 and 899. That is not enough for the 10-step hold requirement, so the video correctly looks unsuccessful."],
      ["Takeaway", "The checkpoint can sometimes move the hammer near the target, but this is not yet a reliable visual trajectory follower. For report screenshots/videos, use 5cm gate metrics and sequence completion, not reward-tolerance success_rate_any."],
      ["Next evals", "Evaluate fixed, line, circle, and strike again with the same 5cm visual threshold. Then select videos by render_env_visual_success_frame_rate, not by aggregate any-success."],
    ],
  },
  {
    id: "isaaclab-richer-log-5080",
    title: "IsaacLab last-episode metrics",
    date: "2026-05-15",
    status: "last episode metrics",
    objective:
      "Reproduce Isaac Gym simple one-object hammer in Isaac Lab and make success logging less ambiguous.",
    result:
      "Adds last-episode success summaries: whether each env had at least one success, and the mean number of successes in the previous episode.",
    conclusion:
      "Use these metrics to separate visual success frequency from average repeated-success count.",
    nextStep:
      "Compare video behavior against episode/success_rate_any and episode/success_count_mean.",
    runName: "isaaclab_simple_hammer_oneobj_metrics_5080_20260515_010906",
    tags: ["isaaclab", "simple-hammer", "one-object", "5080", "richer-log"],
    links: [
      {
        label: "W&B project",
        href: "https://wandb.ai/waybabag/simtoolreal?nw=nwuserwaybaba",
      },
    ],
    facts: [
      ["envs", "8,190"],
      ["gpu", "RTX 5080"],
      ["object", "single hammer"],
      ["video", "on"],
      ["trajectory", "on"],
      ["any success", "episode/success_rate_any"],
      ["mean count", "episode/success_count_mean"],
      ["distribution", "episode/success_count_distribution"],
      ["active count", "train/active_success_count_mean"],
      ["peak any", "~79.9%"],
      ["peak mean count", "~10.44"],
    ],
    videoDir: {
      title: "Live richer-log videos",
      path: "isaaclab_cleanrl_train/simtoolreal/2026-05-15/isaaclab_simple_hammer_oneobj_metrics_5080_20260515_010906/videos/train",
      caption: "Local mp4 captures from the richer-log Isaac Lab run.",
      minSizeBytes: 100000,
    },
    curveDescription:
      "episode/success_rate_any: percentage of envs whose previous episode had at least one success. Parsed from the local CleanRL log.",
    metricLabel: "episode any-success rate",
    metricSuffix: "%",
    yDomain: [0, 100],
    curve: [
      { step: "0M", value: 0, syncStep: 16 },
      { step: "181M", value: 1.76, syncStep: 22048 },
      { step: "361M", value: 7.86, syncStep: 44080 },
      { step: "541M", value: 42.34, syncStep: 66112 },
      { step: "722M", value: 55.92, syncStep: 88144 },
      { step: "902M", value: 61.2, syncStep: 110176 },
      { step: "1.08B", value: 66.76, syncStep: 132208 },
      { step: "1.26B", value: 73.39, syncStep: 154240 },
      { step: "1.44B", value: 70.55, syncStep: 176272 },
      { step: "1.62B", value: 74.7, syncStep: 198304 },
      { step: "1.80B", value: 76.74, syncStep: 220336 },
      { step: "1.99B", value: 76.17, syncStep: 242384 },
      { step: "2.17B", value: 79.1, syncStep: 264416 },
      { step: "2.35B", value: 73.83, syncStep: 286448 },
      { step: "2.53B", value: 36.4, syncStep: 308480 },
      { step: "2.71B", value: 19.29, syncStep: 330512 },
      { step: "2.89B", value: 29.05, syncStep: 352544 },
      { step: "3.07B", value: 37.47, syncStep: 374576 },
      { step: "3.25B", value: 30.24, syncStep: 396608 },
      { step: "3.43B", value: 38.14, syncStep: 418640 },
      { step: "3.61B", value: 31.32, syncStep: 440672 },
      { step: "3.79B", value: 20.59, syncStep: 462704 },
    ],
    extraCurves: [
      {
        curveDescription:
          "episode/success_count_mean: mean number of successes in the previous episode across envs. Parsed from the same local CleanRL log.",
        metricLabel: "episode mean success count",
        metricSuffix: "",
        yDomain: [0, 11],
        curve: [
          { step: "0M", value: 0, syncStep: 16 },
          { step: "181M", value: 0.0237, syncStep: 22048 },
          { step: "361M", value: 0.1313, syncStep: 44080 },
          { step: "541M", value: 1.8643, syncStep: 66112 },
          { step: "722M", value: 3.7623, syncStep: 88144 },
          { step: "902M", value: 5.5902, syncStep: 110176 },
          { step: "1.08B", value: 7.1184, syncStep: 132208 },
          { step: "1.26B", value: 8.5139, syncStep: 154240 },
          { step: "1.44B", value: 8.1967, syncStep: 176272 },
          { step: "1.62B", value: 8.8969, syncStep: 198304 },
          { step: "1.80B", value: 9.4031, syncStep: 220336 },
          { step: "1.99B", value: 9.6954, syncStep: 242384 },
          { step: "2.17B", value: 9.662, syncStep: 264416 },
          { step: "2.35B", value: 8.8427, syncStep: 286448 },
          { step: "2.53B", value: 4.306, syncStep: 308480 },
          { step: "2.71B", value: 2.0557, syncStep: 330512 },
          { step: "2.89B", value: 3.2451, syncStep: 352544 },
          { step: "3.07B", value: 4.1713, syncStep: 374576 },
          { step: "3.25B", value: 3.2799, syncStep: 396608 },
          { step: "3.43B", value: 4.5222, syncStep: 418640 },
          { step: "3.61B", value: 3.4103, syncStep: 440672 },
          { step: "3.79B", value: 2.3099, syncStep: 462704 },
        ],
      },
    ],
    noteSections: [
      ["Why this matters", "A policy can have a high average success count while some envs still fail. The any-success metric asks whether each env succeeded at least once in its previous episode."],
      ["Exact fields", "The trainer logs episode/success_rate_any, episode/success_count_mean, episode/success_count_distribution, and train/active_success_count_mean to W&B."],
      ["Distribution", "The trainer also logs episode/success_count_distribution as a W&B histogram over each env's previous-episode success count. It answers how many envs had 0, 1, 2, ... successes in that last episode. The local text log has the mean/rate curves, but not raw histogram bins, so this canvas records the field name instead of drawing fake bins."],
      ["Logging target", "Keep videos and trajectory logs together so visual inspection and replay data point to the same training segment."],
    ],
  },
  {
    id: "isaaclab-simple-hammer-oneobj-5080",
    title: "IsaacLab hammer baseline",
    date: "2026-05-12",
    status: "positive baseline",
    objective:
      "Reproduce the Isaac Gym simple one-object-variant result in Isaac Lab.",
    result:
      "Videos show the policy finishing the task. Final snapshot: ~5.31B steps, ~8.6 avg successes, peak ~9.6.",
    conclusion:
      "This is the first strong IsaacLab baseline for the simplified task.",
    nextStep:
      "Keep this as the first positive Isaac Lab baseline, then compare against Isaac Gym simple hammer and run controlled simplification variants.",
    runName: "isaaclab_simple_hammer_oneobj_video_5080_20260512_155247",
    tags: ["isaaclab", "simple-hammer", "one-object", "5080", "video"],
    links: [
      {
        label: "W&B run",
        href: "https://wandb.ai/waybabag/simtoolreal/runs/uid_isaaclab_simple_hammer_oneobj_video_5080_20260512_155247",
      },
      {
        label: "W&B project",
        href: "https://wandb.ai/waybabag/simtoolreal?nw=nwuserwaybaba",
      },
    ],
    facts: [
      ["envs", "8,190"],
      ["gpu", "RTX 5080"],
      ["global step", "~5.31B"],
      ["avg successes", "~8.6"],
      ["peak avg successes", "~9.6"],
      ["reward mean", "~11.6"],
      ["object", "single hammer"],
      ["videos", "109"],
    ],
    videoDir: {
      title: "Isaac Lab training videos",
      path: "isaaclab_cleanrl_train/simtoolreal/2026-05-12/isaaclab_simple_hammer_oneobj_video_5080_20260512_155247/videos/train",
      caption: "Local mp4 captures from the finished Isaac Lab single-hammer run.",
      defaultStep: 474000,
      minSizeBytes: 100000,
    },
    curveDescription:
      "The finished run logged average success count, not true episode success rate. Future runs should log both metrics.",
    metricLabel: "avg successes / episode",
    metricSuffix: "",
    yDomain: [0, 10],
    curve: [
      { step: "0.00B", value: 0.0 },
      { step: "0.35B", value: 0.077 },
      { step: "0.71B", value: 3.378 },
      { step: "1.06B", value: 6.56 },
      { step: "1.42B", value: 8.157 },
      { step: "1.77B", value: 8.095 },
      { step: "2.12B", value: 8.595 },
      { step: "2.48B", value: 8.696 },
      { step: "2.83B", value: 8.246 },
      { step: "3.19B", value: 5.896 },
      { step: "3.54B", value: 4.373 },
      { step: "3.89B", value: 5.021 },
      { step: "4.25B", value: 3.906 },
      { step: "4.60B", value: 4.91 },
      { step: "4.96B", value: 8.598 },
      { step: "5.31B", value: 8.635 },
    ],
    noteSections: [],
  },
  {
    id: "isaacgym-simple-hammer-3090",
    title: "IsaacGym hammer debug",
    date: "2026-05-12",
    status: "debug baseline",
    objective:
      "Track the intended simple-hammer experiment as the main simplified IsaacGym debug baseline.",
    result:
      "4,092 envs on 3090, fixed hammer, 64+ videos. Snapshot: ~3.8% success at 1.55B frames; peak ~5.9%.",
    conclusion:
      "The hammer simplification learns earlier than the full reproduction and is useful for debugging.",
    nextStep:
      "Inspect videos and curve, then make W&B media sync clean for the next run.",
    runName: "00_isaacgym_simple_hammer_3090_video_2026-05-11_18-21-06",
    tags: ["isaacgym", "simple-hammer", "3090", "reduced-randomization"],
    links: [
      {
        label: "W&B project",
        href: "https://wandb.ai/waybabag/simtoolreal?nw=nwuserwaybaba",
      },
    ],
    facts: [
      ["envs", "4,092"],
      ["gpu", "RTX 3090"],
      ["global step", "~1.55B"],
      ["elapsed", "~584 min"],
      ["current success", "~3.8%"],
      ["observed peak", "~5.9%"],
      ["object type", "hammer only"],
      ["scale noise", "off: [1.0, 1.0]"],
      ["delay/noise", "off"],
      ["local videos", "64+"],
      ["W&B", "needs safe.directory fix"],
    ],
    videoDir: {
      title: "Local training videos",
      path: "train_dir/simtoolreal/2026-05-11/isaacgym_simple_hammer_3090_video_2026-05-11_18-21-06/videos",
      caption: "All local mp4 captures from the intended simple-hammer 3090 video run.",
    },
    curveDescription:
      "Parsed from the local Isaac Gym launcher log. It rises after ~600M frames, peaks near 5.9%, and is about 3.8% in this snapshot.",
    metricLabel: "success rate",
    metricSuffix: "%",
    yDomain: [0, 8],
    curve: [
      { step: "0M", value: 0.0 },
      { step: "103M", value: 0.0 },
      { step: "207M", value: 0.0 },
      { step: "310M", value: 0.0 },
      { step: "413M", value: 0.1 },
      { step: "517M", value: 0.1 },
      { step: "620M", value: 1.0 },
      { step: "723M", value: 3.7 },
      { step: "827M", value: 5.9 },
      { step: "930M", value: 5.7 },
      { step: "1033M", value: 5.2 },
      { step: "1137M", value: 5.6 },
      { step: "1240M", value: 5.4 },
      { step: "1344M", value: 3.7 },
      { step: "1447M", value: 5.2 },
      { step: "1550M", value: 3.8 },
    ],
    noteSections: [
      ["Goal", "Make the task much simpler than the earlier reproduction run, using the hammer object intended for this simplified-object experiment."],
      ["Changes", "One fixed hammer distribution, fixed geometry and density, no scale noise, no action/observation delay, no object-state noise, and no joint-velocity observation noise."],
    ],
  },
  {
    id: "isaacgym-simple-marker-3090",
    title: "IsaacGym marker comparison",
    date: "2026-05-11",
    status: "comparison",
    objective:
      "Test a single thin-object setting before settling on the hammer debug baseline.",
    result:
      "The long no-video marker run reached about 5.4% success ratio near 3.0B frames. A later short video check with 4,092 envs stayed near 0% over the first ~128M frames.",
    conclusion:
      "The marker path is useful evidence, but hammer is the cleaner one-object baseline for IsaacLab comparison.",
    nextStep:
      "Keep marker as a comparison node; use hammer as the first main IsaacLab parity target.",
    runName: "00_isaacgym_simple_marker_3090_2026-05-11_00-45-22",
    tags: ["isaacgym", "simple-marker", "3090", "single-object"],
    links: [
      {
        label: "W&B project",
        href: "https://wandb.ai/waybabag/simtoolreal?nw=nwuserwaybaba",
      },
    ],
    facts: [
      ["long run envs", "12,288"],
      ["video check envs", "4,092"],
      ["gpu", "RTX 3090"],
      ["object type", "marker only"],
      ["scale noise", "off: [1.0, 1.0]"],
      ["delay/noise", "off"],
      ["long-run frame", "~3.01B"],
      ["long-run success", "~5.4%"],
      ["short video check", "~0% at ~128M frames"],
    ],
    videoDir: {
      title: "Marker video check",
      path: "train_dir/simtoolreal/2026-05-11/isaacgym_simple_marker_3090_video_2026-05-11_12-16-19/videos",
      caption: "Short local mp4 captures from the marker video check.",
    },
    curveDescription:
      "The marker comparison is mixed: the long no-video run eventually reaches a low success ratio, while the short video check is too early to show learning.",
    metricLabel: "success rate",
    metricSuffix: "%",
    yDomain: [0, 8],
    curve: [
      { step: "0.0B", value: 0.0 },
      { step: "0.5B", value: 0.0 },
      { step: "1.0B", value: 0.4 },
      { step: "1.5B", value: 2.2 },
      { step: "2.0B", value: 3.9 },
      { step: "2.5B", value: 5.0 },
      { step: "3.0B", value: 5.4 },
    ],
    noteSections: [
      ["Why it is separate", "This is not the same evidence as the hammer baseline. It removes object-category variance but keeps a different, thinner geometry."],
      ["Decision", "Use it as a comparison node. The hammer task remains the main one-object baseline for IsaacLab."],
    ],
  },
  {
    id: "isaacgym-3090-repro",
    title: "IsaacGym reproduction",
    date: "2026-05-11",
    status: "reference run",
    objective:
      "Reproduce the original SimToolReal Isaac Gym result on the RTX 3090 run from May 9.",
    result:
      "Learning starts around 2.5B frames and reaches about 5.4% success by 7.12B frames.",
    conclusion:
      "This run is a reference point; simplified tasks are better for iteration.",
    nextStep:
      "Move back to a simpler Isaac Gym environment first: reduce object count, remove randomization, and get a clean baseline before adding difficulty back.",
    runName: "00_isaacgym_repro_3090_video_2026-05-09_19-58-53",
    tags: ["isaacgym", "reproduction", "3090", "12288-envs"],
    links: [
      {
        label: "W&B run",
        href: "https://wandb.ai/waybabag/simtoolreal/runs/uid_00_isaacgym_repro_3090_video_2026-05-09_19-58-53?nw=nwuserwaybaba",
      },
      {
        label: "W&B project",
        href: "https://wandb.ai/waybabag/simtoolreal?nw=nwuserwaybaba",
      },
    ],
    facts: [
      ["envs", "12,288"],
      ["gpu", "RTX 3090"],
      ["global step", "7.12B"],
      ["elapsed", "~1,726 min"],
      ["final success", "~5.4%"],
      ["media", "97 local videos"],
    ],
    videoDir: {
      title: "Local training videos",
      path: "train_dir/simtoolreal/2026-05-09/isaacgym_repro_3090_video_2026-05-09_19-58-53/videos",
      caption: "All local mp4 captures from the Isaac Gym reproduction run.",
    },
    curveDescription:
      "Parsed from the local Isaac Gym run log. Success starts moving only after billions of frames and remains low by the end.",
    metricLabel: "success rate",
    metricSuffix: "%",
    yDomain: [0, 8],
    curve: [
      { step: "0.0B", value: 0.0 },
      { step: "0.4B", value: 0.0 },
      { step: "0.8B", value: 0.0 },
      { step: "1.3B", value: 0.0 },
      { step: "1.7B", value: 0.0 },
      { step: "2.1B", value: 0.0 },
      { step: "2.5B", value: 0.2 },
      { step: "2.9B", value: 1.3 },
      { step: "3.4B", value: 2.2 },
      { step: "3.8B", value: 3.5 },
      { step: "4.2B", value: 3.7 },
      { step: "4.6B", value: 4.8 },
      { step: "5.0B", value: 5.1 },
      { step: "5.4B", value: 5.2 },
      { step: "5.9B", value: 4.9 },
      { step: "6.3B", value: 5.4 },
      { step: "6.7B", value: 5.5 },
      { step: "7.1B", value: 5.4 },
    ],
    noteSections: [
      ["Observation", "The run remains at 0% success for the first ~2.5B frames and only climbs to ~5.4% by 7.12B frames."],
      ["Conclusion", "The reproduction is useful as a reference, while simplified tasks are better for routine iteration."],
    ],
  },
];

const parentByStep: Record<string, string> = {
  "1.1": "1",
  "1.1.1": "1.1",
  "1.1.2": "1.1",
  "1.1.3": "1.1",
  "1.2": "1",
  "1.2.1": "1.2",
  "1.2.2": "1.2",
  "1.2.3": "1.2",
  "1.3": "1",
};

function clampScale(value: number) {
  return Math.min(1.45, Math.max(0.58, value));
}

function loadView() {
  if (typeof window === "undefined") {
    return defaultView;
  }
  try {
    const saved = window.localStorage.getItem(storageKey);
    return saved ? { ...defaultView, ...JSON.parse(saved) } : defaultView;
  } catch {
    return defaultView;
  }
}

export default function ResearchStudioPage() {
  const [view, setView] = useState(defaultView);
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [collapsedSteps, setCollapsedSteps] = useState<Set<string>>(() => new Set());
  const dragMovedRef = useRef(false);
  const [drag, setDrag] = useState<{
    pointerId: number;
    startX: number;
    startY: number;
    originX: number;
    originY: number;
  } | null>(null);
  const activeNode = useMemo(
    () => nodes.find((node) => node.id === activeNodeId) ?? null,
    [activeNodeId],
  );
  const activeNodeParent = activeNode
    ? (nodes.find((node) => node.step === parentByStep[activeNode.step]) ?? null)
    : null;

  function isHiddenByCollapsedAncestor(step: string) {
    let parent = parentByStep[step];
    while (parent) {
      if (collapsedSteps.has(parent)) {
        return true;
      }
      parent = parentByStep[parent];
    }
    return false;
  }

  function toggleCollapsed(step: string) {
    setCollapsedSteps((current) => {
      const next = new Set(current);
      if (next.has(step)) {
        next.delete(step);
      } else {
        next.add(step);
      }
      return next;
    });
  }

  const visibleNodes = nodes.filter((node) => !isHiddenByCollapsedAncestor(node.step));
  const childCountByStep = nodes.reduce<Record<string, number>>((acc, node) => {
    const parent = parentByStep[node.step];
    if (parent) {
      acc[parent] = (acc[parent] ?? 0) + 1;
    }
    return acc;
  }, {});

  useEffect(() => {
    setView(loadView());
  }, []);

  useEffect(() => {
    window.localStorage.setItem(storageKey, JSON.stringify(view));
  }, [view]);

  function updateScale(delta: number) {
    setView((current) => ({ ...current, scale: clampScale(current.scale + delta) }));
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || (event.target as HTMLElement).closest("[data-no-pan='true']")) {
      return;
    }
    dragMovedRef.current = false;
    setDrag({
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: view.x,
      originY: view.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    if (Math.abs(event.clientX - drag.startX) > 3 || Math.abs(event.clientY - drag.startY) > 3) {
      dragMovedRef.current = true;
    }
    setView((current) => ({
      ...current,
      x: drag.originX + event.clientX - drag.startX,
      y: drag.originY + event.clientY - drag.startY,
    }));
  }

  function stopDrag(event: PointerEvent<HTMLDivElement>) {
    if (!drag || drag.pointerId !== event.pointerId) {
      return;
    }
    setDrag(null);
    event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    if (event.ctrlKey || event.metaKey) {
      setView((current) => ({
        ...current,
        scale: clampScale(current.scale * Math.exp(-event.deltaY * 0.001)),
      }));
      return;
    }
    setView((current) => ({
      ...current,
      x: current.x - event.deltaX / current.scale,
      y: current.y - event.deltaY / current.scale,
    }));
  }

  return (
    <main className="studio-workbench">
      <section
        className={`research-canvas ${drag ? "is-panning" : ""}`}
        aria-label="SimToolReal migration canvas"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={stopDrag}
        onPointerCancel={stopDrag}
        onWheel={handleWheel}
      >
        <div
          className="board"
          style={{
            transform: `translate3d(${view.x}px, ${view.y}px, 0) scale(${view.scale})`,
          }}
        >
          <TodoBoard />
          <Axis />
          <Connectors visibleNodes={visibleNodes} />
          {visibleNodes.map((node) => (
            <TimelineNode
              key={node.id}
              node={node}
              childCount={childCountByStep[node.step] ?? 0}
              collapsed={collapsedSteps.has(node.step)}
              onToggle={() => toggleCollapsed(node.step)}
              onSelect={() => {
                if (!dragMovedRef.current) {
                  setActiveNodeId(node.id);
                }
              }}
            />
          ))}
        </div>

        <div className="canvas-controls" data-no-pan="true">
          <button type="button" onClick={() => updateScale(-0.08)}>
            -
          </button>
          <span>{Math.round(view.scale * 100)}%</span>
          <button type="button" onClick={() => updateScale(0.08)}>
            +
          </button>
          <button type="button" onClick={() => setView(defaultView)}>
            reset
          </button>
        </div>
      </section>

      {activeNode ? (
        <DetailDialog
          node={activeNode}
          parentNode={activeNodeParent}
          onClose={() => setActiveNodeId(null)}
        />
      ) : null}
    </main>
  );
}

function TodoBoard() {
  return (
    <aside className="todo-board" aria-label="Todo" data-no-pan="true">
      <div className="todo-title-row">
        <div>
          <h2>Todo</h2>
        </div>
        <span className="todo-icon" aria-hidden="true">☷</span>
      </div>
      <section className="todo-group">
        <h3>Main</h3>
        <TodoCard
          label="Goal"
          text="Train manipulation skill first; use it for real-controller path later."
        />
        <TodoCard
          label="Reproduce"
          text="Use IsaacGym full, marker, and hammer runs as the reference ladder."
        />
      </section>
      <section className="todo-group">
        <h3>Near term</h3>
        <TodoCard
          label="IsaacLab"
          text="Match one-hammer IsaacGym before adding object variants."
        />
        <TodoCard
          label="Evidence"
          text="Keep videos, W&B metrics, and trajectory logs in the same run cards."
        />
      </section>
    </aside>
  );
}

function TodoCard({ label, text }: { label: string; text: string }) {
  return (
    <article className="todo-card">
      <strong>{label}</strong>
      <span>{text}</span>
    </article>
  );
}

function Axis() {
  return (
    <div className="axis" aria-hidden="true">
      <span className="axis-dot" />
      <span className="axis-line" />
      <span className="axis-label start">start</span>
      <span className="axis-label next">next</span>
    </div>
  );
}

function Connectors({ visibleNodes }: { visibleNodes: CanvasNode[] }) {
  const visibleSteps = new Set(visibleNodes.map((node) => node.step));

  return (
    <svg className="connectors" viewBox="0 0 1300 1500" aria-hidden="true">
      {visibleNodes.map((node) => {
        const parentStep = parentByStep[node.step];
        if (!parentStep || !visibleSteps.has(parentStep)) {
          return null;
        }
        const parent = nodes.find((candidate) => candidate.step === parentStep);
        if (!parent) {
          return null;
        }
        const startX = parent.x + 24;
        const startY = parent.y + 24;
        const endX = node.x + 24;
        const endY = node.y + 24;
        const midY = startY + (endY - startY) * 0.55;
        return (
          <path
            key={`${parent.step}-${node.step}`}
            className={`line ${node.kind}`}
            d={`M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`}
          />
        );
      })}
    </svg>
  );
}

function TimelineNode({
  node,
  childCount,
  collapsed,
  onToggle,
  onSelect,
}: {
  node: CanvasNode;
  childCount: number;
  collapsed: boolean;
  onToggle: () => void;
  onSelect: () => void;
}) {
  const clickable = Boolean(node.experimentIds?.length);
  const cardContent = (
    <h3>
      <span className="node-keyword">{node.label}</span>
      <span>{node.title}</span>
    </h3>
  );

  return (
    <section
      className={`node ${node.kind}`}
      style={
        {
          "--x": `${node.x}px`,
          "--y": `${node.y}px`,
          "--card-width": `${node.width}px`,
        } as CSSProperties
      }
    >
      {childCount > 0 ? (
        <button
          className={`node-icon node-icon-button ${collapsed ? "is-collapsed" : ""}`}
          type="button"
          data-no-pan="true"
          onClick={onToggle}
          aria-label={collapsed ? "Expand node" : "Collapse node"}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {node.label[0]}
        </button>
      ) : (
        <div className="node-icon">{node.label[0]}</div>
      )}
      {clickable ? (
        <button className="node-card" type="button" onClick={onSelect}>
          {cardContent}
        </button>
      ) : (
        <div className="node-card node-card-static" aria-label={`${node.label} ${node.title}`}>
          {cardContent}
        </div>
      )}
    </section>
  );
}

function DetailDialog({
  node,
  parentNode,
  onClose,
}: {
  node: CanvasNode;
  parentNode: CanvasNode | null;
  onClose: () => void;
}) {
  const relatedExperiments =
    node.experimentIds
      ?.map((experimentId) => experiments.find((experiment) => experiment.id === experimentId))
      .filter((experiment): experiment is ExperimentNote => Boolean(experiment)) ?? [];
  const [activeExperimentId, setActiveExperimentId] = useState<string | null>(
    relatedExperiments[0]?.id ?? null,
  );
  const selectedExperiment =
    relatedExperiments.find((experiment) => experiment.id === activeExperimentId) ??
    relatedExperiments[0] ??
    null;

  useEffect(() => {
    setActiveExperimentId(relatedExperiments[0]?.id ?? null);
  }, [node.id]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div className="dialog-backdrop" data-no-pan="true" role="presentation" onClick={onClose}>
      <section
        className="detail-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="node-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-header">
          <div>
            <h2 id="node-dialog-title">{node.detail.title}</h2>
          </div>
          <button className="close-detail" type="button" onClick={onClose} aria-label="Close details">
            ×
          </button>
        </div>

        {parentNode ? (
          <div className="parent-question">
            <strong>Parent question: </strong>
            <span>{parentNode.title}</span>
          </div>
        ) : null}

        <div className="dialog-grid">
          <div className="dialog-main">
            <div>
              <p className="panel-kicker">Motivation</p>
              <p>{node.detail.body}</p>
            </div>
            <div className="branch-box">
              <p className="branch-title">Takeaway</p>
              <ul>
                {node.branches.map((branch) => (
                  <li key={branch}>{branch}</li>
                ))}
              </ul>
            </div>
          </div>
          <div className="dialog-results">
            <p className="panel-kicker">Details / Evidence</p>
            {relatedExperiments.length > 0 ? (
              <div className="experiment-stack">
                {relatedExperiments.length > 1 ? (
                  <div className="experiment-tabs" role="tablist" aria-label="Experiment results">
                    {relatedExperiments.map((experiment) => (
                      <button
                        key={experiment.id}
                        type="button"
                        className={experiment.id === selectedExperiment?.id ? "is-active" : ""}
                        onClick={() => setActiveExperimentId(experiment.id)}
                      >
                        <strong>{experiment.title}</strong>
                      </button>
                    ))}
                  </div>
                ) : null}
                {selectedExperiment ? <ExperimentPanel experiment={selectedExperiment} /> : null}
              </div>
            ) : (
              <div className="result-section">
                <p>Details</p>
                <ul>
                  {node.detail.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function findClosestSyncStep(
  targetStep: number | null,
  points: { syncStep?: number }[],
) {
  if (targetStep === null) {
    return null;
  }
  const syncPoints = points.filter((point) => point.syncStep !== undefined);
  if (syncPoints.length === 0) {
    return null;
  }
  return syncPoints.reduce((best, point) =>
    Math.abs(point.syncStep! - targetStep) < Math.abs(best.syncStep! - targetStep) ? point : best,
  ).syncStep!;
}

function findClosestVideoIndex(videos: VideoRecord[], targetStep: number) {
  if (videos.length === 0) {
    return 0;
  }
  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  videos.forEach((video, index) => {
    if (video.step < 0) {
      return;
    }
    const distance = Math.abs(video.step - targetStep);
    if (distance < bestDistance) {
      bestIndex = index;
      bestDistance = distance;
    }
  });
  return bestIndex;
}

function clampNumber(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function ExperimentPanel({ experiment }: { experiment: ExperimentNote }) {
  const initialSyncStep = experiment.curve.findLast((point) => point.syncStep !== undefined)?.syncStep ?? null;
  const [selectedSyncStep, setSelectedSyncStep] = useState<number | null>(initialSyncStep);

  useEffect(() => {
    setSelectedSyncStep(experiment.curve.findLast((point) => point.syncStep !== undefined)?.syncStep ?? null);
  }, [experiment.id, experiment.curve]);

  return (
    <article className="experiment-card">
      <div className="experiment-header">
        <div>
          <h3>{experiment.title}</h3>
          <p className="experiment-run">{experiment.date} · {experiment.runName}</p>
        </div>
      </div>

      <div className="experiment-copy-grid">
        <NoteBlock title="Motivation" text={experiment.objective} priority />
        <NoteBlock title="Takeaway" text={experiment.conclusion} priority />
        <NoteBlock title="Result" text={experiment.result} />
        <NoteBlock title="Next" text={experiment.nextStep} />
      </div>

      <FactGrid facts={experiment.facts} />

      {experiment.links.length > 0 ? (
        <div className="experiment-links">
          {experiment.links.map((link) => (
            <a key={link.href} href={link.href} target="_blank" rel="noreferrer">
              {link.label}
            </a>
          ))}
        </div>
      ) : null}

      <details className="experiment-evidence" open>
        <summary>
          <span>Video, curve, and notes</span>
          <strong>Evidence</strong>
        </summary>

        <div className="experiment-media-grid">
          <div className="chart-stack">
            <MiniChart
              experiment={experiment}
              selectedSyncStep={selectedSyncStep}
              onSelectSyncStep={setSelectedSyncStep}
            />
            {experiment.extraCurves?.map((curve) => (
              <MiniChart
                key={curve.metricLabel}
                experiment={{
                  ...experiment,
                  curveDescription: curve.curveDescription,
                  metricLabel: curve.metricLabel,
                  metricSuffix: curve.metricSuffix,
                  yDomain: curve.yDomain,
                  curve: curve.curve,
                }}
                selectedSyncStep={selectedSyncStep}
                onSelectSyncStep={setSelectedSyncStep}
              />
            ))}
          </div>
          {selectedSyncStep !== null ? (
            <div className="sync-step-readout">
              <strong>Synced control step</strong>
              <span>{selectedSyncStep.toLocaleString()}</span>
            </div>
          ) : null}
          <VideoDirectoryPanel
            experiment={experiment}
            selectedSyncStep={selectedSyncStep}
            onSelectSyncStep={setSelectedSyncStep}
          />
          {experiment.extraVideoDirs?.map((videoDir) => (
            <VideoDirectoryPanel
              key={videoDir.path}
              experiment={experiment}
              videoDirOverride={videoDir}
              selectedSyncStep={selectedSyncStep}
              onSelectSyncStep={setSelectedSyncStep}
            />
          ))}
        </div>

        {experiment.noteSections.length > 0 ? (
          <div className="note-section-grid">
            {experiment.noteSections.map(([title, text]) => (
              <NoteBlock key={title} title={title} text={text} />
            ))}
          </div>
        ) : null}
      </details>
    </article>
  );
}

function NoteBlock({ title, text, priority = false }: { title: string; text: string; priority?: boolean }) {
  return (
    <section className={`note-block ${priority ? "note-block-priority" : ""}`}>
      <p>{title}</p>
      <span>{text}</span>
    </section>
  );
}

function FactGrid({ facts }: { facts: [string, string][] }) {
  return (
    <dl className="fact-grid">
      {facts.map(([label, value]) => (
        <div key={`${label}-${value}`}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function MiniChart({
  experiment,
  selectedSyncStep,
  onSelectSyncStep,
}: {
  experiment: ExperimentNote;
  selectedSyncStep: number | null;
  onSelectSyncStep: (syncStep: number) => void;
}) {
  const points = experiment.curve;
  const [minY, maxY] = experiment.yDomain;
  const width = 560;
  const height = 240;
  const plot = { left: 46, right: 536, top: 24, bottom: 196 };
  const closestSelectedSyncStep = findClosestSyncStep(selectedSyncStep, points);
  const selectedIndex =
    closestSelectedSyncStep === null
      ? -1
      : points.findIndex((point) => point.syncStep === closestSelectedSyncStep);
  const spanY = maxY - minY || 1;
  const usesStepScale = points.length > 0 && points.every((point) => point.syncStep !== undefined);
  const xValues = points.map((point, index) => (usesStepScale ? point.syncStep! : index));
  const minX = xValues.length > 0 ? Math.min(...xValues) : 0;
  const maxX = xValues.length > 0 ? Math.max(...xValues) : 1;
  const spanX = maxX - minX || 1;
  const toXValue = (value: number) =>
    plot.left + ((value - minX) / spanX) * (plot.right - plot.left);
  const toX = (index: number) => toXValue(xValues[index] ?? index);
  const toY = (value: number) =>
    plot.bottom - ((value - minY) / spanY) * (plot.bottom - plot.top);
  const polyline = points
    .map((point, index) => `${toX(index).toFixed(2)},${toY(point.value).toFixed(2)}`)
    .join(" ");
  const latest = points.at(-1);
  const selectedPoint = selectedIndex >= 0 ? points[selectedIndex] : latest;
  const selectedX =
    usesStepScale && selectedSyncStep !== null
      ? toXValue(clampNumber(selectedSyncStep, minX, maxX))
      : selectedIndex >= 0
        ? toX(selectedIndex)
        : null;

  return (
    <section className="chart-card">
      <div className="chart-heading">
        <div>
          <p>Curve</p>
          <strong className="chart-label">{experiment.metricLabel}</strong>
          <span>{experiment.curveDescription}</span>
        </div>
        {selectedPoint ? (
          <strong>
            {selectedPoint.value.toFixed(2)}
            {experiment.metricSuffix}
            <small>{selectedPoint.step}</small>
          </strong>
        ) : null}
      </div>

      {points.length > 0 ? (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={experiment.metricLabel}>
          {[0, 1, 2, 3].map((tick) => {
            const value = minY + ((maxY - minY) * tick) / 3;
            const y = toY(value);
            return (
              <g key={tick}>
                <line x1={plot.left} y1={y} x2={plot.right} y2={y} className="chart-grid" />
                <text x={plot.left - 8} y={y + 4} textAnchor="end">
                  {value.toFixed(maxY <= 10 ? 1 : 0)}
                </text>
              </g>
            );
          })}
          <line x1={plot.left} y1={plot.bottom} x2={plot.right} y2={plot.bottom} className="chart-axis" />
          <line x1={plot.left} y1={plot.top} x2={plot.left} y2={plot.bottom} className="chart-axis" />
          {selectedX !== null ? (
            <line
              x1={selectedX}
              y1={plot.top}
              x2={selectedX}
              y2={plot.bottom}
              className="chart-selected-line"
            />
          ) : null}
          <polyline points={polyline} className="chart-line" />
          {points.map((point, index) => {
            const canSync = point.syncStep !== undefined;
            const selected = index === selectedIndex;
            return (
              <g
                key={`${point.step}-${index}`}
                className={`chart-point ${selected ? "is-selected" : ""} ${canSync ? "is-clickable" : ""}`}
                role={canSync ? "button" : undefined}
                tabIndex={canSync ? 0 : undefined}
                aria-label={
                  canSync
                    ? `${experiment.metricLabel} ${point.value.toFixed(2)} at ${point.step}`
                    : undefined
                }
                onClick={canSync ? () => onSelectSyncStep(point.syncStep!) : undefined}
                onKeyDown={
                  canSync
                    ? (event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          onSelectSyncStep(point.syncStep!);
                        }
                      }
                    : undefined
                }
              >
                {canSync ? <circle cx={toX(index)} cy={toY(point.value)} r="9" className="chart-hit-point" /> : null}
                <circle cx={toX(index)} cy={toY(point.value)} r={selected ? "4.6" : "2.7"} />
              </g>
            );
          })}
          {points.map((point, index) => {
            if (index !== 0 && index !== points.length - 1 && index % Math.ceil(points.length / 4) !== 0) {
              return null;
            }
            return (
              <text key={`x-${point.step}`} x={toX(index)} y={plot.bottom + 22} textAnchor="middle">
                {point.step}
              </text>
            );
          })}
        </svg>
      ) : (
        <div className="chart-empty">
          The original note tracked this run live. This canvas records the run identity and videos;
          add parsed scalar rows here when the run is summarized.
        </div>
      )}
    </section>
  );
}

function VideoDirectoryPanel({
  experiment,
  videoDirOverride,
  selectedSyncStep,
  onSelectSyncStep,
}: {
  experiment: ExperimentNote;
  videoDirOverride?: VideoDirectoryConfig;
  selectedSyncStep: number | null;
  onSelectSyncStep: (syncStep: number) => void;
}) {
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const videoDir = videoDirOverride ?? experiment.videoDir;

  useEffect(() => {
    if (!videoDir) {
      setVideos([]);
      setSelectedIndex(0);
      setError(null);
      return;
    }

    let cancelled = false;
    fetch(`/api/simtoolreal-media/list-videos?dir=${encodeURIComponent(videoDir.path)}`)
      .then(async (response) => {
        const payload = (await response.json()) as { videos?: VideoRecord[]; error?: string };
        if (!response.ok || !payload.videos) {
          throw new Error(payload.error ?? "failed to list videos");
        }
        return payload.videos;
      })
      .then((videoList) => {
        if (cancelled) {
          return;
        }
        const filteredVideos = videoDir.minSizeBytes
          ? videoList.filter((video) => video.sizeBytes >= videoDir.minSizeBytes!)
          : videoList;
        const nextVideos = filteredVideos.length > 0 ? filteredVideos : videoList;
        const defaultIndex =
          videoDir.defaultStep === undefined
            ? Math.max(0, nextVideos.length - 1)
            : nextVideos.reduce((bestIndex, video, index) => {
                const bestVideo = nextVideos[bestIndex];
                return Math.abs(video.step - videoDir.defaultStep!) <
                  Math.abs(bestVideo.step - videoDir.defaultStep!)
                  ? index
                  : bestIndex;
              }, 0);
        setVideos(nextVideos);
        setSelectedIndex(defaultIndex);
        setError(null);
      })
      .catch((loadError: unknown) => {
        if (!cancelled) {
          setVideos([]);
          setSelectedIndex(0);
          setError(loadError instanceof Error ? loadError.message : "failed to list videos");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [experiment.id, videoDir]);

  useEffect(() => {
    if (selectedSyncStep === null || videos.length === 0) {
      return;
    }
    const closestIndex = findClosestVideoIndex(videos, selectedSyncStep);
    setSelectedIndex((currentIndex) => (currentIndex === closestIndex ? currentIndex : closestIndex));
  }, [selectedSyncStep, videos]);

  const selectedVideo = videos[selectedIndex];
  const videoSteps = videos.map((video) => video.step).filter((step) => step >= 0);
  const usesVideoStepScale = videoSteps.length > 0;
  const minVideoStep = usesVideoStepScale ? Math.min(...videoSteps) : 0;
  const maxVideoStep = usesVideoStepScale ? Math.max(...videoSteps) : Math.max(0, videos.length - 1);
  const sliderStep =
    usesVideoStepScale
      ? clampNumber(selectedSyncStep ?? selectedVideo?.step ?? minVideoStep, minVideoStep, maxVideoStep)
      : selectedIndex;
  const videoSrc = selectedVideo
    ? `/api/simtoolreal-media/video?path=${encodeURIComponent(selectedVideo.path)}`
    : undefined;

  return (
    <section className="video-card">
      <div className="video-heading">
        <p>{videoDir?.title ?? "Video"}</p>
        <span>{videoDir?.caption ?? experiment.videoMissingReason ?? "No video configured."}</span>
      </div>

      <div className="video-frame">
        {videoSrc ? (
          <video key={videoSrc} controls loop muted playsInline preload="metadata" src={videoSrc} />
        ) : (
          <div>
            <strong>No mp4 loaded</strong>
            <span>{error ?? experiment.videoMissingReason ?? "No local video path is configured."}</span>
          </div>
        )}
      </div>

      {error ? <p className="video-error">{error}</p> : null}

      {videos.length > 0 ? (
        <label className="video-slider">
          <div>
            <span>{selectedVideo?.label}</span>
            <span>
              video {selectedIndex + 1} / {videos.length}
              {selectedVideo?.step >= 0 ? ` · nearest ${selectedVideo.step}` : ""}
              {selectedSyncStep !== null && selectedVideo?.step >= 0
                ? ` · target ${selectedSyncStep.toLocaleString()}`
                : ""}
            </span>
          </div>
          {usesVideoStepScale ? (
            <input
              type="range"
              min={minVideoStep}
              max={maxVideoStep}
              step={1}
              value={sliderStep}
              onChange={(event) => {
                const targetStep = Number(event.target.value);
                const nextIndex = findClosestVideoIndex(videos, targetStep);
                setSelectedIndex(nextIndex);
                onSelectSyncStep(targetStep);
              }}
            />
          ) : (
            <input
              type="range"
              min={0}
              max={Math.max(0, videos.length - 1)}
              value={selectedIndex}
              onChange={(event) => setSelectedIndex(Number(event.target.value))}
            />
          )}
        </label>
      ) : null}
    </section>
  );
}
