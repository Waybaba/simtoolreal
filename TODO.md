

reproduce the results
migrate to isaaclab and reproduce the results

create a high level policy with lowlevel policy to env to train the high level
Re-enable action/object-state/observation delay later to match Isaac Gym parity.
Check the maximum render capacity for Isaac Lab video/render runs.
Learn to debug RL training.
Document and revisit the `--no-adaptive_lr` debug option: it was added after Isaac Lab easy-goal/simple-object fine-tuning showed that adaptive KL LR can raise the learning rate too aggressively and destabilize a policy after an initial success spike. Default adaptive LR remains enabled; use the flag only for controlled debugging or fixed-LR reproduction runs.
