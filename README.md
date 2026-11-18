# OnePassDiff: Train With Diffusion, Walk in One Pass

Code for the paper under double-anonymous review. It trains a diffusion actor online with PPO through a single deterministic pass, adds an advantage-weighted denoising loss that trains the denoising pathway, and evaluates the policy under iterative and single-pass inference on ten simulated terrains.

## Contents

```
envs/                      Isaac Gym environment and training (based on legged_env, see LICENSE)
  run.sh, exp.sh           launcher: bash run.sh <experiment>
  exp_onepass.sh           every controller and ablation of the paper (VARIANT=...)
  train.py                 Hydra + rl_games entry point
  tasks/legged_terrain.py  the biped task: rewards, observations, terrain curriculum, randomization
  common/diffusion_policy.py   actor: encoder, step predictor, denoiser (x0 prediction, cosine schedule)
  common/onepass_agent.py      PPO agent with the denoising loss (M1), step-predictor loss (M2), joint-limit loss (M3)
  common/joint_mlp_agent.py    PPO-MLP with the same joint-limit loss (PPO-MLP+L_joint)
  cfg/train/BipedPPOOnePass.yaml   loss weights and diffusion settings
  cfg/task/Biped.yaml      task configuration (overridden per run in exp.sh / exp_onepass.sh)
assets/urdf/v6biped_urdf_v4_aug29/   10-DoF biped model (from the open-source Duke Humanoid, legged_env)
experiments/               evaluation, baselines, latency, statistics, figures
  onepass_kit.py           simulator-independent pieces: losses, DDIM sampler, inference modes
  my_adapter.py, run_e0.py evaluation harness: same seeded episodes for every inference mode
  launch_train.sh, runs.txt    all training runs
  launch_eval.sh           main evaluation of one checkpoint (ten terrains, 200 episodes each)
  launch_extra_eval.sh     learning curves, commanded speeds, slope / stair sweeps, speed x difficulty grid
  collect_demos.py, train_bc.py, bc_baselines.py, run_baselines.sh   BiRoDiff-R and DiffuseLoco-R
  latency_onboard.py       batch-1 latency per controller (GPU and single CPU core)
  analyze_results.py, ral_numbers.py   statistics: two-level bootstrap over seeds and episodes, Holm correction
  ral_figs.py, plot_ral.py, plot_extra.py   figures and tables
```

## Requirements

Tested on Ubuntu 20.04 with one NVIDIA RTX 3070 (8 GB).

- Python 3.8, PyTorch 2.4.1 (CUDA 12.1)
- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym) (`isaacgym` 1.0rc4)
- [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) 1.5.1 and rl_games 1.6.1 (the version IsaacGymEnvs pins)
- hydra-core 1.3.8, omegaconf 2.3.1, numpy 1.23.5, scipy 1.10.1, pandas 1.3.5, matplotlib 3.5.3, tensorboard 2.14.0,
  trimesh 3.23.5, yourdfpy 0.0.60, pycollada 0.6, networkx 2.2, msgpack 1.1.1, orjson 3.10.15

`envs/setup/conda_env.yaml` lists a matching environment. Install Isaac Gym and IsaacGymEnvs as their own instructions describe, then run everything from a shell where that environment is active.

## Training

Every run is one function call of `envs/exp_onepass.sh`. One 5,000-iteration run takes about 5 h on an RTX 3070.

```bash
cd envs
VARIANT=full SEED=1 bash run.sh onepass        # OnePassDiff
VARIANT=full SEED=1 bash run.sh onepass -r     # dry run: print the full command only
```

| `VARIANT`              | Controller in the paper                                   | Seeds |
|------------------------|-----------------------------------------------------------|-------|
| `full`                 | OnePassDiff                                               | 1-5   |
| `ppo_only`             | OnePassDiff-PPO-only (no denoising loss)                  | 1-5   |
| `ppo_mlp`              | PPO-MLP (capacity-matched)                                | 1-5   |
| `ppo_mlp_joint`        | PPO-MLP+L_joint                                           | 1-3   |
| `abl_self_distill`     | ablation: self-distillation target                        | 1-3   |
| `abl_no_enc_grad`      | ablation: no denoising gradient into the encoder          | 1-3   |
| `abl_no_terrain_latent`| ablation: terrain latent set to zero                      | 1-3   |
| `abl_no_stair_data`    | ablation: no stairs in training                           | 1-3   |
| `abl_linear_schedule`  | ablation: linear noise schedule                           | 1-3   |
| `abl_no_resblocks`     | ablation: no residual connections in the denoiser         | 1-3   |
| `abl_no_joint_loss`    | ablation: no joint-limit loss                             | 1-3   |

`experiments/launch_train.sh` runs every line of `experiments/runs.txt` (`PARALLEL` runs at a time on one GPU; one run needs about 6 GB). Checkpoints are written to `outputs/Biped/train/onepass_<variant>_seed<s>/`.

The two retrained diffusion baselines (BiRoDiff-R, DiffuseLoco-R) are trained by behaviour cloning on demonstrations of PPO-MLP seed 1, three seeds each:

```bash
cd experiments
./run_baselines.sh          # collect demonstrations, then train both baselines
```

## Evaluation

All evaluations replay the same seeded terrains and initial states for every controller and inference mode. Isaac Gym allows one simulation per process, so the scripts start one process per terrain.

```bash
cd experiments
CK=$(python -c "from my_adapter import Adapter; print(Adapter.find_checkpoint('../outputs/Biped/train/onepass_full_seed1', 5000))" | tail -n 1)
./launch_eval.sh my_adapter.py "$CK" full 1 results/            # all inference modes, ten terrains, 200 episodes each
PASS1=ORIG PASS2="" EXTRA="--no-sp-deviation" ./launch_eval.sh my_adapter.py <ppo_mlp checkpoint> ppo_mlp 1 results/
PASS1=ORIG PASS2="" EXTRA="--no-sp-deviation" ./launch_eval.sh my_adapter.py ../outputs/Biped/train/bc_birodiff_seed1/bc.pt birodiff 1 results/
PART=all ./launch_extra_eval.sh my_adapter.py                    # learning curves, speeds, difficulty sweeps
PART=grid ./launch_extra_eval.sh my_adapter.py                   # speed x difficulty grid
```

Inference modes (`run_e0.py --modes`): `SP` single pass, `D<N>` DDIM with fixed N, `AD` learned step count, `AD-mean` / `AD-shuffle` / `AD-random` compute-matched controls, `ES` early stopping, `@0` suffix for a zero initial sample, `ORIG` the controller's own policy (action mean for PPO-MLP, own sampler for the baselines). The header of `run_e0.py` describes all of them.

## Latency

```bash
cd experiments
python latency_onboard.py export --out bundle.pt --onepass <full seed-1 .pth> --mlp <ppo_mlp seed-1 .pth> \
    --bc ../outputs/Biped/train/bc_birodiff_seed1/bc.pt ../outputs/Biped/train/bc_diffuseloco_seed1/bc.pt \
    --obs results/obs_*.pt                                           # observations recorded by run_e0.py --dump-obs
python latency_onboard.py time --bundle bundle.pt --device cuda --out latency_cuda.json
OMP_NUM_THREADS=1 taskset -c 2 python latency_onboard.py time --bundle bundle.pt --device cpu --threads 1 \
    --iters 1000 --warmup 100 --out latency_cpu1_rep1.json          # single CPU core, three repetitions, median reported
```

## Statistics and figures

```bash
cd experiments
python export_scalars.py --logdir ../outputs/Biped/train --runs 'onepass_*' --out scalars_out
python ral_numbers.py --results results/ --curve results_curve/ --sweep results_sweep/ --scalars scalars_out/scalars --out ral_out/
G="0.3=results_grid/v0.3 0.5=results_sweep 0.8=results_grid/v0.8"; C="full:SP ppo_mlp:ORIG ppo_mlp_joint:ORIG birodiff:ORIG diffuseloco:ORIG"
python plot_ral.py grid --results unused --grid $G --cells $C --metric success  --name grid_success  --out grid/
python plot_ral.py grid --results unused --grid $G --cells $C --metric progress --name grid_progress --out grid/
python plot_ral.py sweep --results results_sweep/ --out grid/
python ral_figs.py --numbers ral_out/ral_numbers.json --train-dir ral_out/ --results results/ \
    --sweep-csv grid/heatmap_difficulty.csv --grid-success grid/grid_success.csv --grid-progress grid/grid_progress.csv \
    --only train modes traj3d combo3d --out figs/
```

`ral_numbers.py` computes every number of the Results section (means over seeds, 95% two-level bootstrap intervals, Holm-corrected paired tests) and writes them to `ral_numbers.json`, plus the CSVs the training and trajectory figures read. `ral_figs.py` draws the training curves (`train`), the inference-mode heatmaps (`modes`), the stair-ascent trajectories (`traj3d`) and the speed x difficulty figure (`combo3d`). Analysis scripts run on the CPU.

## License

MIT (see `LICENSE`). The environment and launcher derive from [legged_env](https://github.com/generalroboticslab/legged_env) (MIT, General Robotics Lab, Duke University), which includes the biped model. `envs/train.py` derives from IsaacGymEnvs (BSD-3-Clause, NVIDIA; notice kept in the file).
