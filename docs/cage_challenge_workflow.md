# CAGE Challenge 4 Workflow

This repo keeps the optimized CC4 environment fixes and the explainability
pipeline together.

## Local Smoke Tests

Run from the repository root:

```bash
source .venv/bin/activate
python marl/train.py mini_curriculum_smoke --mini --strategy curriculum --warmup_updates 1 --log_per_profile --device cpu
```

Expected outputs:

- `logs/mini_curriculum_smoke.pt`
- `logs/mini_curriculum_smoke_per_profile.pt`
- `checkpoints/mini_curriculum_smoke-0_checkpoint.pt` through `-4_checkpoint.pt`

Run a short explainability pass:

```bash
python explain.py \
  --mode single \
  --strategy single \
  --single_profile fsm_default \
  --max-eps 1 \
  --episode-length 5 \
  --non-heuristic \
  --shap \
  --seed 1337 \
  --phase_reward_mode red_only \
  --output SmokeResults/FULL_$(date +%Y%m%d_%H%M) \
  --submission-path .
```

Expected outputs under the created run directory:

- `<profile>/reward_log.jsonl`
- `<profile>/scalar_rewards.jsonl`
- `<profile>/actions.jsonl`
- `<profile>/summary_scalar.json`
- `<profile>/SHAPAnalysis/`

## GPU Training

For a larger GPU box, start with:

```bash
source .venv/bin/activate
python marl/train.py exp_curriculum_gpu \
  --KIServer \
  --strategy curriculum \
  --warmup_updates 200 \
  --harden_updates 200 \
  --log_per_profile \
  --phase_reward_mode red_only \
  --device cuda
```

For a fixed mixture:

```bash
python marl/train.py exp_mix_gpu \
  --KIServer \
  --strategy mixture \
  --profile_weights "fsm_default=0.35,discovery=0.15,stealth_pivot=0.20,lateral_spread=0.15,impact_rush=0.10,deception_aware=0.05" \
  --log_per_profile \
  --phase_reward_mode red_only \
  --device cuda
```

The profile registry is tolerant: profiles missing in the checked-out CybORG
tree are ignored and the remaining weights are renormalized.

## Explainability Features

Reward decomposition is logged by `CybORG/Shared/BlueRewardMachine.py` via:

- `CYBORG_REWARD_LOG_PATH`
- `CYBORG_ATTACK_PROFILE`
- `CYBORG_RUN_TAG`
- `CYBORG_PHASE_REWARD_MODE`
- `CYBORG_REWARD_BLUE`

`explain.py` sets these variables per profile and writes the JSONL files used by
`plotting/plot_rew_decomp.py`.

Action statistics are logged by `plotting/plot_actions.py::log_actions_jsonl`.
The graph wrapper returns translated CybORG action objects plus
`chosen_action_type`, `chosen_action_str`, and `shap_features` in `info`, which
lets SHAP and action statistics use the same episode run.
