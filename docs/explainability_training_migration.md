# Explainability and Training Migration

This note maps the old `ExplainabilityCC4` implementation onto the current
`cc4-optimized-explained` repo without touching the already modified core files.

## Current Integration Status

- SHAP: implemented through `explain.py`, `SHAP/policy_shap_heuristic.py`,
  `SHAP/shap_gnn.py`, and `marl/wrappers/graph_wrapper.py`.
- Reward decomposition: the new `CybORG/Shared/BlueRewardMachine.py` already
  emits JSONL entries with `phase`, `reward_list`, `total`, `profile`, and
  `run_tag` to `CYBORG_REWARD_LOG_PATH`.
- Action statistics: `explain.py` writes per-profile `actions.jsonl` via
  `plotting.plot_actions.log_actions_jsonl`.
- Missing glue: `explain.py` writes the needed logs but does not yet call the
  reward decomposition and action statistics postprocessing automatically.

## Old To New Map

| Old repo source | Old role | New target |
| --- | --- | --- |
| `old/explainability.py::run_explainability` | Run episodes, collect SHAP rows, write reward/action logs | `explain.py::run_explainability_profiles` |
| `old/explainability.py::build_shap_dataset_all_features` | Convert wrapper `info` dicts into tabular GNN SHAP input | `explain.py::build_shap_dataset_all_features` |
| `plotting/plot_rew_decomp.py` | Parse reward JSONL and generate paper figures | `plotting/plot_rew_decomp.py`, invoked by `scripts/postprocess_explainability.py` |
| `plotting/plot_actions.py` | Log and plot per-agent action frequencies | `plotting/plot_actions.py`, invoked by `scripts/postprocess_explainability.py` |
| `train.py` | PPO training with single/mixture/curriculum profile sampling | New `train.py` or `scripts/train_gnn_ppo.py` after conflict review |
| `train_curriculum.py` | Sequential curriculum by red profile | New `train_curriculum.py` or `scripts/train_curriculum_gnn_ppo.py` after conflict review |

## Expected Explainability Workflow

Mini local smoke run:

```bash
cd /data/.openclaw/workspace/cc4-optimized-explained
source .venv/bin/activate
python explain.py \
  --mode sweep \
  --max-eps 1 \
  --episode-length 3 \
  --non-heuristic \
  --seed 1337 \
  --phase_reward_mode red_only \
  --submission-path .
```

Postprocess the newest run directory:

```bash
python scripts/postprocess_explainability.py Results/<run-dir>
```

Expected artifacts:

- `Postprocessed/explainability_log_validation.csv`
- `Postprocessed/reward_decomposition_avg.csv`
- `Postprocessed/ActionStatistics/action_freq_attacker_agent_pct.csv`
- `Postprocessed/ActionStatistics/*.png`
- `Postprocessed/RewardDecomposition/*.png`

## Training Migration Target

The old training scripts should move into the new repo only after checking
imports against the current package layout:

- `models.cage4.InductiveGraphPPOAgent` -> likely `marl.models.cage4.InductiveGraphPPOAgent`
- `models.memory_buffer.MultiPPOMemory` -> likely `marl.models.memory_buffer.MultiPPOMemory`
- `wrappers.graph_wrapper.GraphWrapper` -> `marl.wrappers.graph_wrapper.GraphWrapper`
- `wrappers.observation_graph.ObservationGraph` -> `marl.wrappers.observation_graph.ObservationGraph`

Required CLI/API:

```bash
python train.py mini_curriculum \
  --debug \
  --strategy curriculum \
  --warmup_updates 1 \
  --harden_updates 0 \
  --phase_reward_mode red_only \
  --log_per_profile
```

GPU/company run shape:

```bash
python train.py exp_curriculum_gpu \
  --KIServer \
  --strategy curriculum \
  --warmup_updates 200 \
  --harden_updates 200 \
  --profile_weights "fsm_default=0.35,stealth_pivot=0.25,lateral_spread=0.20,impact_rush=0.15,deception_aware=0.05" \
  --phase_reward_mode red_only \
  --log_per_profile
```

Sequential curriculum target:

```bash
python train_curriculum.py exp_curriculum_seq \
  --KIServer \
  --curriculum_order "fsm_default,stealth_pivot,lateral_spread,impact_rush,deception_aware,discovery" \
  --stage_episodes 5000 \
  --phase_reward_mode red_only \
  --log_per_profile
```

## Tests To Add After Training Port

- `python -m py_compile train.py train_curriculum.py`
- `python train.py smoke --debug --strategy single --single_profile fsm_default`
- `python train.py smoke_curr --debug --strategy curriculum --warmup_updates 1 --log_per_profile`
- Assert creation of:
  - `logs/<name>.pt`
  - `logs/<name>_per_profile.pt` when requested
  - `checkpoints/<name>-0_checkpoint.pt` through `checkpoints/<name>-4_checkpoint.pt`

## Notes

The environment here runs Python 3.13, so local smoke tests need modern package
versions. Company GPU runs should prefer Python 3.10 or 3.11 if exact older
training pins are required.
