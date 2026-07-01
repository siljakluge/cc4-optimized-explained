# Submission Registry

`explain.py` can load submissions by name:

```bash
python explain.py --list-agents
python explain.py --agent heuristic_v11b --is-heuristic
python explain.py --agent gpu_test --non-heuristic
```

Registered agents live in `registry.json`.

For heuristic submissions, point `path` at a folder containing `submission.py`.

For MARL submissions that reuse `marl/submission.py`, point `path` at `../marl` and set `checkpoint_prefix` to the prefix of the five checkpoint files:

```text
<checkpoint_prefix>-0_checkpoint.pt
<checkpoint_prefix>-1_checkpoint.pt
<checkpoint_prefix>-2_checkpoint.pt
<checkpoint_prefix>-3_checkpoint.pt
<checkpoint_prefix>-4_checkpoint.pt
```

Example:

```json
"my_new_marl_agent": {
  "path": "../marl",
  "is_heuristic": false,
  "checkpoint_prefix": "my_new_marl_agent",
  "env": {
    "KEEP_CKPT_DIR": "checkpoints"
  }
}
```

Then run:

```bash
python explain.py --agent my_new_marl_agent
```

For large explainability runs, keep the evaluation size high but sample SHAP rows:

```bash
python explain.py \
  --agent heuristic_v11b \
  --mode single \
  --single_profile fsm_default \
  --max-eps 100 \
  --episode-length 500 \
  --shap \
  --shap-episode-stride 2 \
  --shap-step-stride 10 \
  --shap-max-rows-per-profile 2000 \
  --shap-background-samples 50 \
  --shap-explain-samples 100
```

This still evaluates all 100 episodes, but SHAP only trains/explains on a deterministic subset.
