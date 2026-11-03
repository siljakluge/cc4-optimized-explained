# Explainability pipeline (ported to cc4-optimized-explained)

Team-level explainability for the CC4 blue agents: **exact Shapley-value
attribution**, **pairwise Shapley interaction indices**, and **communication
ablation**.



## Run it
```bash
PY=/opt/homebrew/Caskroom/miniconda/base/envs/cc4/bin/python

# 1. coalition values + exact Shapley (32 coalitions)   -> shapley_heuristic_fsm_default.json
$PY explainability/shapley.py --backend heuristic --seeds 20 --profile fsm_default

# 2. interaction index point estimates
$PY explainability/shapley_interactions.py explainability/shapley_heuristic_fsm_default.json

# 3. bootstrap 95% CIs for phi_i and I(i,j)
$PY explainability/shapley_bootstrap.py explainability/shapley_heuristic_fsm_default.json 5000

# 4. Holm-corrected significance of the interactions
$PY explainability/interaction_multiplicity.py explainability/shapley_heuristic_fsm_default.json 5000

# 5. communication ablation (ON / CONTENT / EMPTY, paired per seed)
$PY explainability/comms_ablation.py --backend heuristic --seeds 20 --profile fsm_default

# 6. paired stats for the ablation (pass several profiles to get the family)
$PY explainability/comms_stats.py explainability/comms_heuristic_*.json
```

Red profiles for `--profile`: `fsm_default`, `verbose`, `discovery`,
`stealth_pivot`, `impact_rush`, `deception_aware`, `lateral_spread`.

## Design notes / adaptations

- **Heuristic backend** uses `EnterpriseHeuristicAgentV11a` via the official
  `Submission` wrapper (`BlueFlatWrapperV2` + v9 messaging), with flat integer
  actions. An absent agent in a coalition submits its per-agent `Sleep` index.
- **Plain `EnterpriseScenarioGenerator`**
- **Communication ablation is meaningful here.** V11a reads peer messages (the
  paper's heuristic did not), so ablating them changes behaviour. Ablation is
  applied by zeroing each agent's outgoing message before the env forwards it.
- **`v(S)` is the scalar team return.** Per-reward-component decomposition
  (LWF/ASF/RIA/...)
- **MARL** (`harness.run_episode_marl`): trained `gnn_ppo`
  checkpoints for the fixed environment do not exist yet.
  When they do, wire it against `marl/models/cage4.py` + `graph_wrapper`,
  mirroring the heuristic coalition logic.

## Files

| file | role |
|---|---|
| `harness.py` | shared coalition runner; all repo-specific adaptation lives here |
| `shapley.py` | 32-coalition exact Shapley → `v_raw` JSON |
| `shapley_interactions.py` | interaction index point estimates (reads JSON) |
| `shapley_bootstrap.py` | bootstrap 95% CIs for φ and I (reads JSON) |
| `interaction_multiplicity.py` | bootstrap p-values + Holm correction (reads JSON) |
| `comms_ablation.py` | ON/CONTENT/EMPTY paired ablation → JSON |
| `comms_stats.py` | paired t / permutation / Holm / rm-ANOVA (reads JSON) |
