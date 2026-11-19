"""Load the five trained MARL policies for evaluation (CPU)."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MARL_ROOT = REPO_ROOT / "marl"
# GraphWrapper still uses legacy absolute imports (wrappers.*).
for directory in (REPO_ROOT, MARL_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from marl.models.cage4 import load
from wrappers.graph_wrapper import GraphWrapper


class Submission:
    NAME = "GNN_marl_02"
    TEAM = "Team.KI"
    TECHNIQUE = "Graph-based PPO with inter-agent communication"

    def __init__(self):
        weights_dir = Path(os.environ.get(
            "CC4_WEIGHTS_DIR", str(REPO_ROOT / "weights" / "marl_run_02")
        )).expanduser().resolve()
        paths = [weights_dir / f"gnn_ppo-{i}.pt" for i in range(5)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing MARL weights: " + ", ".join(missing))
        self.weights_dir = str(weights_dir)
        self.AGENTS = {
            f"blue_agent_{i}": load(str(path), map_location="cpu")
            for i, path in enumerate(paths)
        }
        print(f"Loaded five MARL policies from {weights_dir}", flush=True)

    @staticmethod
    def wrap(env):
        return GraphWrapper(env)
