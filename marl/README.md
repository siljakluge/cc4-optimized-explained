# Multi-Agent Autonomous Cyber Defence (KEEP)

Solution to the [TTCP CAGE Challenge 4](https://github.com/cage-challenge/cage-challenge-4) using multi-agent reinforcement learning with Graph Neural Networks and PPO.

Adapted to work with the **optimized CybORG environment** in this repository, which includes significant speed improvements to the simulator, scenario generation, and observation handling.

---

## Setup

Install dependencies (from the repo root):

```bash
pip install -r requirements.txt
pip install -r marl/Requirements.txt
```

Ensure the `CybORG` module is discoverable:

```bash
pip install -e .
# or
export PYTHONPATH=/path/to/cage-challenge-4-optimized:$PYTHONPATH
```

### Key Dependencies

- `torch` and `torch_geometric` (GNN layers)
- `joblib` (parallel rollouts)
- `numpy`

---

## Evaluation

```bash
cd marl
python evaluation.py --distribute 4 --max-eps 100
```

| Flag | Description |
|------|-------------|
| `--distribute N` | Number of parallel workers (default: 1) |
| `--max-eps E` | Number of evaluation episodes (default: 100) |
| `--seed S` | RNG seed for reproducibility |

## Training

```bash
cd marl
python train.py run_name --hidden 256 --embedding 128
```

For a quick smoke test:

```bash
python train.py test_run --debug
```

Debug mode uses 2 workers, 50-step episodes, and 20 total training episodes.

Outputs:
- `logs/` — reward curves (`.pt` files)
- `checkpoints/` — model weights per agent

---

## Optimizations (vs. original KEEP)

This fork includes several speed and compatibility improvements:

### Environment
- **FastEnterpriseScenarioGenerator**: Pre-builds a pool of scenarios; resets return deep-copies instead of rebuilding the full graph each time.
- **Dynamic thread management**: Thread count adapts to `os.cpu_count()` instead of hardcoded 36.
- **Loky backend**: Joblib uses `prefer='loky'` for better process reuse across training iterations.
- **CUDA cuDNN autotuner**: `torch.backends.cudnn.benchmark = True` when GPU is available.

### Graph Construction (`observation_graph.py`)
- **Edge-version caching**: `nid_map` and subnet masks are cached and only recomputed when edges actually change.
- **Pre-allocated tensor buffers**: Feature matrices (`_x_buf`, `_tr_buf`) are reused instead of allocated fresh each call.
- **Vectorized one-hot assignment**: Node type features use scatter instead of a Python loop.
- **Batched feature extraction**: `get_features()` calls are grouped by node type and converted with a single `np.stack` + `torch.from_numpy` per type.

### Wrapper (`graph_wrapper.py`)
- **Pre-allocated step buffers**: Phase, tabular, padding, and message tensors are reused across steps.
- **Removed unused imports**: `Sleep` action import removed (Monitor used via globals).

### PPO Training (`cage4.py`, `memory_buffer.py`, `utils.py`)
- **O(n) discount accumulation**: Replaced `rewards.insert(0, ...)` (O(n^2) total) with reversed append + flip.
- **In-place list operations**: `extend()` instead of `+=` in memory buffer aggregation.
- **Vectorized offsets**: `combine_marl_states` uses `torch.cumsum` for graph offset computation.

---

## How It Works

### The Big Picture

CAGE Challenge 4 simulates an enterprise network under attack. Five **blue agents** defend
different subnets against autonomous **red agents** that scan, exploit, and escalate across
hosts. The blue team must detect intrusions and respond — but each agent only sees its
own subnet and can only communicate via short messages.

```
                          CAGE Challenge 4 — The Battlefield
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         Enterprise Network                         │
  │                                                                    │
  │   ┌──────────┐    ┌──────────┐    ┌──────────┐                     │
  │   │ Restricted│    │Operational│    │  Public   │                    │
  │   │  Zone A   │    │  Zone A   │    │  Access   │    ┌──────────┐   │
  │   │  Agent 0  │    │  Agent 1  │    │  Zone     │    │  Admin   │   │
  │   └─────┬─────┘    └─────┬─────┘    │          ├────┤  Network │   │
  │         │                │          │  Agent 4  │    │          │   │
  │   ┌─────┴─────┐    ┌─────┴─────┐    │ manages   ├────┤  Office  │   │
  │   │ Restricted│    │Operational│    │ 3 subnets │    │  Network │   │
  │   │  Zone B   │    │  Zone B   │    └─────┬─────┘    └──────────┘   │
  │   │  Agent 2  │    │  Agent 3  │          │                        │
  │   └───────────┘    └───────────┘    ┌─────┴─────┐                  │
  │                                     │ Internet  │                  │
  │          Red agents attack  ------> │  + Contr. │                  │
  │          from the outside           └───────────┘                  │
  └─────────────────────────────────────────────────────────────────────┘
```

Each blue agent can take **three kinds of actions** on its subnet:

| Level | Actions | What It Does |
|-------|---------|--------------|
| **Node** (per host) | Analyse, Remove, Restore, DeployDecoy | Inspect, clean, reimage hosts, or plant honeypots |
| **Edge** (per subnet pair) | AllowTraffic, BlockTraffic | Open or close firewall rules between subnets |
| **Global** | Monitor | Do nothing / observe passively |

---

### Step 1: Turning Observations into Graphs

Standard RL uses flat vectors. Our approach converts each agent's observation into a
**dynamic graph** — a natural representation for network topology:

```
  Raw CybORG Observation                    Observation Graph
  ┌─────────────────────┐                  ┌─────────────────────┐
  │ {                   │                  │                     │
  │   'host_0': {       │   GraphWrapper   │   [Router]          │
  │     OS, processes,  │ ──────────────>  │    / | \            │
  │     connections...  │                  │ [H0] [H1] [H2]     │
  │   },                │                  │  |         |        │
  │   'host_1': {...},  │                  │ [Port:80] [Port:443]│
  │   success: TRUE,    │                  │  |                  │
  │   messages: [...]   │                  │ [Malware.exe]       │
  │ }                   │                  │                     │
  └─────────────────────┘                  └─────────────────────┘

  Dictionary with nested           Graph with typed nodes and edges
  host/process/file info           compatible with PyTorch Geometric
```

**Four node types** capture different aspects of the network:

```
  Node Types and Their Features
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  [System]  Hosts, servers, routers                               │
  │  ┌──────────────────────────────────────────────────────┐        │
  │  │ OS type | distro | arch | patches | server? | router?│        │
  │  └──────────────────────────────────────────────────────┘        │
  │                                                                  │
  │  [Connection]  Processes with open ports                         │
  │  ┌──────────────────────────────────────────────────────┐        │
  │  │ process name | type | suspicious? | decoy? | default?│        │
  │  └──────────────────────────────────────────────────────┘        │
  │                                                                  │
  │  [File]  Files observed on hosts (usually malware)               │
  │  ┌──────────────────────────────────────────────────────┐        │
  │  │ file type | vendor | version | density | signed?     │        │
  │  └──────────────────────────────────────────────────────┘        │
  │                                                                  │
  │  [Internet]  Structural node — no features, just connectivity    │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  Every node also gets:
    +  4-dim node type one-hot     (which of the 4 types)
    +  9-dim subnet membership     (which subnet it belongs to)
    +  2-dim tabular features      (from BlueFlatWrapper)
    +  3-dim message features      (compromised? scanned? received?)
```

The graph **updates dynamically** each timestep based on what happened:

| Event | Graph Update |
|-------|-------------|
| New remote connection observed | Add Connection node + edges to local/remote hosts |
| `Restore` succeeds | Remove all transient edges from that host |
| `Remove` succeeds | Remove suspicious session nodes |
| `DeployDecoy` succeeds | Add a new decoy Connection node |
| `Analyse` finds files | Add File nodes linked to the host |

---

### Step 2: The GNN Model (Actor-Critic)

The core insight: since observations are graphs, we use a **Graph Convolutional Network**
(GCN) to process them. Information flows along edges — just like real network traffic.

```
  Actor Network (decides what to do)
  ═══════════════════════════════════

  Input: Graph (x, edge_index) + mission phase (3-dim one-hot)

         x ──────────────────────────────────────────────────>
         │                                                    │
         │  ┌─────────────┐    ┌─────────────┐               │
         ├──│  GCNConv 1  │───>│  GCNConv 2  │──> x' (node   │
         │  │ in -> 256   │    │ 256 -> 64   │    embeddings)│
         │  └──────┬──────┘    └──────┬──────┘               │
         │         │                  │                       │
  phase ─┤  ┌──────┴──────┐    ┌──────┴──────┐    ┌─────┐    │
  (3-dim)├──│ SelfAttn g0 │───>│ SelfAttn g1 │──> │ g2  │──> g (global
         │  │ (hosts+rtrs)│    │ (hosts+rtrs)│    │     │    │  summary)
         │  └─────────────┘    └─────────────┘    └─────┘    │
         │                                                    │
         ▼                                                    ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │  Node Actions    │  │  Edge Actions    │  │  Global Actions  │
  │  x' + g -> MLP   │  │ src*dst + g->MLP │  │     g -> MLP     │
  │  ─────────────   │  │  ─────────────   │  │  ─────────────   │
  │  Per host:       │  │  Per subnet pair:│  │                  │
  │  [Analyse,Remove │  │  [AllowTraffic,  │  │  [Monitor]       │
  │   Restore,Decoy] │  │   BlockTraffic]  │  │                  │
  │  16 hosts x 4    │  │  8 pairs x 2     │  │  1               │
  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
           │                     │                      │
           └─────────── concat ──┴──────────────────────┘
                          │
                     [ Softmax ]
                          │
                   Categorical(probs)
                          │
                    sample action
```

**The self-attention mechanism** is the key innovation. At each GCN layer, it reads
all host+router nodes and computes a **global summary vector** `g` that captures the
overall state of the subnet. This `g` vector is then injected into the action heads,
giving each per-node decision access to the big picture:

```
  Self-Attention (at each GCN layer)
  ──────────────────────────────────

  Input: host/router node embeddings v  (B x N x d)
         previous global vector g       (B x d_g)

              v (all hosts + routers)
              │
       ┌──────┴──────┐
       │  attention   │  att = Softmax(Linear(v))
       │  weights     │
       └──────┬──────┘
              │
       ┌──────┴──────┐
       │   feature    │  feat = Linear(v)
       │   transform  │
       └──────┬──────┘
              │
         att * feat * mask   (mask zeros out padding)
              │
          sum over N          weighted_sum = sum(att * feat, dim=N)
              │
       ┌──────┴──────┐
       │  combine w/  │  g' = tanh(Linear([weighted_sum, g]))
       │  previous g  │
       └──────┬──────┘
              │
          g + g'             residual connection
              │
        updated global g
```

The **critic network** has the same GCN + self-attention structure but outputs a
single scalar value estimate instead of action probabilities:

```
  Critic Network (estimates how good the state is)
  ════════════════════════════════════════════════

  Same GCN + SelfAttention layers as Actor
              │
              g (global summary)
              │
       ┌──────┴──────┐
       │   MLP       │  Linear(g) -> ReLU -> Linear -> V(s)
       │   64 -> 1   │
       └──────┬──────┘
              │
         scalar value
```

---

### Step 3: Multi-Agent Communication

Agents can share 8-bit messages when communications aren't blocked by the
mission phase. Messages are embedded into the graph as features on subnet router nodes:

```
  Agent-to-Agent Messaging
  ════════════════════════

  Agent 0 (Restricted Zone A)          Agent 1 (Operational Zone A)
  ┌──────────────────────┐             ┌──────────────────────┐
  │ Observes own subnet  │             │ Observes own subnet  │
  │ Sees: 2 hosts have   │             │ Sees: 1 host scanned │
  │       suspicious     │             │                      │
  │       connections    │             │                      │
  └──────────┬───────────┘             └──────────┬───────────┘
             │                                    │
     Encodes 8-bit message:               Encodes 8-bit message:
     [1,0, 0,1, 0,0, 0, 1]               [0,0, 1,0, 0,0, 0, 1]
      │ │   │ │           │                │ │   │ │           │
      │ │   │ │           checkbit         │ │   │ │           checkbit
      │ │   │ scanned?                     │ │   │ scanned?
      │ │   compromised?                   │ │   compromised?
      │ per subnet                         │ per subnet
      ▼                                    ▼
  ┌────────────────────────────────────────────────┐
  │              Message Bus (CybORG)              │
  │  (blocked during certain mission phases)       │
  └──────────────────────┬─────────────────────────┘
                         │
                    Other agents receive messages
                    as 3-dim features on router nodes:
                    [compromised?, scanned?, received?]
```

---

### Step 4: Training with PPO

Each of the 5 agents has its own independent actor and critic, trained with
Proximal Policy Optimization (PPO):

```
  Training Loop
  ═════════════

  ┌─────────────────────────────────────────────────────────────┐
  │                   For each training iteration:              │
  │                                                             │
  │  1. COLLECT EPISODES (parallel)                             │
  │     ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐   │
  │     │Worker 0 │ │Worker 1 │ │Worker 2 │ ... │Worker N │   │
  │     │(CybORG) │ │(CybORG) │ │(CybORG) │     │(CybORG) │   │
  │     └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘   │
  │          │           │           │                │        │
  │          └─────────┬─┴───────────┴───┬────────────┘        │
  │                    ▼                 ▼                      │
  │          ┌──────────────────────────────────┐               │
  │          │  Per-Agent Memory Buffers        │               │
  │          │  (states, actions, values,       │               │
  │          │   log_probs, rewards, terminals) │               │
  │          └──────────────┬───────────────────┘               │
  │                         │                                   │
  │  2. PPO UPDATE (parallel, one thread per agent)             │
  │     ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│
  │     │Agent 0 │ │Agent 1 │ │Agent 2 │ │Agent 3 │ │Agent 4 ││
  │     │  PPO   │ │  PPO   │ │  PPO   │ │  PPO   │ │  PPO   ││
  │     └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│
  │                                                             │
  │  3. CHECKPOINT & LOG                                        │
  │     Save model weights + reward curves                      │
  └─────────────────────────────────────────────────────────────┘
```

The PPO update for each agent:

```
  PPO Update (per agent)
  ══════════════════════

  memories = {states, actions, old_values, old_log_probs, rewards, terminals}

  1. Compute discounted returns R_t = r_t + gamma * R_{t+1}
  2. Normalize returns
  3. Advantage A_t = R_t - V(s_t)

  For each epoch (default 4):
    For each minibatch:
      ┌─────────────────────────────────────────────┐
      │ Batch graph states with combine_marl_states │
      │ (offsets edge indices, concatenates nodes)   │
      └──────────────────┬──────────────────────────┘
                         │
                    ┌────┴────┐
                    │  Actor  │──> new_log_probs, entropy
                    │  Critic │──> new_values
                    └────┬────┘
                         │
      ratio = exp(new_log_prob - old_log_prob)
      clipped = clamp(ratio, 1-eps, 1+eps)

      Loss = -min(ratio * A, clipped * A)     actor loss
           + 0.5 * MSE(returns, new_values)   critic loss
           - 0.01 * entropy                   exploration bonus

      Backpropagate and step optimizers
```

### Handling Agent 4 (Multi-Subnet)

Agent 4 is special — it manages **3 subnets** (Admin, Office, Public Access) instead of 1.
The model handles this by processing each subnet's masks separately but sharing the
same graph, then combining the 3 subnet outputs:

```
  Agent 4: Multi-Subnet Handling
  ══════════════════════════════

  Same shared graph (all nodes visible)
        │
  ┌─────┴──────┐──────────┐──────────┐
  │  Admin      │  Office   │  Public  │
  │  subnet     │  subnet   │  Access  │
  │  masks      │  masks    │  masks   │
  └──────┬──────┘─────┬─────┘─────┬────┘
         │            │           │
    action probs  action probs  action probs
    (per subnet)  (per subnet)  (per subnet)
         │            │           │
         └──── concatenate ───────┘
                     │
              [ Softmax over all ]
                     │
              single action choice
              (3x action space)
```

---

## Architecture

### Environment Representation

Observations are constructed as dynamic graphs compatible with PyTorch Geometric:

| Node Type | Features |
|-----------|----------|
| Host/Router | OS, distro, arch, isUser, isServer, isRouter |
| Connection | Process name/type, port, ephemeral, default, decoy |
| File | Version, type, vendor, signed, density |
| Internet | Structural node (no features) |

All nodes also carry: subnet membership (9d one-hot), node type (4d one-hot), tabular features (2d), and message features (3d).

### Inter-Agent Messaging

Agents share 8-bit messages when allowed by the mission phase:
- 2 bits per monitored subnet (compromised/scanned flags)
- 1 checkbit for comms integrity

### Training Loop

Independent PPO per agent over local graph observations:
1. Parallel rollout generation (N episodes across W workers)
2. Per-agent memory collection via `MultiPPOMemory`
3. Parallel backpropagation (threaded, one per agent)

---

## Directory Structure

```
marl/
  train.py                  # Training script (PPO + parallel rollouts)
  evaluation.py             # Evaluation and scoring
  submission.py             # Challenge submission interface
  Requirements.txt          # Python dependencies
  wrappers/
    graph_wrapper.py        # GNN-compatible CybORG wrapper
    observation_graph.py    # Dynamic graph from observations
    globals.py              # Constants, action mappings, subnet topology
    nodes.py                # Graph node classes (System, Connection, File, Internet)
  models/
    cage4.py                # GCN actor/critic + PPO agent
    memory_buffer.py        # PPO experience replay buffer
    utils.py                # Graph batching utilities
  weights/
    contractor_active/      # Trained weights (contractor active scenario)
    contractor_inactive/    # Trained weights (contractor inactive scenario)
  checkpoints/              # Training checkpoints
  logs/                     # Training reward logs
```

---

## Citation

```bibtex
@misc{cage_challenge_4_announcement,
  author = {TTCP CAGE Working Group},
  Title = {TTCP CAGE Challenge 4},
  Publisher = {GitHub},
  Howpublished = {\url{https://github.com/cage-challenge/cage-challenge-4}},
  Year = {2023}
}
```
