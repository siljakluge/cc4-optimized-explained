"""
train.py — Training script for CAGE Challenge 4 using PPO-based GNN agents

This script orchestrates the training process for 5 blue agents operating in the CybORG
simulation environment, leveraging parallel rollout generation, graph-based observations,
and centralized policy optimization.

Key Components:
 - Multi-agent parallel episode generation (joblib)
 - Per-agent memory collection for PPO updates
 - Environment setup using GraphWrapper and FastEnterpriseScenarioGenerator
 - PPO training loop with GNN-based agents
"""

from argparse import ArgumentParser
import os
import sys
from types import SimpleNamespace

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from joblib import Parallel, delayed
import torch
from tqdm import tqdm

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from src.envs.fast_scenario import FastEnterpriseScenarioGenerator

from models.cage4 import InductiveGraphPPOAgent
from models.memory_buffer import MultiPPOMemory
from wrappers.graph_wrapper import GraphWrapper
from wrappers.observation_graph import ObservationGraph

device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print("Using device:", device)

# Enable cuDNN autotuner for faster convolutions when using CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

SEED = 1337
# Workers are loky subprocesses doing CUDA inference on the same GPU.
# 25 replicas × 5 agents saturates a 12 GB card; 8 fits comfortably on a 3080 Ti.
# If you move to a 24 GB+ GPU (or switch to CPU rollouts) you can bump this back up.
HYPER_PARAMS = SimpleNamespace(
    N = 8,              # How many episodes before training
    workers = 8,        # How many envs can run in parallel
    bs = 2500,          # How many steps to learn from at a time
    episode_len = 500,
    training_episodes = 500_000, # Realistically, stops improving around 50k
    epochs = 4
)

N_AGENTS = 5
MAX_THREADS = max(4, os.cpu_count() or 4)  # Dynamic based on available cores
torch.manual_seed(SEED)
torch.set_num_threads(MAX_THREADS)

"""
Run one complete simulation episode.

Parameters:
    agents (List): List of GNN-PPO agent instances
    env (GraphWrapper): Simulation environment wrapper
    hp (SimpleNamespace): Hyperparameter container
    i (int): Process/thread ID

Returns:
    Tuple[List, float]: Per-agent PPO memories and total episode reward
"""
@torch.no_grad()
def generate_episode_job(agents, env, hp, i):
    '''
    Per-process job to generate one episode of memories
    for all 5 agents. Returns `N_AGENTS` memory buffers, 
    and the total reward for the episode. 

    Args: 
        agents:     list of keep.cage4.InductiveGraphAgent objects 
        env:        wrapped cyborg object 
        hp:         hyperparameter namespace 
        i:          process id in range(0, `hp.workers`)
    '''
    torch.set_num_threads(max(1, MAX_THREADS // hp.workers))

    # Initialize environment
    env.reset()
    states = env.last_obs
    blocked_rewards = [0]*N_AGENTS

    tot_reward = 0
    memory_buffers = MultiPPOMemory(hp.bs)

    # Begin episode 
    for ts in tqdm(
            range(hp.episode_len),
            desc=f'Worker {i}',
            disable=(hp.workers > 1)  # <-- disable when many workers
    ):
        actions = dict()
        memories = dict()

        # Get actions for all unblocked agents
        for k,(state,blocked) in states.items():
            i = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action,value,prob = agents[i].get_action((state,blocked))
                memories[i] = (state,action,value,prob)
                actions[k] = action

        next_state, rewards, _,_,_ ,_,_= env.step(actions)
        rewards = list(rewards.values())
        tot_reward += sum(rewards)/N_AGENTS

        # Delay recieving rewards until multi-step actions are completed. 
        # Agents recieve cumulative reward for all the timesteps 
        # they spent performing their action. 
        for i in range(N_AGENTS):
            if i in memories:
                s,a,v,p = memories[i]
                r = rewards[i] + blocked_rewards[i]
                t = 0 if ts < hp.episode_len-1 else 1

                memory_buffers.remember(i, s,a,v,p, r,t)
                blocked_rewards[i] = 0
            else:
                blocked_rewards[i] += rewards[i]

        states = next_state

    return memory_buffers.mems, tot_reward


    """
    Main training loop.

    Spawns multiple simulation environments to collect experiences in parallel
    and trains each agent with PPO using their respective experience.

    Parameters:
        agents (List): List of GNN-PPO agent instances
        hp (SimpleNamespace): Hyperparameter container
        seed (int): RNG seed for reproducibility
    """
def train(agents, hp, seed=SEED):
    [agent.train() for agent in agents]
    log = []

    # Only call constructors once out here to save some time
    envs = []
    for i in range(min(hp.workers, hp.N)):
        sg = FastEnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=hp.episode_len,
            pool_size=4,
        )
        env = CybORG(sg, "sim", seed=seed)
        envs.append(GraphWrapper(env))

    # Define learn function for threads to call later so we can 
    # parallelize the backprop step. Use more threads for Agent 4 
    # because they're managing 3 subnets instead of 1 (bigger graph/matrices)
    # Still not perfectly load-balanced, but close enough
    def learn(i):
            if i < 4:
                torch.set_num_threads(max(1, MAX_THREADS // 9))
            else:
                torch.set_num_threads(max(1, (MAX_THREADS // 9) * N_AGENTS))
            return agents[i].learn()

    # Begin training loop 
    for e in range(hp.training_episodes // hp.N):
        e *= hp.N

        # Generate N episodes in parallel 
        out = Parallel(backend='loky', n_jobs=hp.workers)(
            delayed(generate_episode_job)(agents, envs[i % len(envs)], hp, i) for i in range(hp.N)
        )

        # Concat memories across episodes, and transfer them to agents' 
        # internal memory buffers 
        memories, avg_rewards = zip(*out)

        # transpose: per-agent list of per-episode memory objects
        per_agent_mems = [list(m) for m in zip(*memories)]

        for i in range(N_AGENTS):
            agents[i].memory.mems = per_agent_mems[i]
            # tell MultiPPOMemory how many sub-mems it has now
            agents[i].memory.agents = len(per_agent_mems[i])

        # Use threads because agents are in heap memory
        # Parallel backpropagation 
        print("Updating")
        last_losses = Parallel(prefer='threads', n_jobs=N_AGENTS)(
            delayed(learn)(i) for i in range(N_AGENTS)
        )

        losses = ','.join([f'{last_losses[i]:0.4f}' for i in range(N_AGENTS)])
        print(f"[{e}] Loss: [{losses}]")

        # Log average reward across all episodes 
        avg_reward = sum(avg_rewards) / hp.N
        print(f"Avg reward for episode: {avg_reward}")
        log.append((avg_reward,e,sum(last_losses)/N_AGENTS))
        torch.save(log, f'logs/{hp.fnames}.pt')

        # Checkpoint model states 
        for i in range(N_AGENTS):
            agent = agents[i]
            agent.save(outf=f'checkpoints/{hp.fnames}-{i}_checkpoint.pt')

            if e % 10_000 < hp.N and e > hp.N:
                agent.save(outf=f'checkpoints/{hp.fnames}-{i}_{e//1000}k.pt')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('fname', help='Required: the name to save output files as.')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--embedding', type=int, default=128)
    ap.add_argument('--debug', action='store_true', help='Small, safe config')
    ap.add_argument("--phase_reward_mode", default="default",
                    choices=["default", "contractor_off", "red_only"])
    ap.add_argument("--reward_blue", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use for training")
    args = ap.parse_args()
    print(args)

    os.environ["CYBORG_PHASE_REWARD_MODE"] = args.phase_reward_mode
    os.environ["CYBORG_REWARD_blue"] = "1" if args.reward_blue else "0"

    if args.debug:
        HYPER_PARAMS.N = 2
        HYPER_PARAMS.workers = 2
        HYPER_PARAMS.bs = 64
        HYPER_PARAMS.episode_len = 50
        HYPER_PARAMS.training_episodes = 20   # really tiny
        HYPER_PARAMS.epochs = 1


    # Add directory for log files
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Add directory for model weights 
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Add 5 extra dimensions to observation graph: 
    #   3 for tabular data (gets appended to relevant hosts)
    #   3 for message data (gets appended to relevant subnets): 
    #       1 bit if subnet has comprimised host in it
    #       1 bit if subnet has scanned host in it
    #       1 bit if message was sent successfully 
    #
    # All handled in wrapper.graph_wrapper
    # NOTE: do NOT re-derive `device` here — it overrides the MPS/CUDA/CPU
    # selection at the top of this file. Reuse the module-level `device`.
    if args.device != "auto":
        device = torch.device(args.device)
        print(f"Overriding device selection. Using {device} for training.")

    agents = [InductiveGraphPPOAgent(
        ObservationGraph.DIM + 6,
        bs=HYPER_PARAMS.bs,
        a_kwargs={'lr': 0.0003, 'hidden1': args.hidden, 'hidden2': args.embedding},
        c_kwargs={'lr': 0.001, 'hidden1': args.hidden, 'hidden2': args.embedding},
        clip=0.2,
        epochs=HYPER_PARAMS.epochs,
        device=device,  # <-- important
    ) for _ in range(N_AGENTS)]

    HYPER_PARAMS.fnames = args.fname
    train(agents, HYPER_PARAMS)
