"""
cage4.py — Core PPO Agent Implementation for TTCP CAGE Challenge 4

This module defines the agent architecture, training procedure, and GNN-based actor/critic networks
used to control five cooperating Blue agents in a cybersecurity simulation (CyBORG).

Main Components:
- InductiveActorNetwork: GCN + Self-Attention-based policy network
- InductiveCriticNetwork: GCN-based value estimator
- InductiveGraphPPOAgent: PPO training loop, agent memory, inference
- pad_sequence, extract_hosts: Preprocessing functions for graph batching
- SimpleSelfAttention: Implements subnet-global feature fusion
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from torch_geometric.nn import GCNConv

from models.memory_buffer_constraint import MultiPPOMemory
from models.utils import combine_marl_states

MAX_SERVERS = 6
MAX_USERS = 10
MAX_EDGES = 8


def pad_sequence(seq, lens, padding):
    device = seq.device
    dtype = seq.dtype

    padded = torch.zeros(lens.size(0), padding, seq.size(-1), device=device, dtype=dtype)
    mask = torch.ones(padded.size(0), padded.size(1), device=device, dtype=dtype)

    offset = 0
    for i, length in enumerate(lens):
        st = offset
        en = offset + length

        padded[i][:length] = seq[st:en]
        mask[i][length:] = 0
        offset += length

    return padded, mask.unsqueeze(-1)


def extract_hosts(x, servers, n_servers, users, n_users):
    srv = x[servers]
    srv,s_mask = pad_sequence(srv, n_servers, MAX_SERVERS)  # B x MAX_s x d

    usr = x[users]
    usr,u_mask = pad_sequence(usr, n_users, MAX_USERS)      # B x MAX_u x d

    hosts = torch.cat([srv,usr], dim=1)
    mask = torch.cat([s_mask, u_mask], dim=1)
    return hosts, mask

class SimpleSelfAttention(nn.Module):
    '''
    Implimenting global-node self-attention from
        https://arxiv.org/pdf/2009.12462.pdf
    '''
    def __init__(self, in_dim, h_dim, g_dim):
        super().__init__()

        self.att = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.Softmax(dim=-1)
        )
        self.feat = nn.Linear(in_dim, h_dim)
        self.glb = nn.Sequential(
            nn.Linear(h_dim+g_dim, g_dim),
            nn.Tanh()
        )

        self.g_dim = g_dim
        self.h_dim = h_dim

    def forward(self, v, mask, g=None):
        """
        v:    B x N x d
        mask: B x N x 1
        g:    B x d
        """
        if g is None:
            g = torch.zeros((v.size(0), self.g_dim), device=v.device, dtype=v.dtype)

        att = self.att(v)  # B x N x h
        feat = self.feat(v)  # B x N x h
        out = (att * feat * mask).sum(dim=1)  # B x h

        g_ = self.glb(torch.cat([out, g], dim=-1))  # B x g
        return g + g_  # short circuit


class InductiveActorNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 node_action_space=4, edge_action_space=2, global_action_space=1,
                 hidden1=256, hidden2=64, gdim=64, lr=0.0003, concat_edges=False):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)

        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        # Just learn a good parameter to encode each phase as
        self.global_net = nn.Linear(
            global_state_space, gdim
        )

        self.node_actions = nn.Sequential(
            nn.Linear(hidden2+gdim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
            nn.Linear(hidden2 // 2, node_action_space)
        )

        # If edges should be processed as
        # f(src * dst) or f(src || dst)
        self.concat_edges = concat_edges
        self.edge_actions = nn.Sequential(
            nn.Linear(hidden2 if not concat_edges else hidden2*2, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden2 // 2),
            nn.ReLU(),
        )
        self.edge_out = nn.Linear(hidden2 // 2 + gdim, edge_action_space)

        self.global_out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, global_action_space)
        )

        self.sm = nn.Softmax(dim=1)
        self.opt = Adam(self.parameters(), lr)

        self.node_action_space = node_action_space
        self.edge_action_space = edge_action_space

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        # Always come in groups of 9
        rtrs = action_edges.unique(sorted=True).squeeze(-1)
        bs = rtrs.size(0) // 9
        rtrs = rtrs.reshape(bs, 9)
        if multi_subnet:
            rtrs = rtrs.repeat_interleave(3, 0)

        # rtr_mask on same device as x
        rtr_mask = torch.ones(rtrs.size(0), 9, 1, device=x.device, dtype=x.dtype)

        # Global init features
        g0 = self.global_net(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g0_attn(v,mask, g=g0)

        # Layer 1
        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g1_attn(v,mask, g=g)

        # Layer 2
        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        rtr = x[rtrs]
        v = torch.cat([v, rtr], dim=1)
        mask = torch.cat([mask, rtr_mask], dim=1)
        g = self.g2_attn(v,mask, g=g) # B x d_g

        # B x 16 x d
        z,mask = extract_hosts(x, servers, n_servers, users, n_users)

        # Attach global vec to all nodes in each batch
        z = torch.cat(
            [z, g.unsqueeze(1).repeat(1,z.size(1),1)],
            dim=-1
        )
        node_a = self.node_actions(z) * mask

        # B x 16 x a_n
        nbatches = node_a.size(0)

        # Make rows actions, and columns nodes
        node_a = node_a.transpose(1,2)  # B x a_n x 16
        node_a = node_a.reshape(        # B x 16*a_n
            nbatches,
            (MAX_SERVERS+MAX_USERS)*self.node_action_space
        )

        # Calculate edge-level action probs
        src,dst = action_edges
        src = x[src]; dst = x[dst]

        if self.concat_edges:
            edge_a = self.edge_actions(torch.cat([src,dst], dim=-1))
        else:
            edge_a = self.edge_actions(src) * self.edge_actions(dst)

        # Add in global vector
        edge_a = torch.cat([
            edge_a,
            g.repeat_interleave(MAX_EDGES,0)
        ], dim=1)
        edge_a = self.edge_out(edge_a)

        # Assume edge actions are always in groups of 8 (as they are in CAGE4)
        edge_a = edge_a.reshape(        # B x 8 x a_e
            node_a.size(0),
            MAX_EDGES,
            edge_a.size(-1)
        )
        edge_a = edge_a.transpose(1,2)  # B x a_e x 8 (columns are nodes, rows are actions)
        edge_a = edge_a.reshape(        # B x 8*a_e
            nbatches, edge_a.size(1)*MAX_EDGES
        )

        # Finally, compute prob of taking a global action
        # (Not an action upon a node or an edge. E.g. sleep)
        glb_a = self.global_out(g) # B x d

        out = torch.cat([node_a, edge_a, glb_a], dim=-1)

        # blue_agent_4 sends in groups of 3 subnets.
        # Really makes batching tricky
        if multi_subnet:
            out = out.reshape(out.size(0)//3, out.size(1)*3)

        out[out == 0] = -float('inf')   # So softmax prob is 0
        out = self.sm(out)

        return Categorical(out)


class InductiveCriticNetwork(nn.Module):
    def __init__(self, in_dim, global_state_space=3,
                 hidden1=256, hidden2=64, gdim=64, lr=0.001):
        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.out = nn.Sequential(
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, 1)
        )

        self.gs = nn.Linear(
            global_state_space, gdim
        )
        self.g0_attn = SimpleSelfAttention(in_dim, hidden1, gdim)
        self.g1_attn = SimpleSelfAttention(hidden1, hidden1, gdim)
        self.g2_attn = SimpleSelfAttention(hidden2, hidden2, gdim)

        self.out = nn.Sequential(
            nn.Linear(gdim, gdim//2),
            nn.ReLU(),
            nn.Linear(gdim//2, 1)
        )
        self.opt = Adam(self.parameters(), lr)

    def forward(self, x, ei, global_vec, servers, n_servers, users, n_users, action_edges, multi_subnet):
        g0 = self.gs(global_vec)

        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g0_attn(v, mask, g=g0)

        x = torch.relu(self.conv1(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g1_attn(v, mask, g=g)

        x = torch.relu(self.conv2(x, ei))
        v,mask = extract_hosts(x, servers, n_servers, users, n_users)
        g = self.g2_attn(v, mask, g=g)

        # I guess just average the three global vectors together?
        if multi_subnet:
            g = g.reshape(g.size(0) // 3, 3, g.size(-1))
            g = g.mean(dim=1)

        return self.out(g)


class InductiveGraphPPOAgent():
    '''
    Class to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''

    def __init__(self, in_dim, gamma=0.99, lmbda=0.95, clip=0.1, bs=5, epochs=6,
                 a_kwargs=dict(), c_kwargs=dict(), training=True, concat_edges=False, device=None):

        self.device = device or torch.device("cpu")
        # GNN original Actor Critic
        self.actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)

        # heuristic Actor Critic
        self.h_actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.h_critic = InductiveCriticNetwork(in_dim, **c_kwargs)

        self.actor.to(self.device)
        self.critic.to(self.device)
        self.h_actor.to(self.device)
        self.h_critic.to(self.device)

        self.memory = MultiPPOMemory(bs, agents=5)
        self.h_memory = MultiPPOMemory(bs, agents=5)

        self.args = (in_dim,)
        self.kwargs = dict(
            gamma=gamma, lmbda=lmbda, clip=clip, bs=bs, epochs=epochs,
            a_kwargs=a_kwargs, c_kwargs=c_kwargs, training=training, concat_edges=concat_edges
        )

        # PPO Hyperparams
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss()

        # Lagrange multiplier for constrained PPO
        self.k = 8
        self.lagrange_multiplier_history = [0.0] * self.k
        self.lagrange_multiplier = 0.0
        self.lagrange_lr = a_kwargs.get('lr', 0.0003)

    # Required by CAGE but not utilized
    def end_episode(self):
        pass

    # Required by CAGE but not utilized
    def set_initial_values(self, action_space, observation):
        pass

    def train(self):
        '''
        Set modules to training mode
        '''
        self.actor.train()
        self.critic.train()
        self.h_actor.train()
        self.h_critic.train()

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()
        self.critic.eval()
        self.h_actor.eval()
        self.h_critic.eval()

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def _zero_grad_h(self):
        '''
        Reset opt for heuristic
        '''
        self.h_actor.opt.zero_grad()
        self.h_critic.opt.zero_grad()

    def _step(self):
        '''
        Call opt autograd
        '''
        self.actor.opt.step()
        self.critic.opt.step()

    def _step_h(self):
        '''
        Call opt autograd on heuristic
        '''
        self.h_actor.opt.step()
        self.h_critic.opt.step()


    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems, h_mems):
        self.memory.mems = mems
        self.h_memory.mems = h_mems

    def save(self, outf='saved_models/ppo.pt'):
        me = (self.args, self.kwargs)

        torch.save({
            'actor': self.actor.state_dict(),
            'h_actor': self.h_actor.state_dict(),
            'critic': self.critic.state_dict(),
            'h_critic': self.h_critic.state_dict(),
            'agent': me
        }, outf)

    @torch.no_grad()
    def get_values(self, states):
        return self.critic(*states), self.h_critic(*states)

    @torch.no_grad()
    def get_action(self, obs, *args):
        """
        Sample an action from the actor's distribution given the current state.

        If eval(), only returns the action.
        If train() returns action, value, and log prob.
        """
        state, is_blocked = obs
        if is_blocked:
            return None

        # state is a tuple coming from the env: (x, ei, global_vec, servers, ...)
        # Keep the original CPU state for memory, but create a device copy for the networks.
        state_device = tuple(
            x.to(self.device) if torch.is_tensor(x) else x
            for x in state
        )

        distro = self.actor(*state_device)

        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value, h_value = self.get_values(state_device)
        prob = distro.log_prob(action)

        # Return scalars for logging / memory
        return action.item(), value.item(), prob.item(), h_value.item()
    
    @torch.no_grad()
    def get_action_h(self, obs, *args):
        """
        Sample an action from the heuristic actor's distribution given the current state.

        If eval(), only returns the action.
        If train() returns action, value, and log prob.
        """
        state, is_blocked = obs
        if is_blocked:
            return None

        # state is a tuple coming from the env: (x, ei, global_vec, servers, ...)
        # Keep the original CPU state for memory, but create a device copy for the networks.
        state_device = tuple(
            x.to(self.device) if torch.is_tensor(x) else x
            for x in state
        )

        distro = self.h_actor(*state_device)

        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value, h_value = self.get_values(state_device)
        prob = distro.log_prob(action)

        # Return scalars for logging / memory
        return action.item(), value.item(), prob.item(), h_value.item()

    def remember(self, idx, s, a, v, p, r, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s,a,v,p,r,t)

    def remember_h(self, idx, s, a, v, p, r, t):
        '''
        Save an observation to the heuristic agent's memory buffer
        '''
        self.h_memory.remember(idx, s,a,v,p,r,t)  

    def discount_rewards(self, r, t):
        '''
        Compute discounted rewards for a trajectory, given rewards and terminal flags.
        '''
        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(r), reversed(t)):
            if is_terminal:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        return rewards

    def compute_gae(self, rewards, values, terminals):
        '''
        Compute GAE for a trajectory, given rewards, value estimates, and terminal flags.
        Always returns a list of Python floats so callers can safely call torch.tensor() on it.
        '''
        T = len(rewards)
        advantages = [0.0] * T
        gae = 0.0

        for t in reversed(range(T)):
            rv   = rewards[t].item()  if torch.is_tensor(rewards[t])  else float(rewards[t])
            vt   = values[t].item()   if torch.is_tensor(values[t])   else float(values[t])
            vt1  = (values[t+1].item() if torch.is_tensor(values[t+1]) else float(values[t+1])) if t < T - 1 else 0.0
            term = float(terminals[t])

            delta = rv + self.gamma * vt1 * (1 - term) - vt
            gae   = delta + self.gamma * self.lmbda * (1 - term) * gae
            advantages[t] = gae

        return advantages

    def learn(self, verbose: bool = False):
        """
        This runs the PPO update algorithm on memories stored in self.memory.
        Assumes that an external process is adding memories to the buffer.

        Nameing conventions:
        - s: state
        - a: action
        - v: value (critic output)
        - p: log prob (actor output)
        - r: reward
        - t: terminal flag
        - h_r: heuristic reward (for constrained PPO)
        - h_v: heuristic value (for constrained PPO)
        
        *_{policy}_{rewardfunction}
        - Policies: j for main PPO, h for heuristic PPO
        - Reward functions: r for main env reward, h for heuristic reward
        """
        last_total_loss = None  # to have something safe to return at the end
        last_total_loss_h = None  # to have something safe to return at the end

        # Normalize helper function
        def _normed(lst):
            t_ = torch.tensor(lst, dtype=torch.float32, device=self.device)
            return (t_ - t_.mean()) / (t_.std() + 1e-5)

        for e in range(self.epochs):
            s, a, v, p, r, t, h_r, h_v, batches = self.memory.get_batches()
            s_h, a_h, v_h, p_h, r_h, t_h, h_r_h, h_v_h, batches_h = self.h_memory.get_batches() #all under heuristic policy

            # --- 1. Discounted returns (reversed accumulation, then flip) ---
            rewards_j_r = self.discount_rewards(r, t)[::-1]
            rewards_j_h = self.discount_rewards(h_r, t)[::-1]
            rewards_h_r = self.discount_rewards(r_h, t_h)[::-1]
            rewards_h_h = self.discount_rewards(h_r_h, t_h)[::-1]

            # --- 2. Convert to tensors ON DEVICE and normalize ---
            r_tensor_j_r = _normed(rewards_j_r)
            r_tensor_j_h = _normed(rewards_j_h)
            r_tensor_h_r = _normed(rewards_h_r)
            r_tensor_h_h = _normed(rewards_h_h)

            # v is typically a list/np array of value estimates
            v_tensor_j_r = torch.tensor(v, dtype=torch.float32, device=self.device)
            v_tensor_j_h = torch.tensor(h_v, dtype=torch.float32, device=self.device)
            v_tensor_h_r = torch.tensor(v_h, dtype=torch.float32, device=self.device)
            v_tensor_h_h = torch.tensor(h_v_h, dtype=torch.float32, device=self.device)

            # Advantage on device
            advantages_j_r = r_tensor_j_r - v_tensor_j_r
            advantages_j_h = r_tensor_j_h - v_tensor_j_h
            advantages_h_r = r_tensor_h_r - v_tensor_h_r
            advantages_h_h = r_tensor_h_h - v_tensor_h_h

            #Alternative implementaion with GAE
            """
            # --- 2. Convert to tensors ON DEVICE and normalize ---
            r_tensor_j_r = torch.tensor(r, dtype=torch.float32, device=self.device)
            r_tensor_j_r = _normed(r_tensor_j_r)

            r_tensor_j_h = torch.tensor(h_r, dtype=torch.float32, device=self.device)
            r_tensor_j_h = _normed(r_tensor_j_h)

            r_tensor_h_r = torch.tensor(r_h, dtype=torch.float32, device=self.device)
            r_tensor_h_r = _normed(r_tensor_h_r)

            r_tensor_h_h = torch.tensor(h_r_h, dtype=torch.float32, device=self.device)
            r_tensor_h_h = _normed(r_tensor_h_h)

            # v is typically a list/np array of value estimates
            v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)
            h_v_tensor = torch.tensor(h_v, dtype=torch.float32, device=self.device)
            v_h_tensor = torch.tensor(v_h, dtype=torch.float32, device=self.device)
            h_v_h_tensor = torch.tensor(h_v_h, dtype=torch.float32, device=self.device)


            # Advantage on device
            # advantages: Advantages_{policy}_{rewardfunction} = {rewardfunction} - {valuefunciton of policy}
            advantages_j_r = r_tensor_j_r - v_tensor
            advantages_j_h = r_tensor_j_h - h_v_tensor
            advantages_h_r = r_tensor_h_r - v_h_tensor
            advantages_h_h = r_tensor_h_h - h_v_h_tensor
            
            # Compute GAE advantages (returns Python float lists — safe for torch.tensor())
            # Value function pairing (per HEPO Appendix A.1):
            #   gae_j_r : j-traj, main reward   → baseline = j-critic (v)        = V^pi_r
            #   gae_j_h : j-traj, heuristic rew → baseline = j-critic (v)        = V^pi_h (shared critic)
            #   gae_h_r : h-traj, main reward   → baseline = h-critic (h_v_h)   = V^piH_r
            #   gae_h_h : h-traj, heuristic rew → baseline = h-critic (h_v_h)   = V^piH_h
            gae_j_r = torch.tensor(self.compute_gae(rewards=r_tensor_j_r, values=v_tensor, terminals=t), dtype=torch.float32, device=self.device)
            gae_j_h = torch.tensor(self.compute_gae(rewards=r_tensor_j_h, values=h_v_tensor, terminals=t), dtype=torch.float32, device=self.device)
            gae_h_r = torch.tensor(self.compute_gae(rewards=r_tensor_h_r, values=v_h_tensor, terminals=t_h), dtype=torch.float32, device=self.device)
            gae_h_h = torch.tensor(self.compute_gae(rewards=r_tensor_h_h, values=h_v_h_tensor, terminals=t_h), dtype=torch.float32, device=self.device)

            # Critic targets: return_t = GAE_t + V(s_t)  (unnormalized — critic regresses onto scale of rewards)
            v_np     = v_tensor.tolist()
            h_v_h_np = h_v_h_tensor.tolist()
            returns_j = torch.tensor([gae_j_r[i] + v_np[i]     for i in range(len(gae_j_r))], dtype=torch.float32, device=self.device)
            returns_h = torch.tensor([gae_h_h[i] + h_v_h_np[i] for i in range(len(gae_h_h))], dtype=torch.float32, device=self.device)

            closs, aloss, eloss = 0.0, 0.0, 0.0
            h_closs, h_aloss, h_eloss = 0.0, 0.0, 0.0

            gae_j_r = _normed(gae_j_r)
            gae_j_h = _normed(gae_j_h)
            gae_h_r = _normed(gae_h_r)
            gae_h_h = _normed(gae_h_h)

            returns_j = _normed(returns_j)
            returns_h = _normed(returns_h)
            """

            closs, aloss, eloss = 0.0, 0.0, 0.0
            h_closs, h_aloss, h_eloss = 0.0, 0.0, 0.0

            # --- 3. Optimize for clipped advantage for each minibatch ---
            for b_idx, (b, b_h) in enumerate(zip(batches, batches_h)):

                # batches are not always the same size, check later if this is an issue.
                if len(b) != len(b_h):
                    #print(f"Warning: Unequal number of batches between main and heuristic PPO at epoch {e}, batch index {b_idx}. Shortening this batch.")
                    cut = min(len(b), len(b_h))
                    b = b[:cut]
                    b_h = b_h[:cut]

                b = b.tolist()
                b_h = b_h.tolist()

                # Combine graphs from minibatches so GNN is called once
                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                s_h_ = [s_h[idx] for idx in b_h]
                a_h_ = [a_h[idx] for idx in b_h]


                batched_states = combine_marl_states(s_)
                batched_states_h = combine_marl_states(s_h_)

                # If combine_marl_states returns tensors, move them to device
                # (assumes batched_states is a tuple/list of tensors)
                batched_states = tuple(
                    x.to(self.device) if torch.is_tensor(x) else x
                    for x in batched_states
                )

                batched_states_h = tuple(
                    x.to(self.device) if torch.is_tensor(x) else x
                    for x in batched_states_h
                )

                self._zero_grad()
                self._zero_grad_h()

                # --- 4. Forward pass on device ---
                dist       = self.actor(*batched_states)    # j actor on j states
                dist_j_on_h = self.actor(*batched_states_h) # j actor on h states (cross-trajectory)
                dist_h_on_j = self.h_actor(*batched_states) # h actor on j states (cross-trajectory)
                dist_h     = self.h_actor(*batched_states_h) # h actor on h states
                critic_vals   = self.critic(*batched_states)
                h_critic_vals = self.h_critic(*batched_states_h)

                # Actions + old logprobs as tensors on device
                a_tensor = torch.tensor(a_, dtype=torch.long, device=self.device)
                a_h_tensor = torch.tensor(a_h_, dtype=torch.long, device=self.device)

                # new_probs_{policy}_{trajectory}
                new_probs_j_j = dist.log_prob(a_tensor)
                new_probs_j_h = dist_j_on_h.log_prob(a_h_tensor)  # j policy on h states/actions

                new_probs_h_j = dist_h_on_j.log_prob(a_tensor)    # h policy on j states/actions
                new_probs_h_h = dist_h.log_prob(a_h_tensor)

                old_probs = torch.tensor(
                    [p[i] for i in b],
                    dtype=torch.float32,
                    device=self.device,
                )
                
                old_probs_h = torch.tensor(
                    [p_h[i] for i in b_h],
                    dtype=torch.float32,
                    device=self.device,
                )

                entropy = dist.entropy()  # on device
                entropy_h = dist_h.entropy()  # on device

                a_t_j_r = advantages_j_r[b]  # slice on device
                a_t_j_h = advantages_j_h[b]  # slice on device
                a_t_h_r = advantages_h_r[b_h]  # slice on device
                a_t_h_h = advantages_h_h[b_h]  # slice on device

                r_b = r_tensor_j_r[b]  # returns for this batch
                r_b_h = r_tensor_h_h[b_h]  # returns for this batch

                # Calculate constraint coefficient
                U_j = (1 + self.lagrange_multiplier) * a_t_j_r + a_t_j_h
                U_h = (1 + self.lagrange_multiplier) * a_t_h_r + a_t_h_h

                # Alternative implementation with GAE advantages and returns as critic targets
                """
                # Advantages for this batch — gae_* are already normalized tensors
                #b_t   = torch.tensor(b,   dtype=torch.long, device=self.device)
                #b_h_t = torch.tensor(b_h, dtype=torch.long, device=self.device)
                
                g_j_r = gae_j_r[b]
                g_j_h = gae_j_h[b]
                g_h_r = gae_h_r[b_h]
                g_h_h = gae_h_h[b_h]

                # Critic targets: returns (GAE + V), not raw rewards
                r_b   = returns_j[b]
                r_b_h = returns_h[b_h]

                # Calculate constraint coefficient
                U_j = (1 + self.lagrange_multiplier) * g_j_r + g_j_h
                U_h = (1 + self.lagrange_multiplier) * g_h_r + g_h_h"""


                # --- 5. PPO objective ---
                # ratio = exp(new_log_prob - old_log_prob)
                r_theta_j_j = (new_probs_j_j - old_probs).exp()
                clipped_r_theta_j_j = torch.clamp(
                    r_theta_j_j, min=1 - self.clip, max=1 + self.clip
                )

                r_theta_j_h = (new_probs_j_h - old_probs_h).exp()
                clipped_r_theta_j_h = torch.clamp(
                    r_theta_j_h, min=1 - self.clip, max=1 + self.clip
                )

                # r_theta for h on trajectory of j with new probs from h
                r_theta_h_j = (new_probs_h_j - old_probs_h).exp()
                clipped_r_theta_h_j = torch.clamp(
                    r_theta_h_j, min=1 - self.clip, max=1 + self.clip
                )

                # r_theta for h on trajectory of h with new probs from h
                r_theta_h_h = (new_probs_h_h - old_probs_h).exp()
                clipped_r_theta_h_h = torch.clamp(
                    r_theta_h_h, min=1 - self.clip, max=1 + self.clip
                )

                ### Actor loss
                # Update original PPO
                actor_loss_j = torch.min(r_theta_j_j * U_j, clipped_r_theta_j_j * U_j)
                actor_loss_j = -actor_loss_j.mean()

                actor_loss_h = torch.min(r_theta_j_h * U_h, clipped_r_theta_j_h * U_h)
                actor_loss_h = -actor_loss_h.mean()

                actor_loss = actor_loss_j + actor_loss_h

                # Update the heuristic PPO
                actor_h_loss_j = torch.min(r_theta_h_j * a_t_j_h, clipped_r_theta_h_j * a_t_j_h)
                # actor_h_loss_j = torch.min(r_theta_h_j * g_j_h, clipped_r_theta_h_j * g_j_h) # Alternative implementation with GAE advantages
                actor_h_loss_j = -actor_h_loss_j.mean()

                actor_h_loss_h = torch.min(r_theta_h_h * a_t_h_h, clipped_r_theta_h_h * a_t_h_h)
                # actor_h_loss_h = torch.min(r_theta_h_h * g_h_h, clipped_r_theta_h_h * g_h_h) # Alternative implementation with GAE advantages
                actor_h_loss_h = -actor_h_loss_h.mean()

                actor_h_loss = actor_h_loss_j + actor_h_loss_h

                # Critic loss (MSE between discounted returns and value estimates)
                critic_loss = self.mse(r_b.unsqueeze(-1), critic_vals)
                critic_h_loss = self.mse(r_b_h.unsqueeze(-1), h_critic_vals)

                # Entropy (encourage exploration)
                entropy_loss = entropy.mean()
                entropy_h_loss = entropy_h.mean()   

                # Total loss and backprop on device
                total_loss_j = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                total_loss_h = actor_h_loss + 0.5 * critic_h_loss - 0.01 * entropy_h_loss

                total_loss_j.backward()
                self._step()

                total_loss_h.backward()
                self._step_h()

                last_total_loss_j = total_loss_j  # keep track of the most recent loss for main PPO
                last_total_loss_h = total_loss_h  # keep track of the most recent loss for heuristic PPO

                if verbose:
                    print(
                        f'[{e}] C-Loss: {0.5 * critic_loss.item():0.4f}  '
                        f'A-Loss: {actor_loss.item():0.4f} '
                        f'E-loss: {-entropy_loss.item() * 0.01:0.4f}'
                    )
                    print(
                        f'[{e}] C-Loss_h: {0.5 * critic_h_loss.item():0.4f}  '
                        f'A-Loss_h: {actor_h_loss.item():0.4f} '
                        f'E-loss_h: {-entropy_h_loss.item() * 0.01:0.4f}'
                    )

                closs += critic_loss.item()
                aloss += actor_loss.item()
                eloss += entropy_loss.item()

                h_closs += critic_h_loss.item()
                h_aloss += actor_h_loss.item()
                h_eloss += entropy_h_loss.item()

                # Update Lagrange multiplier via primal-dual gradient ascent
                # grad = E_pi[A^piH_r] - E_piH[A^pi_r]
                #      ≈ a_t_j_r.mean()  - a_t_h_r.mean()
                # alpha increases when J(pi) < J(piH), emphasising task reward.
                constraint_violation = (a_t_j_r.mean() - a_t_h_r.mean()).item()
                self.lagrange_multiplier_history.append(constraint_violation)
                recent = self.lagrange_multiplier_history[-self.k:]
                # Use median of last k records to smooth
                recent_sorted = sorted(recent)
                mid = len(recent_sorted) // 2
                smoothed = (recent_sorted[mid] + recent_sorted[~mid]) / 2.0
                self.lagrange_multiplier = max(0.0, self.lagrange_multiplier - self.lagrange_lr * smoothed)

            # --- 6. Print average loss across minibatches ---
            closs /= len(batches)
            aloss /= len(batches)
            eloss /= len(batches)

            h_closs /= len(batches)
            h_aloss /= len(batches)
            h_eloss /= len(batches)
            print(
                f'[{e}] C-Loss: {0.5 * closs:0.4f}  '
                f'A-Loss: {aloss:0.4f} E-loss: {-eloss * 0.01:0.4f}'
            )
            print(
                f'[{e}] C-Loss_h: {0.5 * h_closs:0.4f}  '
                f'A-Loss_h: {h_aloss:0.4f} E-loss_h: {-h_eloss * 0.01:0.4f}'
            )

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        self.h_memory.clear()

        # In case nothing ran, guard against None
        return last_total_loss_j.item() if last_total_loss_j is not None else 0.0, \
               last_total_loss_h.item() if last_total_loss_h is not None else 0.0


def load(in_f):
    '''
    Loads model checkpoint file 
    '''
    data = torch.load(in_f)
    args,kwargs = data['agent']

    agent = InductiveGraphPPOAgent(*args, **kwargs)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])
    agent.h_actor.load_state_dict(data['h_actor'])
    agent.h_critic.load_state_dict(data['h_critic'])

    agent.eval()
    return agent

