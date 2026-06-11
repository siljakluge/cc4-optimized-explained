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

from models.memory_buffer import MultiPPOMemory
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

        self.actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.memory = MultiPPOMemory(bs, agents=5)

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

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def _step(self):
        '''
        Call opt autograd
        '''
        self.actor.opt.step()
        self.critic.opt.step()


    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems):
        self.memory.mems = mems

    def save(self, outf='saved_models/ppo.pt'):
        me = (self.args, self.kwargs)

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'agent': me
        }, outf)

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

        value = self.critic(*state_device)
        prob = distro.log_prob(action)

        # Return scalars for logging / memory
        return action.item(), value.item(), prob.item()

    def remember(self, idx, s, a, v, p, r, t):
        '''
        Save an observation to the agent's memory buffer
        '''
        self.memory.remember(idx, s,a,v,p,r,t)

    def learn(self, verbose: bool = False):
        """
        This runs the PPO update algorithm on memories stored in self.memory.
        Assumes that an external process is adding memories to the buffer.
        """
        last_total_loss = None  # to have something safe to return at the end

        for e in range(self.epochs):
            s, a, v, p, r, t, batches = self.memory.get_batches()

            # --- 1. Discounted returns (reversed accumulation, then flip) ---
            rewards_rev = []
            discounted_reward = 0.0
            for reward, is_terminal in zip(reversed(r), reversed(t)):
                if is_terminal:
                    discounted_reward = 0.0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards_rev.append(discounted_reward)
            rewards = rewards_rev[::-1]

            # --- 2. Convert to tensors ON DEVICE and normalize ---
            r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            r_tensor = (r_tensor - r_tensor.mean()) / (r_tensor.std() + 1e-5)

            # v is typically a list/np array of value estimates
            v_tensor = torch.tensor(v, dtype=torch.float32, device=self.device)

            # Advantage on device
            advantages = r_tensor - v_tensor

            closs, aloss, eloss = 0.0, 0.0, 0.0

            # --- 3. Optimize for clipped advantage for each minibatch ---
            for b_idx, b in enumerate(batches):
                b = b.tolist()

                # Combine graphs from minibatches so GNN is called once
                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]

                batched_states = combine_marl_states(s_)

                # If combine_marl_states returns tensors, move them to device
                # (assumes batched_states is a tuple/list of tensors)
                batched_states = tuple(
                    x.to(self.device) if torch.is_tensor(x) else x
                    for x in batched_states
                )

                self._zero_grad()

                # --- 4. Forward pass on device ---
                dist = self.actor(*batched_states)  # actor already on self.device
                critic_vals = self.critic(*batched_states)

                # Actions + old logprobs as tensors on device
                a_tensor = torch.tensor(a_, dtype=torch.long, device=self.device)
                new_probs = dist.log_prob(a_tensor)

                old_probs = torch.tensor(
                    [p[i] for i in b],
                    dtype=torch.float32,
                    device=self.device,
                )

                entropy = dist.entropy()  # on device

                a_t = advantages[b]  # slice on device
                r_b = r_tensor[b]  # returns for this batch

                # --- 5. PPO objective ---
                # ratio = exp(new_log_prob - old_log_prob)
                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clamp(
                    r_theta, min=1 - self.clip, max=1 + self.clip
                )

                # Actor loss
                actor_loss = torch.min(r_theta * a_t, clipped_r_theta * a_t)
                actor_loss = -actor_loss.mean()

                # Critic loss (MSE between discounted returns and value estimates)
                critic_loss = self.mse(r_b.unsqueeze(-1), critic_vals)

                # Entropy (encourage exploration)
                entropy_loss = entropy.mean()

                # Total loss and backprop on device
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                total_loss.backward()
                self._step()

                last_total_loss = total_loss  # keep track of the most recent loss

                if verbose:
                    print(
                        f'[{e}] C-Loss: {0.5 * critic_loss.item():0.4f}  '
                        f'A-Loss: {actor_loss.item():0.4f} '
                        f'E-loss: {-entropy_loss.item() * 0.01:0.4f}'
                    )

                closs += critic_loss.item()
                aloss += actor_loss.item()
                eloss += entropy_loss.item()

            # --- 6. Print average loss across minibatches ---
            closs /= len(batches)
            aloss /= len(batches)
            eloss /= len(batches)
            print(
                f'[{e}] C-Loss: {0.5 * closs:0.4f}  '
                f'A-Loss: {aloss:0.4f} E-loss: {-eloss * 0.01:0.4f}'
            )

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()

        # In case nothing ran, guard against None
        return last_total_loss.item() if last_total_loss is not None else 0.0


def load(in_f):
    '''
    Loads model checkpoint file
    '''
    data = torch.load(in_f, map_location='cpu', weights_only=False)
    args,kwargs = data['agent']

    agent = InductiveGraphPPOAgent(*args, **kwargs)
    agent.actor.load_state_dict(data['actor'])
    agent.critic.load_state_dict(data['critic'])

    agent.eval()
    return agent

