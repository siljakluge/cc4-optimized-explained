"""
graph_wrapper.py — Graph-based Wrapper for CybORG Environment (TTCP CAGE Challenge 4)

This wrapper enables graph-based observations for Blue agents using PyTorch Geometric-compatible
data structures. It supports observation construction, message passing simulation,
tabular-to-graph fusion, and action translation for reinforcement learning.

Features:
- Builds graph observations from CybORG simulator state
- Encodes subnet and host interactions as node and edge features
- Allows integration of PPO-based GNN agents with tabular state variables
- Manages inter-agent messaging and subnet connectivity

Primary Class:
- GraphWrapper: Extends EnterpriseMAE, interfacing CybORG with RL agents
"""

from copy import deepcopy

import numpy as np
import torch

from CybORG.env import CybORG
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from CybORG.Shared.Enums import TernaryEnum

from wrappers.observation_graph import ObservationGraph
from wrappers.globals import *

class GraphWrapper(EnterpriseMAE):
    def __init__(self, env: CybORG, *args, **kwargs):
        # Force padded obs so each subnet block has a fixed SN_BLOCK_SIZE=59
        # (9 subnet-id + 9 blocked + 9 comms + 16 compromised + 16 scanned).
        # Without pad_spaces=True, BlueFlatWrapper returns variable-length obs per
        # scenario (actual host counts), which breaks the hardcoded block-size
        # indexing in _parse_tabular.
        kwargs.setdefault('pad_spaces', True)
        super().__init__(env, *args, **kwargs)

        self.graphs = dict()
        self.env = env
        self.agent_names = [f'blue_agent_{i}' for i in range(5)]
        self.ts = 0

        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        # Pre-allocated buffers for performance (reused each step)
        self._phase_buf = torch.zeros((1, 3))
        self._msg_buf_6x3 = torch.zeros((6, 3))
        self._msg_buf_4x3 = torch.zeros((4, 3))
        self._tabular_x_buf = None  # Lazily allocated per graph size
        self._combine_pad_buf = None  # Lazily allocated per graph size
        self._all_msg_buf = None  # Lazily allocated per graph size

    def _get_malfile(self, hostname: str) -> float:
        state = self.env.environment_controller.state
        if hostname not in state.hosts:
            return 0.0

        return float(any(
            f.density >= 0.9 and not f.signed
            for f in state.hosts[hostname].files
        ))

    def action_translator(self, agent_name, a_id):
        '''
        Translates output of PPO model to an action for the CybORG env. 
        Model provides output as
        Node-actions, edge-actions, global-actions, per-subnet.
        '''
        session = 0 # Seems the same every time?

        if a_id is None:
            return Monitor(session, agent_name)

        agent_id = int(agent_name[-1])
        which_subnet = MY_SUBNETS[agent_id][a_id // MAX_ACTIONS]
        a_id %= MAX_ACTIONS

        # Node action
        if a_id < N_NODE_ACTIONS*MAX_HOSTS:
            a = NODE_ACTIONS[a_id // MAX_HOSTS]
            target = a_id % MAX_HOSTS

            if target > 5:
                target = f'{which_subnet}_user_host_{target-6}'
            else:
                target = f'{which_subnet}_server_host_{target}'

            return a(session=session, agent=agent_name, hostname=target)

        # Edge action
        elif (a_id := a_id - (N_NODE_ACTIONS*MAX_HOSTS)) < (N_EDGE_ACTIONS*POSSIBLE_NEIGHBORS):
            a = EDGE_ACTIONS[a_id // POSSIBLE_NEIGHBORS]
            target = [r for r in ROUTERS if which_subnet not in r][a_id % POSSIBLE_NEIGHBORS]

            return a(session, agent_name, target.replace('_router',''), which_subnet)

        # Global action (only one)
        else:
            return Monitor(session, agent_name)


    def step(self, action):
        '''
        Take an action, execute it, and update internal approximation of the
        environment based on new observation.

        Args: 
            action: dict of {agent_id (int) : action_id (int)}
        '''
        # Convert from model out to Action objects
        action = {
            k:self.action_translator(k,v)
            for k,v in action.items()
        }

        # Gets the info from the tabular wrapper (4 dims per host, in order)
        observation, reward, term, trunc, info = super().step(
            action_dict=action, messages=self.msg
        )

        # Tell ObservationGraph what happened and update
        graph_obs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = observation[agent]
            g = self.graphs[agent]

            # Get the raw observation dictionary for this agent
            dict_obs = self.env.environment_controller.get_last_observation(agent).data
            msg = dict_obs.pop('message')
            msg = np.stack(msg, axis=0)

            # Indicates if msg was recieved or comms are blocked
            # This way we differentiate between feature for 0 and unknown
            recieved_msg = msg[:, -1:]
            if i != 4:
                # Repeat agent 4's 'is_recieved' message across 2 more subnets
                recieved_msg = np.concatenate([recieved_msg, np.zeros((2,1))], axis=0)
                recieved_msg[-2:] = recieved_msg[-3]

                # Pull out messages for 'was_scanned' and 'was_comprimised'
                msg_small = msg[:-1, :2]
                msg_big = msg[-1, :6].reshape(3,2)
                msg = np.concatenate([msg_small, msg_big], axis=0)
            else:
                msg = msg[:, :2]

            msg = np.concatenate([msg, recieved_msg], axis=1)

            # Update the graph based on the raw dictionary 
            g.parse_observation(dict_obs)

            # Pull node features from tabular observation, and also update 
            # graph subnet connectivity edges. 
            tab_x,phase,new_msg = self._parse_tabular(o, g) 
            
            self.msg[agent] = new_msg

            # Combine node features from graph source, and tabular source
            x,ei,masks = g.get_state(MY_SUBNETS[i])
            x = self._combine_data(x, tab_x)

            # Mask/pack into conviniently sized tensors for the GNN models 
            obs = self._to_obs(x,ei,masks,phase,msg,new_msg)

            # During training, we need to know if the agent is still mid-action
            # If so, we don't bother calculating an action next turn 
            is_blocked = dict_obs['success'] == TernaryEnum.IN_PROGRESS
            graph_obs[agent] = (obs, is_blocked)

        self.ts += 1
        self.last_obs = graph_obs
        return graph_obs, reward, term, trunc, info, action, observation

    def reset(self):
        '''
        Rebuild internal graph representation with parameters of new environment
        '''
        self.ts = 0

        obs_tab, action_mask = super().reset()
        g = ObservationGraph()

        # I don't *think* this is cheating, because FixedActionWrapper gets
        # to manipulate the obs returned by env.reset() which is the same thing.
        # Graph updates after intialization will all be using partial knowledge
        # known only to the agents.
        obs_dict = self.env.environment_controller.init_state
        g.setup(obs_dict)

        # Set message to empty for all agents
        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        my_state = dict()
        self.graphs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = obs_tab[agent]

            # Message from agent 4 has 2 extra subnet infos
            if i != 4:
                dummy_msg = (np.zeros((6,3)), np.zeros(8))
            else:
                dummy_msg = (np.zeros((4,3)), np.zeros(8))

            # Duplicate shared observation of the initial graph across
            # all agents (but make sure not to pass by reference)
            g_ = deepcopy(g)
            self.graphs[agent] = g_

            # Get tabular features and update connectivity graph 
            tab_x,phase,_ = self._parse_tabular(o,g_)

            # Combine all node features together and package for agents
            x,ei,masks = g_.get_state(MY_SUBNETS[int(agent[-1])])
            x = self._combine_data(x, tab_x)
            obs = self._to_obs(x,ei,masks,phase, *dummy_msg)

            # By default, agents are not blocked on turn 0
            my_state[agent] = (obs, False)

        self.last_obs = my_state
        return my_state, action_mask


    def _parse_tabular(self, x, g):
        '''
        Pull the per-host data out of the tabular observation. 
        Data is formatted as 
        
        Mission phase           (1d, 0,1, or 2)
        Subnet info (x8)
            Subnet id           (9d)
            Blocked subnets     (9d)
            Comm policy         (9d)
            Comprimised hosts   (16d)
            Scanned hosts       (16d)
        Messages (4x8)

        We pull out the host data and append it to the host nodes, 
        pass the phase as a global vector, 
        and add messages to the subnet nodes they originate from 
        '''
        # First bit is phase
        # Last 4x8 are messages from other agents
        phase_idx = int(x[0])
        sn_block = x[1:-(4*8)]
        subnets = sn_block.shape[0] // SN_BLOCK_SIZE

        relevant_subnets = []
        src = []
        dst = []

        # Reuse pre-allocated buffer when size matches, otherwise reallocate
        n_nodes = g.n_permenant_nodes
        if self._tabular_x_buf is None or self._tabular_x_buf.size(0) != n_nodes:
            self._tabular_x_buf = torch.zeros(n_nodes, 3)
        else:
            self._tabular_x_buf.zero_()
        x = self._tabular_x_buf
        msgs = []

        # Only affects agent4 but may as well be generalizeable
        for i in range(subnets):
            block = sn_block[SN_BLOCK_SIZE*i : SN_BLOCK_SIZE*(i+1)]

            # Pull out edges between subnets
            sn = block[:18]
            # pad_spaces=True pads short-agent obs to the long (agent_4) layout
            # with trailing zeros. Zero-padded blocks have no one-hot subnet id;
            # skip them so we only process real subnet blocks.
            nz = sn[:9].nonzero()[0]
            if len(nz) == 0:
                continue
            me = ROUTERS[ nz[0] ]
            can_maybe_connect_to = (sn[9:18] == 0).nonzero()[0]

            # Logic for subnet routing 
            if INTERNET in can_maybe_connect_to:
                can_connect_to = [ROUTERS[i] for i in can_maybe_connect_to]
            else:
                # Can connect to anything in LAN
                can_connect_to = [
                    ROUTERS[i] for i in can_maybe_connect_to
                    if ROUTERS[i] in ACCESSABLE_OFFLINE[me]
                ]

            router_name = me
            me = [me] * len(can_connect_to)
            src += can_connect_to
            dst += me

            # Pull out features for servers/hosts that exist
            hosts = torch.from_numpy(block[27:]).reshape(2, 16).T.float()
            n_srv, n_usr = g.subnet_size[router_name]

            srv_idx = list(range(n_srv))
            usr_idx = list(range(6, n_usr + 6))

            # Insert into rows corresponding w server/host nodes in graph
            # (Always directly after node for subnet they are on)
            start_usr_idx = g.nids[router_name] + 1
            start_srv_idx = start_usr_idx + len(usr_idx)
            end_srv_idx = start_srv_idx + len(srv_idx)

            # Bestehende zwei Features
            x[start_usr_idx:start_srv_idx, :2] = hosts[usr_idx]
            x[start_srv_idx:end_srv_idx, :2] = hosts[srv_idx]

            # Malfile-Flags in derselben Reihenfolge erzeugen
            usr_malfiles = []
            for j in range(n_usr):
                h = f'{router_name.replace("_router", "")}_user_host_{j}'
                usr_malfiles.append(self._get_malfile(h))

            srv_malfiles = []
            for j in range(n_srv):
                h = f'{router_name.replace("_router", "")}_server_host_{j}'
                srv_malfiles.append(self._get_malfile(h))

            if usr_malfiles:
                x[start_usr_idx:start_srv_idx, 2] = torch.tensor(usr_malfiles, dtype=torch.float32)
            if srv_malfiles:
                x[start_srv_idx:end_srv_idx, 2] = torch.tensor(srv_malfiles, dtype=torch.float32)

            msg = list((hosts.sum(dim=0) > 0).long())
            msgs += msg

            relevant_subnets.append(router_name)

        g.set_firewall_rules(src,dst)
        self._phase_buf.zero_()
        self._phase_buf[0, phase_idx] = 1
        phase = self._phase_buf.clone()

        # Make messages all 8-dim and add checkbit to the end
        padding = 8-len(msgs)
        msgs += [0]*padding
        msgs[-1] = 1
        msg = np.array(msgs)

        return x,phase,msg

    def _combine_data(self, graph_x, tabular_x):
        '''
        Stick the tabular data onto the node feature matrix
        on the appropriate rows--those corresponding with
        the hosts the tabular data is referencing
        '''
        # Tabular x only accounts for subnets and workstations
        # Processes/connections have higher indices, but no features
        # from the FlatActionWrapper, so need to be padded before combined
        pad_rows = graph_x.size(0) - tabular_x.size(0)
        pad_cols = tabular_x.size(1)
        if self._combine_pad_buf is None or self._combine_pad_buf.size(0) != pad_rows or self._combine_pad_buf.size(1) != pad_cols:
            self._combine_pad_buf = torch.zeros(pad_rows, pad_cols)
        # No need to zero -- buffer is always the same zeros
        tabular_x = torch.cat([tabular_x, self._combine_pad_buf], dim=0)

        return torch.cat([graph_x, tabular_x], dim=1)

    def _to_obs(self, x,ei,masks,phase, other_msg,my_msg):
        '''
        Prepare for GNN injestion (assumes unbatched. E.g. this is
        called during inference, not training)

        Args:  
            x: feature matrix           (Nxd tensor)
            ei: edge index              (2xE tensor)
            masks: list of bitmaps for servers, users, 
                   subnet edges, and routers 
            phase: global state vector  (1x3 tensor)
            other_msg: messages from other agents
                                        (4x8 tensor)
            my_msg: the message this agent sends to 
                    other agents        (1x8 tensor)
        '''

        # Reuse or allocate message buffer
        n_nodes = x.size(0)
        if self._all_msg_buf is None or self._all_msg_buf.size(0) != n_nodes:
            self._all_msg_buf = torch.zeros(n_nodes, 3)

        # Happens in all cases except agent_4
        if len(masks) == 1:
            (srv,usr,edge,rtrs) = masks[0]

            all_msg = self._all_msg_buf
            all_msg.zero_()
            all_msg[rtrs] = torch.from_numpy(other_msg).float()

            # Edge[0][0] is always the subnet node managed by this agent
            all_msg[edge[0][0], :2] = torch.from_numpy(my_msg[:2]).float()

            # Set 'is_recieved' to a special value to indicate this is self
            all_msg[edge[0][0], 2] = -1

            x = torch.cat([x, all_msg], dim=1)

            return (
                x,ei,phase,
                srv,torch.tensor([srv.size(0)]),
                usr,torch.tensor([usr.size(0)]),
                edge, False
            )

        # If this is agent 4's state, we have to be careful to batch
        # three observations together. The graph info is the same,
        # so we don't duplicate that, but we have to concat the
        # masks together and scatter the messages we recieved
        # properly to the feature matrix
        srv,usr,edges = [],[],[]
        n_srv,n_usr = [],[]
        my_ids = []
        for (s,u,e,_) in masks:
            my_ids.append(e[0][0].item())

            srv.append(s)
            usr.append(u)
            edges.append(e)

            n_srv.append(s.size(0))
            n_usr.append(u.size(0))

        rtrs = masks[0][3]
        other_rtrs = [o.item() for o in rtrs if o.item() not in my_ids]

        all_msg = self._all_msg_buf
        all_msg.zero_()

        # Cache numpy-to-tensor conversions
        other_msg_t = torch.from_numpy(other_msg).float()
        my_msg_t = torch.from_numpy(my_msg[:6].reshape(3, 2)).float()

        all_msg[other_rtrs] = other_msg_t
        all_msg[my_ids, :2] = my_msg_t
        all_msg[my_ids, 2] = -1

        x = torch.cat([x, all_msg], dim=1)

        return (
            x,ei,phase.repeat_interleave(3,0),
            torch.cat(srv), torch.tensor(n_srv),
            torch.cat(usr), torch.tensor(n_usr),
            torch.cat(edges, dim=1), True
        )
    
    def malfile_obs_change(self, agent_name: str, observation) -> np.ndarray:
        """Build the standard obs then append one malfile bit per host."""
        base_obs = observation
        state = self.env.environment_controller.state

        malfile_flags: list[float] = []
        for sn in self.subnets(agent_name):
            subnet_hosts = self._cached_subnet_hosts.get(sn, [])
            for h in subnet_hosts:
                malfile_flags.append(
                    float(self._get_malfile(h)) if h in state.hosts else 0.0
                )

        if not malfile_flags:
            return base_obs

        return np.concatenate(
            [base_obs, np.array(malfile_flags, dtype=np.float32)]
        )
