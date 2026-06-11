"""Graph data model for CybORG environment state visualization.

Converts CybORG simulation state into a JSON graph structure compatible
with the 3d-force-graph library for real-time 3D network visualization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
COLORS = {
    "host_clean": "#00ff41",
    "host_user_compromised": "#ffb300",
    "host_root_compromised": "#ff4444",
    "host_restoring": "#00ffff",
    "host_decoy_active": "#bf5fff",
    "agent_blue": "#0088ff",
    "agent_red": "#ff2222",
    "agent_green": "#44ff44",
    "traffic_blocked": "#ff4444",
    "traffic_active": "#00ff41",
    "edge_contains": "#555555",
    "edge_session": "#ffb300",
    "edge_action": "#ffffff",
    "subnet_default": "#888888",
}

# ---------------------------------------------------------------------------
# Default 3-D positions for subnet nodes (logical network topology)
# ---------------------------------------------------------------------------
SUBNET_POSITIONS: dict[str, dict[str, float]] = {
    "internet_subnet":                {"x": 0,    "y": 220,  "z": 0},
    "public_access_zone_subnet":      {"x": 0,    "y": 110,  "z": 0},
    "contractor_network_subnet":      {"x": -200, "y": 80,   "z": 40},
    "office_network_subnet":          {"x": -90,  "y": 10,   "z": -40},
    "admin_network_subnet":           {"x": 90,   "y": 10,   "z": 40},
    "operational_zone_a_subnet":      {"x": -140, "y": -190, "z": 50},
    "operational_zone_b_subnet":      {"x": 140,  "y": -190, "z": -50},
    "restricted_zone_a_subnet":       {"x": -140, "y": -90,  "z": -30},
    "restricted_zone_b_subnet":       {"x": 140,  "y": -90,  "z": 30},
}

SUBNET_ROLES: dict[str, str] = {
    "internet_subnet": "external",
    "public_access_zone_subnet": "dmz",
    "contractor_network_subnet": "external",
    "office_network_subnet": "internal",
    "admin_network_subnet": "internal",
    "operational_zone_a_subnet": "operational",
    "operational_zone_b_subnet": "operational",
    "restricted_zone_a_subnet": "restricted",
    "restricted_zone_b_subnet": "restricted",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GraphNode:
    """A single node in the network graph."""

    id: str
    type: str  # "subnet" | "host" | "agent"
    label: str
    group: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """A single edge in the network graph."""

    source: str
    target: str
    type: str  # "contains" | "connection" | "session" | "action_target" | "traffic"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSnapshot:
    """Full graph state at a single simulation step."""

    episode_id: int
    step: int
    mission_phase: int
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- convenience look-ups -----------------------------------------------
    def node_by_id(self, node_id: str) -> GraphNode | None:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def edges_of_type(self, edge_type: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.type == edge_type]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class GraphBuilder:
    """Constructs and updates ``GraphSnapshot`` instances."""

    # -- static topology -----------------------------------------------------

    @staticmethod
    def build_base_topology(
        subnet_hosts: dict[str, list[str]],
        subnet_connections: list[tuple[str, str]],
    ) -> GraphSnapshot:
        """Build the static network structure (no dynamic state yet).

        Parameters
        ----------
        subnet_hosts:
            Mapping of subnet name -> list of host names belonging to it.
        subnet_connections:
            Pairs of subnet names that are directly connected.
        """
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        # Subnet nodes
        for subnet_name in subnet_hosts:
            pos = SUBNET_POSITIONS.get(subnet_name, {"x": 0, "y": 0, "z": 0})
            role = SUBNET_ROLES.get(subnet_name, "internal")
            nodes.append(GraphNode(
                id=f"subnet:{subnet_name}",
                type="subnet",
                label=subnet_name.replace("_subnet", "").replace("_", " ").title(),
                group=subnet_name,
                properties={
                    "role": role,
                    "is_isolated": False,
                    "fx": pos["x"],
                    "fy": pos["y"],
                    "fz": pos["z"],
                },
            ))

        # Host nodes + containment edges
        for subnet_name, hosts in subnet_hosts.items():
            for host_name in hosts:
                nodes.append(GraphNode(
                    id=f"host:{host_name}",
                    type="host",
                    label=host_name.replace("_", " ").title(),
                    group=subnet_name,
                    properties={
                        "compromise_level": "none",
                        "has_red_session": False,
                        "decoy_count": 0,
                        "is_restoring": False,
                        "service_reliability": 1.0,
                    },
                ))
                edges.append(GraphEdge(
                    source=f"subnet:{subnet_name}",
                    target=f"host:{host_name}",
                    type="contains",
                    properties={"color": COLORS["edge_contains"], "width": 1},
                ))

        # Subnet-to-subnet connections
        for src, tgt in subnet_connections:
            edges.append(GraphEdge(
                source=f"subnet:{src}",
                target=f"subnet:{tgt}",
                type="connection",
                properties={
                    "is_blocked": False,
                    "is_active": True,
                    "color": COLORS["traffic_active"],
                    "width": 2,
                },
            ))

        return GraphSnapshot(
            episode_id=0,
            step=0,
            mission_phase=0,
            nodes=nodes,
            edges=edges,
            metadata={},
        )

    # -- dynamic update ------------------------------------------------------

    @staticmethod
    def update_from_state(
        base: GraphSnapshot,
        step: int,
        mission_phase: int,
        host_states: dict[str, dict[str, Any]],
        agent_actions: dict[str, dict[str, Any]],
        sessions: list[dict[str, Any]],
        traffic: list[dict[str, Any]],
        reward: float,
    ) -> GraphSnapshot:
        """Create a new snapshot with dynamic state overlaid on *base*.

        Parameters
        ----------
        base:
            The topology snapshot (usually from ``build_base_topology``).
        step:
            Current simulation step.
        mission_phase:
            Current mission phase (affects subnet isolation).
        host_states:
            ``{host_name: {compromise_level, has_red_session, decoy_count,
            is_restoring, service_reliability}}``
        agent_actions:
            ``{agent_name: {team, current_action, last_action_target}}``
        sessions:
            List of ``{agent, host, type}`` dicts describing active sessions.
        traffic:
            List of ``{source_subnet, target_subnet, is_blocked}`` dicts.
        reward:
            Cumulative or step reward for metadata.
        """
        snap = GraphSnapshot(
            episode_id=base.episode_id,
            step=step,
            mission_phase=mission_phase,
            nodes=copy.deepcopy(base.nodes),
            edges=copy.deepcopy(base.edges),
            metadata={},
        )

        # -- update host nodes -----------------------------------------------
        for host_name, state in host_states.items():
            node = snap.node_by_id(f"host:{host_name}")
            if node is None:
                continue
            node.properties.update(state)

        # -- update traffic edges --------------------------------------------
        for t in traffic:
            src_id = f"subnet:{t['source_subnet']}"
            tgt_id = f"subnet:{t['target_subnet']}"
            for edge in snap.edges:
                if edge.type != "connection":
                    continue
                ids = {edge.source, edge.target}
                if ids == {src_id, tgt_id}:
                    edge.properties["is_blocked"] = t.get("is_blocked", False)
                    edge.properties["is_active"] = not t.get("is_blocked", False)
                    edge.properties["color"] = (
                        COLORS["traffic_blocked"]
                        if t.get("is_blocked")
                        else COLORS["traffic_active"]
                    )

        # -- remove previous dynamic nodes/edges (agents, sessions, actions) -
        snap.nodes = [n for n in snap.nodes if n.type != "agent"]
        snap.edges = [
            e for e in snap.edges
            if e.type not in ("session", "action_target")
        ]

        # -- add agent nodes -------------------------------------------------
        for agent_name, info in agent_actions.items():
            team = info.get("team", "blue")
            color_key = f"agent_{team}"
            snap.nodes.append(GraphNode(
                id=f"agent:{agent_name}",
                type="agent",
                label=agent_name.replace("_", " ").title(),
                group=f"team_{team}",
                properties={
                    "team": team,
                    "current_action": info.get("current_action", "Sleep"),
                    "last_action_target": info.get("last_action_target"),
                    "color": COLORS.get(color_key, COLORS["agent_blue"]),
                },
            ))

            # action-target edge
            target = info.get("last_action_target")
            if target:
                target_id = (
                    f"host:{target}" if not target.startswith("host:") else target
                )
                if snap.node_by_id(target_id) is None:
                    continue
                snap.edges.append(GraphEdge(
                    source=f"agent:{agent_name}",
                    target=target_id,
                    type="action_target",
                    properties={
                        "color": COLORS["edge_action"],
                        "width": 2,
                        "is_active": True,
                    },
                ))

        # -- ensure agent nodes exist for all session agents -------------------
        existing_agents = {n.id for n in snap.nodes if n.type == "agent"}
        for sess in sessions:
            aid = f"agent:{sess['agent']}"
            if aid not in existing_agents:
                aname = sess["agent"]
                team = "red" if "red" in aname else "green" if "green" in aname else "blue"
                color_key = f"agent_{team}"
                # Green/red agents not in agent_actions: show session-based status
                default_action = {
                    "green": "UserActivity",
                    "red": "Attacking",
                }.get(team, "Sleep")
                snap.nodes.append(GraphNode(
                    id=aid,
                    type="agent",
                    label=aname.replace("_", " ").title(),
                    group=f"team_{team}",
                    properties={
                        "team": team,
                        "current_action": default_action,
                        "last_action_target": sess.get("host"),
                        "color": COLORS.get(color_key, COLORS["agent_blue"]),
                    },
                ))
                existing_agents.add(aid)

        # -- add session edges -----------------------------------------------
        existing_node_ids = {n.id for n in snap.nodes}
        for sess in sessions:
            agent_id = f"agent:{sess['agent']}"
            host_id = f"host:{sess['host']}"
            if agent_id not in existing_node_ids or host_id not in existing_node_ids:
                continue
            snap.edges.append(GraphEdge(
                source=agent_id,
                target=host_id,
                type="session",
                properties={
                    "session_type": sess.get("type", "unknown"),
                    "color": COLORS["edge_session"],
                    "width": 1,
                    "is_active": True,
                },
            ))

        # -- metadata --------------------------------------------------------
        actions_summary = {
            name: info.get("current_action", "Sleep")
            for name, info in agent_actions.items()
        }
        snap.metadata = {
            "total_reward": reward,
            "active_actions": actions_summary,
            "mission_phase": mission_phase,
        }

        return snap

    # -- serialization -------------------------------------------------------

    @staticmethod
    def to_json(snapshot: GraphSnapshot) -> dict[str, Any]:
        """Serialize a snapshot for the 3d-force-graph frontend.

        Returns a dict with ``nodes``, ``links``, and ``metadata`` keys.
        """
        json_nodes: list[dict[str, Any]] = []
        for node in snapshot.nodes:
            entry: dict[str, Any] = {
                "id": node.id,
                "name": node.label,
                "group": node.group,
                "type": node.type,
                "val": _node_value(node),
                "color": _node_color(node),
            }
            # Merge all extra properties so the frontend can access them.
            entry.update(node.properties)
            json_nodes.append(entry)

        json_links: list[dict[str, Any]] = []
        for edge in snapshot.edges:
            entry = {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "color": edge.properties.get("color", COLORS["edge_contains"]),
                "width": edge.properties.get("width", 1),
            }
            entry.update(edge.properties)
            json_links.append(entry)

        return {
            "nodes": json_nodes,
            "links": json_links,
            "metadata": {
                "episode_id": snapshot.episode_id,
                "step": snapshot.step,
                "mission_phase": snapshot.mission_phase,
                **snapshot.metadata,
            },
        }

    # -- diff ----------------------------------------------------------------

    @staticmethod
    def diff_snapshots(
        prev: GraphSnapshot,
        curr: GraphSnapshot,
    ) -> dict[str, Any]:
        """Return only the nodes and edges that changed between snapshots.

        Useful for efficient incremental WebSocket updates.
        """
        prev_nodes = {n.id: n for n in prev.nodes}
        curr_nodes = {n.id: n for n in curr.nodes}
        prev_edges = {_edge_key(e): e for e in prev.edges}
        curr_edges = {_edge_key(e): e for e in curr.edges}

        changed_nodes: list[dict[str, Any]] = []
        removed_node_ids: list[str] = []
        changed_links: list[dict[str, Any]] = []
        removed_link_keys: list[str] = []

        # Nodes added or changed
        for nid, node in curr_nodes.items():
            prev_node = prev_nodes.get(nid)
            if prev_node is None or node.properties != prev_node.properties:
                changed_nodes.append({
                    "id": node.id,
                    "name": node.label,
                    "group": node.group,
                    "type": node.type,
                    "val": _node_value(node),
                    "color": _node_color(node),
                    **node.properties,
                })

        # Nodes removed
        for nid in prev_nodes:
            if nid not in curr_nodes:
                removed_node_ids.append(nid)

        # Edges added or changed
        for key, edge in curr_edges.items():
            prev_edge = prev_edges.get(key)
            if prev_edge is None or edge.properties != prev_edge.properties:
                changed_links.append({
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "color": edge.properties.get("color", COLORS["edge_contains"]),
                    "width": edge.properties.get("width", 1),
                    **edge.properties,
                })

        # Edges removed
        for key in prev_edges:
            if key not in curr_edges:
                removed_link_keys.append(key)

        return {
            "changed_nodes": changed_nodes,
            "removed_node_ids": removed_node_ids,
            "changed_links": changed_links,
            "removed_link_keys": removed_link_keys,
            "metadata": {
                "episode_id": curr.episode_id,
                "step": curr.step,
                "mission_phase": curr.mission_phase,
                **curr.metadata,
            },
        }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _node_color(node: GraphNode) -> str:
    """Derive the display color for a node based on its type and state."""
    if node.type == "agent":
        return node.properties.get("color", COLORS["agent_blue"])

    if node.type == "subnet":
        if node.properties.get("is_isolated"):
            return COLORS["traffic_blocked"]
        return COLORS["subnet_default"]

    # Host nodes -- priority: restoring > root > user > decoy > clean
    props = node.properties
    if props.get("is_restoring"):
        return COLORS["host_restoring"]
    compromise = props.get("compromise_level") or props.get("compromised_level") or "none"
    if compromise == "root":
        return COLORS["host_root_compromised"]
    if compromise == "user":
        return COLORS["host_user_compromised"]
    if props.get("decoy_count", 0) > 0:
        return COLORS["host_decoy_active"]
    return COLORS["host_clean"]


def _node_value(node: GraphNode) -> int:
    """Derive the visual size value for a node."""
    if node.type == "subnet":
        return 8
    if node.type == "agent":
        return 4
    # Host -- larger if compromised
    compromise = node.properties.get("compromise_level") or node.properties.get("compromised_level") or "none"
    if compromise == "root":
        return 3
    if compromise == "user":
        return 2
    return 1


def _edge_key(edge: GraphEdge) -> str:
    """Create a unique string key for an edge."""
    return f"{edge.source}|{edge.target}|{edge.type}"
