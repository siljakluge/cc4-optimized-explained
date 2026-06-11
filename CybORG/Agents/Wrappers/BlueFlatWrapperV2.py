"""BlueFlatWrapperV2 — extends BlueFlatWrapper with a malicious-file flag.

Background
----------
ExploitRemoteService ALWAYS drops a file (cmd.exe / cmd.sh) with density=0.9
and signed=False on the target host — including the 5% of exploits that fire
no process_creation event (the "silent 5%").

PrivilegeEscalate ALWAYS drops escalate.exe / escalate.sh with the same
attributes and fires NO events.

Neither green agent actions nor PhishingEmail drop high-density files, so
the malfile flag has zero false positives.

Remove NOW clears malware files (density>=0.9 unsigned) as well as killing
suspicious processes. Root sessions survive Remove. A malfile flag with no
proc/conn events is the PrivilegeEscalate signature (root session) and should
trigger Restore rather than Remove.
Only Restore resets host.files to the scenario-start originals (full reimage).

Observation format
------------------
This wrapper appends a malfile section AFTER the standard BlueFlatWrapper obs:

  obs[:base_len]   identical to BlueFlatWrapper output
                     [phase(1)] [subnet_blocks...] [messages(32)]
  obs[base_len:]   malfile flags — one bit per actual host, in the same
                     subnet/host order as the proc/conn flags in the base obs.

The agent can find base_len as:
    len(obs) - sum(n_hosts_i for each controlled subnet)
or equivalently:
    1 + sum(27 + 2*n_hosts_i) + 32
"""
from __future__ import annotations

import numpy as np

from CybORG.Agents.Wrappers.BlueFlatWrapper import BlueFlatWrapper
from CybORG.Simulator.State import State


class BlueFlatWrapperV2(BlueFlatWrapper):
    """BlueFlatWrapper extended with per-host malicious-file detection.

    Drop-in replacement for BlueFlatWrapper.  All existing behaviour is
    preserved; the only difference is that each observation vector has
    ``sum(n_hosts_i)`` extra float32 bits appended at the end.

    A malfile bit is 1.0 if the host currently holds any file with
    ``density >= 0.9 and not signed`` — the fingerprint left by
    ExploitRemoteService and PrivilegeEscalate.
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_malfile(self, state: State, hostname: str) -> bool:
        """Return True if the host holds a high-density unsigned file."""
        return any(
            f.density >= 0.9 and not f.signed
            for f in state.hosts[hostname].files
        )

    # ------------------------------------------------------------------
    # Observation override
    # ------------------------------------------------------------------

    def observation_change(self, agent_name: str, observation: dict) -> np.ndarray:
        """Build the standard obs then append one malfile bit per host."""
        base_obs = super().observation_change(agent_name, observation)
        state = self.env.environment_controller.state

        malfile_flags: list[float] = []
        for sn in self.subnets(agent_name):
            subnet_hosts = self._cached_subnet_hosts.get(sn, [])
            for h in subnet_hosts:
                malfile_flags.append(
                    float(self._get_malfile(state, h)) if h in state.hosts else 0.0
                )

        if not malfile_flags:
            return base_obs

        return np.concatenate(
            [base_obs, np.array(malfile_flags, dtype=np.float32)]
        )
