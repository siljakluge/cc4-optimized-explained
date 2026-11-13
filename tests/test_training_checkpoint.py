"""Run with: python -m unittest discover -s tests -p 'test_training_checkpoint.py'"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from marl.training_checkpoint import save_training_checkpoint, load_training_checkpoint, atomic_torch_save


def fake_agent():
    actor = torch.nn.Linear(3, 2)
    critic = torch.nn.Linear(3, 1)
    actor.opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic.opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for net in (actor, critic):
        net.opt.zero_grad()
        net(torch.ones(1, 3)).sum().backward()
        net.opt.step()
    return SimpleNamespace(actor=actor, critic=critic, device=torch.device('cpu'))


class CheckpointTests(unittest.TestCase):
    def test_roundtrip(self):
        agents = [fake_agent() for _ in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'joint.pt'
            save_training_checkpoint(path, agents, 10, {'N': 10}, [('reward',)], min_free_gb=0)
            expected = [[p.detach().clone() for p in a.actor.parameters()] for a in agents]
            for agent in agents:
                for p in agent.actor.parameters():
                    p.data.zero_()
            recovered = load_training_checkpoint(path, agents, {'N': 10})
            self.assertEqual(recovered['next_update'], 10)
            for agent, params in zip(agents, expected):
                for p, target in zip(agent.actor.parameters(), params):
                    torch.testing.assert_close(p, target)
            self.assertTrue(agents[0].actor.opt.state)

    def test_atomic_failed_save_keeps_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'joint.pt'
            atomic_torch_save({'number': 42}, path, min_free_gb=0)
            with self.assertRaises(OSError):
                atomic_torch_save({'number': 100}, path, min_free_gb=10**9)
            self.assertEqual(torch.load(path, weights_only=False)['number'], 42)

    def test_mismatched_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            agents = [fake_agent() for _ in range(5)]
            path = Path(tmp) / 'joint.pt'
            save_training_checkpoint(path, agents, 2, {'N': 10}, [], min_free_gb=0)
            with self.assertRaises(ValueError):
                load_training_checkpoint(path, agents, {'N': 2})


if __name__ == '__main__':
    unittest.main()
