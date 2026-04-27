"""Active-investigation hook (Section 4.3, citing Anaedevha & Trofimov, ElCon
2025 -- the DQN+actor-critic poisoning detector).

When a client's FLTrust score falls below the configured threshold, the
suspicious updates are routed here for adapter-level fuzzing in a sandboxed
environment.  This file ships a lightweight DQN agent + a fuzzing-action
space that the RobustIDPS.ai platform's active-defence module imports
directly via ``from fedloraguard.investigation import DQNInvestigator``.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import torch
from torch import nn


@dataclass
class FuzzAction:
    """Symbolic fuzzing actions on a suspicious adapter."""
    name: str

ACTIONS: Tuple[FuzzAction, ...] = (
    FuzzAction("scan_spectrum"),
    FuzzAction("inject_random_trigger"),
    FuzzAction("merge_with_benign"),
    FuzzAction("differential_test"),
    FuzzAction("flag_malicious"),
    FuzzAction("clear_suspicion"),
)


class _QNet(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class DQNInvestigator:
    """Tabular-style DQN with experience replay (cf. our prior ElCon paper).

    The agent exposes a ``decide(state)`` method that returns the chosen
    action, and an ``update(transition)`` method to absorb feedback from
    the sandbox fuzzing pipeline.  The mapping from `state` (the verifier's
    feature vector + the FLTrust margin) to action is the production
    investigation policy.
    """

    def __init__(self, state_dim: int, num_actions: int = len(ACTIONS),
                 buffer: int = 5000, gamma: float = 0.95, lr: float = 1e-3,
                 epsilon_start: float = 0.5, epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000) -> None:
        self.q = _QNet(state_dim, num_actions)
        self.target = _QNet(state_dim, num_actions)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.replay: Deque = deque(maxlen=buffer)
        self.gamma = gamma
        self.num_actions = num_actions
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self._steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self._steps / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def decide(self, state: torch.Tensor) -> int:
        self._steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q = self.q(state.unsqueeze(0)).squeeze(0)
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay.append((state, action, reward, next_state, done))

    def train_step(self, batch_size: int = 32) -> float:
        if len(self.replay) < batch_size:
            return float("nan")
        batch = random.sample(self.replay, batch_size)
        states = torch.stack([b[0] for b in batch])
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * self.target(next_states).max(dim=-1).values
        q = self.q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        loss = nn.functional.smooth_l1_loss(q, target_q)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.q.state_dict())
