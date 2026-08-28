"""PPO 的框架无关教学骨架。

这不是可直接训练大模型的完整实现，而是用于定位 TRL/自研系统中各组件的职责。
真正训练还需要分布式策略、稳定的奖励模型、KL 控制、checkpoint 和离线评测。
"""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class PPOBackend(Protocol):
    """将具体框架隐藏在最小接口之后。"""

    def generate(self, prompts: Sequence[str]) -> Sequence[Any]: ...

    def score(self, prompts: Sequence[str], responses: Sequence[Any]) -> Sequence[float]: ...

    def ppo_step(
        self, prompts: Sequence[str], responses: Sequence[Any], rewards: Sequence[float]
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class PPOBatchResult:
    rewards: Sequence[float]
    metrics: dict[str, float]


def train_batch(backend: PPOBackend, prompts: Sequence[str]) -> PPOBatchResult:
    """执行一次 on-policy PPO batch：采样、打分、更新。"""
    if not prompts:
        raise ValueError("prompts 不能为空")
    responses = backend.generate(prompts)
    rewards = backend.score(prompts, responses)
    if len(responses) != len(prompts) or len(rewards) != len(prompts):
        raise ValueError("prompt、response 和 reward 数量必须一致")
    metrics = backend.ppo_step(prompts, responses, rewards)
    return PPOBatchResult(rewards=rewards, metrics=metrics)
