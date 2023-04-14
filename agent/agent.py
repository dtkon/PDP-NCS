from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod
import torch
from tensorboard_logger import Logger as TbLogger

from nets.actor_network import Actor_N2S, Actor_Construct
from nets.critic_network import Critic_N2S, Critic_Construct
from options import Option
from problems.problem_pdp import PDP


class Agent(ABC):
    opts: Option
    actor: Actor_N2S
    critic: Critic_N2S
    actor_construct: Actor_Construct
    critic_construct: Critic_Construct
    optimizer: torch.optim.Optimizer
    optimizer_sc: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR

    @abstractmethod
    def __init__(self, problem_name: str, size: int, opts: Option) -> None:
        pass

    @abstractmethod
    def load(self, load_path: str) -> None:
        pass

    @abstractmethod
    def save(self, epoch: int) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def rollout(
        self,
        problem: PDP,
        val_m: int,
        batch: Dict[str, torch.Tensor],
        show_bar: bool,
        zoom: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def start_inference(
        self,
        problem: PDP,
        val_dataset: Optional[str],
        tb_logger: Optional[TbLogger],
        load_path: Optional[str],
        zoom: bool,
    ) -> None:
        pass

    @abstractmethod
    def start_training(
        self,
        problem: PDP,
        val_dataset: Optional[str],
        tb_logger: Optional[TbLogger],
        load_path: Optional[str],
    ) -> None:
        pass
