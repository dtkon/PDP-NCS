from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import math
from torch import nn
import torch

from problems.problem_pdp import PDP

from .graph_layers import (
    N2SEncoder,
    N2SDecoder,
    EmbeddingNet,
    MHA_Self_Score_WithoutNorm,
    ConstructEncoder,
    ConstructDecoder,
    HeterEmbedding,
)

if TYPE_CHECKING:
    from agent.agent import Agent


class mySequential(nn.Sequential):

    __call__: Callable[..., Union[Tuple[torch.Tensor], torch.Tensor]]

    def forward(
        self, *inputs: Union[Tuple[torch.Tensor], torch.Tensor]
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs  # type: ignore


class Actor_N2S(nn.Module):
    def __init__(
        self,
        problem_name: str,
        embedding_dim: int,
        ff_hidden_dim: int,
        n_heads_actor: int,
        n_layers: int,
        normalization: str,
        v_range: float,
        seq_length: int,
        embedding_type: str,
        removal_type: str,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.v_range = v_range
        self.seq_length = seq_length
        self.calc_stacks = bool(problem_name == 'pdtspl')
        self.node_dim = 2

        # networks
        self.embedder = EmbeddingNet(
            self.node_dim, self.embedding_dim, self.seq_length, embedding_type
        )

        self.pos_emb_encoder = MHA_Self_Score_WithoutNorm(
            self.n_heads_actor, self.embedding_dim
        )  # for PFEs

        self.encoder = mySequential(
            *(
                N2SEncoder(
                    self.n_heads_actor,
                    self.embedding_dim,
                    self.ff_hidden_dim,
                    self.normalization,
                )
                for _ in range(self.n_layers)
            )
        )  # for NFEs

        self.decoder = N2SDecoder(
            self.n_heads_actor, self.embedding_dim, self.v_range, removal_type
        )  # the two propsoed decoders

        print('Actor_N2S:', self.get_parameter_number())

    def get_parameter_number(self) -> Dict[str, int]:
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    @staticmethod
    def _get_action_removal_recent(
        action_removal_record: List[torch.Tensor],
    ) -> torch.Tensor:
        action_removal_record_tensor = torch.stack(
            action_removal_record
        )  # (len_action_record, batch_size, graph_size/2)
        return torch.cat(
            (
                action_removal_record_tensor[-3:].transpose(0, 1),
                action_removal_record_tensor.mean(0).unsqueeze(1),
            ),
            1,
        )  # (batch_size, 4, graph_size/2)

    __call__: Callable[
        ..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]

    def forward(
        self,
        problem: PDP,
        x_in: torch.Tensor,
        solution: torch.Tensor,
        pre_action: Optional[torch.Tensor],
        action_removal_record: List[torch.Tensor],
        fixed_action: Optional[torch.Tensor] = None,
        require_entropy: bool = False,
        to_critic: bool = False,
        only_critic: bool = False,
        only_fea: bool = False,
    ):
        # the embedded input x
        # batch_size, graph_size+1, node_dim = x_in.size()

        if only_fea:
            h_fea = self.embedder(x_in, None, False)[0]
            return h_fea.detach(), None, None, None

        h_fea, g_pos, visit_index, top2 = self.embedder(
            x_in, solution, self.calc_stacks
        )

        if h_fea is None:  # share or together
            h_fea = self.agent.actor_construct(x_in, only_fea=True)[0]

        # pass through encoder
        aux_att = self.pos_emb_encoder(g_pos)
        h_wave = self.encoder(h_fea, aux_att)[0]

        if only_critic:
            return h_wave, None, None, None

        # pass through decoder
        action, log_ll, entropy = self.decoder(
            problem=problem,
            h_wave=h_wave,
            solution=solution,
            x_in=x_in,
            top2=top2,
            visit_index=visit_index,
            pre_action=pre_action,
            selection_recent=Actor_N2S._get_action_removal_recent(
                action_removal_record
            ).to(x_in.device),
            fixed_action=fixed_action,
            require_entropy=require_entropy,
        )

        return (
            action,
            log_ll.squeeze(),
            h_wave if to_critic else None,
            entropy if require_entropy else None,
        )

    def hook_agent(self, agent: 'Agent'):
        self.agent = agent


class Actor_Construct(nn.Module):  # kool suggest 8 heads and 128 dim
    def __init__(
        self,
        problem_name: str,
        embedding_dim: int,
        n_heads: int,
        n_layers: int,
        normalization: str,
        type_select: str,
        embedding_type: str,
        attn_type: str,
    ) -> None:
        super().__init__()

        self.stack_is_lifo = bool(problem_name == 'pdtspl')

        self.together = embedding_type == 'together'

        # self.embedder = nn.Linear(2, embedding_dim, bias=False)
        if embedding_type == 'pair' or embedding_type == 'together':
            self.embedder = HeterEmbedding(2, embedding_dim)
        elif embedding_type == 'share':
            self.embedder = None  # type: ignore
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(
            *(
                ConstructEncoder(n_heads, embedding_dim, normalization, attn_type)
                for _ in range(n_layers)
            )
        )
        self.decoder = ConstructDecoder(
            n_heads, embedding_dim, self.stack_is_lifo, type_select
        )

        self.init_parameters()

        print('Actor_SC:', self.get_parameter_number())

    def init_parameters(self) -> None:
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    __call__: Callable[..., Tuple[torch.Tensor, torch.Tensor]]

    def forward(
        self,
        x_in: torch.Tensor,
        fixed_sol: Optional[torch.Tensor] = None,
        temperature: float = 1,
        only_fea: bool = False,
    ):

        if self.embedder is None:  # share
            h_fea = self.agent.actor(None, x_in, None, None, None, only_fea=True)[0]
        else:
            h_fea = self.embedder(x_in)  # (batch_size, graph_size+1, embedding_dim)

        if only_fea:
            return h_fea if self.together else h_fea.detach(), None

        hN: torch.Tensor = self.encoder(h_fea)
        hN_mean = hN.mean(1)

        batch_size, graph_size_plus1, _ = h_fea.size()

        init_sol = (
            torch.arange(graph_size_plus1).repeat((batch_size, 1)).to(h_fea.device)
        )
        cur_sol = init_sol.clone()

        stack = (
            torch.zeros((batch_size, graph_size_plus1 // 2 + 1)).to(h_fea.device) - 1
        )
        stack[:, 0] = 0

        direct_fixed_sol = (
            PDP.direct_solution(fixed_sol) if fixed_sol is not None else None
        )

        log_ll_list = []
        for step in range(graph_size_plus1 - 1):
            cur_sol, log_p = self.decoder(
                hN,
                hN_mean,
                cur_sol,
                init_sol,
                step,
                stack,
                direct_fixed_sol,
                temperature,
            )
            log_ll_list.append(log_p.view(-1))

        log_ll = torch.stack(log_ll_list, 1).sum(1)  # (batch_size,)

        return cur_sol, log_ll

    def get_parameter_number(self) -> Dict[str, int]:
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def hook_agent(self, agent: 'Agent'):
        self.agent = agent
