from typing import Any, Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboard_logger import Logger as TbLogger
import math
import random

from utils import clip_grad_norms
from nets.actor_network import Actor_N2S, Actor_Construct
from nets.critic_network import Critic_N2S, Critic_Construct
from utils import torch_load_cpu, get_inner_model, move_to, batch_picker
from utils.logger import log_to_tb_train
from problems.problem_pdp import PDP
from options import Option

from .agent import Agent
from .utils import validate, batch_augments, mem_test, zoom_feature


class Memory:
    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.best_obj: List[torch.Tensor] = []
        self.action_removal_record: List[List[torch.Tensor]] = []

    def clear_memory(self) -> None:
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.best_obj[:]
        del self.action_removal_record[:]


class PPO(Agent):
    def __init__(self, problem_name: str, size: int, opts: Option) -> None:
        # figure out the options
        self.opts = opts

        # figure out the actor
        self.actor = Actor_N2S(
            problem_name=problem_name,
            embedding_dim=opts.embedding_dim,
            ff_hidden_dim=opts.ff_hidden_dim,
            n_heads_actor=opts.actor_head_num,
            n_layers=opts.n_encode_layers,
            normalization=opts.normalization,
            v_range=opts.v_range,
            seq_length=size + 1,
            embedding_type=opts.embed_type_n2s,
            removal_type=opts.removal_type,
        )

        if opts.shared_critic:
            self.actor_construct = Actor_Construct(
                problem_name,
                opts.embedding_dim,
                8,
                3,
                opts.sc_normalization,
                opts.sc_decoder_select_type,
                opts.embed_type_sc,
                opts.sc_attn_type,
            )
            self.critic_construct = Critic_Construct()

            self.actor.hook_agent(self)
            self.actor_construct.hook_agent(self)

        if not opts.eval_only:
            # figure out the critic
            self.critic = Critic_N2S(
                embedding_dim=opts.embedding_dim,
                ff_hidden_dim=opts.ff_hidden_dim,
                n_heads=opts.critic_head_num,
                n_layers=opts.n_encode_layers,
                normalization=opts.normalization,
            )

            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
                [{'params': self.actor.parameters(), 'lr': opts.lr_model}]
                + [{'params': self.critic.parameters(), 'lr': opts.lr_critic}]
            )
            if opts.shared_critic:
                self.optimizer_sc = torch.optim.Adam(
                    [
                        {
                            'params': self.actor_construct.parameters(),
                            'lr': opts.lr_construct,
                        }
                    ]
                    + [
                        {
                            'params': self.critic_construct.parameters(),
                            'lr': opts.lr_trust_degree,
                        }
                    ]
                )

            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                opts.lr_decay,
                last_epoch=-1,
            )

        print(f'Distributed: {opts.distributed}')
        if opts.use_cuda and not opts.distributed:
            self.actor.to(opts.device)
            if opts.shared_critic:
                self.actor_construct.to(opts.device)
            if not opts.eval_only:
                self.critic.to(opts.device)
                if opts.shared_critic:
                    self.critic_construct.to(opts.device)

    def load(self, load_path: str) -> None:
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.actor)
        if self.opts.load_original_n2s:
            print(
                ' [*] Loading original N2S data from {}'.format(
                    self.opts.load_original_n2s
                )
            )
            n2s_load_data = torch_load_cpu(self.opts.load_original_n2s)
            model_actor.load_state_dict(n2s_load_data['actor'])
        else:
            model_actor.load_state_dict(load_data['actor'])
        if self.opts.shared_critic:
            model_actor_cons = get_inner_model(self.actor_construct)
            model_actor_cons.load_state_dict(load_data['actor_construct'])
        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict(load_data['critic'])
            if self.opts.shared_critic:
                model_critic_cons = get_inner_model(self.critic_construct)
                model_critic_cons.load_state_dict(load_data['critic_construct'])
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            if self.opts.shared_critic:
                self.optimizer_sc.load_state_dict(load_data['optimizer_sc'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                if isinstance(load_data['cuda_rng_state'], torch.Tensor):
                    torch.cuda.set_rng_state(load_data['cuda_rng_state'])
                else:
                    if len(load_data['cuda_rng_state']) > 1:
                        torch.cuda.set_rng_state(load_data['cuda_rng_state'][1])
                    else:
                        torch.cuda.set_rng_state(load_data['cuda_rng_state'][0])
            try:
                random.setstate(load_data['random_state'])
            except KeyError:
                print('Error type: random_state')
        # done
        print(' [*] Loading data from {}'.format(load_path))

    def save(self, epoch: int) -> None:
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_sc': self.optimizer_sc.state_dict()
                if self.opts.shared_critic
                else {},
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'actor_construct': get_inner_model(self.actor_construct).state_dict()
                if self.opts.shared_critic
                else {},
                'critic_construct': get_inner_model(self.critic_construct).state_dict()
                if self.opts.shared_critic
                else {},
                'random_state': random.getstate(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch)),
        )

    def eval(self) -> None:
        torch.set_grad_enabled(False)
        self.actor.eval()
        if self.opts.shared_critic:
            self.actor_construct.eval()
        if not self.opts.eval_only:
            self.critic.eval()
            if self.opts.shared_critic:
                self.critic_construct.eval()

    def train(self) -> None:
        torch.set_grad_enabled(True)
        self.actor.train()
        if self.opts.shared_critic:
            self.actor_construct.train()
        if not self.opts.eval_only:
            self.critic.train()
            if self.opts.shared_critic:
                self.critic_construct.train()

    def rollout(
        self,
        problem: PDP,
        val_m: int,
        batch: Dict[str, torch.Tensor],
        show_bar: bool,
        zoom: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = move_to(batch, self.opts.device)
        batch_size, graph_size_plus1, node_dim = batch['coordinates'].size()

        batch_augments(val_m, batch, graph_size_plus1, node_dim)

        batch_feature = PDP.input_coordinates(
            batch
        )  # (new_batch_size, graph_size+1, node_dim)
        new_batch_size = batch_feature.size(0)

        if not self.opts.shared_critic:
            solution = move_to(
                problem.get_initial_solutions(batch), self.opts.device
            ).long()

            obj = problem.get_costs(batch_feature, solution, zoom)  # (new_batch_size,)
        else:
            solution_list = []
            obj_list = []

            if self.opts.inference_sample_size >= self.opts.inference_sample_batch:
                ms_batch_feature = batch_feature.unsqueeze(1).repeat(
                    1, self.opts.inference_sample_batch, 1, 1
                )
                ms_batch_feature = ms_batch_feature.view(-1, graph_size_plus1, node_dim)

            pbar = tqdm(
                total=math.ceil(
                    self.opts.inference_sample_size / self.opts.inference_sample_batch
                )
                * len(self.opts.inference_temperature),
                disable=self.opts.no_progress_bar or not show_bar,
                desc='constructing',
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            )

            for sample_batch in batch_picker(
                self.opts.inference_sample_size, self.opts.inference_sample_batch
            ):
                if sample_batch < self.opts.inference_sample_batch:
                    ms_batch_feature = batch_feature.unsqueeze(1).repeat(
                        1, sample_batch, 1, 1
                    )
                    ms_batch_feature = ms_batch_feature.view(
                        -1, graph_size_plus1, node_dim
                    )

                for temperature in self.opts.inference_temperature:
                    if zoom:
                        ms_batch_feature_4actor, _ = zoom_feature(ms_batch_feature)
                    else:
                        ms_batch_feature_4actor = ms_batch_feature
                    solution, _ = self.actor_construct(
                        ms_batch_feature_4actor, temperature=temperature
                    )
                    obj = problem.get_costs(ms_batch_feature, solution, zoom)

                    solution = solution.view(new_batch_size, sample_batch, -1)
                    obj = obj.view(new_batch_size, sample_batch)

                    solution_list.append(solution)
                    obj_list.append(obj)

                    pbar.update(1)
            # pbar.close()

            solution = torch.cat(solution_list, 1)
            obj = torch.cat(obj_list, 1)

            min_sol_index = obj.argmin(dim=1)
            obj = obj[torch.arange(new_batch_size), min_sol_index]
            solution = solution[torch.arange(new_batch_size), min_sol_index]

            if val_m > 1 and False:  # shut down
                obj_aug = obj.reshape(batch_size, val_m)
                solution_aug = solution.reshape(batch_size, val_m, -1)

                min_sol_index_among_val_m = obj_aug.argmin(dim=1)
                obj_val_m = obj_aug[torch.arange(batch_size), min_sol_index_among_val_m]
                solution_val_m = solution_aug[
                    torch.arange(batch_size), min_sol_index_among_val_m
                ]

                obj = obj_val_m.unsqueeze(1).repeat(1, val_m).reshape(-1)
                solution = (
                    solution_val_m.unsqueeze(1)
                    .repeat(1, val_m, 1)
                    .reshape(new_batch_size, -1)
                )

        obj_history = [
            torch.cat((obj[:, None], obj[:, None]), -1)
        ]  # [(new_batch_size, 2)]

        rewards: List[torch.Tensor] = []

        action = None
        action_removal_record = [
            torch.zeros((batch_feature.size(0), problem.size // 2))
            for _ in range(problem.size // 2)  # N2S paper section 4.4 last sentence
        ]

        for _ in tqdm(
            range(self.opts.T_max),
            disable=self.opts.no_progress_bar or not show_bar,
            desc='rollout',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        ):
            # pass through model
            if zoom:
                batch_feature_4actor, _ = zoom_feature(batch_feature)
            else:
                batch_feature_4actor = batch_feature
            action = self.actor(
                problem, batch_feature_4actor, solution, action, action_removal_record
            )[0]

            # new solution
            solution, reward, obj, action_removal_record = problem.step(
                batch, solution, action, obj, action_removal_record, zoom=zoom
            )

            # record informations
            rewards.append(reward)  # [(new_batch_size,), ...]
            obj_history.append(obj)  # [(new_batch_size, 2), ...]

        if self.opts.shared_critic:
            pbar.close()

        out = (
            obj[:, -1].reshape(batch_size, val_m).min(1)[0],  # (batch_size, 1)
            torch.stack(obj_history, 1)[:, :, 0]  # current obj history
            .view(batch_size, val_m, -1)
            .min(1)[0],  # (batch_size, T_max)
            torch.stack(obj_history, 1)[:, :, -1]  # current best obj history
            .view(batch_size, val_m, -1)
            .min(1)[0],  # (batch_size, T_max)
            torch.stack(rewards, 1)
            .view(batch_size, val_m, -1)
            .max(1)[0],  # (batch_size, T_max)
        )

        return out

    def start_inference(
        self,
        problem: PDP,
        val_dataset: Optional[str],
        tb_logger: Optional[TbLogger],
        load_path: Optional[str],
        zoom: bool = False,
    ) -> None:
        if load_path is not None:
            self.load(load_path)
        if self.opts.distributed:
            mp.spawn(
                validate,
                nprocs=self.opts.world_size,
                args=(
                    problem,
                    self,
                    val_dataset,
                    tb_logger,
                    True,
                    None,
                    False,
                    zoom,
                ),
            )
        else:
            validate(
                0, problem, self, val_dataset, tb_logger, distributed=False, zoom=zoom
            )

    def start_training(
        self,
        problem: PDP,
        val_dataset: Optional[str],
        tb_logger: Optional[TbLogger],
        load_path: Optional[str],
    ) -> None:
        if self.opts.distributed:
            mp.spawn(
                train,
                nprocs=self.opts.world_size,
                args=(problem, self, val_dataset, tb_logger, load_path),
            )
        else:
            train(0, problem, self, val_dataset, tb_logger, load_path)


def train(
    rank: int,
    problem: PDP,
    agent: Agent,
    val_dataset: Optional[str],
    tb_logger: Optional[TbLogger],
    load_path: Optional[str],
) -> None:
    opts = agent.opts

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            backend='nccl', world_size=opts.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)

        if opts.normalization == 'batch':
            agent.actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent.actor).to(
                device
            )  # type: ignore
            agent.critic = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                agent.critic
            ).to(
                device
            )  # type: ignore

        if opts.shared_critic:
            agent.actor_construct.to(device)
            agent.critic_construct.to(device)

            if opts.sc_normalization == 'batch':
                agent.actor_construct = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    agent.actor_construct
                ).to(
                    device
                )  # type: ignore

            for state in agent.optimizer_sc.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)

        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        agent.actor = torch.nn.parallel.DistributedDataParallel(
            agent.actor, device_ids=[rank]
        )  # type: ignore
        if opts.shared_critic:
            agent.actor_construct = torch.nn.parallel.DistributedDataParallel(
                agent.actor_construct, device_ids=[rank]
            )  # type: ignore
        if not opts.eval_only:
            agent.critic = torch.nn.parallel.DistributedDataParallel(
                agent.critic, device_ids=[rank]
            )  # type: ignore
            if opts.shared_critic:
                agent.critic_construct = torch.nn.parallel.DistributedDataParallel(
                    agent.critic_construct, device_ids=[rank]
                )  # type: ignore

        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(
                os.path.join(
                    opts.log_dir,
                    "{}_{}".format(opts.problem, opts.graph_size),
                    opts.run_name,
                )
            )
    else:
        for state in agent.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)
        if opts.shared_critic:
            for state in agent.optimizer_sc.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)

    # check cuda memory
    if opts.use_cuda:
        if rank == 0:
            training_dataset_test = PDP.make_dataset(
                size=opts.graph_size,
                num_samples=opts.batch_size // opts.world_size,
                silence=True,
            )
            training_dataloader_test = DataLoader(
                training_dataset_test,
                batch_size=opts.batch_size // opts.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            mem_test(agent, problem, next(iter(training_dataloader_test)))
    if opts.distributed:
        dist.barrier()

    # set or restore seed
    if load_path is None:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
    else:
        agent.load(load_path)

    if opts.distributed:
        dist.barrier()

    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        agent.lr_scheduler.step(epoch)

        # Training mode
        if rank == 0:
            print('\n\n')
            print("|", format(f" Training epoch {epoch} ", "*^60"), "|")
            print(
                "Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(
                    agent.optimizer.param_groups[0]['lr'],
                    agent.optimizer.param_groups[1]['lr'],
                    opts.run_name,
                ),
                flush=True,
            )
        # prepare training data
        training_dataset = PDP.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size
        )
        if opts.distributed:
            train_sampler: Any = torch.utils.data.distributed.DistributedSampler(
                training_dataset, shuffle=False
            )
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=opts.batch_size // opts.world_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=train_sampler,
            )
        else:
            training_dataloader = DataLoader(
                training_dataset,
                batch_size=opts.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

        if opts.distributed:
            dist.barrier()

        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)
        pbar = tqdm(
            total=(opts.K_epochs)
            * (opts.epoch_size // opts.batch_size)
            * (opts.T_train // opts.n_step),
            disable=opts.no_progress_bar or rank != 0,
            desc='training',
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        )
        if opts.shared_critic:
            opts.cur_temperature = max(
                1, opts.temperature_init * (opts.temperature_decay**epoch)
            )
            if opts.cur_temperature == 1:
                opts.cur_init_sample_size = min(
                    opts.max_init_sample_size,
                    int(
                        opts.init_sample_increase
                        ** (epoch - opts.start_init_sample_epoch)
                    ),
                )
            opts.cur_imitation_augment = max(
                0,
                min(
                    int(epoch * opts.imitation_increase_w + opts.imitation_increase_b),
                    opts.imitation_max_augment,
                ),
            )
        for batch in training_dataloader:
            train_batch(
                rank,
                problem,
                agent,
                epoch,
                step,
                batch,
                tb_logger,
                opts,
                pbar,
                epoch < opts.sc_start_train_epoch,
            )
            step += 1
        pbar.close()

        # save new model after one epoch
        if rank == 0:
            if not opts.no_saving and (
                (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0)
                or epoch == opts.epoch_end - 1
            ):
                agent.save(epoch)

        # validate the new model
        validate(rank, problem, agent, val_dataset, tb_logger, id_=epoch)

        # syn
        if opts.distributed:
            dist.barrier()


def train_batch(
    rank: int,
    problem: PDP,
    agent: Agent,
    epoch: int,
    step: int,
    batch: Dict[str, torch.Tensor],
    tb_logger: TbLogger,
    opts: Option,
    pbar: tqdm,
    no_sc_train: bool,
) -> None:
    # setup
    agent.train()
    memory = Memory()

    # prepare the input
    batch = (
        move_to(batch, rank) if opts.distributed else move_to(batch, opts.device)
    )  # batch_size, graph_size+1, 2
    batch_feature: torch.Tensor = (
        move_to(PDP.input_coordinates(batch), rank)
        if opts.distributed
        else move_to(PDP.input_coordinates(batch), opts.device)
    )
    batch_size, graph_size_plus1, node_dim = batch_feature.size()
    action = (
        move_to(torch.tensor([-1, -1, -1]).repeat(batch_size, 1), rank)
        if opts.distributed
        else move_to(torch.tensor([-1, -1, -1]).repeat(batch_size, 1), opts.device)
    )

    action_removal_record = [
        torch.zeros((batch_feature.size(0), problem.size // 2))
        for _ in range(problem.size)  # N2S paper section 4.4 last sentence
    ]

    # initial solution
    if not opts.shared_critic or opts.no_sample_init:
        solution: torch.Tensor = (
            move_to(problem.get_initial_solutions(batch), rank)
            if opts.distributed
            else move_to(problem.get_initial_solutions(batch), opts.device)
        )
        obj = problem.get_costs(batch_feature, solution)
        best_sol = solution.clone()
    else:
        agent.eval()

        if opts.cur_temperature > 1:
            solution, _ = agent.actor_construct(
                batch_feature, temperature=opts.cur_temperature
            )
            obj = problem.get_costs(batch_feature, solution)
        else:
            solution_list = []
            obj_list = []

            if opts.cur_init_sample_size >= opts.max_init_sample_batch:
                ms_batch_feature = batch_feature.unsqueeze(1).repeat(
                    1, opts.max_init_sample_batch, 1, 1
                )
                ms_batch_feature = ms_batch_feature.view(-1, graph_size_plus1, node_dim)

            for sample_batch in batch_picker(
                opts.cur_init_sample_size, opts.max_init_sample_batch
            ):
                if sample_batch < opts.max_init_sample_batch:
                    ms_batch_feature = batch_feature.unsqueeze(1).repeat(
                        1, sample_batch, 1, 1
                    )
                    ms_batch_feature = ms_batch_feature.view(
                        -1, graph_size_plus1, node_dim
                    )

                solution, _ = agent.actor_construct(ms_batch_feature)
                obj = problem.get_costs(ms_batch_feature, solution)

                solution = solution.view(batch_size, sample_batch, -1)
                obj = obj.view(batch_size, sample_batch)

                solution_list.append(solution)
                obj_list.append(obj)

            solution = torch.cat(solution_list, 1)
            obj = torch.cat(obj_list, 1)

            min_sol_index = obj.argmin(dim=1)
            obj = obj[torch.arange(batch_size), min_sol_index]
            solution = solution[torch.arange(batch_size), min_sol_index]

        best_sol = solution.clone()

        agent.train()

    # warm_up
    if opts.warm_up > 0:
        agent.eval()

        for _ in range(
            min(
                opts.max_warm_up,
                int(max(0, (epoch - opts.start_warm_up_epoch) // opts.warm_up)),
            )
        ):
            # get model output
            action = agent.actor(
                problem, batch_feature, solution, action, action_removal_record
            )[0]

            # state transient
            solution, rewards, obj, action_removal_record = problem.step(
                batch, solution, action, obj, action_removal_record, best_sol
            )

        if opts.warm_up_type == 'update':
            obj = obj.view(batch_size, -1)[:, -1]
            solution = best_sol
        else:
            obj = problem.get_costs(batch_feature, solution)

        agent.train()

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    t = 0
    initial_cost = obj
    best_sol = solution.clone()
    imitation_loss = None
    grad_norms_imi = None

    if opts.sc_map_sample_type == 'augment':
        batch_for_sample = {'coordinates': batch_feature.clone()}
        batch_augments(opts.sc_map_sample_times, batch_for_sample)
        batch_feature_for_sample = batch_for_sample[
            'coordinates'
        ]  # (batch_size, augment, graph_size_plus1, node_dim)
        sample_index = 0

    # sample trajectory
    while t < T:  # t will add n_step next time
        t_s = t
        memory.actions.append(action)

        # data array
        total_cost = torch.tensor(0)

        # for first step
        entropy_list = []
        bl_val_detached_list = []
        bl_val_list = []

        if opts.shared_critic:
            if opts.sc_map_sample_type == 'augment':
                construct_solution, construct_logprobs = agent.actor_construct(
                    batch_feature_for_sample[:, sample_index, :, :]
                )
                sample_index += 1
            else:
                construct_solution, construct_logprobs = agent.actor_construct(
                    batch_feature
                )
            construct_obj = problem.get_costs(batch_feature, construct_solution)
            old_construct_logprobs = construct_logprobs

            obj_of_n2s = []

        while t - t_s < n_step and not (t == T):
            memory.states.append(solution)
            memory.action_removal_record.append(action_removal_record)

            # get model output

            action, log_lh, to_critic_, entro_p = agent.actor(
                problem,
                batch_feature,
                solution,
                action,
                action_removal_record,
                require_entropy=True,
                to_critic=True,
            )

            memory.actions.append(action)
            memory.logprobs.append(log_lh)
            memory.best_obj.append(obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1))

            if opts.shared_critic:
                obj_of_n2s.append(obj.view(obj.size(0), -1)[:, 0])

            entropy_list.append(entro_p.detach().cpu())

            baseline_val_detached, baseline_val = agent.critic(
                to_critic_, obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1)
            )

            bl_val_detached_list.append(baseline_val_detached)
            bl_val_list.append(baseline_val)

            # state transient
            solution, rewards, obj, action_removal_record = problem.step(
                batch, solution, action, obj, action_removal_record, best_sol
            )
            memory.rewards.append(rewards)
            # memory.mask_true = memory.mask_true + info['swaped']

            # store info
            total_cost = total_cost + obj[:, -1]

            # next
            t = t + 1

        # store info
        t_time = t - t_s
        total_cost = total_cost / t_time

        # begin update        =======================

        # convert list to tensor
        all_actions = torch.stack(memory.actions)
        old_states = torch.stack(memory.states).detach().view(t_time, batch_size, -1)
        old_actions = all_actions[1:].view(t_time, -1, 3)
        old_logprobs = torch.stack(memory.logprobs).detach().view(-1)
        old_pre_actions = all_actions[:-1].view(t_time, -1, 3)
        old_action_removal_record = memory.action_removal_record

        old_best_obj = torch.stack(memory.best_obj)

        # Optimize ppo policy for K mini-epochs:
        old_value = None
        if opts.shared_critic:
            old_value_construct = None

        for k_ in range(K_epochs):
            if k_ == 0:
                logprobs_list = memory.logprobs

            else:
                # Evaluating old actions and values :
                logprobs_list = []
                entropy_list = []
                bl_val_detached_list = []
                bl_val_list = []

                if opts.shared_critic and opts.sc_rl_train_type == 'ppo':
                    if opts.sc_map_sample_type == 'augment':
                        _, construct_logprobs = agent.actor_construct(
                            batch_feature_for_sample[:, sample_index - 1, :, :],
                            fixed_sol=construct_solution,
                        )
                    else:
                        _, construct_logprobs = agent.actor_construct(
                            batch_feature, fixed_sol=construct_solution
                        )

                for tt in range(t_time):
                    # get new action_prob
                    _, log_p, to_critic_, entro_p = agent.actor(
                        problem,
                        batch_feature,
                        old_states[tt],
                        old_pre_actions[tt],
                        old_action_removal_record[tt],
                        fixed_action=old_actions[tt],  # take same action
                        require_entropy=True,
                        to_critic=True,
                    )

                    logprobs_list.append(log_p)
                    entropy_list.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = agent.critic(
                        to_critic_, old_best_obj[tt]
                    )

                    bl_val_detached_list.append(baseline_val_detached)
                    bl_val_list.append(baseline_val)

            logprobs = torch.stack(logprobs_list).view(-1)
            entropy = torch.stack(entropy_list).view(-1)
            bl_val_detached = torch.stack(bl_val_detached_list).view(-1)
            bl_val = torch.stack(bl_val_list).view(-1)

            if opts.shared_critic:
                # bl_construct = torch.stack(
                #    obj_of_n2s
                # ) - _baseline_trust_degree * torch.stack(bl_val_detached_list)
                # bl_construct = bl_construct.mean(0)
                (
                    bl_construct_detach,
                    bl_construct,
                    trust_degree,
                ) = agent.critic_construct(obj_of_n2s, bl_val_detached_list)

                # obj_of_n2s = []

            # get traget value for critic
            Reward_list = []
            reward_reversed = memory.rewards[::-1]

            # estimate return
            R = agent.critic(
                agent.actor(
                    problem,
                    batch_feature,
                    solution,
                    action,
                    action_removal_record,
                    only_critic=True,
                )[0],
                obj.view(obj.size(0), -1)[:, -1].unsqueeze(-1),
            )[0]
            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward_list.append(R)

            # clip the target:
            Reward = torch.stack(Reward_list[::-1], 0)  # (n_step, batch_size)
            Reward = Reward.view(-1)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            reinforce_loss = -torch.min(surr1, surr2).mean()

            # define baseline loss
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(
                    bl_val - old_value, -eps_clip, eps_clip
                )
                v_max = torch.max(
                    ((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2)
                )
                baseline_loss = v_max.mean()

            if opts.shared_critic:
                if opts.sc_rl_train_type == 'ppo':
                    ratios_construct = torch.exp(
                        construct_logprobs - old_construct_logprobs.detach()
                    ).view(-1)
                    advantages_construct = (
                        (construct_obj - bl_construct_detach).view(-1).detach()
                    )
                    surr1_construct = ratios_construct * advantages_construct
                    surr2_construct = (
                        torch.clamp(ratios_construct, 1 - eps_clip, 1 + eps_clip)
                        * advantages_construct
                    )
                    reinforce_loss_construct = torch.max(
                        surr1_construct, surr2_construct
                    ).mean()

                    if old_value_construct is None:
                        baseline_loss_construct = (
                            (bl_construct - construct_obj.detach()) ** 2
                        ).mean()
                        old_value_construct = bl_construct.detach()
                    else:
                        vpredclipped_construct = old_value_construct + torch.clamp(
                            bl_construct - old_value_construct, -eps_clip, eps_clip
                        )
                        v_max_construct = torch.max(
                            ((bl_construct - construct_obj.detach()) ** 2),
                            ((vpredclipped_construct - construct_obj.detach()) ** 2),
                        )
                        baseline_loss_construct = v_max_construct.mean()

                elif opts.sc_rl_train_type == 'pg':
                    if k_ == 0:
                        reinforce_loss_construct = (
                            construct_logprobs.view(-1)
                            * (construct_obj - bl_construct_detach).view(-1).detach()
                        ).mean()
                        baseline_loss_construct = (
                            (bl_construct - construct_obj.detach()) ** 2
                        ).mean()
                    else:
                        reinforce_loss_construct = reinforce_loss_construct.detach()
                        baseline_loss_construct = baseline_loss_construct.detach()

            # check K-L divergence
            approx_kl_divergence = (
                (0.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            )
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            # calculate loss
            if opts.shared_critic:
                loss = (
                    baseline_loss
                    + reinforce_loss
                    + reinforce_loss_construct
                    + baseline_loss_construct
                )
            else:
                loss = baseline_loss + reinforce_loss  # - 1e-5 * entropy.mean()

            # update gradient step
            agent.optimizer.zero_grad()
            if opts.shared_critic:
                agent.optimizer_sc.zero_grad()

            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            current_step = int(
                step * T / n_step * K_epochs + (t - 1) // n_step * K_epochs + k_
            )

            grad_norms = clip_grad_norms(
                agent.optimizer.param_groups, opts.max_grad_norm
            )

            # perform gradient descent
            agent.optimizer.step()

            if opts.shared_critic:
                grad_norms_sc_actor = clip_grad_norms(
                    agent.optimizer_sc.param_groups[:1], opts.max_grad_norm_construct
                )
                grad_norms_sc_critic = clip_grad_norms(
                    agent.optimizer_sc.param_groups[1:], opts.max_grad_norm
                )
                grad_norms[0].extend(grad_norms_sc_actor[0] + grad_norms_sc_critic[0])
                grad_norms[1].extend(grad_norms_sc_actor[1] + grad_norms_sc_critic[1])

                if not no_sc_train:
                    agent.optimizer_sc.step()

            # imitation learning
            if (
                opts.shared_critic
                and opts.imitation_step > 0
                and t % opts.imitation_step == 0
                and k_ == K_epochs - 1
                and not no_sc_train
            ):
                is_good = (memory.best_obj[-1].view(-1) < construct_obj).float()
                batch_for_imi = {'coordinates': batch_feature.clone()}
                batch_augments(
                    opts.cur_imitation_augment, batch_for_imi, one_is_keep=False
                )
                batch_feature_for_imi = batch_for_imi[
                    'coordinates'
                ]  # (batch_size, imitation_augment, graph_size_plus1, node_dim)

                # imi_adv = (memory.best_obj[-1].view(-1) - construct_obj).detach()

                for i in range(opts.cur_imitation_augment):
                    _, teaching_logprobs = agent.actor_construct(
                        batch_feature_for_imi[:, i, :, :], fixed_sol=best_sol
                    )
                    imitation_loss = -(
                        (is_good * teaching_logprobs).mean() * opts.imitation_rate
                    )

                    # update gradient step
                    agent.optimizer_sc.zero_grad()
                    imitation_loss.backward()

                    grad_norms_imi = clip_grad_norms(
                        agent.optimizer_sc.param_groups[:1],
                        opts.imitation_max_grad_norm,
                    )

                    # perform gradient descent
                    agent.optimizer_sc.step()

            # Logging to tensorboard
            if (not opts.no_tb) and rank == 0:
                if (current_step + 1) % int(opts.log_step) == 0:
                    log_to_tb_train(
                        tb_logger,
                        batch_feature[0, 0],
                        agent,
                        Reward,
                        ratios,
                        bl_val_detached,
                        total_cost,
                        grad_norms,
                        memory.rewards,
                        entropy,
                        approx_kl_divergence,
                        reinforce_loss,
                        baseline_loss,
                        logprobs,
                        initial_cost,
                        current_step + 1,
                        construct_obj if opts.shared_critic else None,
                        reinforce_loss_construct if opts.shared_critic else None,
                        bl_construct_detach if opts.shared_critic else None,
                        baseline_loss_construct if opts.shared_critic else None,
                        trust_degree if opts.shared_critic else None,
                        ratios_construct
                        if opts.shared_critic and opts.sc_rl_train_type == 'ppo'
                        else None,
                        imitation_loss if opts.shared_critic else None,
                        grad_norms_imi if opts.shared_critic else None,
                    )

            if rank == 0:
                pbar.update(1)

        # end update
        memory.clear_memory()
