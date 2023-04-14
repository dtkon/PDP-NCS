from typing import Dict, Optional, Tuple
import time
import torch
import os
import random
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger

from problems.problem_pdp import PDP
from utils.logger import log_to_screen, log_to_tb_val
from utils import rotate_tensor, move_to

from .agent import Agent


def gather_tensor_and_concat(tensor: torch.Tensor) -> torch.Tensor:
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)


def validate(
    rank: int,
    problem: PDP,
    agent: Agent,
    val_dataset_str: Optional[str] = None,
    tb_logger: Optional[TbLogger] = None,
    distributed: bool = False,
    id_: Optional[int] = None,
    mem_test: bool = False,
    zoom: bool = False,
) -> None:
    # Validate mode
    if rank == 0 and not mem_test:
        print('\nValidating...', flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = agent.opts
    agent.eval()

    random_state_backup = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        random.getstate(),
    )

    torch.manual_seed(opts.seed)
    random.seed(opts.seed)

    val_dataset = PDP.make_dataset(
        size=opts.graph_size,
        num_samples=opts.val_size,
        filename=val_dataset_str,
        silence=mem_test,
    )

    if distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(
            backend='nccl', world_size=opts.world_size, rank=rank
        )
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        if opts.normalization == 'batch':
            agent.actor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent.actor).to(
                device
            )  # type: ignore
            agent.actor = torch.nn.parallel.DistributedDataParallel(
                agent.actor, device_ids=[rank]
            )  # type: ignore
        if opts.shared_critic:
            agent.actor_construct.to(device)
            if opts.sc_normalization == 'batch':
                agent.actor_construct = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    agent.actor_construct
                ).to(
                    device
                )  # type: ignore
            agent.actor_construct = torch.nn.parallel.DistributedDataParallel(
                agent.actor_construct, device_ids=[rank]
            )  # type: ignore
        if not opts.no_tb and rank == 0 and not mem_test:
            tb_logger = TbLogger(
                os.path.join(
                    opts.log_dir,
                    "{}_{}".format(opts.problem, opts.graph_size),
                    opts.run_name,
                )
            )

        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )  # type: ignore
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opts.val_batch_size // opts.world_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=train_sampler,
        )
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=opts.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    s_time = time.time()
    bv_list = []
    cost_hist_list = []
    best_hist_list = []
    r_list = []
    for batch in tqdm(
        val_dataloader,
        desc='inference',
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        disable=mem_test or (rank != 0),
    ):
        bv_, cost_hist_, best_hist_, r_ = agent.rollout(
            problem, opts.val_m, batch, show_bar=(rank == 0 and not mem_test), zoom=zoom
        )
        bv_list.append(bv_)
        cost_hist_list.append(cost_hist_)
        best_hist_list.append(best_hist_)
        r_list.append(r_)

        if mem_test:
            break
    bv = torch.cat(bv_list, 0)
    cost_hist = torch.cat(cost_hist_list, 0)
    best_hist = torch.cat(best_hist_list, 0)
    r = torch.cat(r_list, 0)

    if distributed:
        dist.barrier()

        initial_cost = gather_tensor_and_concat(cost_hist[:, 0].contiguous())
        time_used = gather_tensor_and_concat(
            torch.tensor([time.time() - s_time]).cuda()
        )
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        search_history = gather_tensor_and_concat(best_hist.contiguous())
        reward = gather_tensor_and_concat(r.contiguous())

        dist.barrier()
    else:
        initial_cost = cost_hist[:, 0]  # bs
        time_used = torch.tensor([time.time() - s_time])  # bs
        bv = bv
        costs_history = cost_hist
        search_history = best_hist
        reward = r

    # save costs_history and search_history
    if opts.save_infer_dir:
        torch.save(
            {'current': costs_history, 'best': search_history, 'option': vars(opts)},
            opts.save_infer_dir + '/infer_' + opts.run_name + '.pt',
        )

    # log to screen
    if rank == 0 and not mem_test:
        log_to_screen(
            time_used,
            initial_cost,
            bv,
            reward,
            costs_history,
            search_history,
            batch_size=opts.val_size,
            dataset_size=len(val_dataset),
            T=opts.T_max,
        )

    # log to tb
    if (not opts.no_tb) and rank == 0 and not mem_test:
        log_to_tb_val(
            tb_logger,
            time_used,
            initial_cost,
            bv,
            reward,
            costs_history,
            search_history,
            batch_size=opts.val_size,
            val_size=opts.val_size,
            dataset_size=len(val_dataset),
            T=opts.T_max,
            epoch=id_,
        )

    torch.set_rng_state(random_state_backup[0])
    torch.cuda.set_rng_state(random_state_backup[1])
    random.setstate(random_state_backup[2])

    if distributed:
        dist.barrier()


def batch_augments(
    val_m: int,
    batch: Dict[str, torch.Tensor],
    graph_size_plus1: Optional[int] = None,
    node_dim: Optional[int] = None,
    one_is_keep: bool = True,
) -> None:
    batch['coordinates'] = batch['coordinates'].unsqueeze(1).repeat(1, val_m, 1, 1)
    augments = ['Rotate', 'Flip_x-y', 'Flip_x_cor', 'Flip_y_cor']

    if val_m > 1 or (not one_is_keep and val_m == 1):
        for i in range(val_m):
            random.shuffle(augments)
            id_ = torch.rand(4)
            for aug in augments:
                if aug == 'Rotate':
                    batch['coordinates'][:, i] = rotate_tensor(
                        batch['coordinates'][:, i], int(id_[0] * 4 + 1) * 90
                    )
                elif aug == 'Flip_x-y':
                    if int(id_[1] * 2 + 1) == 1:
                        data = batch['coordinates'][:, i].clone()
                        batch['coordinates'][:, i, :, 0] = data[:, :, 1]
                        batch['coordinates'][:, i, :, 1] = data[:, :, 0]
                elif aug == 'Flip_x_cor':
                    if int(id_[2] * 2 + 1) == 1:
                        batch['coordinates'][:, i, :, 0] = (
                            1 - batch['coordinates'][:, i, :, 0]
                        )
                elif aug == 'Flip_y_cor':
                    if int(id_[3] * 2 + 1) == 1:
                        batch['coordinates'][:, i, :, 1] = (
                            1 - batch['coordinates'][:, i, :, 1]
                        )

    if graph_size_plus1 is not None and node_dim is not None:
        batch['coordinates'] = batch['coordinates'].view(-1, graph_size_plus1, node_dim)


def mem_test(agent: Agent, problem: PDP, batch: Dict[str, torch.Tensor]) -> None:
    random_state_backup = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state(),
        random.getstate(),
    )

    opts = agent.opts

    batch = (
        move_to(batch, 0) if opts.distributed else move_to(batch, opts.device)
    )  # batch_size, graph_size+1, 2
    batch_feature: torch.Tensor = (
        move_to(PDP.input_coordinates(batch), 0)
        if opts.distributed
        else move_to(PDP.input_coordinates(batch), opts.device)
    )
    _, graph_size_plus1, node_dim = batch_feature.size()

    agent.eval()

    if opts.shared_critic and not opts.no_sample_init:
        print('testing memory restriction for init construct sample...', end=' ')
        train_sample_size = min(opts.max_init_sample_size, opts.max_init_sample_batch)

        ms_batch_feature = batch_feature.unsqueeze(1).repeat(1, train_sample_size, 1, 1)
        ms_batch_feature = ms_batch_feature.view(-1, graph_size_plus1, node_dim)

        agent.actor_construct(ms_batch_feature)

        print('pass')

    print('testing memory restriction for validate...', end=' ')

    opts_backup = opts.T_max, opts.inference_sample_size
    opts.T_max = 1
    opts.inference_sample_size = min(
        opts.inference_sample_size, opts.inference_sample_batch
    )
    validate(0, problem, agent, mem_test=True)
    opts.T_max, opts.inference_sample_size = opts_backup

    print('pass')

    torch.set_rng_state(random_state_backup[0])
    torch.cuda.set_rng_state(random_state_backup[1])
    random.setstate(random_state_backup[2])


def zoom_feature(feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    max_c = feature.max(1)[0]
    min_c = feature.min(1)[0]
    x_gap = max_c[:, 0] - min_c[:, 0]  # (10,)
    y_gap = max_c[:, 1] - min_c[:, 1]  # (10,)
    xy_gap = torch.cat([x_gap[None, :], y_gap[None, :]])  # (2,10)
    gap = xy_gap.max(0)[0]  # (10,)
    new_fea = (feature - min_c[:, None, :]) / gap[:, None, None]
    return new_fea, gap
