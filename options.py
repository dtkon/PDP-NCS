from typing import Optional, List
import os
import time
import argparse
import math
import torch


class Option(argparse.Namespace):
    # shared critic
    shared_critic: bool

    sc_decoder_select_type: str
    sc_rl_train_type: str
    sc_map_sample_type: str
    sc_normalization: str
    no_sample_init: bool

    lr_trust_degree: float
    lr_construct: float
    max_grad_norm_construct: float
    sc_start_train_epoch: int

    inference_sample_size: int
    inference_sample_batch: int
    inference_temperature: List[float]

    temperature_init: float
    temperature_decay: float
    max_init_sample_size: int
    max_init_sample_batch: int
    init_sample_increase: float
    max_warm_up: int

    imitation_rate: float
    imitation_max_grad_norm: float
    imitation_step: int
    imitation_max_augment: int
    imitation_increase_w: float
    imitation_increase_b: float

    sc_attn_type: str
    embed_type_n2s: str
    embed_type_sc: str
    removal_type: str
    warm_up_type: str

    load_original_n2s: Optional[str]
    save_infer_dir: str
    zoom: bool
    # dynamic
    start_init_sample_epoch: int
    sc_map_sample_times: int
    cur_temperature: float
    cur_init_sample_size: int
    cur_imitation_augment: int
    start_warm_up_epoch: int

    # overall settings
    problem: str
    graph_size: int
    init_val_method: str
    no_cuda: bool
    no_tb: bool
    no_saving: bool
    use_assert: bool
    no_DDP: bool
    seed: int
    DDP_port_offset: int

    # N2S parameters
    v_range: float
    actor_head_num: int
    critic_head_num: int
    embedding_dim: int
    ff_hidden_dim: int
    n_encode_layers: int
    normalization: str

    # Training parameters
    RL_agent: str
    gamma: float
    K_epochs: int
    eps_clip: float
    T_train: int
    n_step: int
    warm_up: float
    batch_size: int
    epoch_end: int
    epoch_size: int
    lr_model: float
    lr_critic: float
    lr_decay: float
    max_grad_norm: float

    # Inference and validation parameters
    T_max: int
    eval_only: bool
    val_size: int
    val_batch_size: int
    val_dataset: Optional[str]
    val_m: int

    # resume and load models
    load_path: Optional[str]
    resume: Optional[str]
    epoch_start: int

    # logs/output settings
    no_progress_bar: bool
    log_dir: str
    log_step: int
    output_dir: str
    run_name: str
    checkpoint_epochs: int

    # add later
    world_size: int
    distributed: bool
    use_cuda: bool
    save_dir: str
    device: torch.device


def get_options(args: Optional[List[str]] = None) -> Option:
    parser = argparse.ArgumentParser(description="Neural Neighborhood Search")

    # shared critic
    parser.add_argument(
        '--shared_critic', action='store_true', help='enable shared critic mechanism'
    )
    parser.add_argument(
        '--sc_decoder_select_type',
        default='sample',
        choices=('sample', 'greedy'),
        help='next node select type for sc actor',
    )
    parser.add_argument(
        '--sc_rl_train_type',
        default='ppo',
        choices=('ppo', 'pg'),
        help='RL Training algorithm for sc actor',
    )
    parser.add_argument(
        '--sc_map_sample_type',
        default='augment',
        choices=('origin', 'augment'),
        help='RL sample type for sc actor',
    )
    parser.add_argument(
        '--sc_normalization',
        default='layer',
        help="normalization type for sc actor, 'layer' (default) or 'batch'",
    )
    parser.add_argument(
        '--no_sample_init', action='store_true', help='sc use random init'
    )
    parser.add_argument(
        '--lr_trust_degree',
        type=float,
        default=0.01,
        help='learning rate for trust degree',
    )
    parser.add_argument(
        '--lr_construct',
        type=float,
        default=1e-4,
        help='learning rate for trust degree',
    )
    parser.add_argument(
        '--max_grad_norm_construct',
        type=float,
        default=1,
        help='maximum L2 norm for gradient clipping of actor-construct',
    )
    parser.add_argument(
        '--sc_start_train_epoch',
        type=int,
        default=0,
        help='sc actor start train epoch',
    )
    parser.add_argument(
        '--inference_sample_size',
        type=int,
        default=128,
        help='actor-construct total sample size when inference',
    )
    parser.add_argument(
        '--inference_sample_batch',
        type=int,
        default=1,
        help='actor-construct sample size per batch when inference',
    )
    parser.add_argument(
        '--inference_temperature',
        type=float,
        nargs='*',
        default=[1],
        help='control the diversity of the sampled tours when inference',
    )
    parser.add_argument(
        '--temperature_init',
        type=float,
        default=10,
        help='part of adapative init solution generation when training',
    )
    parser.add_argument(
        '--temperature_decay',
        type=float,
        default=-1,  # variable default
        help='part of adapative init solution generation when training',
    )
    parser.add_argument(
        '--max_init_sample_size',
        type=int,
        default=-1,  # variable default
        help='part of adapative init solution generation when training',
    )
    parser.add_argument(
        '--max_init_sample_batch',
        type=int,
        default=128,
        help='part of adapative init solution generation when training',
    )
    parser.add_argument(
        '--init_sample_increase',
        type=float,
        default=-1,  # variable default
        help='part of adapative init solution generation when training',
    )
    parser.add_argument(
        '--max_warm_up',
        type=int,
        default=-1,  # variable default
        help='max warm up time',
    )
    parser.add_argument(
        '--imitation_rate',
        type=float,
        default=1,
        help='tunable parameter for imitation learning',
    )
    parser.add_argument(
        '--imitation_max_grad_norm',
        type=float,
        default=-1,  # variable default
        help='maximum L2 norm for gradient clipping of imitation',
    )
    parser.add_argument(
        '--imitation_step',
        type=int,
        default=250,
        help='if t %% imitation_step == 0, imitation loss run, set <=0 to off',
    )
    parser.add_argument(
        '--imitation_max_augment',
        type=int,
        default=-1,  # variable default
        help='max time of instance augment when performing imitaion learning',
    )
    parser.add_argument(
        '--imitation_increase_w',
        type=float,
        default=1,
        help='increase speed of instance augment when performing imitaion learning',
    )
    parser.add_argument(
        '--imitation_increase_b',
        type=float,
        default=-0.01,  # variable default
        help='increase speed of instance augment when performing imitaion learning',
    )
    parser.add_argument(
        '--sc_attn_type',
        default='typical',
        choices=('typical', 'heter'),
        help='MHA type for ConstructEncoder',
    )
    parser.add_argument(
        '--embed_type_n2s',
        default='origin',
        choices=('origin', 'pair', 'share', 'sep'),
        help='n2s actor graph embedding type',
    )
    parser.add_argument(
        '--embed_type_sc',
        default='pair',
        choices=('pair', 'share'),
        help='construction actor graph embedding type',
    )
    parser.add_argument(
        '--removal_type',
        default='glitch',
        choices=(
            'origin',
            'glitch',
            'update1',
            'update2',
        ),
        help='N2S NodePairRemovalDecoder type',
    )
    parser.add_argument(
        '--warm_up_type',
        default='origin',  # variable default
        choices=('origin', 'update'),
        help='N2S curriculum learning type',
    )
    parser.add_argument('--load_original_n2s', help='path to load original N2S actor')
    parser.add_argument(
        '--save_infer_dir', help='save costs_history and search_history'
    )
    parser.add_argument('--zoom', action='store_true', help='zoom')

    # overall settings
    parser.add_argument(
        '--problem',
        default='pdtsp',
        choices=['pdtsp', 'pdtspl'],
        help="The targeted problem to solve, default 'pdp'",
    )
    parser.add_argument(
        '--graph_size',
        type=int,
        default=20,
        help="T number of customers in the targeted problem (graph size)",
    )
    parser.add_argument(
        '--init_val_method',
        choices=['greedy', 'random'],
        default='random',
        help='method to generate initial solutions for inference',
    )
    parser.add_argument('--no_cuda', action='store_true', help='disable GPUs')
    parser.add_argument(
        '--no_tb', action='store_true', help='disable Tensorboard logging'
    )
    parser.add_argument(
        '--no_saving', action='store_true', help='disable saving checkpoints'
    )
    parser.add_argument('--use_assert', action='store_true', help='enable assertion')
    parser.add_argument(
        '--no_DDP', action='store_true', help='disable distributed parallel'
    )
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use')
    parser.add_argument(
        '--DDP_port_offset',
        type=int,
        default=0,
        help="os.environ['MASTER_PORT'] = 4869 + this_arg",
    )

    # N2S parameters
    parser.add_argument(
        '--v_range', type=float, default=6.0, help='to control the entropy'
    )
    parser.add_argument(
        '--actor_head_num', type=int, default=4, help='head number of N2S actor'
    )
    parser.add_argument(
        '--critic_head_num', type=int, default=4, help='head number of N2S critic'
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=128,
        help='dimension of input embeddings (NEF & PFE)',
    )
    parser.add_argument(
        '--ff_hidden_dim',
        type=int,
        default=128,
        help='dimension of hidden layers in Enc/Dec',
    )
    parser.add_argument(
        '--n_encode_layers',
        type=int,
        default=3,
        help='number of stacked layers in the encoder',
    )
    parser.add_argument(
        '--normalization',
        default='layer',
        help="normalization type, 'layer' (default) or 'batch'",
    )

    # Training parameters
    parser.add_argument(
        '--RL_agent', default='ppo', choices=['ppo'], help='RL Training algorithm'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.999,
        help='reward discount factor for future rewards',
    )
    parser.add_argument('--K_epochs', type=int, default=3, help='mini PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO clip ratio')
    parser.add_argument(
        '--T_train', type=int, default=250, help='number of itrations for training'
    )
    parser.add_argument(
        '--n_step', type=int, default=5, help='n_step for return estimation'
    )
    parser.add_argument(
        '--warm_up',
        type=float,
        default=-1,  # variable default
        help='hyperparameter of CL scalar $\rho^{CL}$',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=600,
        help='number of instances per batch during traingammaing',
    )
    parser.add_argument(
        '--epoch_end', type=int, default=200, help='maximum training epoch'
    )
    parser.add_argument(
        '--epoch_size',
        type=int,
        default=12000,
        help='number of instances per epoch during training',
    )
    parser.add_argument(
        '--lr_model',
        type=float,
        default=8e-5,
        help="learning rate for the actor network",
    )
    parser.add_argument(
        '--lr_critic',
        type=float,
        default=2e-5,
        help="learning rate for the critic network",
    )
    parser.add_argument(
        '--lr_decay', type=float, default=-1, help='learning rate decay per epoch'
    )  # variable default
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=-1,  # variable default
        help='maximum L2 norm for gradient clipping',
    )

    # Inference and validation parameters
    parser.add_argument(
        '--T_max', type=int, default=1500, help='number of steps for inference'
    )
    parser.add_argument(
        '--eval_only', action='store_true', help='switch to inference mode'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=1000,
        help='number of instances for validation/inference',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=1000,
        help='Number of instances per batch for validation/inference',
    )
    parser.add_argument(
        '--val_dataset',
        type=str,
        default=None,  # variable default
        help='dataset file path',
    )
    parser.add_argument(
        '--val_m', type=int, default=1, help='number of data augments in Algorithm 2'
    )

    # resume and load models
    parser.add_argument(
        '--load_path',
        default=None,
        help='path to load model parameters and optimizer state from',
    )
    parser.add_argument(
        '--resume', default=None, help='resume from previous checkpoint file'
    )
    parser.add_argument(
        '--epoch_start',
        type=int,
        default=0,
        help='start at epoch # (relevant for learning rate decay)',
    )

    # logs/output settings
    parser.add_argument(
        '--no_progress_bar', action='store_true', help='disable progress bar'
    )
    parser.add_argument(
        '--log_dir',
        default='logs',
        help='directory to write TensorBoard information to',
    )
    parser.add_argument(
        '--log_step',
        type=int,
        default=50,
        help='log info every log_step gradient steps',
    )
    parser.add_argument(
        '--output_dir', default='outputs', help='directory to write output models to'
    )
    parser.add_argument(
        '--run_name', default='run_name', help='name to identify the run'
    )
    parser.add_argument(
        '--checkpoint_epochs',
        type=int,
        default=1,
        help='save checkpoint every n epochs (default 1), 0 to save no checkpoints',
    )

    opts = Option()
    parser.parse_args(args, namespace=opts)

    # variable default
    if opts.temperature_decay == -1:
        if opts.graph_size == 50:
            opts.temperature_decay = 0.94
        elif opts.graph_size == 100:
            opts.temperature_decay = 0.93
        else:
            opts.temperature_decay = 0.95
    if opts.max_init_sample_size == -1:
        if opts.graph_size == 50:
            opts.max_init_sample_size = 256
        elif opts.graph_size == 100:
            opts.max_init_sample_size = 512
        else:
            opts.max_init_sample_size = 128
    if opts.init_sample_increase == -1:
        if opts.graph_size == 50:
            opts.init_sample_increase = 1.2
        elif opts.graph_size == 100:
            opts.init_sample_increase = 1.3
        else:
            opts.init_sample_increase = 1.1
    if opts.max_warm_up == -1:
        if opts.shared_critic and not opts.no_sample_init:
            opts.max_warm_up = 25
        else:
            opts.max_warm_up = 250
    if opts.imitation_max_grad_norm == -1:
        if opts.graph_size <= 50:
            opts.imitation_max_grad_norm = 0.1
        else:
            opts.imitation_max_grad_norm = 0.01
    if opts.imitation_max_augment == -1:
        if opts.graph_size >= 50:
            opts.imitation_max_augment = 25
        else:
            opts.imitation_max_augment = 10
    # if opts.imitation_increase_w == -1:
    #        opts.imitation_increase_w = 1
    if opts.imitation_increase_b == -0.01:
        if opts.graph_size == 100:
            opts.imitation_increase_b = -2
        else:
            opts.imitation_increase_b = 0
    if opts.warm_up_type == 'origin':
        if opts.shared_critic and not opts.no_sample_init:
            opts.warm_up_type = 'update'
    if opts.warm_up == -1:
        if opts.shared_critic and not opts.no_sample_init:
            opts.warm_up = 0
            # if opts.graph_size == 100:
            #    opts.warm_up = 2
        elif opts.graph_size == 50:
            opts.warm_up = 1.5
        elif opts.graph_size == 100:
            opts.warm_up = 1
        else:
            opts.warm_up = 2
    if opts.lr_decay == -1:
        if opts.shared_critic and not opts.no_sample_init:
            opts.lr_decay = 0.99
        else:
            opts.lr_decay = 0.985
    if opts.max_grad_norm == -1:
        if opts.graph_size == 50:
            opts.max_grad_norm = 0.15
        elif opts.graph_size == 100:
            opts.max_grad_norm = 0.3
        else:
            opts.max_grad_norm = 0.05
    if opts.val_dataset is None:
        if opts.graph_size == 20:
            opts.val_dataset = './datasets/pdp_20.pkl'
        elif opts.graph_size == 50:
            opts.val_dataset = './datasets/pdp_50.pkl'
        elif opts.graph_size == 100:
            opts.val_dataset = './datasets/pdp_100.pkl'

    ### figure out whether to use distributed training
    opts.world_size = torch.cuda.device_count()
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.distributed = (
        opts.use_cuda and (torch.cuda.device_count() > 1) and (not opts.no_DDP)
    )
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(4869 + opts.DDP_port_offset)

    # assert opts.val_m <= opts.graph_size // 2
    assert opts.epoch_size % opts.batch_size == 0
    if opts.distributed:
        assert opts.batch_size % opts.world_size == 0

    opts.run_name = (
        "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
        if not opts.resume
        else opts.resume.split('/')[-2]
    )
    opts.save_dir = (
        os.path.join(
            opts.output_dir,
            "{}_{}".format(opts.problem, opts.graph_size),
            opts.run_name,
        )
        if not opts.no_saving
        else 'no_saving'
    )

    assert opts.temperature_init >= 1 and 0 < opts.temperature_decay < 1
    opts.start_init_sample_epoch = math.ceil(
        math.log(1 / opts.temperature_init, opts.temperature_decay)
    )
    if opts.shared_critic and not opts.no_sample_init:
        opts.start_warm_up_epoch = opts.start_init_sample_epoch + math.ceil(
            math.log(opts.max_init_sample_size, opts.init_sample_increase)
        )
    else:
        opts.start_warm_up_epoch = 0

    opts.sc_map_sample_times = math.ceil(opts.T_train / opts.n_step)

    if opts.embed_type_n2s == 'share' and opts.embed_type_sc == 'share':
        opts.embed_type_n2s = 'together'
        opts.embed_type_sc = 'together'

    return opts
