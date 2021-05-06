import gym
import time
import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI

# from stable_baselines.common.schedules import PiecewiseSchedule, LinearSchedule
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule, linear_interpolation

from ppo_decay import PPO2_DECAY

import argparse

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False, log_dir='./model7-linear-decay', comm=None, scheduler):
    learning_rate = 5e-4
    if scheduler is None:
        ent_coef = .01
    elif scheduler == "linear":
        print("linear")
        ent_coef = LinearSchedule(timesteps_per_proc, 1e-2, 1e-5).value,
    elif scheduler == "piecewise":
        print("piecewise")
        ent_coef = PiecewiseSchedule([1e-2, 1e-5]).value,
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # @register("impala_cnn")
    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2.learn(
        env=venv,
        # policy=conv_fn,
        # policy="impala_cnn"
        network=conv_fn,                        # 'network' for baselines, 'policy' for stable-baselines
        total_timesteps=timesteps_per_proc,
        save_interval=1,
        nsteps=nsteps,
        # n_steps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        # learning_rate=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log = "./tensorboard_logs/"
    )
    
    # model = PPO2_DECAY(
    #     env=venv,
    #     # policy=conv_fn,
    #     policy="impala_cnn"
    #     # network=conv_fn,                        # 'network' for baselines, 'policy' for stable-baselines
    #     # total_timesteps=timesteps_per_proc,
    #     # save_interval=1,
    #     # nsteps=nsteps,
    #     n_steps=nsteps,
    #     nminibatches=nminibatches,
    #     lam=lam,
    #     gamma=gamma,
    #     noptepochs=ppo_epochs,
    #     # log_interval=1,
    #     ent_coef=ent_coef,
    #     # mpi_rank_weight=mpi_rank_weight,

    #     # clip_vf=use_vf_clipping,

    #     # comm=comm,
    #     # lr=learning_rate,
    #     learning_rate=learning_rate,
    #     cliprange=clip_range,
    #     # update_fn=None,
    #     # init_fn=None,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    #     tensorboard_log = "./tensorboard_logs/"
    # )

    # model.learn(timesteps_per_proc)

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='fruitbot')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='easy', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=500)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=2)
    parser.add_argument('--timesteps_per_proc', type=int, default=5_000_000)
    parser.add_argument('--scheduler', type=str, choices=["linear", "piecewise"])

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    tic = time.perf_counter()

    print("USING ", args.scheduler)

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        is_test_worker=is_test_worker,
        comm=comm,
        args.scheduler)

    toc = time.perf_counter()
    num_hours = (toc - tic) // 3600
    num_minutes = ((toc - tic) % 3600) // 60
    num_seconds = ((toc - tic) % 3600) % 60
    print(f"Train time: {num_hours} hours, {num_minutes} minutes, {num_seconds} seconds")

if __name__ == '__main__':
    main()
