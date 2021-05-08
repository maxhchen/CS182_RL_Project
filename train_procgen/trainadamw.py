# import gym
# import time
import tensorflow as tf
from baselines.ppo2 import ppo2
from baselines.common.models import build_impala_cnn, impala_cnn
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
import ppo_adamw

from baselines.common.schedules import LinearSchedule, PiecewiseSchedule, linear_interpolation

# from ppo_decay import PPO2_DECAY

import argparse

class ExponentialSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Exponential interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        self.endpoint = -np.log(self.final_p / self.initial_p)

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0) * self.endpoint
        return self.initial_p * np.exp(-fraction)

def zoh_interpolation(l, r, alpha):
    return l

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, scheduler, high_entropy, is_test_worker=False, log_dir='./model-12-adamw', comm=None):
    learning_rate = 5e-4
    if high_entropy == False:
        if scheduler == "none":
            print("Constant Entropy Coeff")
            ent_coef = 1e-2
        elif scheduler == "linear":
            print("Linear Scheduler -- creating function...")
            ent_coef = LinearSchedule(schedule_timesteps = timesteps_per_proc, final_p = 1e-5, initial_p = 1e-2)
        elif scheduler == "exponential":
            print("Exponential Scheduler -- creating function...")
            ent_coef = ExponentialSchedule(schedule_timesteps = timesteps_per_proc, final_p = 1e-5, initial_p = 1e-2)
        elif scheduler == "piecewise":
            print("Piecewise Scheduler -- creating function...")
            ent_coef = PiecewiseSchedule(endpoints = [(0, 1e-2),
                                                        (timesteps_per_proc // 10, 7e-3),
                                                        (timesteps_per_proc // 10 * 2, 4e-3),
                                                        (timesteps_per_proc // 10 * 3, 1e-3),
                                                        (timesteps_per_proc // 10 * 4, 7e-4),
                                                        (timesteps_per_proc // 10 * 5, 4e-4),
                                                        (timesteps_per_proc // 10 * 6, 1e-4),
                                                        (timesteps_per_proc // 10 * 7, 7e-5),
                                                        (timesteps_per_proc // 10 * 8, 4e-5),
                                                        (timesteps_per_proc, 1e-5)],
                                            interpolation = zoh_interpolation)
    else:
        if scheduler == "none":
            print("Constant Entropy Coeff")
            ent_coef = 1e-1
        elif scheduler == "linear":
            print("Linear Scheduler -- creating function...")
            ent_coef = LinearSchedule(schedule_timesteps = timesteps_per_proc, final_p = 1e-4, initial_p = 1e-1)
        elif scheduler == "exponential":
            print("Exponential Scheduler -- creating function...")
            ent_coef = ExponentialSchedule(schedule_timesteps = timesteps_per_proc, final_p = 1e-4, initial_p = 1e-1)
        elif scheduler == "piecewise":
            print("Piecewise Scheduler -- creating function...")
            ent_coef = PiecewiseSchedule(endpoints = [(0, 1e-1),
                                                        (timesteps_per_proc // 10, 7e-2),
                                                        (timesteps_per_proc // 10 * 2, 4e-2),
                                                        (timesteps_per_proc // 10 * 3, 1e-2),
                                                        (timesteps_per_proc // 10 * 4, 7e-3),
                                                        (timesteps_per_proc // 10 * 5, 4e-3),
                                                        (timesteps_per_proc // 10 * 6, 1e-3),
                                                        (timesteps_per_proc // 10 * 7, 7e-4),
                                                        (timesteps_per_proc // 10 * 8, 4e-4),
                                                        (timesteps_per_proc, 1e-4)],
                                            interpolation = zoh_interpolation)
    # ent_coef = .1

    # print(type(ent_coef))
    # print(type(ent_coef.value))
    # print(ent_coef)
    # print(ent_coef.value)

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

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo_adamw.learn(
        env=venv,
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
        #################################################################################
        ent_coef=ent_coef,
        #################################################################################
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
    #     policy="impala_cnn",
    #     # policy="impala_cnn",
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
    parser.add_argument('--scheduler', type=str, default="none", choices=["none", "linear", "exponential", "piecewise"])
    parser.add_argument('--log_dir', type=str, default="TEST")
    parser.add_argument('--high_entropy', type=bool, default=False)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    # tic = time.perf_counter()

    print("Using", args.scheduler, "Scheduler for Entropy Decay")
    print("Saving to dir:", args.log_dir)
    print("Using high entropy?", args.high_entropy)

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        ####################################
        args.scheduler,
        args.high_entropy,
        ####################################
        is_test_worker=is_test_worker,
        ####################################
        log_dir = args.log_dir,
        ####################################
        comm=comm,
        )

    # toc = time.perf_counter()
    # num_hours = (toc - tic) // 3600
    # num_minutes = ((toc - tic) % 3600) // 60
    # num_seconds = ((toc - tic) % 3600) % 60
    # print(f"Train time: {num_hours} hours, {num_minutes} minutes, {num_seconds} seconds")

if __name__ == '__main__':
    main()
