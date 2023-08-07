'''Example showing how we can train PPO on LiH molecule.'''

import argparse
from typing import Dict, Tuple
import os
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import ray
from ray import air, tune
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.experiment.trial import Trial
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks, make_multi_callbacks
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.utils.filter import RunningStat

from nonlocalgames.envs import NPSEnv

class CustomMetrics(DefaultCallbacks):
    '''This class logs custom information about our agent'''
    track = ['energy']

    def on_episode_step(self, *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy] | None = None,
        episode: EpisodeV2,
        env_index: int | None = None,
        **kwargs) -> None:

        # Log special reward types like `reward_hardware`
        agent_id = episode.get_agents()[0]
        for k, v in episode._last_infos[agent_id].items():
            if k in self.track:
                episode.user_data.setdefault(k, []).append(v)
    
    def on_episode_end(self, *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: EpisodeV2 | Exception,
        env_index: int | None = None,
        **kwargs) -> None:
        
        for k, v in episode.user_data.items():
            if k in self.track:
                episode.custom_metrics[k] = np.mean(v)
                episode.custom_metrics[f'{k}_min'] = np.min(v)
                episode.custom_metrics[f'{k}_max'] = np.max(v)
                episode.custom_metrics[f'{k}_std'] = np.std(v)

class RewardNormalizer(DefaultCallbacks):
    GAMMA = 0.99
    CLIP = 10
    EPSILON = 1e-8

    def __init__(self):
        super().__init__()

        self.shape = (1,)
        self.running_stats = RunningStat(self.shape)
        self.ret = np.zeros(self.shape)

    def on_postprocess_trajectory(self, *,
                                  worker: RolloutWorker,
                                  episode: Episode,
                                  agent_id: AgentID,
                                  policy_id: PolicyID,
                                  policies: Dict[PolicyID, Policy],
                                  postprocessed_batch: SampleBatch,
                                  original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
                                  **kwargs) -> None:

        batch = postprocessed_batch[SampleBatch.REWARDS]
        
        for timestep in range(len(batch)):
            batch[timestep] = self.reward(batch[timestep])

    def reward(self, reward):
        return self._normalized_reward(reward)

    def _normalized_reward(self, reward):
        self.ret = self.GAMMA * self.ret + reward
        self.running_stats.push(self.ret)

        reward = reward / (self.running_stats.std + self.EPSILON)
        return reward

def register_nps_env(args: argparse.Namespace):
    '''Registers ADAPT with the RLLib environment registry'''

    # This is the env creator for rllib
    def env_creator(config: dict):
        steps = config.get('players', 2) * (config.get('layers', 1) + 2)
        env = NPSEnv(config)
        env = TimeLimit(env, max_episode_steps=steps)

        return env

    # Pass the creator function to rllib
    register_env('NPSGame', env_creator)

def get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', type=int, default=2000,
                        help='The number of timesteps to train')
    parser.add_argument('--num-cpus', type=int, default=4,
                        help='The number of CPUs available for training')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='The number of GPUs for training. Probably only need 1')

    parser.add_argument('--resume', type=str,
                        help='Resumes training from an unfinished checkpoint')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--num-samples', type=int,
                        help='Number of hyperparameter samples to try')

    parser.add_argument('--lstm', action='store_true',
                        help='Use LSTM model')
    parser.add_argument('--gtrxl', action='store_true',
                        help='Use transformer model (GTrXL)')
    
    parser.add_argument('--players', type=int, default=2)
    parser.add_argument('--layers', type=int, default=1)

    args = parser.parse_args()
    return args

def trial_name_creator(trial: Trial):
    env_config: dict = trial.config['env_config']

    s = '{trainable}_{env}_{players}_{trial_id}'.format(
        trainable=trial.trainable_name,
        env=trial.config['env'],
        players=env_config['players'],
        trial_id=trial.trial_id
    )

    return s

default_hparams = {
    'gamma': 0.99,
    'train_batch_size': 4500,
    'lr': 1e-4,
    'entropy_coeff': 0,
    # 'lambda_': 0.95
}

if __name__ == '__main__':
    args = get_cli_args()

    # Start the ray cluster
    ray.init(num_cpus=args.num_cpus)

    # Register NPS environment
    register_nps_env(args)

    # Default configuration using PyTorch
    config = PPOConfig()
    config.framework('torch')

    config.environment('NPSGame',
        env_config={
            'players': args.players,
            'layers': args.layers,
            'questions': 2
        }
    )

    steps = (args.layers + 2) * args.players
    num_rollout_workers = args.num_cpus - 1
    rollout_fragment_length = steps
    min_batch_size = num_rollout_workers * rollout_fragment_length

    training_params = default_hparams
    training_params['train_batch_size'] = 5 * steps * num_rollout_workers

    # Specify PPO architecture
    model_cfg = {
        'fcnet_hiddens': [256, 256],
        'fcnet_activation': 'relu'
    }
    if args.lstm:
        model_cfg |= {
            # This flag causes rllib to use
            # an LSTM wrapper, and it defaults to using a fully-connected
            # net inside
            'use_lstm': True,
            'max_seq_len': steps,
            'lstm_cell_size': 256
        }
    elif args.gtrxl:
        model_cfg |= {
            # This flag causes rllib to use
            # a GTrXL wrapper, and it defaults to using a fully-connected
            # net inside
            'use_attention': True,
            'max_seq_len': steps,
            'attention_num_transformer_units': 2,
            'attention_dim': 64,
            'attention_num_heads': 4,
            'attention_head_dim': 64,
            'attention_memory_inference': 150,
            'attention_memory_training': 150,
            'attention_position_wise_mlp_dim': 64
        }
    else:
        # Default fcnet
        pass

    config.training(
        model=model_cfg,
        # lr_schedule=[[0, 1e-4], [1000000, 0]],
        clip_param=0.2,
        sgd_minibatch_size=128,
        num_sgd_iter=4,
        **training_params,
        _enable_learner_api=False
    )

    config.rl_module(_enable_rl_module_api=False)

    # Allocate the GPU to the process holding our model (trainer worker #0),
    # since AdaptEnv cannot use the GPU
    config.resources(
        num_gpus=args.num_gpus,
        num_gpus_per_learner_worker=args.num_gpus
    )

    # Number of simultaneous NPSEnv we train on. We set this to be
    # num_cpus - 1 so that one worker process is left to drive everything, while
    # all the other cores run environments.
    config.rollouts(
        num_rollout_workers=num_rollout_workers,
        num_envs_per_worker=1,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode='truncate_episodes'
    )

    # Log our custom metrics
    config.callbacks(CustomMetrics)
    # config.callbacks(make_multi_callbacks([CustomMetrics, RewardNormalizer]))

    # Optional: upload logs to AWS S3. This is useful for
    # clusters that don't support SSH port-forwarding.
    upload_dir = os.environ.get('NLG_BUCKET', None)

    # Stop if we hit our maximum timesteps or if we can
    # solve the non-local game for 6 players reliably. We
    # divide by steps here because it's the mean of a sparse
    # reward.
    stop ={
        'timesteps_total': args.num_iters,
        'episode_reward_mean': 0.258 / steps
    }

    if args.resume is None:
        name = 'NLG_PPO'

        if args.lstm:
            name += '_LSTM'
        elif args.gtrxl:
            name += '_GTrXL'

        if args.tune:
            name += '_tune'

        # This is the part where everything happens
        tune.Tuner(
            'PPO',
            run_config=air.RunConfig(
                name=name,
                storage_path=str(Path.home() / 'ray_results' / name),
                stop=stop,
                sync_config=tune.SyncConfig(
                    upload_dir=upload_dir
                ),
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_frequency=250
                )
            ),
            param_space=config.to_dict()
        ).fit()
    else:
        tune.Tuner.restore(path=args.resume, resume_unfinished=True).fit()

    # Cleanup
    ray.shutdown()
