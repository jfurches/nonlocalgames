import pytest

import numpy as np
from ray.rllib.utils.pre_checks.env import check_env

from nonlocalgames.envs import NPSEnv

class TestNPSEnv:
    @pytest.mark.parametrize('players', (2, 4))
    def test_env(self, players: int):
        config = {
            'players': players
        }
        env = NPSEnv(config)

        # Run rllib compatibility check, this function errors
        # if there's a problem
        check_env(env)

        # Parameters: N for 1 layer + 2N for measurement settings
        max_iter = 3 * players
        iter_ = 0

        # Start one episode
        obs, _ = env.reset()
        done = False
        assert np.allclose(obs, 0, rtol=0)

        # Randomly choose our actions for the episode
        params: np.ndarray = env.observation_space.sample()
        params.setflags(write=False)
        
        while not done:
            obs, reward, done, *_ = env.step(params[iter_])
            iter_ += 1

            # Check that our parameters up to the current action
            # match up
            assert (iter_ < max_iter) != done, 'Ended too early'
            assert np.allclose(params[:iter_], obs[:iter_])

            # We currently use a sparse reward, so reward should be 0
            # at all time steps except at the end, which could possibly be 0,
            # but it's very low chance. If it fails because of this, just rerun
            # the test.
            if done:
                assert reward != 0
            else:
                assert reward == 0
