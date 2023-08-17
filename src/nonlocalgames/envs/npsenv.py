from typing import Dict, Any, Tuple

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box

from qiskit import QuantumCircuit, transpile, Aer
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveStatevector

from adaptgym.hamiltonians import Hamiltonian

from ..hamiltonians import NPartiteSymmetricNLG

class NPSEnv(gym.Env):
    '''RL environment for N-partite symmetric non-local game
    
    Configuration:
    ```
    {
        'players': Number of players, minimum 2,
        'layers': [Optional] Number of hardware layers, default 1,
        'questions': [Optional] Number of questions (currently only 2)
    }
    ```
    '''

    # Don't support rendering the environment for the time being
    # Todo: Possibly support drawing qiskit circuit
    metadata = {
        'render_modes': []
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self._config = config
        self._validate_config()

        self._sim: AerSimulator = Aer.get_backend('aer_simulator_statevector')
        self._qc, self._theta = self._create_ansatz()
        self._ham, self._phi = self._create_hamiltonian()

        self._params = np.zeros(len(self._theta) + self._phi.size)
        self._iter = 0

        self.observation_space = Box(
            low=-np.pi,
            high=np.pi,
            shape=self._params.shape
        )

        self.action_space = Box(low=-np.pi, high=np.pi, shape=(1,))
    
    def reset(self, seed: int | None = None, options: dict = None):
        super().reset(seed=seed)

        self._params[:] = 0
        self._ham.params[:] = 0
        self._iter = 0

        return self._params, {}
    
    def step(self, action: float):
        # Set new circuit parameter
        action = action + np.pi
        action %= 2 * np.pi
        action -= np.pi
        self._params[self._iter] = action
        self._iter += 1

        obs = self._params
        reward = 0
        done = self._iter == self._params.size
        info = {}

        if done:
            energy = self._get_energy()
            # Invert the energy since the g.s. is optimal violation
            reward = -energy
            info['energy'] = energy
        
        return obs, reward, done, False, info
    
    def render(self):
        pass

    def _validate_config(self):
        '''Validates the environment configuration'''

        assert 'players' in self._config
        assert self._config['players'] >= 2, 'Game requires multiple players'

        if self._config.setdefault('questions', 2) != 2:
            raise NotImplementedError("Don't support more than 2 questions")
        
        assert self._config.setdefault('layers', 1) >= 1, \
            "Need to specify positive number of circuit layers"

    def _create_ansatz(self) -> Tuple[QuantumCircuit, ParameterVector]:
        '''Constructs the shared quantum circuit'''

        players = self._config['players']
        layers = self._config['layers']

        qc = QuantumCircuit(players, name='q')
        params = ParameterVector('θ', players * layers)

        param_idx = 0
        for _ in range(layers):
            # Ry block
            for i in range(players):
                qc.ry(params[param_idx], i)
                param_idx += 1
            
            # CNOT gates
            for i in range(players):
                qc.cx(i, (i + 1) % players)
        
        qc.append(SaveStatevector(players), qargs=range(players))
        qc = transpile(qc, self._sim)
        return qc, params
    
    def _create_hamiltonian(self) -> Tuple[Hamiltonian, np.ndarray]:
        '''Constructs hamiltonian'''
        N = self._config['players']
        ham = NPartiteSymmetricNLG(N)
        phi = ham.params

        return ham, phi

    def _get_energy(self) -> float:
        n_theta = len(self._theta)
        theta = self._params[:n_theta]
        phi = self._params[n_theta:]

        # First prepare quantum state
        qc = self._qc.bind_parameters({self._theta: theta})
        job = self._sim.run(qc)
        result = job.result()

        # Get ket and bra as numpy array from statevector simulator
        ket: np.ndarray = np.asarray(result.get_statevector()).reshape(-1, 1)
        bra = ket.T.conj()
        
        # Regenerate the hamiltonian based on the parameters chosen
        self._ham.params = phi

        # Compute energy
        E = bra @ self._ham.mat @ ket
        assert E.size == 1
        E = E.item()
        assert np.isclose(E.imag, 0)
        E = E.real

        assert not np.isnan(E)
        return E
