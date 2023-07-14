from typing import Tuple, Dict, Union
import itertools

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.opflow import I, X, Y, Z
from qiskit_aer import AerSimulator, StatevectorSimulator

ObsType = np.ndarray
InfoType = dict

class CHSHEnv(gym.Env):
    metadata = {
        'render_modes': []
    }

    ansatze = ['adapt', 'arxiv']
    players = 2

    chsh_rules = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ], dtype=float)

    default_config = {
        'ansatz': 'adapt',
        'simulator': {
            'type': AerSimulator,
            'shots': 1024
        }
    }

    def __init__(self, config: dict, **kwargs):
        super().__init__(**kwargs)

        config = self.default_config | config

        self.iter = 0

        # The qiskit_params object is a parameter vector where the last
        # N parameters (for CHSH N=2) control the measurement angles
        self._prepare_simulator(config['simulator'])
        self.qc, self.qiskit_params = self._get_circuit(config['ansatz'])
        self.state_params = np.zeros(len(self.qiskit_params) - self.players)
        self.measurement_params = np.zeros(2 * self.players)

        # Observation space and action space are the parameters in the circuit
        self.observation_space = spaces.Box(low=-np.pi, high=np.pi,
                        shape=(len(self.state_params) + len(self.measurement_params),))
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,))
    
    def reset(self, seed: int | None = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        
        self.state_params[:] = 0
        self.measurement_params[:] = 0
        self.iter = 0
    
        return self._get_obs()

    def step(self, action: float) -> Tuple[ObsType, float, bool, bool, InfoType]:
        # First actions set the state preparation circuit
        if self.iter < len(self.state_params):
            self.state_params[self.iter] = action
        # Last actions set the measurements
        else:
            self.measurement_params[self.iter - len(self.state_params)] = action
        
        obs, info = self._get_obs()
        reward = self._measure()

        self.iter += 1
        # Episode ends when all parameters have been set
        done = self.iter == (len(self.state_params) + len(self.measurement_params))

        return obs, reward, done, False, info
    
    def _get_obs(self) -> Tuple[ObsType, InfoType]:
        return np.concatenate((self.state_params, self.measurement_params)), {}
    
    def _measure(self) -> float:
        '''Measures the probability of winning the game'''

        q_set = (0, 1)
        q_prob = 1 / len(q_set) ** self.players # π(x, y)
        p_win = 0
        for questions in itertools.product(q_set, repeat=self.players):
            # phi[i] = params[2 * p + q]
            phi = self.measurement_params[2 * np.arange(self.players) + np.array(questions)]
            circuit_params = np.concatenate((self.state_params, phi))

            qc = self.qc.bind_parameters({self.qiskit_params: circuit_params})
            job = self.sim.run(qc)
            result = job.result()

            if isinstance(self.sim, AerSimulator):
                counts: Dict[str, int] = result.get_counts(0)
                p_vec = np.zeros(len(q_set) ** self.players)

                for bits, count in counts.items():
                    p_vec[int(bits, 2)] = count
                
                p_vec /= p_vec.sum()

            elif isinstance(self.sim, StatevectorSimulator):
                state = result.get_state(0)
                p_vec = np.abs(state) ** 2
            
            # p_vec is the vector p(a,b|x,y)
            # V_i = V(a,b|x,y)p(a,b|x,y)
            v = self.chsh_rules @ p_vec

            # Convert the question (a binary string) to integer to index into
            # the game value.
            q_index = int(np.dot(2 ** np.arange(self.players), np.array(questions)))
            p_win += q_prob * v[q_index]
        
        return p_win
    
    def _get_circuit(self, ansatz: str) -> Tuple[QuantumCircuit, ParameterVector]:
        assert ansatz in self.ansatze

        # Initialize quantum circuit of 2 qubits, 2 classical bits, and
        # 2 measurement parameters
        qc = QuantumCircuit(2, 2)
        params = ParameterVector('θ', 2)

        # State preparation
        if ansatz == 'adapt':
            # Add 1 parameter for our state exp(-itYX)|00>
            params.resize(len(params) + 1)
            op = PauliEvolutionGate(Y ^ X, params[0])
            qc.append(op, range(self.players))
        
        elif ansatz == 'arxiv':
            # todo: implement the hardware ansatz from the paper
            raise NotImplementedError()
    
        # Add measurement layer
        for i in range(self.players):
            qc.ry(params[-self.players + i], i)

        # Adding measurement causes collapse for StatevectorSimulator
        if isinstance(self.sim, AerSimulator):
            mapping = list(range(self.players))
            qc.measure(mapping, mapping)
        
        qc = transpile(qc, self.sim)
        return qc, params

    def _prepare_simulator(self, config: dict):
        sim_type = config.pop('type')
        assert sim_type in (AerSimulator, StatevectorSimulator)
        self.sim: Union[AerSimulator, StatevectorSimulator] = sim_type()
        self.sim.set_options(**config)

if __name__ == '__main__':
    env = CHSHEnv({})
    env.reset()
    done = False
    while not done:
        obs, reward, done, *_ = env.step(np.random.normal())
        print(reward)