import json

import numpy as np
from qiskit_ibm_runtime import Session, Estimator, QiskitRuntimeService, Options
from qiskit.quantum_info import Operator, Statevector

from nonlocalgames.hamiltonians import G14
from nonlocalgames.circuit import NLGCircuit, load_adapt_ansatz
from nonlocalgames.measurement import MeasurementLayer

def get_pcc() -> Operator:
    proj = 0
    for c in range(4):
        s = f'{c:02b}'[::-1]
        s = s + s
        cc = Statevector.from_label(s)
        pcc = cc.to_operator()
        proj += pcc
    
    return pcc

def load_g14_circuit(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qubits = 2
    players = 2
    qc = load_adapt_ansatz(
        data['state'],
        '++++',
        [qubits] * players,
        adapt_order=False,
        min_theta = 1e-4)

    # Don't need to transform phi since qinfo's Ry gate follows the qiskit convention
    phi = np.array(data['phi'])

    constrained = False
    if 'constrained' in path:
        constrained = True

    if 'metadata' in data:
        layer = data['metadata'].get('layer', 'ry')
        constrained = data['metadata'].get('constrain_phi', constrained)
    else:
        layer = 'ry'
    
    ml = MeasurementLayer.get(layer, players=2, questions=14, qubits=2)
    
    if constrained:
        if phi.ndim == 1:
            phi = np.concatenate([phi, phi]).reshape(ml.shape)
        else:
            phi = np.concatenate([phi, phi])
            assert phi.shape == ml.shape
    
    ml.params = phi

    if constrained:
        ml.conj(1)

    return qc, ml

def main():
    qc, ml = load_g14_circuit('../../data/g14_constrained_u3ry/g14_state.json')
    nlg = NLGCircuit(qc, measurement_layer=ml)

    options = Options()
    options.execution.shots = 1024
    options.optimization_level = 0  # No optimization
    options.resilience_level = 2  # ZNE

    service = QiskitRuntimeService(name='full')
    with Session(service=service, backend='ibm_hanoi') as session:
        # Prepare questions
        vertices = [(v, v) for v in range(14)]
        edges = G14._get_graph().edge_links.tolist()
        questions = vertices + edges

        estimator = Estimator(session=session, options=options)
        eval_qc = nlg._prepare_question((0, 0))
        pcc = get_pcc()
        job = estimator.run(circuits=[eval_qc], observables=[pcc])
        print('Job:', job.job_id())
        result = job.result()
        
        with open('zne_results_optim{optim}_zne{zne}.json'.format(
            optim=options.optimization_level,
            zne=options.resilience_level
        ))


if __name__ == '__main__':
    main()