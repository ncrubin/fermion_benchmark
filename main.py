"""
Main interface to running the Givens rotation benchmark.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
from numpy.random import random

import matplotlib.pyplot as plt
from scipy.linalg import expm
import cirq
import cirq.google as cg

import analysis
from circuits import optimal_givens_decomposition
from opdm_functional import RDMCollector


def fidelity_measurement(qubits: List[cirq.Qid], sampler: cirq.Sampler,
                         num_samples: int) -> Dict:
    # this is ~ half filling checkboard pattern
    # lower filling fractions can be added.
    # at half filling this will provide the worst case performance
    # of the Givens rotation
    num_qubits = len(qubits)
    filling_pattern = np.array(
        [0, 1] * (num_qubits // 2) + [0] * (num_qubits % 2))
    num_excitations = sum(filling_pattern)
    initial_circuit = cirq.Circuit(
        [cirq.X.on(qubits[ii]) for ii in np.where(filling_pattern == 1)[0]])
    initial_opdm = np.diag(filling_pattern)

    # 2. generate random real unitary
    generator = random((num_qubits, num_qubits))
    generator = generator - generator.T
    unitary = expm(generator)

    # simple checks to make sure unitary is gennerate properly
    assert np.allclose(1j * generator, (1j * generator).conj().T)
    assert np.allclose(unitary.conj().T @ unitary, np.eye(num_qubits))

    # 3. Build circuits for opdm measurement
    # NOTE: ij_rotations = (Qubit_i, Qubit_j, theta, phi)
    # Qubit_i and Qubit_j are nearest neighbors on the chip
    opdm_gen = RDMCollector(sampler, num_samples, qubits, cg.optimized_for_sycamore)
    raw_opdm, raw_var, ps_opdm, ps_var = opdm_gen.calculate_rdm(unitary,
                                                                initial_circuit,
                                                                num_excitations)
    true_opdm = unitary @ initial_opdm @ unitary.conj().T
    print(raw_opdm)
    print()
    print(ps_opdm)
    print()
    print(true_opdm)

    # 6. compute fidelity fidelity witness and error bars
    raw_fidelity_witnesses = []
    ps_fidelity_witnesses = []
    for _ in range(1000):
        raw_opdm_resampled = analysis.resample_opdm(raw_opdm, raw_var)
        ps_opdm_resampled = analysis.resample_opdm(ps_opdm, ps_var)

        raw_fidelity_witnesses.append(
            analysis.fidelity_witness(target_unitary=unitary,
                                      omega=filling_pattern.tolist(),
                                      measured_opdm=raw_opdm_resampled))
        ps_fidelity_witnesses.append(
            analysis.fidelity_witness(target_unitary=unitary,
                                      omega=filling_pattern.tolist(),
                                      measured_opdm=ps_opdm_resampled))

    raw_fidelity_witness = np.mean(raw_fidelity_witnesses)
    raw_fidelity_witness_err = np.std(raw_fidelity_witnesses, ddof=1)

    ps_fidelity_witness = np.mean(ps_fidelity_witnesses)
    ps_fidelity_witness_err = np.std(ps_fidelity_witnesses, ddof=1)

    results = {'raw_fidelity_witness': raw_fidelity_witness,
               'raw_fidelity_witness_err': raw_fidelity_witness_err,
               'ps_fidelity_witness': ps_fidelity_witness,
               'ps_fidelity_witness_err': ps_fidelity_witness_err}
    return results


def time_evolve(qubits: List[cirq.Qid], sampler: cirq.Sampler,
                num_samples: int = 5000):
    dim = len(qubits)
    hamiltonain_matrix = np.diag([-1] * (dim - 1), k=1)
    hamiltonain_matrix = hamiltonain_matrix + hamiltonain_matrix.T

    # set initial occupations we'll have 1 excitation at 0 and 1 excitation at dim // 2
    occ = [0] * dim
    occ[0] = 1
    occ[dim // 2] = 1
    initial_opdm = np.diag(occ)
    print(initial_opdm.diagonal())
    print(qubits)

    # set up time evolution and get exact answer by evolving the 1-RDM
    num_circuits = 100
    evolve_times = np.linspace(0, 20, num_circuits)
    evolve_unitaries = [expm(-1j * tt * hamiltonain_matrix) for tt in
                        evolve_times]
    evolved_rdms = [u @ initial_opdm @ u.conj().T for u in evolve_unitaries]
    onsite_charge = [np.diagonal(rdm) for rdm in evolved_rdms]
    onsite_charge_mat = np.vstack(onsite_charge).real

    # Now construct all the circuits we want to run as a batch
    print("Constructing circuits")
    circuits = []
    for idx, uu in enumerate(evolve_unitaries):
        circuit = cirq.Circuit(
            optimal_givens_decomposition(qubits, uu.T.copy()))
        measurement_moment = cirq.Moment(
            [cirq.measure(x, key=repr(x)) for x in qubits])
        initial_moment = cirq.Moment(
            [cirq.X.on(qubits[0]), cirq.X.on(qubits[dim // 2])])

        circuit = cirq.Circuit(initial_moment) + cg.optimized_for_sycamore(
            circuit)
        circuit += measurement_moment
        circuits.append(circuit)

    print("Running batches")
    results = sampler.run_batch(circuits,
                                repetitions=num_samples,
                                params_list=[cirq.ParamResolver()] * num_circuits)

    # collect the onsite expectation values
    occs = []
    for res in results:
        # add post-selection here!
        occs.append(res[0].data.mean(axis=0))

    # get the onsite expectation ordered properly for plotting
    onsite_charge_mat_syc = pd.concat(occs, axis=1).transpose()
    onsite_charge_mat_syc = np.hstack(
        [onsite_charge_mat_syc[repr(qq)].to_numpy().reshape((-1, 1)) for qq in
         qubits])

    # plot!
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(onsite_charge_mat, aspect='auto')
    ax[0].set_title('Theory')
    ax[1].imshow(onsite_charge_mat_syc, aspect='auto')
    ax[1].set_title('Rainbow')
    plt.savefig("{}_qubit_rainbow.png".format(dim), format="PNG", dpi=300)
    plt.show()


def get_sampler(project_id='q-engine-v1', processor='rainbow'):
    """The project id should be changed if people other than Nick Rubin 
    use this code"""
    engine = cirq.google.Engine(project_id=project_id)
    sampler = cirq.google.QuantumEngineSampler(engine=engine, processor_id=processor,
                                               gate_set=cirq.google.SQRT_ISWAP_GATESET)
    return sampler

def get_device(project_id='q-engine-v1', processor='rainbow'):
    """The project id should be changed if people other than Nick Rubin 
    use this code"""
    engine = cirq.google.Engine(project_id=project_id,
                                proto_version=cirq.google.ProtoVersion.V2)

    engine_proc_rainbow = engine.get_processor(processor)
    rainbow = engine_proc_rainbow.get_device([cirq.google.SQRT_ISWAP_GATESET])
    return rainbow



if __name__ == "__main__":
    num_qubits = 4
    qubits = [cirq.GridQubit(n, 5) for n in range(0, num_qubits)]
    sampler = cirq.Simulator(dtype=np.complex128)
    # sampler = get_sampler(processor='weber')
    # device = get_device(processor='weber')
    # print(device)
    print("using qubits")
    print(qubits)

    num_samples = 100_000
    results = []
    for _ in range(1):
        result = fidelity_measurement(qubits, sampler, num_samples)
        results.append(result)
    print(results)

    # time_evolve(qubits, sampler)
