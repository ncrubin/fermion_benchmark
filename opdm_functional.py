import numpy as np
import cirq

import util
from circuits import (build_rotation_circuits_virtual_swaps,
                      circuits_with_measurements, align_givens_circuit)
import analysis


class RDMCollector:

    def __init__(self, sampler, num_samples, qubits, compiler):
        self.sampler = sampler
        self.num_samples = num_samples
        self.qubits = qubits
        self.compiler = compiler

    def calculate_data(self, unitary, initial_circuit, num_excitations):
        bare_circuits = build_rotation_circuits_virtual_swaps(
            unitary=unitary, qubits=self.qubits,
            initial_circuit=cirq.Circuit())
        circuit_dict = circuits_with_measurements(self.qubits, bare_circuits)

        # 4. Take data
        data_dict = {
            'z': {},
            'xy_even': {},
            'xy_odd': {},
            'qubits': [f'({q.row}, {q.col})' for q in self.qubits],
            'qubit_permutations': util.generate_permutations(len(self.qubits)),
            'circuits': bare_circuits,
            'circuits_with_measurement': circuit_dict
        }
        ps_data_dict = {
            'z': {},
            'xy_even': {},
            'xy_odd': {},
            'qubits': [f'({q.row}, {q.col})' for q in self.qubits],
            'qubit_permutations': util.generate_permutations(len(self.qubits)),
            'circuits': bare_circuits,
            'circuits_with_measurement': circuit_dict
        }
        circuits_to_run_dict = {}
        circuit_list = []
        for measure_type in circuit_dict.keys():
            circuits = circuit_dict[measure_type]
            for circuit_index in circuits.keys():
                circuit = circuits[circuit_index]
                if measure_type == 'z' and circuit_index != 0:
                    continue
                # This is where we take the data
                circuit_to_run = initial_circuit + align_givens_circuit(circuit)
                circuits_to_run_dict[(measure_type, circuit_index)] = circuit_to_run
                circuit_list.append(circuit_to_run)
                print("circuit measure_type ", measure_type)
                print("permutation ", circuit_index)
                print(circuit_to_run.to_text_diagram(transpose=True,
                                                     qubit_order=self.qubits))
                print()

        results = self.sampler.run_batch(
            circuit_list, repetitions=self.num_samples,
            params_list=[cirq.ParamResolver()] * len(circuit_list))
        res_cnt = 0
        for measure_type in circuit_dict.keys():
            circuits = circuit_dict[measure_type]
            for circuit_index in circuits.keys():
                circuit = circuits[circuit_index]
                if measure_type == 'z' and circuit_index != 0:
                    continue
                # data = results[res_cnt]
                data = self.sampler.run(circuit, repetitions=self.num_samples)
                data_dict[measure_type][circuit_index] = data.data.copy()
                # PostSelect the data
                good_indices = \
                    np.where(np.sum(np.array(data.data), axis=1) ==
                             num_excitations)[0]
                good_data = data.data[data.data.index.isin(good_indices)]
                ps_data_dict[measure_type][circuit_index] = good_data
                res_cnt += 1

        return data_dict, ps_data_dict

    def calculate_rdm(self, unitary, initial_circuit, num_excitations):
        data_dict, ps_data_dict = self.calculate_data(unitary, initial_circuit,
                                                      num_excitations)
        raw_opdm, raw_var = analysis.compute_opdm(data_dict,
                                                  return_variance=True)
        ps_opdm, ps_var = analysis.compute_opdm(ps_data_dict,
                                                return_variance=True)
        return raw_opdm, raw_var, ps_opdm, ps_var
