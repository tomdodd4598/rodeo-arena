import doddq.helpers as helpers
import doddq.rodeo as rodeo

import matplotlib.pyplot as pyplot
import numpy as np
import re
import scipy.signal as signal

from qiskit import IBMQ
from qiskit.providers import Backend
from qiskit.providers.aer import AerSimulator
from typing import Any, Callable


def rodeo_run(
        eval_repeats: int,
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        target_energy: float,
        time_stddev: float,
        qsp_eigvec: np.ndarray,
        run_id: int,
        energy_sample_num: int,
        backend: Backend,
        shots: int
) -> (float, float):
    zeros, overlap, successes = 0.0, 0.0, 0
    qsp = qsp_eigvec is not None
    all_zero = '0' * num_cycles

    print(f'\r[{1 + run_id}/{energy_sample_num}] Rodeo run targeting energy {target_energy:0.05f}...', end='')

    if qsp:
        swap_zero = '0 ' + all_zero
        swap_one = '1 ' + all_zero

        for _ in range(eval_repeats):
            circuit = rodeo.rodeo_qsp(
                num_cycles,
                hamiltonian,
                initial_state,
                qsp_eigvec,
                target_energy,
                helpers.rand_gaussian_array(num_cycles, time_stddev),
            )
            result = helpers.default_job(circuit, backend, shots, False)
            counts = result.get_counts(circuit)
            success_zero = counts.get(swap_zero, 0)
            success_either = success_zero + counts.get(swap_one, 0)
            successes += success_either
            overlap += 2.0 * success_zero - success_either
    else:
        for _ in range(eval_repeats):
            circuit = rodeo.rodeo_qpe(
                num_cycles,
                hamiltonian,
                initial_state,
                target_energy,
                helpers.rand_gaussian_array(num_cycles, time_stddev),
            )
            result = helpers.default_job(circuit, backend, shots, False)
            counts = result.get_counts(circuit)
            zeros += counts.get(all_zero, 0)

    return zeros / (eval_repeats * shots), 0.0 if successes == 0 else overlap / successes


def rodeo_sequence(
        eval_repeats: int,
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        time_stddev_generator: Callable[[int], float],
        qsp_eigvec: np.ndarray,
        min_energy: float,
        max_energy: float,
        energy_sample_num: int,
        next_targets: Callable[[int, float, float, [float], [float]], Any],
        narrowing_factor: float,
        max_iterations: int,
        backend: Backend,
        shots: int
) -> None:
    qsp = qsp_eigvec is not None

    def rodeo_internal(min_e: float, max_e: float, iteration: int) -> None:
        mean_e = (min_e + max_e) / 2.0
        energy_arr = np.linspace(min_e, max_e, energy_sample_num)
        iteration += 1

        time_stddev = time_stddev_generator(iteration)

        print()
        run_arr = helpers.mp_starmap(
            rodeo_run,
            [(
                eval_repeats,
                num_cycles,
                hamiltonian,
                initial_state,
                energy,
                time_stddev,
                qsp_eigvec,
                run_id,
                energy_sample_num,
                backend,
                shots
            ) for run_id, energy in enumerate(energy_arr)],
            chunksize=1
        )

        print()
        # print(f'{iteration} ({mean_e}): {run_arr}')

        targets = next_targets(iteration, min_e, max_e, [x[0] for x in run_arr], [x[1] for x in run_arr])

        if iteration < max_iterations:
            range_offset = (max_e - min_e) / (2.0 * (narrowing_factor ** iteration))
            for target in targets:
                target_energy = target if qsp else energy_arr[target]
                rodeo_internal(target_energy - range_offset, target_energy + range_offset, iteration)
        else:
            if qsp:
                print(f'Mean overlap: {np.mean([x[1] for x in run_arr])}')
            else:
                num_estimations = len(targets)
                if num_estimations == 1:
                    print(f'Eigenvalue estimation: {energy_arr[targets[0]]}')
                elif num_estimations > 0:
                    def energy_str(x): return f'{x:0.05f}'
                    print(f'Eigenvalue estimations: {[energy_str(energy_arr[target]) for target in targets]}')
                else:
                    print(f'No eigenvalue found near {mean_e}!')

    rodeo_internal(min_energy, max_energy, 0)


def write_mathematica_file(file_name: str, x_values: Any, y_values: Any) -> None:
    with open(f'{file_name}.txt', 'w') as fp:
        fp.write('{')
        fp.write(','.join(f'{{{x_value},{y_value}}}' for x_value, y_value in zip(x_values, y_values)))
        fp.write('}')


def rodeo_zeros_plot(
        iteration: int,
        min_e: float,
        max_e: float,
        energy_sample_num: int,
        zeros_arr: [float],
        targets: np.ndarray
) -> None:
    x_values = np.linspace(min_e, max_e, energy_sample_num)
    y_values = zeros_arr
    helpers.list_histogram(
        x_values,
        y_values,
        f'Rodeo All Zero: iteration {iteration}, range [{min_e:.2f}, {max_e:.2f}]',
        'Bin Index',
        'Success Rate'
    )
    file_name = f'rodeo_qpe_scan_{iteration}_{helpers.list_underscore_str(targets)}'
    pyplot.savefig(f'{file_name}.svg')
    write_mathematica_file(file_name, x_values, y_values)


def rodeo_overlap_plot(
        iteration: int,
        min_e: float,
        max_e: float,
        energy_sample_num: int,
        overlap_arr: [float],
        target_energy: float
) -> None:
    x_values = np.linspace(min_e, max_e, energy_sample_num)
    y_values = overlap_arr
    helpers.list_histogram(
        x_values,
        y_values,
        f'Rodeo Target Overlaps: iteration {iteration}, range [{min_e:.2f}, {max_e:.2f}]',
        'Bin Index',
        'Overlap'
    )
    suffix = f'{target_energy:.2f}'.replace('.', '_')
    file_name = f'rodeo_qsp_scan_{iteration}_{suffix}'
    pyplot.savefig(f'{file_name}.svg')
    write_mathematica_file(file_name, x_values, y_values)


def rodeo_qpe_auto(
        eval_repeats: int,
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        time_stddev_initial: float,
        min_energy: float,
        max_energy: float,
        energy_sample_num: int,
        narrowing_factor: float,
        max_iterations: int,
        min_targets: int,
        backend: Backend,
        shots: int
) -> None:
    def next_targets(iteration: int, min_e: float, max_e: float, zeros_arr: [float], _: [float]) -> Any:
        targets = []
        width = (2 * energy_sample_num) // 3
        target_count = min_targets if iteration == 1 else 1
        while (len(targets) < target_count) and width > 0:
            targets, _ = signal.find_peaks(zeros_arr, width=width)
            width -= 1
        if len(targets) > target_count:
            targets = helpers.index_nlargest(target_count, targets, zeros_arr)

        str_end = 'next target(s)...' if iteration < max_iterations else 'estimation(s)...'
        print(f'[{iteration}/{max_iterations}] Selected {len(targets)} {str_end}')
        rodeo_zeros_plot(iteration, min_e, max_e, energy_sample_num, zeros_arr, targets)
        return targets

    rodeo_sequence(
        eval_repeats,
        num_cycles,
        hamiltonian,
        initial_state,
        lambda iteration: time_stddev_initial * (narrowing_factor ** (iteration - 1)),
        None,
        min_energy,
        max_energy,
        energy_sample_num,
        next_targets,
        narrowing_factor,
        max_iterations,
        backend,
        shots
    )


def rodeo_qpe_manual(
        eval_repeats: int,
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        min_energy: float,
        max_energy: float,
        energy_sample_num: int,
        narrowing_factor: float,
        max_iterations: int,
        backend: Backend,
        shots: int
) -> None:
    def next_targets(iteration: int, min_e: float, max_e: float, zeros_arr: [float], _: [float]) -> Any:
        targets = input(f'[{iteration}/{max_iterations}] Select array index: ')
        rodeo_zeros_plot(iteration, min_e, max_e, energy_sample_num, zeros_arr, targets)
        return [int(x) for x in re.split(r'[\s,]+', targets) if x.strip().isdigit()]

    rodeo_sequence(
        eval_repeats,
        num_cycles,
        hamiltonian,
        initial_state,
        lambda iteration: float(input(f'[{iteration}/{max_iterations}] Select standard deviation: ')),
        None,
        min_energy,
        max_energy,
        energy_sample_num,
        next_targets,
        narrowing_factor,
        max_iterations,
        backend,
        shots
    )


def rodeo_qsp_auto(
        eval_repeats: int,
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        time_stddev_initial: float,
        qsp_eigval_approx: float,
        min_energy: float,
        max_energy: float,
        energy_sample_num: int,
        narrowing_factor: float,
        max_iterations: int,
        backend: Backend,
        shots
) -> None:
    eigval, eigvec = helpers.nearest_hermitian_eigentuple(hamiltonian, qsp_eigval_approx)
    print(f'Rodeo QSP targeting eigenvalue {eigval:0.05f}...')

    def next_targets(iteration: int, min_e: float, max_e: float, _: [float], overlap_arr: [float]) -> Any:
        print(f'Iteration {iteration}/{max_iterations}...')
        rodeo_overlap_plot(iteration, min_e, max_e, energy_sample_num, overlap_arr, eigval)
        return [eigval]

    rodeo_sequence(
        eval_repeats,
        num_cycles,
        hamiltonian,
        initial_state,
        lambda iteration: time_stddev_initial * (narrowing_factor ** (iteration - 1)),
        eigvec,
        min_energy,
        max_energy,
        energy_sample_num,
        next_targets,
        narrowing_factor,
        max_iterations,
        backend,
        shots
    )


def main():
    provider = IBMQ.load_account()
    backend = provider.get_backend('ibm_nairobi')
    backend_sim = AerSimulator.from_backend(backend)

    phi = 0.0
    h_0 = helpers.one_qubit_hamiltonian(-0.08496, -0.89134, 0.26536, 0.57205)
    h_1 = helpers.one_qubit_hamiltonian(-0.84537, 0.00673, -0.29354, 0.18477)
    hamiltonian = h_0 + phi * h_1

    '''hamiltonian = helpers.two_qubit_hamiltonian(
        -0.68388,
        0.00054,
        0.63435,
        0.25985,
        0.16555,
        -0.72323,
        0.48794,
        0.11261,
        0.06378,
        -0.51002,
        -0.78572,
        -0.64711,
        -0.29152,
        -0.89286,
        -0.32324,
        -0.95432
    )'''

    helpers.print_hermitian_eigensystem(hamiltonian)
    ground_energy, ground_state = helpers.ground_hermitian_eigentuple(hamiltonian)

    # rodeo_qpe_auto(50, 8, hamiltonian, None, 3.0, -4.0, 4.0, 50, 3.0, 1, 2, None, 2048)
    # rodeo_qpe_manual(50, 4, hamiltonian, None, -3.0, 3.0, 50, 3.0, 3, None, 2048)
    for i in range(1, 21):
        rodeo_qsp_auto(50 * i, i, hamiltonian, None, 10000.0, *((ground_energy,) * 3), 1, 1.0, 1, None, 2048)
    # for i in [1, 2, 5, 10, 20]:
        # rodeo_qsp_auto(50 * i, i, hamiltonian, None, 10000.0, *((ground_energy,) * 3), 1, 1.0, 1, backend_sim, 2048)

    # hamiltonian = helpers.heisenberg_chain_hamiltonian(8, 1, 3)
    # zero, one = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    # initial_state = helpers.recursive_kron(zero, one, zero, one, zero, one, zero, one)
    # helpers.print_hermitian_extremal_eigenvalues(hamiltonian)

    '''rodeo_qpe_auto(
        25,
        6,
        hamiltonian,
        initial_state,
        5.0,
        -22.0,
        32.0,
        250,
        1.0,
        1,
        1,
        None,
        2048
    )'''


if __name__ == '__main__':
    main()
