class Result:

    def __init__(self, name):
        self.name = name

    def result(self):
        pass

    def __repr__(self):
        return "Result"
#    Result(
#            backend_name='qasm_simulator',
#            backend_version='0.12.0',
#            qobj_id='',
#            job_id='adf40e66-2c70-4ddd-8f7b-063d5972f648',
#            success=True,
#            results=[
#                ExperimentResult(shots=1024,
#                                      success=True,
#                                      meas_level=2,
#                                      data=ExperimentResultData(),
#                                      header=QobjExperimentHeader(creg_sizes=[],
#                                                                  global_phase=0.0,
#                                                                  memory_slots=0,
#                                                                  n_qubits=4,
#                                                                  name='circuit-88',
#                                                                  qreg_sizes=[['q',
#                                                                               4]]),
#              status=DONE,
#              seed_simulator=3797578995,
#              metadata={'batched_shots_optimization': False,
#                        'measure_sampling': False,
#                        'parallel_shots': 1,
#                        'remapped_qubits': False,
#                        'active_input_qubits': [],
#                        'num_clbits': 0,
#                        'parallel_state_update': 32,
#                        'num_qubits': 0,
#                        'device': 'CPU',
#                        'input_qubit_map': [],
#                        'method': 'stabilizer'},
#              time_taken=2.5577e-05)
#                ],
#            date=2023-04-10T08:00:24.884872,
#            status=COMPLETED,
#            header=None,
#            metadata={'time_taken_execute': 5.6509e-05,
#                      'mpi_rank': 0,
#                      'num_mpi_processes': 1,
#                      'max_gpu_memory_mb': 0,
#                      'max_memory_mb': 112763,
#                      'parallel_experiments': 1,
#                      'num_processes_per_experiments': 1,
#                      'omp_enabled': True},
#            time_taken=0.0005645751953125)  
