import timeit
from alternative_phase_1 import create_bus_admittance_matrix as function1
from create_bus_admittance_matrix import create_bus_admittance_matrix as function2

filename = "ieee300bus.json"
num_runs = 150

# Measure the total execution time for the specified number of runs
total_time1 = timeit.timeit('function1(filename)', globals=globals(), number=num_runs)
average_time1 = total_time1 / num_runs
print(f"First Function Average execution time over {num_runs} runs: {average_time1:.8f} seconds")

total_time2 = timeit.timeit('function2(filename)', globals=globals(), number=num_runs)
average_time2 = total_time2 / num_runs
print(f"Second Function Average execution time over {num_runs} runs: {average_time2:.8f} seconds")

