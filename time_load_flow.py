import time
from load_flow import load_flow

# Function to time the execution
def time_function(filename, maxIterations, tolerance, runs=100):
    total_time = 0  # Variable to accumulate total time

    # Run the function multiple times
    for _ in range(runs):
        # Start the timer
        start_time = time.time()

        # Call the function
        a, b, c, d = load_flow(filename, maxIterations=maxIterations, tolerance=tolerance)

        # Stop the timer
        end_time = time.time()

        # Accumulate elapsed time
        total_time += (end_time - start_time)

    # Calculate average time
    average_time = total_time / runs

    # Print the result
    print(f"{filename}, {runs}, {average_time:.4f}, {tolerance}")

# Parameters
initial_tolerance = 10
tolerance_log_stepsize = 10
max_tolerance_runs = 10
maxIterations = 100
filename = "ieee57bus.json"

print("Filename, Number of Runs, Average Execution Time(s), Tolerance")
# Run the timing function with different tolerances
for i in range(max_tolerance_runs):
    tolerance = initial_tolerance / (tolerance_log_stepsize**i)
    time_function(filename, maxIterations, tolerance, runs=100)

