import jax
import jax.numpy as jnp
import jax.random
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--matrix",
    help="Size of the matrix to be used in the tests",
    default=4096,
    type=int
)

parser.add_argument(
    "--tests",
    help="Number of tests to run per operertion",
    default=50,
    type=int
)

parser.add_argument(
    "--gpu_cost",
    help="Cost per GPU per hour",
    default="0.56",
    type=float
)

parser.add_argument(
    "--machine_cost",
    help="Cost per machine per hour. USED ONLY WITH GPU.",
    default=3.57,
    type=float
)

parser.add_argument(
    "--tpu_cost",
    help="Cost per TPU chip per hour",
    default=1.20,
    type=float
)

args = parser.parse_args()

# ==========================================================
#                      TEST CONFIGURATION
# ==========================================================
MATRIX_SIZE = args.matrix
NUM_TESTS = args.tests

COST_PER_GPU = args.gpu_cost # per GPU per hour
COST_MACHINE = args.machine_cost # GPU only. Associated machine per hour  
COST_PER_TPU = args.tpu_cost # per chip per hour 
# ==========================================================

# Functions to represent the operation to be executed
def simple_addition(a, b):
    return a + b

def matrix_multiplication(m1, m2):
    return jnp.matmul(m1, m2)

def complex_neural_op(m1, m2, m3):
    temp = jnp.matmul(m1, m2)
    temp = temp + m3
    output = jax.nn.relu(temp)
    return output

# Execution and Time Measurement Function with Multiple Tests
def run_and_time_jax_jitted(operation_name, operation_func, num_tests, *args):
    """
    Executes a JAX compiled operation (JIT and pmap) N times and measures the times,
    calculating the average, standard deviation, minimum, and maximum.
    Args:
        operation_name (str): Name of the operation being executed.
        operation_func (callable): Function to be executed.
    Returns:
        Dictionary with the statistics.
    """

    print(f"\nExecuting '{operation_name}' on {jax.default_backend()} with {num_devices} devices...")
    print(f"Performing {num_tests} tests...")

    # JAX's 'pmap' (parallel map) transforms the function for SPMD (Single Program, Multiple Data) execution.
    # 'in_axes=(0,)' * len(args) specifies that the first dimension (axis 0) of each input tensor
    # should be split and distributed across all available devices.
    # 'out_axes=0' specifies that the results from each device should be concatenated back
    # along the first dimension, yielding a single output tensor containing results from all devices.
    jitted_op = jax.jit(operation_func)

    # Warm-up (executes a few times to ensure JIT/pmap compilation occurs before statistics)
    print("Starting warm-up...")
    for _ in range(10):
        _ = jitted_op(*args).block_until_ready()
    print("Warm-up complete.")
    print()

    execution_times = []
    # Repeat the tests in order to get average execution time  
    for i in range(num_tests):
        start_time = time.time()

        # _ = pmapped_op(*args)
        # In non-benchmark applications, '.block_until_ready()' (or similar explicit synchronizations)
        # is typically removed to allow asynchronous execution. This maximizes overall throughput
        # by letting the CPU prepare the next tasks while the accelerator is busy.
        # Synchronization will happen implicitly only when the result is actually needed (e.g., fetching to CPU).
        _ = jitted_op(*args).block_until_ready()

        end_time = time.time()
        execution_times.append(end_time - start_time)
        print(f"\rTest {i+1}/{num_tests}: {execution_times[-1]*1000:.6f} ms", end="")
    print()

    print()

    operation_time = np.sum(execution_times)
    cost = operation_time * cost_per_second #cost calculated according to estimation. Not actual cost of the number of tests

    return {
        "operation_name": operation_name,
        "operation_time": operation_time,
        "cost": cost,
    }


# Automatic Configuration
print(f"JAX backend: {jax.default_backend()}")
devices = jax.devices()

# TPU chips or individuals GPUs
print(f"Available devices:")
for i in range(len(devices)):
    print(f"Device {i}: {devices[i]}")
    
num_devices = len(devices)
print()
print(f"Number of detected devices: {num_devices}")

if num_devices == 0:
    raise RuntimeError("ERROR: No JAX device (TPU/GPU) detected. Ensure the environment is configured correctly.")

if jax.default_backend() == 'tpu':
    cost_per_second = (num_devices * COST_PER_TPU)/3600
else:
    cost_per_second = (num_devices * COST_PER_GPU + COST_MACHINE)/3600


# Tests preparation
matrix_size = MATRIX_SIZE
num_benchmark_tests = NUM_TESTS

print(f"\nUsing matrix size: {matrix_size}x{matrix_size}")
print(f"Number of benchmark tests per operation: {num_benchmark_tests}")

batch_matrix_shape = (num_devices, matrix_size, matrix_size)

key = jax.random.PRNGKey(0)
key_A, key_B, key_C = jax.random.split(key, 3)

tensor_A = jax.random.normal(key_A, batch_matrix_shape, dtype=jnp.float32)
tensor_B = jax.random.normal(key_B, batch_matrix_shape, dtype=jnp.float32)
tensor_C = jax.random.normal(key_C, batch_matrix_shape, dtype=jnp.float32)

# Test Execution
print("\n--- Starting Multiple Tests ---")

t_start = time.time()

all_benchmark_results = [] # List to store results for each operation

sum_stats = run_and_time_jax_jitted("Tensor Addition", simple_addition, num_benchmark_tests, tensor_A, tensor_B)
all_benchmark_results.append(sum_stats)

matmul_stats = run_and_time_jax_jitted("Matrix Multiplication", matrix_multiplication, num_benchmark_tests, tensor_A, tensor_B)
all_benchmark_results.append(matmul_stats)

complex_op_stats = run_and_time_jax_jitted("Neural Simulation", complex_neural_op, num_benchmark_tests, tensor_A, tensor_B, tensor_C)
all_benchmark_results.append(complex_op_stats)

t_final = time.time()

total_time = t_final - t_start

print("\n--- Tests Completed ---")

print("\n" + "="*60)
print("BENCHMARK SUMMARY")
print("="*60)

header = f"{'Operation':<30} | {'Time (seconds)'} | {'Cost (USD)'}"
print(header)
for stats in all_benchmark_results:
    print(f"{stats['operation_name']:<30} | {stats['operation_time']:06.2f} \t| {stats['cost']:.5f}")

print("="*60)
print(f"Total Time w/ Warm-ups: {total_time:.2f} seconds | Cost (USD): {total_time * cost_per_second:.4f}")
print("="*60)
print(f"JAX backend: {jax.default_backend()} | Devices: {num_devices}")
print(f"Matrix size: {matrix_size}x{matrix_size} | Tests: {num_benchmark_tests}")
print("="*60 + "\n")
