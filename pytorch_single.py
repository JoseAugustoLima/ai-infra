import torch
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


# Try to import torch_xla for TPU detection
try:
    import torch_xla.core.xla_model as xm
    # import torch_xla.distributed.xla_multiprocessing as xmp # Not used for this single-device benchmark
    use_xla = True
except ImportError:
    use_xla = False

# Device Configuration ---
def get_pytorch_device():
    """Detects and configures the best available PyTorch device."""
    device = torch.device('cpu')
    accelerator_type = 'cpu' #fallback
    xm_module = None # To return the xm module if it's a TPU
    num_devices_detected = 1 # Default for CPU

    # Try TPU
    if use_xla:
        try:
            device = xm.xla_device() # Attempts to allocate the XLA device
            accelerator_type = 'tpu'
            xm_module = xm
            # If xm.xla_device() succeeded, we can now count the supported devices
            num_devices_detected = len(xm.get_xla_supported_devices()) 
            print(f"Detected and configured for TPU: {device} ({num_devices_detected} TPUs available)")
            return device, accelerator_type, num_devices_detected, xm_module
        except RuntimeError as e:
            # Catches the error if xm.xla_device() cannot find a TPU
            print(f"INFO: Could not configure TPU via xm.xla_device(): {e}")
            pass # Continues to the next attempt (GPU)
    
    # Try GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        accelerator_type = 'gpu'
        num_devices_detected = torch.cuda.device_count() # Actual number of GPUs in the system
        print(f"Detected and configured for GPU: {device} ({num_devices_detected} GPUs available)")
        return device, accelerator_type, num_devices_detected, None
    
    raise RuntimeError("ERROR: No TPU/GPU detected. Ensure the environment is configured correctly.")

# Execution and Time Measurement Function with Multiple Tests ---
def run_and_time_pytorch(operation_name, operation_func, num_tests, *args):
    """
    Executes a JAX compiled operation (JIT and pmap) N times and measures the times,
    calculating the average, standard deviation, minimum, and maximum.
    Args:
        operation_name (str): Name of the operation being executed.
        operation_func (callable): Function to be executed.
    Returns:
        Dictionary with the statistics.
    """

    print(f"\nExecuting '{operation_name}' on {accelerator_type}...")
    print(f"Performing {num_tests} tests...")

    # Move tensors to the device before warm-up and tests
    casted_args = [arg.to(device) for arg in args]

    # Warm-up
    print("Starting warm-up...")
    for _ in range(10):
        _ = operation_func(*casted_args)
        if accelerator_type == 'tpu' and xm_module:
            xm_module.mark_step() # Synchronizes TPU
        elif accelerator_type == 'gpu':
            torch.cuda.synchronize(device) # Synchronizes GPU
    print("Warm-up complete.")
    print()

    execution_times = []
    for i in range(num_tests):
        start_time = time.time()
        _ = operation_func(*casted_args)
        if accelerator_type == 'tpu' and xm_module:
            xm_module.mark_step() # Synchronizes TPU
        elif accelerator_type == 'gpu':
            torch.cuda.synchronize(device) # Synchronizes GPU
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


# Operations and Tensor Size Definition ---

matrix_size = MATRIX_SIZE
num_benchmark_tests = NUM_TESTS

# Get the device, accelerator type, total number of detected devices, and xm module (if TPU)
device, accelerator_type, num_devices_detected_total, xm_module = get_pytorch_device()

print(f"PyTorch backend: {accelerator_type}")
print(f"This script will run on a single device, processing a batch of size {num_devices_detected_total})")

if accelerator_type == 'tpu':
    cost_per_second = COST_PER_TPU/3600
else:
    cost_per_second = (COST_PER_GPU + COST_MACHINE)/3600

# The batch_matrix_shape uses the total number of detected devices,
# even if the script runs on a single core for PyTorch.
# This ensures that the total workload (total processed elements)
# per test is the same as the JAX benchmark.
batch_matrix_shape = (num_devices_detected_total, matrix_size, matrix_size)


print(f"\nUsing matrix size: {matrix_size}x{matrix_size}")
print(f"Number of benchmark tests per operation: {num_benchmark_tests}")
print(f"Input tensor shape: {batch_matrix_shape}")

tensor_A = torch.randn(batch_matrix_shape, dtype=torch.float32)
tensor_B = torch.randn(batch_matrix_shape, dtype=torch.float32)
tensor_C = torch.randn(batch_matrix_shape, dtype=torch.float32)

# Functions to be executed (operate on PyTorch tensors)
def simple_addition_pytorch(a, b):
    return torch.add(a, b)

def matrix_multiplication_pytorch(m1, m2):
    return torch.matmul(m1, m2)

def complex_neural_op_pytorch(m1, m2, m3):
    temp = torch.matmul(m1, m2)
    temp = torch.add(temp, m3) # Use explicit torch.add
    output = torch.relu(temp)
    return output

# --- 4. Test Execution and Statistics Collection ---
print("\n--- Starting Multiple PyTorch Tests ---")

t_start = time.time()
all_benchmark_results = []

sum_stats = run_and_time_pytorch("Tensor Addition", simple_addition_pytorch, num_benchmark_tests, tensor_A, tensor_B)
all_benchmark_results.append(sum_stats)

matmul_stats = run_and_time_pytorch("Matrix Multiplication", matrix_multiplication_pytorch, num_benchmark_tests, tensor_A, tensor_B)
all_benchmark_results.append(matmul_stats)

complex_op_stats = run_and_time_pytorch("Neural Simulation", complex_neural_op_pytorch, num_benchmark_tests, tensor_A, tensor_B, tensor_C)
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
print(f"Total Time w/ Warm-ups: {total_time:.2f} seconds | Cost (USD): {total_time * cost_per_second:.6f}")
print("="*60)
print(f"PyTorch backend: {accelerator_type} | Devices: 1 | Batch Size: {num_devices_detected_total}")
print(f"Matrix size: {matrix_size}x{matrix_size} | Tests: {num_benchmark_tests}")
print("="*60 + "\n")
