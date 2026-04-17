import time
import RootFinding as rf

def benchmark():
    temp_min, temp_max = 220, 320
    steps = int(temp_max-temp_min)+1
    
    # Benchmark Explicit
    start = time.perf_counter()
    _ = rf.analyze_convergence_range(temp_min, temp_max, steps, 'newton', 'explicit', damping=True)
    end = time.perf_counter()
    time_explicit = end - start
    
    # Benchmark FD
    start = time.perf_counter()
    _ = rf.analyze_convergence_range(temp_min, temp_max, steps, 'newton', 'fd', damping=True)
    end = time.perf_counter()
    time_fd = end - start
    
    print(f"\n--- Performance Results ({steps} runs) ---")
    print(f"Explicit Jacobian: {time_explicit:.4f} seconds")
    print(f"FD Jacobian:       {time_fd:.4f} seconds")
    print(f"Speedup Factor:    {time_fd / time_explicit:.2f}x")

if __name__ == "__main__":
    benchmark()