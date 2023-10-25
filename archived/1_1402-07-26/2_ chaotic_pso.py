import numpy as np
import matplotlib.pyplot as plt

# Define the benchmark function
def benchmark_function(x):
    return np.sin(x) + np.cos(0.5 * x)

# Define the PSO algorithm with chaotic parameter adaptation
def chaotic_pso(benchmark_func, num_particles, num_iterations):
    # Define the chaotic map function
    def chaotic_map(x, a):
        return (4 * a * x * (1 - x))  # Logistic map equation

    # Initialize particles' positions and velocities
    positions = np.random.uniform(low=-10, high=10, size=(num_particles,))
    velocities = np.random.uniform(low=-1, high=1, size=(num_particles,))
    
    # Initialize global best position and corresponding value
    global_best_pos = positions[0]
    global_best_val = benchmark_func(global_best_pos)
    
    # Initialize lists to store best positions and values at each iteration
    best_positions = [global_best_pos]
    best_values = [global_best_val]
    
    # PSO main loop
    for _ in range(num_iterations):
        a = np.random.uniform(low=2.9, high=4.0)  # Chaotic map parameter
        
        for i in range(num_particles):
            # Update particle's velocity using chaotic map
            velocities[i] = velocities[i] + 2 * chaotic_map(best_positions[-1] - positions[i], a) \
                            + 2 * chaotic_map(global_best_pos - positions[i], a)
            
            # Update particle's position
            positions[i] = positions[i] + velocities[i]
            
            # Update global best position and value
            particle_val = benchmark_func(positions[i])
            if particle_val < global_best_val:
                global_best_pos = positions[i]
                global_best_val = particle_val
        
        # Store best position and value at each iteration
        best_positions.append(global_best_pos)
        best_values.append(global_best_val)
    
    return best_positions, best_values

# Example usage
num_particles = 20
num_iterations = 50

best_positions, best_values = chaotic_pso(benchmark_function, num_particles, num_iterations)

# Print the best solutions
print("Best Solutions:")
for i, pos in enumerate(best_positions):
    print(f"Iteration {i+1}: {pos}")

# Plotting the optimization process
x = np.linspace(-10, 10, 100)
y_benchmark = benchmark_function(x)

plt.plot(x, y_benchmark, label="Benchmark Function")
plt.scatter(best_positions, best_values, c='r', label="Particle Best")
plt.xlabel("Position")
plt.ylabel("Objective Value")
plt.title("Chaotic PSO")
plt.legend()
plt.show()