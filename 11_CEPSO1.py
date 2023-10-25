import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    return np.sum(np.square(x))

def logistic_map(x, r):
    return r * x * (1 - x)
x0 = np.random.uniform(0, 1)
r = np.random.uniform(0, 4)

def optimize_sphere(dimension, x_min, x_max, v_min, v_max, max_iterations, population_size, c1, c2):
    class Particle:
        def __init__(self):
            #self.position = np.random.uniform(x_min, 50, (population_size, dimension))
            self.position = np.zeros((population_size, dimension))
            x = x0
            for i in range(population_size):
                for j in range(dimension):
                     x = logistic_map(x, r)
                     self.position[i, j] = x
            
            #self.velocity = np.random.uniform(v_min, v_max, (population_size, dimension))
            self.velocity = np.zeros((population_size, dimension))
            x = x0
            for i in range(population_size):
                for j in range(dimension):
                     x = logistic_map(x, r)
                     self.position[i, j] = x

            self.cost = np.array([sphere(p) for p in self.position])
            self.pbest = np.copy(self.position)
            self.pbest_cost = np.copy(self.cost)
            self.gbest_idx = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.gbest_idx]
            self.gbest_cost = self.pbest_cost[self.gbest_idx]
            self.best_cost = np.zeros(max_iterations)

        def limit_velocity(self):
            self.velocity = np.clip(self.velocity, v_min, v_max)
        def limit_position(self):
            self.position = np.clip(self.position, x_min, x_max)

        def update(self, w):
            self.velocity = (w * self.velocity +
                             c1 * np.random.rand(population_size, dimension) * (self.pbest - self.position) +
                             c2 * np.random.rand(population_size, dimension) * (self.gbest - self.position))
            self.limit_velocity()
            self.position += self.velocity
            self.limit_position()
            self.cost = np.array([sphere(p) for p in self.position])
            improved_pbest = self.cost < self.pbest_cost
            self.pbest[improved_pbest] = self.position[improved_pbest]
            self.pbest_cost[improved_pbest] = self.cost[improved_pbest]
            self.gbest_idx = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.gbest_idx]
            self.gbest_cost = self.pbest_cost[self.gbest_idx]

        def optimize(self):
            for iteration in range(max_iterations):
                w = 0.9 - ((0.9 - 0.4) / max_iterations) * iteration
                self.update(w)
                self.best_cost[iteration] = self.gbest_cost

        def plot(self):
            plt.figure(figsize=(12, 16))

            # Plot fitness function
            plt.subplot(4, 2, 1)
            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(x_min, x_max, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i, j] = sphere(np.array([X[i, j], Y[i, j]]))
            levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), num=50)
            plt.contour(X, Y, Z, levels=levels, cmap='jet')
            plt.scatter(self.position[:, 0], self.position[:, 1], color='b', alpha=0.5, label='Particles')
            plt.scatter(self.gbest[0], self.gbest[1], color='r', label='Global Best')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Fitness Function')
            plt.legend()

            # Plot particle positions
            plt.subplot(4, 2, 2)
            for i in range(population_size):
                plt.scatter(self.position[i, 0], self.position[i, 1], color='b', alpha=0.5)
            plt.scatter(self.gbest[0], self.gbest[1], color='r', label='Global Best')
            plt.xlim([x_min, x_max])
            plt.ylim([x_min, x_max])
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Particle Positions')
            plt.legend()


            # Plot global best
            plt.subplot(4, 2, 3)
            plt.scatter(self.gbest[0], self.gbest[1], color='r', label='Global Best')
            plt.xlim([x_min, x_max])
            plt.ylim([x_min, x_max])
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title('Global Best')


            plt.subplot(4, 2, 4)
            plt.semilogy(self.best_cost)
            plt.ylim([10e-120, 10e20])
            plt.xlim([0, max_iterations])
            plt.ylabel('Best Function Value')
            plt.xlabel('Number of Iteration')
            plt.title('Particle Swarm Optimization of Sphere Function')
            print('Best fitness value =', self.gbest_cost)
            print('Best position value =', self.gbest)

            plt.tight_layout()
            plt.show()

    p = Particle()
    p.optimize()
    p.plot()

dimension=10
x_min, x_max = -100, 100
v_min, v_max = -0.2 * (x_max - x_min), 0.2 * (x_max - x_min)
max_iterations = 3000
population_size = 10
c1 = 2
c2 = 2

optimize_sphere(dimension=dimension, x_min=x_min, x_max=x_max, v_min=v_min, v_max=v_max, max_iterations=max_iterations, population_size=population_size, c1=c1, c2=c2)