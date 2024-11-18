import math
import numpy as np
import matplotlib.pyplot as plt


# Ackley function definition
def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.e


# Visualize the Ackley function
def visualize_ackley():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = ackley_function(X, Y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_title("Ackley Function")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


# Particle class for PSO
class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.best_position = np.copy(self.position)
        self.best_value = ackley_function(*self.position)


# PSO algorithm to minimize Ackley function
def pso_ackley(population_size=20, max_iter=100, bounds=(-5, 5)):
    particles = []
    global_best_position = None
    global_best_value = float('inf')

    # Initialize particles
    for _ in range(population_size):
        position = np.random.uniform(bounds[0], bounds[1], 2)
        velocity = np.random.uniform(-1, 1, 2)
        particle = Particle(position, velocity)
        particles.append(particle)
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = np.copy(particle.position)

    # PSO iterations
    for iteration in range(max_iter):
        for particle in particles:
            # Update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            inertia = 0.5
            cognitive = 2 * r1 * (particle.best_position - particle.position)
            social = 2 * r2 * (global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive + social

            # Update position
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, bounds[0], bounds[1])

            # Evaluate function at new position
            value = ackley_function(*particle.position)
            if value < particle.best_value:
                particle.best_value = value
                particle.best_position = np.copy(particle.position)
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = np.copy(particle.position)

        if iteration % 10 == 0:
            print(f"PSO Iteration {iteration}: Best Value = {global_best_value}")

    print(f"PSO Optimal Solution: {global_best_position} with Value: {global_best_value}")
    return global_best_position, global_best_value


# GWO implementation
def gwo_ackley(population_size=20, max_iter=100, bounds=(-5, 5)):
    # Initialize wolves
    wolves = np.random.uniform(bounds[0], bounds[1], (population_size, 2))
    fitness = np.array([ackley_function(w[0], w[1]) for w in wolves])

    # Sort wolves based on fitness
    alpha, beta, gamma = sorted(wolves, key=lambda w: ackley_function(w[0], w[1]))[:3]
    alpha_value = ackley_function(*alpha)
    beta_value = ackley_function(*beta)
    gamma_value = ackley_function(*gamma)

    for iteration in range(max_iter):
        a = 2 - (iteration / max_iter) * 2  # linearly decreases from 2 to 0

        for i in range(population_size):
            for j in range(2):  # for each dimension
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i, j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i, j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_gamma = abs(C3 * gamma[j] - wolves[i, j])
                X3 = gamma[j] - A3 * D_gamma

                wolves[i, j] = (X1 + X2 + X3) / 3

        # Recalculate fitness
        fitness = np.array([ackley_function(w[0], w[1]) for w in wolves])
        alpha, beta, gamma = sorted(wolves, key=lambda w: ackley_function(w[0], w[1]))[:3]
        alpha_value = ackley_function(*alpha)
        beta_value = ackley_function(*beta)
        gamma_value = ackley_function(*gamma)

        if iteration % 10 == 0:
            print(f"GWO Iteration {iteration}: Alpha Value = {alpha_value}")

    print(f"GWO Optimal Solution: {alpha} with Value: {alpha_value}")
    return alpha, alpha_value


if __name__ == "__main__":
    # Visualize the Ackley function
    visualize_ackley()

    # Run PSO on Ackley function
    print("\nRunning PSO...")
    pso_position, pso_value = pso_ackley()

    # Run GWO on Ackley function
    print("\nRunning GWO...")
    gwo_position, gwo_value = gwo_ackley()

    # Compare results
    print("\nComparison of Results:")
    print(f"PSO Optimal Position: {pso_position}, Value: {pso_value}")
    print(f"GWO Optimal Position: {gwo_position}, Value: {gwo_value}")
