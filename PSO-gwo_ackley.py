import math
import numpy as np
import matplotlib.pyplot as plt
from random import random

# Define o tamanho da população de lobos
POPULATION_SIZE = 5  # ou qualquer número que você deseja usar

# Função Ackley
def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    term2 = -np.exp(0.5 * (np.cos(c * x) + np.cos(c * y)))
    return term1 + term2 + a + np.e

# Visualize a função Ackley
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

# Implementação PSO
class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.best_position = np.copy(self.position)
        self.best_value = ackley_function(*self.position)

def pso_ackley(population_size=20, max_iter=100, bounds=(-5, 5), inertia=0.3, cognitive_coefficient=2.0, social_coefficient=2.0):
    particles = []
    global_best_position = None
    global_best_value = float('inf')

    # Inicialização das partículas
    for _ in range(population_size):
        position = np.random.uniform(bounds[0], bounds[1], 2)
        velocity = np.random.uniform(-1, 1, 2)
        particle = Particle(position, velocity)
        particles.append(particle)
        if particle.best_value < global_best_value:
            global_best_value = particle.best_value
            global_best_position = np.copy(particle.position)

    # Algoritmo de Otimização por Enxame de Partículas (PSO)
    for iteration in range(max_iter):
        for particle in particles:
            r1, r2 = np.random.rand(), np.random.rand()

            # Atualiza a velocidade com base nos parâmetros ajustáveis
            cognitive = cognitive_coefficient * r1 * (particle.best_position - particle.position)
            social = social_coefficient * r2 * (global_best_position - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive + social

            # Atualiza a posição da partícula
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, bounds[0], bounds[1])

            # Calcula o valor da função Ackley na nova posição
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

# Nova Implementação do GWO para Ackley
def initialize_wolves(num_wolves):
    wolves = []
    for _ in range(num_wolves):
        x = random() * 10 - 5  # Limites de -5 a 5
        y = random() * 10 - 5
        wolves.append([x, y])
    return wolves

def update_position(wolf, alpha, beta, delta, a):
    new_position = []
    for i in range(2):  # Duas dimensões: x e y
        r1, r2 = random(), random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha[i] - wolf[i])
        X1 = alpha[i] - A1 * D_alpha

        r1, r2 = random(), random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta[i] - wolf[i])
        X2 = beta[i] - A2 * D_beta

        r1, r2 = random(), random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_gamma = abs(C3 * delta[i] - wolf[i])
        X3 = delta[i] - A3 * D_gamma

        new_position.append((X1 + X2 + X3) / 3)
    return new_position

def gwo_ackley(num_wolves=5, num_iterations=10):
    wolves = initialize_wolves(num_wolves)
    alpha, beta, delta = None, None, None

    for iteration in range(num_iterations):
        fitness_scores = [(ackley_function(*wolf), wolf) for wolf in wolves]
        fitness_scores.sort()
        alpha, beta, delta = fitness_scores[0][1], fitness_scores[1][1], fitness_scores[2][1]

        a = 2 * (1 - iteration / num_iterations)
        for i in range(num_wolves):
            wolves[i] = update_position(wolves[i], alpha, beta, delta, a)
        if iteration % 2 == 0:
            print(f"GWO Iteration {iteration}: Alpha Value = {ackley_function(*alpha)}")

    print(f"GWO Optimal Solution: {alpha} with Value: {ackley_function(*alpha)}")
    return alpha, ackley_function(*alpha)

# Executa PSO e GWO e compara resultados
visualize_ackley()
print("\nRunning PSO...")
pso_position, pso_value = pso_ackley()

print("\nRunning GWO...")
gwo_position, gwo_value = gwo_ackley()

print("\nComparison of Results:")
print(f"PSO Optimal Position: {pso_position}, Value: {pso_value}")
print(f"GWO Optimal Position: {gwo_position}, Value: {gwo_value}")
