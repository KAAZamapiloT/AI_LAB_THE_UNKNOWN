import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
OUTPUT_DIR = "lab6_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Processing... Images will be saved to: ./{OUTPUT_DIR}/")


# ==========================================
# PART 1: Error Correcting (Associative Memory)
# ==========================================

class HopfieldMemory:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.n_neurons = np.prod(input_shape)
        self.weights = np.zeros((self.n_neurons, self.n_neurons))

    def train(self, patterns):
        """Hebbian Learning: W_ij = (1/N) * sum(xi * xj)"""
        print(f"[Part 1] Training on {len(patterns)} patterns...")
        for p in patterns:
            p_flat = p.flatten()
            self.weights += np.outer(p_flat, p_flat)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.n_neurons

    def predict(self, pattern, steps=5):
        """Asynchronous update rule"""
        state = pattern.flatten().copy()
        indices = np.arange(self.n_neurons)
        for _ in range(steps):
            np.random.shuffle(indices)
            for i in indices:
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
        return state.reshape(self.input_shape)


def run_part1_associative_memory():
    # Define Patterns (5x5)
    p1 = -1 * np.ones((5, 5))
    np.fill_diagonal(p1, 1)
    np.fill_diagonal(np.fliplr(p1), 1)  # Pattern X

    p2 = -1 * np.ones((5, 5))
    p2[:, 2] = 1
    p2[2, :] = 1  # Pattern +

    patterns = [p1, p2]

    # Train
    net = HopfieldMemory((5, 5))
    net.train(patterns)

    # Corrupt Pattern 1
    test_pattern = patterns[0].copy()
    noise_mask = np.random.choice([True, False], test_pattern.shape, p=[0.3, 0.7])  # 30% noise
    test_pattern[noise_mask] *= -1

    # Recover
    recovered = net.predict(test_pattern)

    # Plot and Save
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(patterns[0], cmap='gray')
    ax[0].set_title("Original Stored Pattern")
    ax[1].imshow(test_pattern, cmap='gray')
    ax[1].set_title("Corrupted Input (30% Noise)")
    ax[2].imshow(recovered, cmap='gray')
    ax[2].set_title("Recovered Output")

    save_path = os.path.join(OUTPUT_DIR, "1_error_correction.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Part 1] Image saved to {save_path}")


# ==========================================
# PART 2: Eight-Rook Problem (Combinatorial)
# ==========================================

def run_part2_n_rooks():
    print("[Part 2] Solving 8-Rooks problem...")
    n = 8
    # Weights Logic:
    # We don't train weights; we define energy penalties.
    # A (Row), B (Col) = Penalties for >1 rook in line.
    # C (Global) = Penalty for not having exactly N rooks.
    A, B, C = 1.0, 1.0, 1.0

    state = np.random.choice([0, 1], size=(n, n))
    min_energy = float('inf')
    best_state = state.copy()

    # Simple stochastic optimization loop (Simulated Annealing-like)
    for k in range(2000):
        row_sum = np.sum(state, axis=1)
        col_sum = np.sum(state, axis=0)
        total = np.sum(state)

        # Energy E
        E = (A * np.sum(row_sum * (row_sum - 1)) +
             B * np.sum(col_sum * (col_sum - 1)) +
             C * (total - n) ** 2)

        if E == 0:
            best_state = state
            print(f"[Part 2] Solution found at iteration {k}")
            break

        if E < min_energy:
            min_energy = E
            best_state = state.copy()

        # Perturb state: Flip a random bit
        i, j = np.random.randint(0, n), np.random.randint(0, n)
        state[i, j] = 1 - state[i, j]

        # Greedy check: if Energy worsened significantly, revert (with slight noise)
        r_new = np.sum(state, axis=1)
        c_new = np.sum(state, axis=0)
        t_new = np.sum(state)
        E_new = (A * np.sum(r_new * (r_new - 1)) +
                 B * np.sum(c_new * (c_new - 1)) +
                 C * (t_new - n) ** 2)

        if E_new >= E and np.random.rand() > 0.05:  # 5% chance to accept bad move
            state[i, j] = 1 - state[i, j]  # Revert

    # Plot and Save
    plt.figure(figsize=(6, 6))
    plt.imshow(best_state, cmap='binary')
    plt.title(f"8-Rooks Solution\n(Black=Rook, Valid={min_energy == 0})")
    plt.grid(which='both', color='gray', linestyle='-', linewidth=1)
    plt.xticks(np.arange(-.5, 8, 1), [])
    plt.yticks(np.arange(-.5, 8, 1), [])

    save_path = os.path.join(OUTPUT_DIR, "2_eight_rooks.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Part 2] Image saved to {save_path}")


# ==========================================
# PART 3: Traveling Salesman Problem (TSP)
# ==========================================

class HopfieldTSP:
    def __init__(self, city_coords):
        self.coords = city_coords
        self.n_cities = len(city_coords)
        self.dist_matrix = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                self.dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

        # Hopfield-Tank Parameters
        self.A, self.B, self.C, self.D = 500.0, 500.0, 200.0, 50.0

        # Initialize potentials with noise to break symmetry
        self.u = np.random.uniform(-0.02, 0.02, (self.n_cities, self.n_cities))
        self.V = self.sigmoid(self.u)

    def sigmoid(self, u, u0=0.02):
        return 0.5 * (1 + np.tanh(u / u0))

    def update(self, dt=0.0001):
        # 1. Constraints: Each city visited once, each position filled once
        term_A = np.tile(np.sum(self.V, axis=1, keepdims=True) - 1, (1, self.n_cities))
        term_B = np.tile(np.sum(self.V, axis=0, keepdims=True) - 1, (self.n_cities, 1))

        # 2. Distance Optimization
        term_D = np.zeros_like(self.V)
        for x in range(self.n_cities):
            for i in range(self.n_cities):
                left, right = (i - 1) % self.n_cities, (i + 1) % self.n_cities
                d_sum = 0
                for y in range(self.n_cities):
                    d_sum += self.dist_matrix[x, y] * (self.V[y, left] + self.V[y, right])
                term_D[x, i] = d_sum

        # Update potentials
        du = -self.A * term_A - self.B * term_B - self.D * term_D - self.C * (np.sum(self.V) - self.n_cities)
        self.u += du * dt
        self.V = self.sigmoid(self.u)


def run_part3_tsp():
    print("[Part 3] Solving TSP for 10 cities...")
    np.random.seed(42)
    cities = np.random.rand(10, 2)

    tsp = HopfieldTSP(cities)
    for _ in range(5000):  # Iterations
        tsp.update()

    # Decode output
    city_order = np.argsort(np.argmax(tsp.V, axis=1))
    ordered_coords = cities[city_order]
    ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])  # Close loop

    # Plot and Save
    plt.figure(figsize=(12, 5))

    # Subplot 1: Matrix
    plt.subplot(1, 2, 1)
    plt.imshow(tsp.V, cmap='hot', interpolation='nearest')
    plt.title("TSP Activation Matrix")
    plt.xlabel("Order in Tour")
    plt.ylabel("City Index")

    # Subplot 2: Map
    plt.subplot(1, 2, 2)
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=2)
    plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'b-', linewidth=2, zorder=1)
    for i in range(len(cities)):
        plt.annotate(str(i), (cities[i, 0] + 0.01, cities[i, 1] + 0.01))
    plt.title("Resulting Tour")

    save_path = os.path.join(OUTPUT_DIR, "3_tsp_solution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Part 3] Image saved to {save_path}")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    run_part1_associative_memory()
    run_part2_n_rooks()
    run_part3_tsp()
    print("\n--- Lab 6 Complete ---")
    print(f"Please check the folder '{OUTPUT_DIR}' for your generated images.")