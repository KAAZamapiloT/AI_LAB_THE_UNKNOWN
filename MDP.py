import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# ==========================================
# PART 1: FILE HANDLING
# ==========================================
ZIP_FILENAME = 'gbike.zip'
EXTRACT_DIR = 'gbike_extracted'


def setup_environment():
    print(f">>> Checking for {ZIP_FILENAME}...")
    if not os.path.exists(ZIP_FILENAME):
        print(f"Warning: {ZIP_FILENAME} not found. Running in simulation mode.")
        return
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR)
    with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
        print(f">>> Extracted {ZIP_FILENAME}")


# ==========================================
# PART 2: MODIFIED GBIKE DYNAMICS
# ==========================================
MAX_BIKES = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
PARKING_COST = 4
DISCOUNT = 0.9
THETA = 1e-4

# Loc 1: Rent=3, Ret=3 | Loc 2: Rent=4, Ret=2
LAMBDA_RENT_1, LAMBDA_RET_1 = 3, 3
LAMBDA_RENT_2, LAMBDA_RET_2 = 4, 2


def precompute_dynamics(lambda_rent, lambda_ret):
    POISSON_UPPER = 21
    rewards = np.zeros(MAX_BIKES + 1)
    transitions = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))

    for s in range(MAX_BIKES + 1):
        for req in range(POISSON_UPPER):
            prob_req = poisson.pmf(req, lambda_rent)
            actual_rent = min(s, req)
            rewards[s] += prob_req * (actual_rent * RENTAL_REWARD)

            for ret in range(POISSON_UPPER):
                prob_ret = poisson.pmf(ret, lambda_ret)
                next_s = min(MAX_BIKES, max(0, s - actual_rent + ret))
                transitions[s, next_s] += prob_req * prob_ret
    return rewards, transitions


def solve_modified_gbike():
    print(">>> Precomputing State Dynamics...")
    R1, T1 = precompute_dynamics(LAMBDA_RENT_1, LAMBDA_RET_1)
    R2, T2 = precompute_dynamics(LAMBDA_RENT_2, LAMBDA_RET_2)

    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)

    def calculate_expected_return(state, action, V_current):
        n1, n2 = state
        real_action = action
        if action > 0:
            real_action = min(action, n1)
        else:
            real_action = max(action, -n2)

        n1_post = min(MAX_BIKES, n1 - real_action)
        n2_post = min(MAX_BIKES, n2 + real_action)

        cost = 0
        if real_action > 0:
            cost += MOVE_COST * (real_action - 1)  # Employee shuttle
        else:
            cost += MOVE_COST * abs(real_action)

        if n1_post > 10: cost += PARKING_COST
        if n2_post > 10: cost += PARKING_COST

        current_reward = R1[n1_post] + R2[n2_post]
        future_val = np.sum(np.outer(T1[n1_post], T2[n2_post]) * V_current)
        return (current_reward - cost) + DISCOUNT * future_val

    iteration = 0
    while True:
        # Policy Evaluation
        while True:
            delta = 0
            # In-place update for speed (Gauss-Seidel style)
            for i in range(MAX_BIKES + 1):
                for j in range(MAX_BIKES + 1):
                    v = V[i, j]
                    action = policy[i, j]
                    V[i, j] = calculate_expected_return((i, j), action, V)
                    delta = max(delta, abs(v - V[i, j]))
            if delta < THETA: break

        # Policy Improvement
        policy_stable = True
        print(f"Iteration {iteration} completed.")
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                old_action = policy[i, j]
                action_values = []
                actions = range(-MAX_MOVE, MAX_MOVE + 1)
                for a in actions:
                    if (a > 0 and i < a) or (a < 0 and j < abs(a)):
                        action_values.append(-np.inf)
                    else:
                        action_values.append(calculate_expected_return((i, j), a, V))

                best_action = actions[np.argmax(action_values)]
                policy[i, j] = best_action
                if old_action != best_action: policy_stable = False

        if policy_stable: break
        iteration += 1
    return V, policy


# ==========================================
# PART 3: EXTENDED VISUALIZATION
# ==========================================
def visualize_results_extended(V, policy):
    print(">>> Generating extended visualizations...")

    # 1. Original Combined Heatmaps (For Latex Compatibility)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(policy, ax=axes[0], cmap="RdBu", center=0, annot=True, fmt="d", cbar_kws={'label': 'Action'})
    axes[0].invert_yaxis()
    axes[0].set_title("Optimal Policy (Heatmap)")
    axes[0].set_xlabel("Loc 2")
    axes[0].set_ylabel("Loc 1")

    sns.heatmap(V, ax=axes[1], cmap="viridis", cbar_kws={'label': 'Value'})
    axes[1].invert_yaxis()
    axes[1].set_title("Optimal Value Function")
    axes[1].set_xlabel("Loc 2")
    axes[1].set_ylabel("Loc 1")
    plt.tight_layout()
    plt.savefig("lab8_gbike_results.png")
    print("  - Saved lab8_gbike_results.png")

    # 2. 3D Value Function Surface [Image of 3D value surface plot]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Create meshgrid: X=Loc2 (cols), Y=Loc1 (rows)
    X, Y = np.meshgrid(range(MAX_BIKES + 1), range(MAX_BIKES + 1))
    surf = ax.plot_surface(X, Y, V, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Bikes at Loc 2')
    ax.set_ylabel('Bikes at Loc 1')
    ax.set_zlabel('Value ($)')
    ax.set_title('3D Value Function Surface')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig("lab8_value_3d.png")
    print("  - Saved lab8_value_3d.png")

    # 3. Policy Boundaries (Contours)
    plt.figure(figsize=(8, 8))
    # Plot contours for actions. We use levels -5 to 5.
    # Adding 0.5 allows the lines to center between integer states
    X, Y = np.meshgrid(range(MAX_BIKES + 1), range(MAX_BIKES + 1))
    cs = plt.contour(X, Y, policy, levels=np.arange(-5, 6), colors='black', linewidths=1.5)
    plt.clabel(cs, inline=True, fontsize=10, fmt='%d')

    # Overlay with faint colors for clarity
    plt.imshow(policy, origin='lower', cmap='RdBu', alpha=0.3, extent=[0, 20, 0, 20])
    plt.xlabel('Bikes at Location 2')
    plt.ylabel('Bikes at Location 1')
    plt.title('Optimal Policy Boundaries (Contour Map)')
    plt.grid(True, alpha=0.3)
    plt.savefig("lab8_policy_boundaries.png")
    print("  - Saved lab8_policy_boundaries.png")

    plt.show()


if __name__ == "__main__":
    setup_environment()
    opt_V, opt_policy = solve_modified_gbike()
    visualize_results_extended(opt_V, opt_policy)