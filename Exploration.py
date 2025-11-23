import numpy as np
import random
import matplotlib.pyplot as plt
import os


# ==========================================
# PART 1: MENACE (Tic-Tac-Toe RL Agent)
# ==========================================

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9

    def reset(self):
        self.board = [' '] * 9
        return tuple(self.board)

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False

    def check_winner(self):
        # Rows, Cols, Diagonals
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != ' ':
                return self.board[a]
        if ' ' not in self.board:
            return 'Draw'
        return None


class MENACE:
    def __init__(self):
        # The "Matchboxes": Dictionary mapping state (tuple) -> list of beads (weights)
        self.matchboxes = {}
        # Memory of current game: (state, move_index)
        self.moves_played = []

    def get_move(self, board_state, available_moves):
        # CRUCIAL: If we haven't seen this board state, initialize a new "Matchbox"
        if board_state not in self.matchboxes:
            # Initialize with equal beads for available moves.
            self.matchboxes[board_state] = {mv: 10 for mv in available_moves}

        # Get beads for this state
        current_beads = self.matchboxes[board_state]

        # Filter only currently valid moves
        valid_beads = {k: v for k, v in current_beads.items() if k in available_moves}

        if not valid_beads:
            return random.choice(available_moves)  # Fallback if empty

        # CRUCIAL: Probabilistic selection based on number of beads
        moves = list(valid_beads.keys())
        weights = list(valid_beads.values())

        # Selection logic
        if sum(weights) == 0:
            chosen_move = random.choice(moves)
        else:
            chosen_move = random.choices(moves, weights=weights, k=1)[0]

        self.moves_played.append((board_state, chosen_move))
        return chosen_move

    def reinforce(self, result):
        # CRUCIAL: The Learning Step
        # Win: Add beads (Reinforce behavior)
        # Lose: Remove beads (Discourage behavior)
        # Draw: Slight reward or neutral

        if result == 'Win':
            delta = 3
        elif result == 'Draw':
            delta = 1
        else:  # Loss
            delta = -1

        for state, move in self.moves_played:
            if state in self.matchboxes:
                # Adjust bead count
                self.matchboxes[state][move] += delta
                # Ensure bead count never drops below 1
                if self.matchboxes[state][move] < 1:
                    self.matchboxes[state][move] = 1

        self.moves_played = []  # Clear memory for next game


def train_menace(episodes=2000):
    print(f"\n--- Training MENACE for {episodes} episodes against Random Player ---")
    menace = MENACE()
    env = TicTacToe()

    stats = {'Win': 0, 'Draw': 0, 'Loss': 0}
    # Store numerical result for plotting: 1=Win, 0=Draw, -1=Loss
    history = []

    for i in range(episodes):
        state = env.reset()
        done = False
        turn = 'MENACE'  # Menace plays X, Random plays O

        while not done:
            avail = env.available_moves()
            if turn == 'MENACE':
                move = menace.get_move(state, avail)
                env.make_move(move, 'X')
            else:
                move = random.choice(avail)
                env.make_move(move, 'O')

            winner = env.check_winner()
            state = tuple(env.board)

            if winner:
                done = True
                if winner == 'X':
                    menace.reinforce('Win')
                    stats['Win'] += 1
                    history.append(1)
                elif winner == 'O':
                    menace.reinforce('Loss')
                    stats['Loss'] += 1
                    history.append(-1)
                else:
                    menace.reinforce('Draw')
                    stats['Draw'] += 1
                    history.append(0)
            else:
                turn = 'Random' if turn == 'MENACE' else 'MENACE'

    print("Training Results:", stats)
    print(f"MENACE learned {len(menace.matchboxes)} distinct board states.")
    return menace, history


# ==========================================
# PART 2: BINARY BANDITS & EPSILON GREEDY
# ==========================================

class BinaryBandit:
    def __init__(self, p1, p2):
        # Probabilities of getting reward 1 for Action 1 and Action 2
        self.probs = [p1, p2]

    def pull(self, action):
        # action is 0 or 1
        if random.random() < self.probs[action]:
            return 1
        return 0


def epsilon_greedy_stationary(bandit, steps=1000, epsilon=0.1):
    # Q-values for 2 arms, initialized to 0
    Q = [0.0, 0.0]
    # Count of times each arm was pulled
    N = [0, 0]

    # Store history for plotting [step, Q0, Q1]
    history = np.zeros((steps, 2))

    print(f"Running Epsilon-Greedy (eps={epsilon}) on Bandit with Probs {bandit.probs}")

    for i in range(steps):
        # Exploration vs Exploitation
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            # Argmax (break ties randomly)
            if Q[0] == Q[1]:
                action = random.choice([0, 1])
            else:
                action = np.argmax(Q)

        # Get Reward
        reward = bandit.pull(action)

        # Update Estimates (Sample Average method)
        N[action] += 1
        Q[action] = Q[action] + (1.0 / N[action]) * (reward - Q[action])

        history[i] = Q

    print(f"Final Q-values: {Q}")
    print(f"Arm counts: {N}\n")
    return history


# ==========================================
# PART 3 & 4: NON-STATIONARY 10-ARM BANDIT
# ==========================================

class NonStationaryBandit:
    def __init__(self, k=10):
        self.k = k
        self.q_true = np.zeros(k)

    def step(self):
        # Random Walk: Add N(0, 0.01) to all means
        self.q_true += np.random.normal(0, 0.01, self.k)

    def pull(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.step()  # Drifts after every step
        return reward

    def get_optimal_action(self):
        return np.argmax(self.q_true)


def run_nonstat_experiment(agent_type, steps=10000, epsilon=0.1, alpha=None):
    bandit = NonStationaryBandit(k=10)

    # Estimates
    Q = np.zeros(10)
    # Counts
    N = np.zeros(10)

    optimal_action_counts = np.zeros(steps)

    for t in range(steps):
        # 1. Select Action (Epsilon Greedy)
        if np.random.rand() < epsilon:
            action = np.random.randint(10)
        else:
            top_actions = np.where(Q == np.max(Q))[0]
            action = np.random.choice(top_actions)

        # 2. Get Reward
        reward = bandit.pull(action)

        # Check if optimal
        if action == bandit.get_optimal_action():
            optimal_action_counts[t] = 1

        # 3. Update Estimates
        if agent_type == 'sample_average':
            N[action] += 1
            Q[action] = Q[action] + (1.0 / N[action]) * (reward - Q[action])

        elif agent_type == 'constant_alpha':
            Q[action] = Q[action] + alpha * (reward - Q[action])

    return optimal_action_counts


# ==========================================
# MAIN EXECUTION & PLOTTING
# ==========================================

if __name__ == "__main__":

    # Setup Figure for all 3 tasks
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))
    plt.subplots_adjust(hspace=0.4)

    # --- PART 1: MENACE VISUALIZATION ---
    print(">>> Excecuting Part 1: MENACE...")
    trained_menace, game_history = train_menace(episodes=3000)

    # Calculate moving average of wins (1=Win, 0=Draw, -1=Loss)
    # We map Loss to 0 for win-rate calculation simplicity, or keep -1 for 'score'
    # Let's visualize "Win Rate" (Win=1, others=0) over a window
    win_binary = [1 if res == 1 else 0 for res in game_history]
    window = 100
    rolling_wins = np.convolve(win_binary, np.ones(window) / window, mode='valid')

    axs[0].plot(rolling_wins, color='blue')
    axs[0].set_title(f'Part 1: MENACE Learning Curve (Moving Avg Window={window})')
    axs[0].set_ylabel('Win Probability')
    axs[0].set_xlabel('Episode')
    axs[0].grid(True)
    axs[0].set_ylim(0, 1.0)

    # --- PART 2: BINARY BANDITS VISUALIZATION ---
    print(">>> Executing Part 2: Binary Bandits...")
    # Bandit B: 0.4 vs 0.6 (Harder case)
    p1, p2 = 0.4, 0.6
    binaryBanditB = BinaryBandit(p1, p2)
    q_hist = epsilon_greedy_stationary(binaryBanditB, steps=1000)

    axs[1].plot(q_hist[:, 0], label='Estimated Q(Arm 0) [True=0.4]', color='red', alpha=0.7)
    axs[1].plot(q_hist[:, 1], label='Estimated Q(Arm 1) [True=0.6]', color='green', alpha=0.7)
    axs[1].axhline(y=p1, color='red', linestyle='--', alpha=0.3)
    axs[1].axhline(y=p2, color='green', linestyle='--', alpha=0.3)
    axs[1].set_title('Part 2: Binary Bandit Q-Value Convergence (Stationary)')
    axs[1].set_ylabel('Estimated Q-Value')
    axs[1].set_xlabel('Steps')
    axs[1].legend()
    axs[1].grid(True)

    # --- PART 3 & 4: NON-STATIONARY VISUALIZATION ---
    print(">>> Executing Part 3: Non-Stationary Bandits...")
    steps = 10000
    runs = 10

    avg_opt_sample = np.zeros(steps)
    avg_opt_alpha = np.zeros(steps)

    for r in range(runs):
        avg_opt_sample += run_nonstat_experiment('sample_average', steps, epsilon=0.1)
        avg_opt_alpha += run_nonstat_experiment('constant_alpha', steps, epsilon=0.1, alpha=0.1)

    avg_opt_sample /= runs
    avg_opt_alpha /= runs

    axs[2].plot(avg_opt_alpha, label='Modified (Constant Alpha=0.1)', color='green', alpha=0.8)
    axs[2].plot(avg_opt_sample, label='Standard (Sample Average)', color='red', alpha=0.6)
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('% Optimal Action')
    axs[2].set_title('Part 3: Non-Stationary Bandit Tracking')
    axs[2].legend()
    axs[2].grid(True)

    print("\nAll experiments complete.")

    # Save the figure
    save_path = 'lab7_results.png'
    plt.savefig(save_path)
    print(f"Graph saved successfully as: {os.path.abspath(save_path)}")

    # Show the plot
    plt.show()