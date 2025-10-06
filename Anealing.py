import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
from scipy.io import savemat, loadmat
import os
import requests
from PIL import Image


# ============================================================================
# PROBLEM FORMULATION (State Space Search)
# ============================================================================
# State: A 2D arrangement matrix of piece indices (rows × cols)
# Actions: Swap any two pieces in the arrangement
# Goal Test: Minimize total edge dissimilarity (energy function)
# Path Cost: Sum of squared differences between adjacent piece edges
# Search Strategy: Simulated Annealing (non-deterministic, randomized search)
# ============================================================================


# ----------------------------------------------------------------------------
# 1. SETUP: GENERATE AND LOAD THE PUZZLE
# ----------------------------------------------------------------------------

def generate_dummy_lena_mat(grid_size=(16, 16), image_size=256):
    """
    Downloads the Lena test image, slices it into a grid, scrambles the pieces,
    and saves them in the .mat format to simulate the provided file.
    This makes the script self-contained and runnable.
    Returns the original image array.
    """
    mat_filename = 'scrambled_lena.mat'

    print("Downloading Lena test image for puzzle generation...")
    try:
        # Standard 256x256 Lena test image URL
        url = 'http://www.lenna.org/len_std.jpg'
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        img = Image.open(response.raw).convert('RGB')
        img = img.resize((image_size, image_size))
        img_array = np.array(img)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        print("Generating a synthetic image instead...")
        # Create a simple gradient image as fallback
        img_array = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for i in range(image_size):
            for j in range(image_size):
                img_array[i, j] = [i % 256, j % 256, (i + j) % 256]

    # Save original image
    original_filename = 'original_lena.npy'
    np.save(original_filename, img_array)
    print(f"Saved original image to '{original_filename}'.")

    if os.path.exists(mat_filename):
        print(f"'{mat_filename}' already exists. Skipping generation.")
        return img_array

    print("Slicing image into pieces and scrambling...")
    piece_h = image_size // grid_size[0]
    piece_w = image_size // grid_size[1]

    pieces = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            piece = img_array[i * piece_h:(i + 1) * piece_h, j * piece_w:(j + 1) * piece_w, :]
            pieces.append(piece)

    # Scramble the pieces
    random.shuffle(pieces)

    # Save in a .mat file format
    mdict = {'scrambled_pieces': np.array(pieces, dtype=object), 'grid_size': np.array(grid_size)}
    savemat(mat_filename, mdict)
    print(f"Generated and saved '{mat_filename}'.")

    return img_array


def load_puzzle_pieces(mat_filename='scrambled_lena.mat'):
    """Loads the scrambled pieces and grid size from the .mat file."""
    try:
        data = loadmat(mat_filename)
        grid_size = tuple(data['grid_size'].flatten())

        scrambled_pieces_array = data['scrambled_pieces']

        # Check the shape and convert to list of pieces
        print(f"Loaded array shape: {scrambled_pieces_array.shape}")

        # Expected shape: (num_pieces, piece_height, piece_width, 3)
        if len(scrambled_pieces_array.shape) == 4:
            # Correct format: convert to list of individual pieces
            # Use np.array() to ensure each piece is a proper numpy array, not a scalar
            pieces = [np.array(scrambled_pieces_array[i], dtype=np.uint8) for i in
                      range(scrambled_pieces_array.shape[0])]
        else:
            # Old format or corrupted file
            print(f"Error: Unexpected array shape {scrambled_pieces_array.shape}")
            print("The .mat file may be corrupted. Deleting and regenerating...")
            os.remove(mat_filename)
            return None, None

        # Verify the number of pieces matches the grid size
        if len(pieces) != grid_size[0] * grid_size[1]:
            print(f"Error: Mismatch between grid size {grid_size} and number of loaded pieces {len(pieces)}.")
            print("Deleting corrupted file...")
            os.remove(mat_filename)
            return None, None

        print(f"Successfully loaded {len(pieces)} pieces with grid size {grid_size}")
        print(f"Each piece shape: {pieces[0].shape}")
        return pieces, grid_size
    except FileNotFoundError:
        print(f"Error: '{mat_filename}' not found.")
        print("Please place the file in the same directory or let the script generate it.")
        return None, None


# ----------------------------------------------------------------------------
# 2. SIMULATED ANNEALING CORE LOGIC
# ----------------------------------------------------------------------------

def calculate_energy(arrangement, pieces, grid_size):
    """
    Calculates the 'energy' of the current arrangement.
    Energy is the sum of dissimilarities between adjacent piece edges.
    Lower energy indicates better piece alignment (better solution).

    This is our objective function to minimize.
    """
    rows, cols = grid_size
    total_dissimilarity = 0.0

    # Horizontal dissimilarity (left-right edges)
    for r in range(rows):
        for c in range(cols - 1):
            piece_idx1 = arrangement[r, c]
            piece_idx2 = arrangement[r, c + 1]

            # Compare right edge of piece1 with left edge of piece2
            right_edge = np.asarray(pieces[piece_idx1][:, -1, :], dtype=np.float32)  # (height, 3)
            left_edge = np.asarray(pieces[piece_idx2][:, 0, :], dtype=np.float32)  # (height, 3)
            total_dissimilarity += np.sum((right_edge - left_edge) ** 2)

    # Vertical dissimilarity (top-bottom edges)
    for r in range(rows - 1):
        for c in range(cols):
            piece_idx1 = arrangement[r, c]
            piece_idx2 = arrangement[r + 1, c]

            # Compare bottom edge of piece1 with top edge of piece2
            bottom_edge = np.asarray(pieces[piece_idx1][-1, :, :], dtype=np.float32)  # (width, 3)
            top_edge = np.asarray(pieces[piece_idx2][0, :, :], dtype=np.float32)  # (width, 3)
            total_dissimilarity += np.sum((bottom_edge - top_edge) ** 2)

    return total_dissimilarity


def solve_puzzle_sa(pieces, grid_size, max_iterations=None):
    """
    Solves the jigsaw puzzle using Simulated Annealing.

    Parameters:
    - pieces: List of puzzle piece arrays
    - grid_size: Tuple (rows, cols)
    - max_iterations: Optional maximum number of iterations (for time control)

    Returns:
    - best_arrangement: The best solution found
    - energy_history: List of (iteration, energy) tuples for visualization
    """
    num_pieces = len(pieces)
    rows, cols = grid_size

    # --- Improved Annealing Parameters ---
    initial_temp = 1e9  # Even higher temperature for better exploration
    final_temp = 1e-3  # Very low final temperature
    alpha = 0.9998  # Much slower cooling rate for better convergence
    iterations_per_temp = rows * cols  # More iterations per temperature

    # Initialize with random arrangement
    current_arrangement = np.arange(num_pieces).reshape(grid_size)
    current_energy = calculate_energy(current_arrangement, pieces, grid_size)

    best_arrangement = np.copy(current_arrangement)
    best_energy = current_energy

    temp = initial_temp
    iteration = 0
    energy_history = [(0, current_energy)]

    # For progress tracking
    last_improvement = 0
    no_improvement_threshold = 2000000

    start_time = time.time()
    print("\n" + "=" * 70)
    print("SIMULATED ANNEALING - JIGSAW PUZZLE SOLVER")
    print("=" * 70)
    print(f"Puzzle Size: {rows}×{cols} ({num_pieces} pieces)")
    print(f"Initial Energy: {current_energy:.2f}")
    print(f"Temperature Range: {initial_temp:.2e} → {final_temp:.2e}")
    print(f"Cooling Rate: {alpha}")
    print("=" * 70 + "\n")

    while temp > final_temp:
        if max_iterations and iteration >= max_iterations:
            print(f"\nReached maximum iterations ({max_iterations})")
            break

        for _ in range(iterations_per_temp):
            iteration += 1

            # Generate neighbor state with bias towards local swaps (70% of the time)
            if random.random() < 0.7:
                # Local swap: swap nearby pieces
                r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
                dr, dc = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
                r2 = max(0, min(r1 + dr, rows - 1))
                c2 = max(0, min(c1 + dc, cols - 1))
            else:
                # Random swap: swap any two pieces
                r1, c1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
                r2, c2 = random.randint(0, rows - 1), random.randint(0, cols - 1)

            # Create neighbor by swapping
            neighbor_arrangement = np.copy(current_arrangement)
            neighbor_arrangement[r1, c1], neighbor_arrangement[r2, c2] = \
                neighbor_arrangement[r2, c2], neighbor_arrangement[r1, c1]

            neighbor_energy = calculate_energy(neighbor_arrangement, pieces, grid_size)
            delta_energy = neighbor_energy - current_energy

            # Acceptance criterion
            if delta_energy < 0 or random.random() < math.exp(-delta_energy / temp):
                current_arrangement = neighbor_arrangement
                current_energy = neighbor_energy

                # Update best solution
                if current_energy < best_energy:
                    best_arrangement = np.copy(current_arrangement)
                    best_energy = current_energy
                    last_improvement = iteration
                    energy_history.append((iteration, best_energy))

        # Cool down
        temp *= alpha

        # Progress update every 1000 iterations
        if iteration % 100 == 0:  # More frequent updates
            elapsed = time.time() - start_time
            improvement_pct = ((energy_history[0][1] - best_energy) / energy_history[0][1]) * 100
            print(f"Iter: {iteration:6d} | Temp: {temp:10.2e} | "
                  f"Current: {current_energy:12.2f} | Best: {best_energy:12.2f} | "
                  f"Improved: {improvement_pct:5.1f}% | Time: {elapsed:5.1f}s")

        # Early stopping if no improvement for a long time
        if iteration - last_improvement > no_improvement_threshold:
            print(f"\nNo improvement for {no_improvement_threshold} iterations. Stopping early.")
            break

    end_time = time.time()
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Total Iterations: {iteration}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Initial Energy: {energy_history[0][1]:.2f}")
    print(f"Final Best Energy: {best_energy:.2f}")
    print(f"Energy Reduction: {((energy_history[0][1] - best_energy) / energy_history[0][1]) * 100:.2f}%")
    print("=" * 70 + "\n")

    return best_arrangement, energy_history


# ----------------------------------------------------------------------------
# 3. VISUALIZATION
# ----------------------------------------------------------------------------

def reconstruct_image(arrangement, pieces, grid_size):
    """Reconstructs the full image from the solved arrangement of pieces."""
    rows, cols = grid_size
    piece_h, piece_w, _ = pieces[0].shape

    full_image = np.zeros((rows * piece_h, cols * piece_w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            piece_idx = arrangement[r, c]
            full_image[r * piece_h:(r + 1) * piece_h, c * piece_w:(c + 1) * piece_w, :] = pieces[piece_idx]

    return full_image


def display_results(scrambled_pieces, solved_arrangement, grid_size, energy_history=None, original_image=None):
    """Displays the original, scrambled and solved puzzles, with optional energy plot."""

    # Create the initial scrambled image
    initial_arrangement = np.arange(len(scrambled_pieces)).reshape(grid_size)
    scrambled_image = reconstruct_image(initial_arrangement, scrambled_pieces, grid_size)

    # Create the solved image
    solved_image = reconstruct_image(solved_arrangement, scrambled_pieces, grid_size)

    # Determine subplot layout
    if energy_history and len(energy_history) > 1:
        if original_image is not None:
            fig = plt.figure(figsize=(24, 6))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1])
            ax2 = fig.add_subplot(gs[2])
            ax3 = fig.add_subplot(gs[3])
        else:
            fig = plt.figure(figsize=(18, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
            ax0 = None
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax3 = fig.add_subplot(gs[2])
    else:
        if original_image is not None:
            fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))
            ax3 = None
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax0 = None
            ax3 = None

    # Original image (if provided)
    if ax0 is not None and original_image is not None:
        ax0.imshow(original_image)
        ax0.set_title("Original Image", fontsize=14, fontweight='bold')
        ax0.axis('off')

    # Scrambled puzzle
    ax1.imshow(scrambled_image)
    ax1.set_title("Initial Scrambled Puzzle", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Solved puzzle
    ax2.imshow(solved_image)
    ax2.set_title("Solved Puzzle (Simulated Annealing)", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Energy convergence plot
    if ax3 and energy_history:
        iterations, energies = zip(*energy_history)
        ax3.plot(iterations, energies, 'b-', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Energy (Lower is Better)', fontsize=12)
        ax3.set_title('Energy Convergence', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    np.random.seed(42)

    # Ensure the puzzle file exists, or generate it
    original_image = generate_dummy_lena_mat(grid_size=(16,16), image_size=256)

    # Try to load original image if it exists
    if os.path.exists('original_lena.npy'):
        original_image = np.load('original_lena.npy')
    else:
        original_image = None

    # Load the puzzle pieces
    pieces, grid_size = load_puzzle_pieces()

    if pieces:
        # Solve the puzzle using Simulated Annealing
        solved_arrangement, energy_history = solve_puzzle_sa(
            pieces,
            grid_size,
            max_iterations=150000  # More iterations for better results
        )

        # Display the results with original image
        display_results(pieces, solved_arrangement, grid_size, energy_history, original_image)
    else:
        print("Failed to load puzzle pieces. Exiting.")