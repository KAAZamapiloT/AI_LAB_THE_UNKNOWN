# ==============================================================================
# Breadth-First Search (BFS) with Detailed Tracing
# For academic presentation and notebook transcription.
#
# Date: 18 August 2025
# ==============================================================================

from collections import deque


def reconstruct_path(parent, start, goal):
    """Helper function to reconstruct the path from the parent dictionary."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]  # Reverse the path to get start -> goal


def calculate_path_cost(path, graph_costs):
    """Helper function to calculate the total cost of the found path."""
    cost = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        # Find the cost for the edge u -> v
        edge_cost = next((c for neighbor, c in graph_costs[u] if neighbor == v), 0)
        cost += edge_cost
    return cost


def bfs_trace(graph, graph_costs, start, goal):
    """
    Performs a Breadth-First Search and prints a detailed trace
    of its execution for academic purposes.
    """
    # 1. Initialize data structures
    frontier = deque([start])  # Use a deque as a FIFO Queue for the frontier
    explored = set()  # A set to store visited states
    parent = {start: None}  # A dictionary to store parent pointers for path reconstruction

    step = 0

    # Print table header
    print("=" * 80)
    print("Executing Breadth-First Search (BFS)")
    print(f"Finding path from '{start}' to '{goal}'")
    print("=" * 80)
    print(f"{'Step':<5} | {'Node Expanded':<15} | {'Frontier (Queue: Head <- Tail)':<50} | {'Explored Set'}")
    print("-" * 80)

    # 2. Begin the search loop
    while frontier:
        step += 1

        # Print the state *before* expanding the node
        # This shows the decision-making process
        frontier_str = str(list(frontier))
        explored_str = str(sorted(list(explored))) if explored else "{}"

        # Pop a node from the head of the queue
        current_node = frontier.popleft()

        print(f"{step:<5} | {current_node:<15} | {frontier_str:<50} | {explored_str}")

        # 3. Goal Check
        if current_node == goal:
            print("\n" + "-" * 80)
            print("Goal reached!")

            # Reconstruct and print the path
            path = reconstruct_path(parent, start, goal)
            cost = calculate_path_cost(path, graph_costs)

            print(f"\nPath Found: {' -> '.join(path)}")
            print(f"Number of Edges: {len(path) - 1}")
            print(f"Total Cost: {cost}")
            print("=" * 80)
            return

        # 4. Add the current node to the explored set
        explored.add(current_node)

        # 5. Expand the node and add its neighbors to the frontier
        for neighbor in sorted(graph[current_node]):  # Sort for consistent output
            if neighbor not in explored and neighbor not in frontier:
                parent[neighbor] = current_node
                frontier.append(neighbor)  # Add to the tail of the queue

    # If the loop finishes without finding the goal
    print("\n" + "=" * 80)
    print("Goal not found. The frontier is empty.")
    print("=" * 80)


# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    # Graph representation for BFS (unweighted, neighbors only)
    graph_bfs = {
        'S': ['D'],
        'A': ['B', 'D', 'H'],
        'B': ['A', 'C', 'K'],
        'C': ['B', 'E'],
        'D': ['S', 'A', 'F', 'I'],
        'E': ['C', 'K'],
        'F': ['D', 'L'],
        'G': ['N', 'T'],
        'H': ['A', 'J', 'K', 'N'],
        'I': ['D', 'L', 'M'],
        'J': ['H', 'M'],
        'K': ['E', 'H', 'N', 'Q', 'B'],
        'L': ['F', 'I', 'O'],
        'M': ['I', 'J', 'P'],
        'N': ['H', 'K', 'Q', 'G'],
        'O': ['L', 'R'],
        'P': ['M'],
        'Q': ['K', 'N'],
        'R': ['O', 'T'],
        'T': ['G', 'R']
    }

    # Graph representation with costs for final calculation
    graph_with_costs = {
        'S': [('D', 25)],
        'A': [('B', 11), ('D', 32), ('H', 36)],
        'B': [('A', 11), ('C', 24), ('K', 42)],
        'C': [('B', 24), ('E', 40)],
        'D': [('S', 25), ('A', 32), ('F', 24), ('I', 26)],
        'E': [('C', 40), ('K', 32)],
        'F': [('D', 24), ('L', 27)],
        'G': [('N', 42), ('T', 32)],
        'H': [('A', 36), ('J', 22), ('K', 28), ('N', 44)],
        'I': [('D', 26), ('L', 21), ('M', 32)],
        'J': [('H', 22), ('M', 20)],
        'K': [('E', 32), ('H', 28), ('N', 27), ('Q', 62),('B',42)],
        'L': [('F', 27), ('I', 21), ('O', 26)],
        'M': [('I', 32), ('J', 20), ('P', 23)],
        'N': [('H', 44), ('K', 27), ('Q', 32), ('G', 42)],
        'O': [('L', 26), ('R', 27)],
        'P': [('M', 23)],
        'Q': [('K', 62), ('N', 32)],
        'R': [('O', 27), ('T', 52)],
        'T': [('G', 32), ('R', 52)]
    }

    # Run the BFS trace
    bfs_trace(graph_bfs, graph_with_costs, 'S', 'G')