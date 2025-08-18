# ==============================================================================
# Breadth-First Search (BFS) with Detailed PUSH/POP Tracing
# For academic presentation and notebook transcription.
#
# Date: 19 August 2025
# ==============================================================================

from collections import deque


def reconstruct_path(parent, start, goal):
    """Helper function to reconstruct the path from the parent dictionary."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]


def calculate_path_cost(path, graph_costs):
    """Helper function to calculate the total cost of the found path."""
    cost = 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_cost = next((c for neighbor, c in graph_costs[u] if neighbor == v), 0)
        cost += edge_cost
    return cost


def bfs_trace_push_pop(graph, graph_costs, start, goal):
    """
    Performs a BFS and prints a detailed trace describing each PUSH and POP operation.
    """
    # 1. Initialize data structures
    frontier_queue = deque([start])  # Use a deque as a FIFO Queue
    explored_set = set()
    parent = {start: None}

    step = 0

    # Print header
    print("=" * 80)
    print("Executing Breadth-First Search (BFS) with PUSH/POP Operations")
    print(f"Finding path from '{start}' to '{goal}'")

    # 2. Begin the search loop
    while frontier_queue:
        step += 1
        print("\n" + f"--- Step {step} ---")

        # Report state before the POP
        print(f"Frontier before action: {list(frontier_queue)}")

        # POP a node from the head of the queue
        current_node = frontier_queue.popleft()
        print(f"Action: POP '{current_node}' from the HEAD of the queue.")

        # Report state after the POP
        print(f"Frontier after action:  {list(frontier_queue)}")

        # 3. Goal Check
        if current_node == goal:
            print("\n" + "-" * 80)
            print("Goal reached!")
            path = reconstruct_path(parent, start, goal)
            cost = calculate_path_cost(path, graph_costs)
            print(f"\nPath Found: {' -> '.join(path)}")
            print(f"Number of Edges: {len(path) - 1}")
            print(f"Total Cost: {cost}")
            print("=" * 80)
            return

        # 4. Add the current node to the explored set
        explored_set.add(current_node)
        print(f"Explored Set updated: {sorted(list(explored_set))}")

        # 5. Expand the node and PUSH its neighbors to the frontier
        print("Discovering neighbors...")
        for neighbor in sorted(graph[current_node]):
            if neighbor not in explored_set and neighbor not in frontier_queue:
                parent[neighbor] = current_node
                # PUSH neighbor to the tail of the queue
                frontier_queue.append(neighbor)
                print(f"  -> Found new node '{neighbor}'. PUSH to the TAIL of the queue.")

    print("\n" + "=" * 80)
    print("Goal not found. The frontier is empty.")
    print("=" * 80)


# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    # Corrected unweighted graph
    graph_bfs = {
        'S': ['D'], 'A': ['B', 'D', 'H'], 'B': ['A', 'C', 'K'], 'C': ['B', 'E'],
        'D': ['S', 'A', 'F', 'I'], 'E': ['C', 'K'], 'F': ['D', 'L'], 'G': ['N', 'T'],
        'H': ['A', 'J', 'K', 'N'], 'I': ['D', 'L', 'M'], 'J': ['H', 'M'],
        'K': ['E', 'H', 'N', 'Q', 'B'], 'L': ['F', 'I', 'O'], 'M': ['I', 'J', 'P'],
        'N': ['H', 'K', 'Q', 'G'], 'O': ['L', 'R'], 'P': ['M'], 'Q': ['K', 'N'],
        'R': ['O', 'T'], 'T': ['G', 'R']
    }

    # Corrected weighted graph
    graph_with_costs = {
        'S': [('D', 25)], 'A': [('B', 11), ('D', 32), ('H', 36)],
        'B': [('A', 11), ('C', 24), ('K', 42)], 'C': [('B', 24), ('E', 40)],
        'D': [('S', 25), ('A', 32), ('F', 24), ('I', 26)], 'E': [('C', 40), ('K', 32)],
        'F': [('D', 24), ('L', 27)], 'G': [('N', 42), ('T', 32)],
        'H': [('A', 36), ('J', 22), ('K', 28), ('N', 44)], 'I': [('D', 26), ('L', 21), ('M', 32)],
        'J': [('H', 22), ('M', 20)], 'K': [('E', 32), ('H', 28), ('N', 27), ('Q', 62), ('B', 42)],
        'L': [('F', 27), ('I', 21), ('O', 26)], 'M': [('I', 32), ('J', 20), ('P', 23)],
        'N': [('H', 44), ('K', 27), ('Q', 32), ('G', 42)], 'O': [('L', 26), ('R', 27)],
        'P': [('M', 23)], 'Q': [('K', 62), ('N', 32)], 'R': [('O', 27), ('T', 52)],
        'T': [('G', 32), ('R', 52)]
    }

    # Run the BFS trace with push/pop details
    bfs_trace_push_pop(graph_bfs, graph_with_costs, 'S', 'G')