# ==============================================================================
# CORRECTED Greedy Best-First Search with Detailed Tracing
# - Fixes the bug in path reconstruction to show the final path.
#
# Date: 19 August 2025
# ==============================================================================

import heapq
import math


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
    if len(path) < 2:
        return 0
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        edge_cost = next((c for neighbor, c in graph_costs[u] if neighbor == v), 0)
        cost += edge_cost
    return cost


def calculate_manhattan_heuristic(node, goal, coordinates):
    """Calculates the Manhattan distance heuristic (h1)."""
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return abs(x2 - x1) + abs(y2 - y1)


def calculate_euclidean_heuristic(node, goal, coordinates):
    """Calculates the Euclidean distance heuristic (h2)."""
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def greedy_trace_push_pop(graph, graph_costs, coordinates, start, goal, heuristic_func):
    """
    Performs a Greedy Best-First Search and prints a detailed trace describing
    each PUSH and POP operation on the Priority Queue.
    """
    h_start = heuristic_func(start, goal, coordinates)
    frontier_pq = [(h_start, start)]
    parent = {start: None}
    explored_set = set()
    frontier_set = {start}
    step = 0

    print("=" * 80)
    print(f"Executing Greedy Best-First Search with {heuristic_func.__name__}")
    print(f"Finding path from '{start}' to '{goal}'")

    while frontier_pq:
        step += 1
        print("\n" + f"--- Step {step} ---")

        print(f"Frontier (Priority Queue ordered by h): {sorted(frontier_pq)}")
        current_h, current_node = heapq.heappop(frontier_pq)
        frontier_set.remove(current_node)
        print(f"Action: POP node '{current_node}' with h={current_h:.2f} (the lowest heuristic in the frontier).")

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

        explored_set.add(current_node)
        print(f"Explored Set updated: {sorted(list(explored_set))}")

        print("Discovering neighbors...")
        for neighbor in sorted(graph[current_node]):
            if neighbor not in explored_set and neighbor not in frontier_set:
                # =====================================================
                # THIS IS THE CORRECTED LINE:
                parent[neighbor] = current_node
                # =====================================================
                h_neighbor = heuristic_func(neighbor, goal, coordinates)
                heapq.heappush(frontier_pq, (h_neighbor, neighbor))
                frontier_set.add(neighbor)
                print(f"  -> Found new node '{neighbor}'. PUSH to queue with h={h_neighbor:.2f}.")

    print("\n" + "=" * 80)
    print("Goal not found. The frontier is empty.")
    print("=" * 80)


# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
    node_coordinates = {
        'S': (0, 8), 'A': (3, 9), 'B': (5, 9), 'C': (8, 9), 'D': (2, 7), 'E': (10, 8),
        'F': (0, 5), 'G': (11, 1), 'H': (5, 6), 'I': (2, 4), 'J': (4, 5), 'K': (8, 6),
        'L': (0, 3), 'M': (4, 3), 'N': (8, 3), 'O': (2, 2), 'P': (5, 2), 'Q': (10, 3),
        'R': (2, 0), 'T': (8, 0)
    }
    graph_bfs = {
        'S': ['D'], 'A': ['B', 'D', 'H'], 'B': ['A', 'C', 'K'], 'C': ['B', 'E'],
        'D': ['S', 'A', 'F', 'I'], 'E': ['C', 'K'], 'F': ['D', 'L'], 'G': ['N', 'T'],
        'H': ['A', 'J', 'K', 'N'], 'I': ['D', 'L', 'M'], 'J': ['H', 'M'],
        'K': ['E', 'H', 'N', 'Q', 'B'], 'L': ['F', 'I', 'O'], 'M': ['I', 'J', 'P'],
        'N': ['H', 'K', 'Q', 'G'], 'O': ['L', 'R'], 'P': ['M'], 'Q': ['K', 'N'],
        'R': ['O', 'T'], 'T': ['G', 'R']
    }
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

    # Running with Euclidean Heuristic as requested
    greedy_trace_push_pop(graph_bfs, graph_with_costs, node_coordinates, 'S', 'G', calculate_euclidean_heuristic)