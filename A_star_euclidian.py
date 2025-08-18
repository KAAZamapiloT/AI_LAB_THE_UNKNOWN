# ==============================================================================
# A* Search with Euclidean Heuristic and Detailed PUSH/POP Tracing
# For academic presentation and notebook transcription.
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


def calculate_heuristic(node, goal, coordinates):
    """Calculates the Euclidean distance heuristic (h2)."""
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def a_star_trace_push_pop(graph_costs, coordinates, start, goal):
    """
    Performs A* Search and prints a detailed trace describing each
    PUSH and POP operation on the Priority Queue.
    """
    # 1. Initialize data structures
    # The frontier stores tuples of (f_score, node_name)
    h_start = calculate_heuristic(start, goal, coordinates)
    frontier_pq = [(h_start, start)]

    explored_set = set()
    parent = {start: None}
    # path_costs stores the g(n) score for each node
    path_costs = {start: 0}

    step = 0

    print("=" * 80)
    print("Executing A* Search with Euclidean Heuristic (h2) and PUSH/POP Operations")
    print(f"Finding path from '{start}' to '{goal}'")

    while frontier_pq:
        step += 1
        print("\n" + f"--- Step {step} ---")

        print(f"Frontier (Priority Queue ordered by f=g+h): {sorted(frontier_pq)}")

        # POP the node with the lowest f_score
        current_f, current_node = heapq.heappop(frontier_pq)
        current_g = path_costs[current_node]

        h_val = current_f - current_g
        print(
            f"Action: POP node '{current_node}' with f={current_f:.2f} (g={current_g}, h={h_val:.2f}) (the lowest in the frontier).")
        print(f"Frontier after action:  {sorted(frontier_pq)}")

        if current_node in explored_set:
            print(f"Note: Node '{current_node}' was already explored via a cheaper path. Skipping.")
            continue

        if current_node == goal:
            print("\n" + "-" * 80)
            print("Goal reached!")
            path = reconstruct_path(parent, start, goal)
            print(f"\nPath Found: {' -> '.join(path)}")
            print(f"Number of Edges: {len(path) - 1}")
            print(f"Total Cost (g): {current_g}")
            print("=" * 80)
            return

        explored_set.add(current_node)
        print(f"Explored Set updated: {sorted(list(explored_set))}")

        print("Discovering neighbors...")
        for neighbor, edge_weight in sorted(graph_costs[current_node]):
            new_g_cost = current_g + edge_weight

            if neighbor not in path_costs or new_g_cost < path_costs[neighbor]:
                path_costs[neighbor] = new_g_cost
                parent[neighbor] = current_node

                h_neighbor = calculate_heuristic(neighbor, goal, coordinates)
                new_f_score = new_g_cost + h_neighbor

                heapq.heappush(frontier_pq, (new_f_score, neighbor))
                print(
                    f"  -> Found new/better path to '{neighbor}' with g={new_g_cost}. PUSH to queue with f={new_f_score:.2f} (g={new_g_cost}, h={h_neighbor:.2f}).")

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

    a_star_trace_push_pop(graph_with_costs, node_coordinates, 'S', 'G')