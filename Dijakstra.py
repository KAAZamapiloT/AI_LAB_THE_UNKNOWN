# ==============================================================================
# Dijkstra's Algorithm with Detailed PUSH/POP Tracing
# For academic presentation and notebook transcription.
#
# Date: 19 August 2025
# ==============================================================================

import heapq


def reconstruct_path(parent, start, goal):
    """Helper function to reconstruct the path from the parent dictionary."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]


def dijkstra_trace_push_pop(graph_costs, start, goal):
    """
    Performs Dijkstra's Algorithm and prints a detailed trace describing
    each PUSH and POP operation on the Priority Queue.
    """
    # 1. Initialize data structures
    # The frontier is a priority queue storing tuples of (cost, node_name)
    frontier_pq = [(0, start)]
    explored_set = set()
    parent = {start: None}
    # A dictionary to store the lowest cost found so far to reach each node
    path_costs = {start: 0}

    step = 0

    # Print header
    print("=" * 80)
    print("Executing Dijkstra's Algorithm with PUSH/POP Operations")
    print(f"Finding path from '{start}' to '{goal}'")

    # 2. Begin the search loop
    while frontier_pq:
        step += 1
        print("\n" + f"--- Step {step} ---")

        # Report state before the POP
        # Note: The 'head' of a priority queue is always the item with the lowest cost
        print(f"Frontier (Priority Queue ordered by cost): {sorted(frontier_pq)}")

        # POP the node with the lowest cost from the priority queue
        current_cost, current_node = heapq.heappop(frontier_pq)
        print(f"Action: POP node '{current_node}' with cost {current_cost} (the lowest in the frontier).")

        # Report state after the POP
        print(f"Frontier after action:  {sorted(frontier_pq)}")

        # If we have already found a cheaper path to this node, skip it.
        # This handles outdated entries in the priority queue.
        if current_node in explored_set:
            print(f"Note: Node '{current_node}' has already been explored via a cheaper path. Skipping.")
            continue

        # 3. Goal Check
        if current_node == goal:
            print("\n" + "-" * 80)
            print("Goal reached!")
            path = reconstruct_path(parent, start, goal)
            print(f"\nPath Found: {' -> '.join(path)}")
            print(f"Number of Edges: {len(path) - 1}")
            print(f"Total Cost: {current_cost}")
            print("=" * 80)
            return

        # 4. Add the current node to the explored set
        explored_set.add(current_node)
        print(f"Explored Set updated: {sorted(list(explored_set))}")

        # 5. Expand the node and PUSH its neighbors to the frontier
        print("Discovering neighbors...")
        for neighbor, edge_weight in sorted(graph_costs[current_node]):
            new_cost = current_cost + edge_weight

            # If we've found a new node or a shorter path to an existing one
            if neighbor not in path_costs or new_cost < path_costs[neighbor]:
                path_costs[neighbor] = new_cost
                parent[neighbor] = current_node
                # PUSH the new (cost, neighbor) pair to the priority queue
                heapq.heappush(frontier_pq, (new_cost, neighbor))
                print(f"  -> Found new/better path to '{neighbor}' with cost {new_cost}. PUSH to queue.")

    print("\n" + "=" * 80)
    print("Goal not found. The frontier is empty.")
    print("=" * 80)


# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == "__main__":
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

    # Run the Dijkstra's trace with push/pop details
    dijkstra_trace_push_pop(graph_with_costs, 'S', 'G')