# Week 1: Lab Assignment - 1
# State Space Search using BFS and DFS

from collections import deque


# ----------------------------------------------------------------------------
# PART 1: GENERIC SEARCH ALGORITHMS (BFS and DFS)
# ----------------------------------------------------------------------------

def reconstruct_path(parent_map, goal_state):
    """
    Reconstructs the path from the goal state back to the initial state
    using the parent map.
    """
    path = []
    current = goal_state
    while current is not None:
        path.append(current)
        current = parent_map.get(current)
    return path[::-1]  # Reverse the path to get it from start to goal


def bfs(initial_state, goal_state_func, get_successors_func):
    """
    Performs a Breadth-First Search.

    Args:
        initial_state: The starting state.
        goal_state_func: A function that returns True if a state is the goal.
        get_successors_func: A function that returns a list of valid successor states.

    Returns:
        The path from the initial state to the goal state, or None if not found.
    """
    frontier = deque([initial_state])
    visited = {initial_state}
    parent_map = {initial_state: None}

    while frontier:
        current_state = frontier.popleft()

        if goal_state_func(current_state):
            return reconstruct_path(parent_map, current_state)

        for successor in get_successors_func(current_state):
            if successor not in visited:
                visited.add(successor)
                parent_map[successor] = current_state
                frontier.append(successor)

    return None  # No solution found


def dfs(initial_state, goal_state_func, get_successors_func):
    """
    Performs a Depth-First Search.

    Args:
        initial_state: The starting state.
        goal_state_func: A function that returns True if a state is the goal.
        get_successors_func: A function that returns a list of valid successor states.

    Returns:
        The path from the initial state to the goal state, or None if not found.
    """
    frontier = [initial_state]  # Using a list as a stack
    visited = {initial_state}
    parent_map = {initial_state: None}

    while frontier:
        current_state = frontier.pop()  # LIFO for DFS

        if goal_state_func(current_state):
            return reconstruct_path(parent_map, current_state)

        # Note: We reverse the successors to maintain a more intuitive
        # exploration order (left-to-right in the conceptual search tree)
        for successor in reversed(get_successors_func(current_state)):
            if successor not in visited:
                visited.add(successor)
                parent_map[successor] = current_state
                frontier.append(successor)

    return None  # No solution found


# ----------------------------------------------------------------------------
# PART 2: PROBLEM-SPECIFIC LOGIC
# ----------------------------------------------------------------------------

### --- Missionaries and Cannibals Problem Logic ---

def is_valid_mc_state(state):
    """
    Checks if a state is valid for the M&C problem.
    State is a tuple (missionaries, cannibals, boat_position)
    """
    m, c, _ = state
    # Check bounds
    if not (0 <= m <= 3 and 0 <= c <= 3):
        return False
    # Check for missionaries being outnumbered on the starting bank
    if m > 0 and m < c:
        return False
    # Check for missionaries being outnumbered on the destination bank
    m_dest = 3 - m
    c_dest = 3 - c
    if m_dest > 0 and m_dest < c_dest:
        return False
    return True


def get_successors_mc(state):
    """
    Generates all valid successor states for the M&C problem.
    """
    successors = []
    m, c, b = state

    # Possible moves (missionaries, cannibals)
    moves = [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]

    for dm, dc in moves:
        if b == 1:  # Boat is on the start bank, moving to destination
            new_state = (m - dm, c - dc, 0)
        else:  # Boat is on the destination bank, moving back to start
            new_state = (m + dm, c + dc, 1)

        if is_valid_mc_state(new_state):
            successors.append(new_state)

    return successors


### --- Rabbit Leap Problem Logic ---

def get_successors_rl(state):
    """
    Generates all valid successor states for the Rabbit Leap problem.
    State is a tuple representing the stones.
    """
    successors = []
    state_list = list(state)
    empty_index = state_list.index('_')

    # Try all possible moves into the empty spot
    # Note: Rabbits only move forward

    # Check for 'E' rabbits that can move
    # Slide E -> _
    if empty_index > 0 and state_list[empty_index - 1] == 'E':
        new_list = state_list[:]
        new_list[empty_index], new_list[empty_index - 1] = new_list[empty_index - 1], new_list[empty_index]
        successors.append(tuple(new_list))
    # Jump E _ W -> _ W E
    if empty_index > 1 and state_list[empty_index - 2] == 'E':
        new_list = state_list[:]
        new_list[empty_index], new_list[empty_index - 2] = new_list[empty_index - 2], new_list[empty_index]
        successors.append(tuple(new_list))

    # Check for 'W' rabbits that can move
    # Slide _ <- W
    if empty_index < 6 and state_list[empty_index + 1] == 'W':
        new_list = state_list[:]
        new_list[empty_index], new_list[empty_index + 1] = new_list[empty_index + 1], new_list[empty_index]
        successors.append(tuple(new_list))
    # Jump W E _ -> _ E W
    if empty_index < 5 and state_list[empty_index + 2] == 'W':
        new_list = state_list[:]
        new_list[empty_index], new_list[empty_index + 2] = new_list[empty_index + 2], new_list[empty_index]
        successors.append(tuple(new_list))

    return successors


# ----------------------------------------------------------------------------
# PART 3: MAIN EXECUTION AND PRINTING
# ----------------------------------------------------------------------------

def print_solution(problem_name, algorithm_name, path):
    """Helper function to print the solution path."""
    print(f"\n--- {problem_name}: Solution found using {algorithm_name} ---")
    if path:
        print(f"Optimal solution: {'Yes' if algorithm_name == 'BFS' else 'Not Guaranteed'}")
        print(f"Total steps: {len(path) - 1}")
        print("Path:")
        for i, state in enumerate(path):
            if isinstance(state, tuple) and len(state) > 5:  # Rabbit Leap state
                print(f"  Step {i}: {''.join(state)}")
            else:  # M&C state
                print(f"  Step {i}: {state}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    # --- Solve Missionaries and Cannibals ---
    mc_initial_state = (3, 3, 1)
    mc_goal_func = lambda s: s == (0, 0, 0)

    bfs_path_mc = bfs(mc_initial_state, mc_goal_func, get_successors_mc)
    print_solution("Missionaries & Cannibals", "BFS", bfs_path_mc)

    dfs_path_mc = dfs(mc_initial_state, mc_goal_func, get_successors_mc)
    print_solution("Missionaries & Cannibals", "DFS", dfs_path_mc)

    # --- Solve Rabbit Leap ---
    # Using tuples for states because they are hashable and can be stored in sets/dicts
    rl_initial_state = ('E', 'E', 'E', '_', 'W', 'W', 'W')
    rl_goal_state = ('W', 'W', 'W', '_', 'E', 'E', 'E')
    rl_goal_func = lambda s: s == rl_goal_state

    bfs_path_rl = bfs(rl_initial_state, rl_goal_func, get_successors_rl)
    print_solution("Rabbit Leap", "BFS", bfs_path_rl)

    dfs_path_rl = dfs(rl_initial_state, rl_goal_func, get_successors_rl)
    print_solution("Rabbit Leap", "DFS", dfs_path_rl)