import random
import time
from collections import namedtuple

# --- Data Structures for Reporting ---
Result = namedtuple('Result', ['solved', 'time', 'path_length', 'nodes_explored', 'penetrance', 'final_score'])


# ----------------------------------------------------------------------------
# 1. k-SAT Problem Generator
# ----------------------------------------------------------------------------
def generate_k_sat_problem(k, m, n):
    """
    Generates a uniform random k-SAT problem.
    - k: number of literals per clause
    - m: number of clauses
    - n: number of variables
    Returns a list of clauses. Each clause is a tuple of integers.
    A positive integer i represents variable x_i.
    A negative integer -i represents the negation of x_i.
    """
    clauses = []
    variables = list(range(1, n + 1))

    for _ in range(m):
        clause = []
        # Choose k distinct variables for the clause
        chosen_vars = random.sample(variables, k)
        for var in chosen_vars:
            # Randomly negate the variable
            if random.choice([True, False]):
                clause.append(var)
            else:
                clause.append(-var)
        clauses.append(tuple(clause))

    return clauses


def verify_solution(assignment, clauses):
    """Verify if an assignment satisfies all clauses."""
    for clause in clauses:
        satisfied = False
        for literal in clause:
            var = abs(literal)
            var_value = assignment[var - 1]
            if (literal > 0 and var_value) or (literal < 0 and not var_value):
                satisfied = True
                break
        if not satisfied:
            return False
    return True


# ----------------------------------------------------------------------------
# 2. Heuristic Functions
# ----------------------------------------------------------------------------
def h1_satisfied_clauses(assignment, clauses):
    """
    Heuristic 1: Counts the number of clauses satisfied by the assignment.
    The goal is to MAXIMIZE this value.
    """
    count = 0
    for clause in clauses:
        for literal in clause:
            var = abs(literal)
            var_value = assignment[var - 1]
            if (literal > 0 and var_value) or (literal < 0 and not var_value):
                count += 1
                break  # Clause is satisfied, move to the next one
    return count


def h2_weighted_satisfiability(assignment, clauses):
    """
    Heuristic 2: A weighted score. A satisfied clause gets score 1.
    An unsatisfied clause gets a partial score based on how many literals are
    "correctly" set (i.e., would satisfy the clause if they were part of a disjunction).
    This creates a smoother gradient for the search to follow.
    Goal: MAXIMIZE.
    """
    total_score = 0
    for clause in clauses:
        satisfied_literals = 0
        is_satisfied = False
        for literal in clause:
            var = abs(literal)
            var_value = assignment[var - 1]
            if (literal > 0 and var_value) or (literal < 0 and not var_value):
                is_satisfied = True
                break
            # Count literals that match their required value
            elif (literal > 0 and not var_value) or (literal < 0 and var_value):
                satisfied_literals += 1

        if is_satisfied:
            total_score += 1
        else:
            # For an unsatisfied clause, give partial credit
            k = len(clause)
            total_score += (k - satisfied_literals) / k

    return total_score


# ----------------------------------------------------------------------------
# 3. Base Solver and Helper Functions
# ----------------------------------------------------------------------------
def get_random_assignment(n):
    return tuple(random.choice([True, False]) for _ in range(n))


class SatSolver:
    """Base class for our SAT solvers."""

    def __init__(self, clauses, n, heuristic_func, max_steps=10000):
        self.clauses = clauses
        self.m = len(clauses)
        self.n = n
        self.heuristic = heuristic_func
        self.max_steps = max_steps
        self.nodes_explored = 0
        self.path_length = 0

    def _evaluate(self, assignment):
        self.nodes_explored += 1
        return self.heuristic(assignment, self.clauses)

    def solve(self):
        raise NotImplementedError


# ----------------------------------------------------------------------------
# 4. Local Search Algorithm Implementations
# ----------------------------------------------------------------------------

class HillClimbingSolver(SatSolver):
    """Hill-Climbing with random restarts."""

    def solve(self, max_restarts=50):
        start_time = time.time()
        best_overall_score = -1

        for restart in range(max_restarts):
            current_assignment = get_random_assignment(self.n)
            current_score = self._evaluate(current_assignment)

            for _ in range(self.max_steps):
                if current_score == self.m:  # Found a solution
                    penetrance = self.path_length / self.nodes_explored if self.nodes_explored > 0 else 0
                    return Result(True, time.time() - start_time, self.path_length,
                                  self.nodes_explored, penetrance, current_score)

                best_neighbor, best_neighbor_score = None, -1

                # Explore neighbors (1-flip)
                for i in range(self.n):
                    neighbor = list(current_assignment)
                    neighbor[i] = not neighbor[i]
                    neighbor = tuple(neighbor)

                    neighbor_score = self._evaluate(neighbor)
                    if neighbor_score > best_neighbor_score:
                        best_neighbor_score = neighbor_score
                        best_neighbor = neighbor

                if best_neighbor_score > current_score:
                    current_assignment = best_neighbor
                    current_score = best_neighbor_score
                    self.path_length += 1
                else:
                    break  # Reached local maximum

            best_overall_score = max(best_overall_score, current_score)

        # Failed to find a solution after all restarts
        return Result(False, time.time() - start_time, self.path_length,
                      self.nodes_explored, 0, best_overall_score)


class BeamSearchSolver(SatSolver):
    """Beam Search implementation."""

    def __init__(self, clauses, n, heuristic_func, max_steps, beam_width):
        super().__init__(clauses, n, heuristic_func, max_steps)
        self.beam_width = beam_width

    def solve(self):
        start_time = time.time()

        # Initialize beam with random assignments
        beam = [get_random_assignment(self.n) for _ in range(self.beam_width)]

        for step in range(self.max_steps):
            # Check current beam for solutions
            for assignment in beam:
                score = self._evaluate(assignment)
                if score == self.m:
                    self.path_length = step
                    penetrance = self.path_length / self.nodes_explored if self.nodes_explored > 0 else 0
                    return Result(True, time.time() - start_time, self.path_length,
                                  self.nodes_explored, penetrance, score)

            # Generate all successors
            all_successors = set()
            for assignment in beam:
                for i in range(self.n):
                    neighbor = list(assignment)
                    neighbor[i] = not neighbor[i]
                    all_successors.add(tuple(neighbor))

            if not all_successors:
                break

            # Evaluate all successors and select the best ones for the new beam
            sorted_successors = sorted(list(all_successors),
                                       key=lambda s: self._evaluate(s), reverse=True)
            beam = sorted_successors[:self.beam_width]
            self.path_length = step + 1

        final_score = self._evaluate(beam[0]) if beam else -1
        return Result(False, time.time() - start_time, self.path_length,
                      self.nodes_explored, 0, final_score)


class VndSolver(SatSolver):
    """Variable Neighborhood Descent implementation."""

    def _find_best_neighbor_n1(self, assignment):
        """Neighborhood 1: 1-flip"""
        best_neighbor, best_score = None, -1
        for i in range(self.n):
            neighbor = list(assignment)
            neighbor[i] = not neighbor[i]
            neighbor = tuple(neighbor)
            score = self._evaluate(neighbor)
            if score > best_score:
                best_neighbor, best_score = neighbor, score
        return best_neighbor, best_score

    def _find_best_neighbor_n2(self, assignment):
        """Neighborhood 2: 2-flip (stochastic sample to be efficient)"""
        best_neighbor, best_score = None, -1
        # Sample n*2 random pairs to flip instead of all nC2 pairs
        sample_size = min(self.n * 2, (self.n * (self.n - 1)) // 2)
        for _ in range(sample_size):
            i, j = random.sample(range(self.n), 2)
            neighbor = list(assignment)
            neighbor[i] = not neighbor[i]
            neighbor[j] = not neighbor[j]
            neighbor = tuple(neighbor)
            score = self._evaluate(neighbor)
            if score > best_score:
                best_neighbor, best_score = neighbor, score
        return best_neighbor, best_score

    def _find_best_neighbor_n3(self, assignment):
        """Neighborhood 3: Targeted flip from a random unsatisfied clause"""
        unsatisfied_clauses = []
        for clause in self.clauses:
            is_sat = False
            for literal in clause:
                var = abs(literal)
                if (literal > 0 and assignment[var - 1]) or (literal < 0 and not assignment[var - 1]):
                    is_sat = True
                    break
            if not is_sat:
                unsatisfied_clauses.append(clause)

        if not unsatisfied_clauses:
            return None, -1

        target_clause = random.choice(unsatisfied_clauses)
        best_neighbor, best_score = None, -1
        for literal in target_clause:
            var_idx_to_flip = abs(literal) - 1
            neighbor = list(assignment)
            neighbor[var_idx_to_flip] = not neighbor[var_idx_to_flip]
            neighbor = tuple(neighbor)
            score = self._evaluate(neighbor)
            if score > best_score:
                best_neighbor, best_score = neighbor, score
        return best_neighbor, best_score

    def solve(self):
        start_time = time.time()
        current_assignment = get_random_assignment(self.n)

        neighborhood_functions = [
            self._find_best_neighbor_n1,
            self._find_best_neighbor_n2,
            self._find_best_neighbor_n3
        ]

        for step in range(self.max_steps):
            current_score = self._evaluate(current_assignment)
            if current_score == self.m:
                self.path_length = step
                penetrance = self.path_length / self.nodes_explored if self.nodes_explored > 0 else 0
                return Result(True, time.time() - start_time, self.path_length,
                              self.nodes_explored, penetrance, current_score)

            improved = False
            k = 0
            while k < len(neighborhood_functions):
                find_neighbor_func = neighborhood_functions[k]
                neighbor, neighbor_score = find_neighbor_func(current_assignment)

                if neighbor and neighbor_score > current_score:
                    current_assignment = neighbor
                    current_score = neighbor_score
                    self.path_length += 1
                    improved = True
                    k = 0  # Go back to the first neighborhood
                else:
                    k += 1  # Try next neighborhood

            if not improved:
                break  # No improvement in any neighborhood

        final_score = self._evaluate(current_assignment)
        return Result(False, time.time() - start_time, self.path_length,
                      self.nodes_explored, 0, final_score)


# ----------------------------------------------------------------------------
# 5. Experiment Runner
# ----------------------------------------------------------------------------
def run_experiments():
    """Main function to run all comparisons."""

    random.seed(42)  # For reproducibility

    # Configurations for 3-SAT problems (m=clauses, n=variables)
    # The hardness of SAT problems is related to the ratio m/n
    # For 3-SAT, the phase transition (hardest problems) is around m/n = 4.26
    problem_configs = [
        {'k': 3, 'm': 60, 'n': 20, 'desc': 'Easy (m/n=3.0)'},
        {'k': 3, 'm': 85, 'n': 20, 'desc': 'Hard (m/n=4.25)'},
        {'k': 3, 'm': 100, 'n': 20, 'desc': 'Harder (m/n=5.0)'}
    ]

    heuristics = {
        'H1 (Satisfied Clauses)': h1_satisfied_clauses,
        'H2 (Weighted Score)': h2_weighted_satisfiability
    }

    results_log = []

    for config in problem_configs:
        print(f"\n{'=' * 80}")
        print(f"Generating Problem: k={config['k']}, m={config['m']}, n={config['n']} ({config['desc']})")
        print(f"{'=' * 80}")

        clauses = generate_k_sat_problem(config['k'], config['m'], config['n'])

        for h_name, h_func in heuristics.items():
            print(f"\n--- Using Heuristic: {h_name} ---")

            # --- Hill-Climbing ---
            print("  Running Hill-Climbing...")
            solver_hc = HillClimbingSolver(clauses, config['n'], h_func)
            result_hc = solver_hc.solve()
            results_log.append(('Hill-Climbing', h_name, config['desc'], result_hc))

            # --- Beam Search (width 3) ---
            print("  Running Beam Search (w=3)...")
            solver_bs3 = BeamSearchSolver(clauses, config['n'], h_func, max_steps=1000, beam_width=3)
            result_bs3 = solver_bs3.solve()
            results_log.append(('Beam Search (w=3)', h_name, config['desc'], result_bs3))

            # --- Beam Search (width 4) ---
            print("  Running Beam Search (w=4)...")
            solver_bs4 = BeamSearchSolver(clauses, config['n'], h_func, max_steps=1000, beam_width=4)
            result_bs4 = solver_bs4.solve()
            results_log.append(('Beam Search (w=4)', h_name, config['desc'], result_bs4))

            # --- Variable Neighborhood Descent ---
            print("  Running VND...")
            solver_vnd = VndSolver(clauses, config['n'], h_func, max_steps=1000)
            result_vnd = solver_vnd.solve()
            results_log.append(('VND', h_name, config['desc'], result_vnd))

    # --- Print Final Report ---
    print(f"\n\n{'=' * 120}")
    print("PERFORMANCE COMPARISON REPORT")
    print(f"{'=' * 120}")
    print(
        f"{'Algorithm':<22} | {'Heuristic':<27} | {'Problem':<18} | {'Solved':<7} | {'Time (s)':<10} | {'Nodes':<10} | {'Penetrance':<12} | {'Final Score'}")
    print(f"{'-' * 120}")

    for r in results_log:
        algo, h, prob, res = r
        solved_str = '✓ Yes' if res.solved else '✗ No'
        score_str = f"{int(res.final_score)}/{problem_configs[0]['m'] if 'Easy' in prob else (problem_configs[1]['m'] if 'Hard' in prob and 'm/n=4.25' in prob else problem_configs[2]['m'])}"
        print(
            f"{algo:<22} | {h:<27} | {prob:<18} | {solved_str:<7} | {res.time:<10.4f} | {res.nodes_explored:<10} | {res.penetrance:<12.6f} | {score_str}")

    print(f"{'=' * 120}")

    # Summary Statistics
    print("\n\nSUMMARY ANALYSIS")
    print("=" * 80)

    # Group results by algorithm
    algo_results = {}
    for algo, h, prob, res in results_log:
        key = f"{algo} + {h}"
        if key not in algo_results:
            algo_results[key] = {'solved': 0, 'total': 0, 'avg_penetrance': []}
        algo_results[key]['total'] += 1
        if res.solved:
            algo_results[key]['solved'] += 1
            algo_results[key]['avg_penetrance'].append(res.penetrance)

    print("\nSuccess Rate and Average Penetrance (for successful runs):")
    print("-" * 80)
    for key, stats in sorted(algo_results.items()):
        success_rate = (stats['solved'] / stats['total']) * 100
        avg_pen = sum(stats['avg_penetrance']) / len(stats['avg_penetrance']) if stats['avg_penetrance'] else 0
        print(f"{key:<50} | Success: {success_rate:5.1f}% | Avg Penetrance: {avg_pen:.6f}")

    print("=" * 80)


if __name__ == "__main__":
    run_experiments()