import heapq
import re
import time


# ----------------------------------------------------------------------------
# 1. Text Preprocessing and Edit Distance
# ----------------------------------------------------------------------------

def preprocess_text(text):
    """
    Tokenizes text into sentences and normalizes them.
    - Splits text into sentences.
    - Converts to lowercase.
    - Removes punctuation.
    - Filters out empty sentences.
    """
    # Split into sentences using punctuation as delimiters
    sentences = re.split(r'[.?!]\s*', text)

    normalized_sentences = []
    for sentence in sentences:
        if not sentence:
            continue
        # Convert to lowercase and remove all non-alphanumeric characters (except spaces)
        clean_sentence = re.sub(r'[^\w\s]', '', sentence).lower()
        normalized_sentences.append(clean_sentence.strip())

    return [s for s in normalized_sentences if s]


def levenshtein_distance(s1_words, s2_words):
    """
    Computes the Levenshtein distance between two sequences of words.
    This is the cost of aligning two sentences.
    """
    m, n = len(s1_words), len(s2_words)

    # Initialize DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1_words[i - 1] == s2_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                           dp[i][j - 1] + 1,  # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[m][n]


# ----------------------------------------------------------------------------
# 2. A* Search Algorithm for Text Alignment
# ----------------------------------------------------------------------------

def a_star_align(doc1_sents, doc2_sents):
    """
    Aligns two documents (lists of sentences) using A* search.
    Returns the alignment path and the total edit distance (g_cost).
    """
    # Tokenize sentences into words for cost calculation
    doc1_words = [s.split() for s in doc1_sents]
    doc2_words = [s.split() for s in doc2_sents]

    initial_state = (0, 0)
    goal_state = (len(doc1_sents), len(doc2_sents))

    def heuristic(i, j):
        return abs((len(doc1_sents) - i) - (len(doc2_sents) - j))

    open_set = [(heuristic(0, 0), 0, initial_state, [])]
    g_costs = {initial_state: 0}

    while open_set:
        f_cost, g_cost, current_state, path = heapq.heappop(open_set)

        if current_state == goal_state:
            return path, g_cost  # Goal reached, return path and final cost

        i, j = current_state
        successors = []

        if i < len(doc1_sents) and j < len(doc2_sents):
            match_cost = levenshtein_distance(doc1_words[i], doc2_words[j])
            successors.append(('align', (i + 1, j + 1), match_cost))

        if i < len(doc1_sents):
            skip_doc1_cost = len(doc1_words[i]) if doc1_words[i] else 1
            successors.append(('skip_doc1', (i + 1, j), skip_doc1_cost))

        if j < len(doc2_sents):
            skip_doc2_cost = len(doc2_words[j]) if doc2_words[j] else 1
            successors.append(('skip_doc2', (i, j + 1), skip_doc2_cost))

        for action, next_state, cost in successors:
            new_g_cost = g_cost + cost
            if next_state not in g_costs or new_g_cost < g_costs[next_state]:
                g_costs[next_state] = new_g_cost
                h_cost = heuristic(next_state[0], next_state[1])
                new_f_cost = new_g_cost + h_cost
                new_path = path + [(action, i, j)]
                heapq.heappush(open_set, (new_f_cost, new_g_cost, next_state, new_path))

    return None, float('inf')  # Return None if no path is found


# ----------------------------------------------------------------------------
# 3. Plagiarism Detection and Evaluation
# ----------------------------------------------------------------------------

def detect_plagiarism(alignment_path, doc1_sents, doc2_sents, similarity_threshold=0.8):
    """
    Analyzes the alignment path to identify potential plagiarism.
    """
    print(f"\nAnalyzing alignment for plagiarism (Threshold: {similarity_threshold * 100:.0f}% similarity)...")
    plagiarized_count = 0

    if not alignment_path:
        print("Could not generate an alignment path.")
        return

    for action, i, j in alignment_path:
        if action == 'align':
            s1 = doc1_sents[i]
            s2 = doc2_sents[j]
            s1_words = s1.split()
            s2_words = s2.split()

            if not s1_words and not s2_words:
                continue

            dist = levenshtein_distance(s1_words, s2_words)
            max_len = max(len(s1_words), len(s2_words))
            similarity = 1 - (dist / max_len) if max_len > 0 else 1.0

            if similarity >= similarity_threshold:
                plagiarized_count += 1
                print(f"\n[Potential Plagiarism Found] (Similarity: {similarity:.2%})")
                print(f"  - DOC 1, Sent {i + 1}: '{s1}'")
                print(f"  - DOC 2, Sent {j + 1}: '{s2}'")

    if plagiarized_count == 0:
        print("No sentences met the plagiarism threshold.")


# ----------------------------------------------------------------------------
# 4. Main Execution with Test Cases
# ----------------------------------------------------------------------------

def run_test_case(case_name, doc1_path, doc2_path, threshold=0.8):
    """Helper function to run a single test case from files."""
    print("\n" + "=" * 50)
    print(f"Test Case: {case_name}")
    print(f"Comparing '{doc1_path}' and '{doc2_path}'")
    print("=" * 50)
    try:
        with open(doc1_path, 'r') as f1, open(doc2_path, 'r') as f2:
            doc1_text = f1.read()
            doc2_text = f2.read()

        sents1 = preprocess_text(doc1_text)
        sents2 = preprocess_text(doc2_text)

        # --- Time the alignment process ---
        start_time = time.time()
        alignment, total_distance = a_star_align(sents1, sents2)
        end_time = time.time()
        duration = end_time - start_time

        # --- Print the new information ---
        if alignment is not None:
            print(f"Alignment completed in {duration:.4f} seconds.")
            print(f"Optimal alignment found with a total edit distance of: {total_distance}")
        else:
            print("Alignment failed or no path was found.")

        # --- Run plagiarism analysis ---
        detect_plagiarism(alignment, sents1, sents2, similarity_threshold=threshold)

    except FileNotFoundError:
        print(f"\nError: Could not find '{doc1_path}' or '{doc2_path}'.")
        print("Please ensure all test files are in the same directory as the script.")


if __name__ == "__main__":
    # Run all test cases by reading from external files
    run_test_case("Identical Documents",
                  "test_case_1_doc1.txt",
                  "test_case_1_doc2.txt")

    run_test_case("Slightly Modified Document",
                  "test_case_2_doc1.txt",
                  "test_case_2_doc2.txt")

    run_test_case("Completely Different Documents",
                  "test_case_3_doc1.txt",
                  "test_case_3_doc2.txt")

    run_test_case("Partial Overlap",
                  "test_case_4_doc1.txt",
                  "test_case_4_doc2.txt",
                  threshold=0.9)

