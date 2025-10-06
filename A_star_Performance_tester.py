import time
import random
import tracemalloc
import os

# --- IMPORTANT ---
# This script MUST be in the same directory as your 'A_star_PlagDetector.py' file.
try:
    from A_star_PlagDetector import preprocess_text, a_star_align
except ImportError:
    print("Error: Could not find 'A_star_PlagDetector.py'.")
    print("Please ensure both scripts are in the same directory.")
    exit()

# A pool of unique sentences to generate test documents from
SAMPLE_SENTENCES = [
    "The sun rises in the east and sets in the west.",
    "Artificial intelligence is a rapidly evolving field of study.",
    "The A* search algorithm guarantees an optimal path if the heuristic is admissible.",
    "Data structures are fundamental to efficient programming.",
    "The quick brown fox jumps over the lazy dog.",
    "Memory management is a critical aspect of system performance.",
    "Natural language processing enables computers to understand human speech.",
    "The complexity of an algorithm describes its resource usage.",
    "Graph theory has many applications in computer science and logistics.",
    "Object-oriented programming is a popular paradigm.",
    "The capital of France is Paris.",
    "Quantum computing promises to solve currently intractable problems.",
    "Machine learning models are trained on large datasets.",
    "A recursive function is one that calls itself.",
    "The internet has revolutionized communication and information sharing."
]


def generate_test_docs(doc1_path, doc2_path, num_sentences):
    """Generates two text files with a specified number of sentences."""
    # Create the first document
    with open(doc1_path, 'w') as f:
        for _ in range(num_sentences):
            f.write(random.choice(SAMPLE_SENTENCES) + "\n")

    # Create the second document with some overlap
    with open(doc2_path, 'w') as f:
        for _ in range(num_sentences):
            # 80% chance to reuse a sentence, 20% to add a different one
            if random.random() < 0.8:
                f.write(random.choice(SAMPLE_SENTENCES) + "\n")
            else:
                # Add a slightly different sentence
                f.write(random.choice(SAMPLE_SENTENCES).replace(" is ", " was ") + "\n")


def run_performance_test():
    """
    Tests the A* alignment function with increasing document sizes
    and reports the time and memory usage for each.
    """
    # Define the sizes of the documents to test (in number of sentences)
    # Be careful with high numbers, as performance degrades exponentially!
    test_sizes = [10, 20, 40, 60, 80, 100]

    results = []

    print("=" * 60)
    print("A* Alignment Performance Test")
    print("=" * 60)
    print("This script will test time and memory usage for A* alignment.")
    print("Note: Runtimes may be long for larger sentence counts.\n")

    for size in test_sizes:
        print(f"[*] Testing with {size} sentences per document...")
        doc1_path = f"test_{size}_doc1.txt"
        doc2_path = f"test_{size}_doc2.txt"

        # 1. Generate the test files
        generate_test_docs(doc1_path, doc2_path, size)

        # 2. Read and preprocess the text
        with open(doc1_path, 'r') as f1, open(doc2_path, 'r') as f2:
            sents1 = preprocess_text(f1.read())
            sents2 = preprocess_text(f2.read())

        # --- 3. Measure Performance ---

        # Start tracking memory allocations
        tracemalloc.start()

        # Record time
        start_time = time.time()

        # Run the A* alignment
        a_star_align(sents1, sents2)

        # Stop time and calculate duration
        end_time = time.time()
        duration = end_time - start_time

        # Get memory usage statistics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            "size": size,
            "time": duration,
            "memory_kb": peak / 1024  # Convert bytes to kilobytes
        })

        # Clean up generated files
        os.remove(doc1_path)
        os.remove(doc2_path)
        print(f"    ... Done in {duration:.4f}s, Peak Memory: {peak / 1024:.2f} KB")

    # --- 4. Print Summary Report ---
    print("\n" + "=" * 60)
    print("Performance Test Summary")
    print("=" * 60)
    print(f"{'Sentences':<15} | {'Time Taken (s)':<20} | {'Peak Memory (KB)':<20}")
    print("-" * 60)
    for res in results:
        print(f"{res['size']:<15} | {res['time']:<20.4f} | {res['memory_kb']:<20.2f}")
    print("=" * 60)
    print("\nAnalysis: As the number of sentences increases, both time and memory")
    print("usage grow at an exponential rate, demonstrating the scaling limitations.")


if __name__ == "__main__":
    run_performance_test()