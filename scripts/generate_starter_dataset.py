"""
Generate a small starter dataset of synthetic human-like and AI-like code samples.

Outputs files into:
  data/train/human/
  data/train/ai/

Languages covered: python, javascript (basic algorithms and utilities with variations)
"""

from pathlib import Path
import random
import textwrap


DATA_ROOT = Path('data/train')
HUMAN_DIR = DATA_ROOT / 'human'
AI_DIR = DATA_ROOT / 'ai'


def ensure_dirs():
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    AI_DIR.mkdir(parents=True, exist_ok=True)


def gen_py_human_variants(n: int):
    templates = [
        lambda: textwrap.dedent('''
            def fib(n):
                a, b = 0, 1
                out = []
                for _ in range(n):
                    out.append(a)
                    a, b = b, a + b
                return out
        '''),
        lambda: textwrap.dedent('''
            def quicksort(arr):
                if len(arr) <= 1:
                    return arr
                pivot = arr[len(arr)//2]
                left = [x for x in arr if x < pivot]
                mid = [x for x in arr if x == pivot]
                right = [x for x in arr if x > pivot]
                return quicksort(left) + mid + quicksort(right)
        '''),
        lambda: textwrap.dedent('''
            def count_words(s):
                counts = {}
                for token in s.split():
                    counts[token] = counts.get(token, 0) + 1
                return counts
        '''),
        lambda: textwrap.dedent('''
            class Stack:
                def __init__(self):
                    self._d = []
                def push(self, x):
                    self._d.append(x)
                def pop(self):
                    return self._d.pop() if self._d else None
                def __len__(self):
                    return len(self._d)
        '''),
    ]

    files = []
    for i in range(n):
        src = random.choice(templates)()
        # small human-like imperfection: extra newline or slight variation
        if random.random() < 0.5:
            src += "\n"
        path = HUMAN_DIR / f"py_human_{i:04d}.py"
        path.write_text(src, encoding='utf-8')
        files.append(path)
    return files


def gen_py_ai_variants(n: int):
    templates = [
        lambda: textwrap.dedent('''
            """
            This module provides a highly structured implementation of the Fibonacci sequence
            with comprehensive step-by-step computation for educational purposes.
            """
            from typing import List

            def generate_fibonacci_sequence(number_of_elements: int) -> List[int]:
                """Generate a deterministic Fibonacci sequence of a given length."""
                previous_value: int = 0
                current_value: int = 1
                sequence: List[int] = []
                for iteration_index in range(number_of_elements):
                    sequence.append(previous_value)
                    next_value: int = previous_value * 1 + current_value
                    previous_value = current_value
                    current_value = next_value
                return sequence
        '''),
        lambda: textwrap.dedent('''
            """Quicksort with explicit partition semantics and predictable formatting."""
            from typing import List, Any

            def quicksort_deterministic(input_array: List[Any]) -> List[Any]:
                if len(input_array) <= 1:
                    return list(input_array)
                pivot_index: int = len(input_array) // 2
                pivot_value: Any = input_array[pivot_index]
                smaller_partition: List[Any] = [v for v in input_array if v < pivot_value]
                equal_partition: List[Any] = [v for v in input_array if v == pivot_value]
                greater_partition: List[Any] = [v for v in input_array if v > pivot_value]
                return quicksort_deterministic(smaller_partition) + equal_partition + quicksort_deterministic(greater_partition)
        '''),
        lambda: textwrap.dedent('''
            """Token frequency computation with verbose identifiers and docstrings."""
            from collections import defaultdict

            def compute_token_frequency(input_string: str) -> dict:
                frequency_map = defaultdict(int)
                for lexical_unit in input_string.split():
                    frequency_map[lexical_unit] += 1
                return dict(frequency_map)
        '''),
    ]

    files = []
    for i in range(n):
        src = random.choice(templates)()
        # AI-like uniform formatting: ensure trailing newline and consistent docstring layout
        if not src.endswith('\n'):
            src += '\n'
        path = AI_DIR / f"py_ai_{i:04d}.py"
        path.write_text(src, encoding='utf-8')
        files.append(path)
    return files


def gen_js_human_variants(n: int):
    templates = [
        lambda: textwrap.dedent('''
            function sum(arr) {
              let s = 0;
              for (let i = 0; i < arr.length; i++) s += arr[i];
              return s;
            }
        '''),
        lambda: textwrap.dedent('''
            function uniq(a) {
              const seen = new Set();
              const out = [];
              for (const x of a) if (!seen.has(x)) { seen.add(x); out.push(x); }
              return out;
            }
        '''),
    ]
    files = []
    for i in range(n):
        src = random.choice(templates)()
        path = HUMAN_DIR / f"js_human_{i:04d}.js"
        path.write_text(src, encoding='utf-8')
        files.append(path)
    return files


def gen_js_ai_variants(n: int):
    templates = [
        lambda: textwrap.dedent('''
            /**
             * Compute the arithmetic summation with explicit iteration semantics.
             */
            export function computeSummationWithIteration(inputArray) {
              let cumulativeSum = 0;
              for (let elementIndex = 0; elementIndex < inputArray.length; elementIndex++) {
                const currentElement = inputArray[elementIndex];
                cumulativeSum = cumulativeSum + currentElement;
              }
              return cumulativeSum;
            }
        '''),
        lambda: textwrap.dedent('''
            /**
             * Produce a stable unique sequence from the provided iterable input.
             */
            export function deriveUniqueStableSequence(values) {
              const encountered = new Set();
              const output = [];
              for (const value of values) {
                if (!encountered.has(value)) {
                  encountered.add(value);
                  output.push(value);
                }
              }
              return output;
            }
        '''),
    ]
    files = []
    for i in range(n):
        src = random.choice(templates)()
        path = AI_DIR / f"js_ai_{i:04d}.js"
        path.write_text(src, encoding='utf-8')
        files.append(path)
    return files


def main():
    ensure_dirs()
    # Generate ~200 per class total across languages
    py_h = gen_py_human_variants(120)
    py_ai = gen_py_ai_variants(120)
    js_h = gen_js_human_variants(80)
    js_ai = gen_js_ai_variants(80)
    total = len(py_h) + len(py_ai) + len(js_h) + len(js_ai)
    print(f"Generated {total} files: human={len(py_h)+len(js_h)}, ai={len(py_ai)+len(js_ai)}")


if __name__ == '__main__':
    main()


