#!/usr/bin/env python3
"""
Collect real AI and human code samples for training.

This script collects:
- Human code: From GitHub repositories (public code)
- AI code: Generated using OpenAI API, Anthropic Claude, or other AI models
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / 'data' / 'train'
HUMAN_DIR = DATA_DIR / 'human'
AI_DIR = DATA_DIR / 'ai'

# Ensure directories exist
HUMAN_DIR.mkdir(parents=True, exist_ok=True)
AI_DIR.mkdir(parents=True, exist_ok=True)


def collect_human_code_from_github(
    languages: List[str] = ['python', 'javascript', 'java'],
    samples_per_lang: int = 200,
    min_lines: int = 10,
    max_lines: int = 200
) -> int:
    """
    Collect human-written code from GitHub public repositories.
    Uses GitHub API to fetch real code samples.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed. Install with: pip install requests")
        return 0
    
    github_token = os.getenv('GITHUB_TOKEN', '')
    headers = {'Authorization': f'token {github_token}'} if github_token else {}
    
    collected = 0
    
    for lang in languages:
        logger.info(f"Collecting {samples_per_lang} {lang} samples from GitHub...")
        
        # Search for popular repositories
        search_url = 'https://api.github.com/search/repositories'
        params = {
            'q': f'language:{lang} stars:>10',
            'sort': 'stars',
            'order': 'desc',
            'per_page': 50
        }
        
        try:
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"GitHub API error: {response.status_code}")
                continue
            
            repos = response.json().get('items', [])
            
            for repo in repos[:20]:  # Limit to top 20 repos
                repo_name = repo['full_name']
                contents_url = f"https://api.github.com/repos/{repo_name}/contents"
                
                try:
                    contents_resp = requests.get(contents_url, headers=headers, timeout=10)
                    if contents_resp.status_code != 200:
                        continue
                    
                    files = contents_resp.json()
                    if not isinstance(files, list):
                        continue
                    
                    for file_info in files[:5]:  # Limit files per repo
                        if file_info['type'] != 'file':
                            continue
                        
                        ext_map = {
                            'python': '.py',
                            'javascript': '.js',
                            'java': '.java'
                        }
                        
                        if not file_info['name'].endswith(ext_map.get(lang, '')):
                            continue
                        
                        # Download file content
                        download_url = file_info['download_url']
                        file_resp = requests.get(download_url, timeout=10)
                        if file_resp.status_code != 200:
                            continue
                        
                        content = file_resp.text
                        lines = content.split('\n')
                        
                        if min_lines <= len(lines) <= max_lines:
                            # Save file
                            filename = f"{lang}_human_{collected:04d}.{ext_map[lang]}"
                            filepath = HUMAN_DIR / filename
                            filepath.write_text(content, encoding='utf-8')
                            collected += 1
                            
                            if collected >= samples_per_lang:
                                break
                        
                        time.sleep(0.5)  # Rate limiting
                    
                    if collected >= samples_per_lang:
                        break
                
                except Exception as e:
                    logger.debug(f"Error processing repo {repo_name}: {e}")
                    continue
                
                time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Error collecting {lang} code: {e}")
    
    logger.info(f"Collected {collected} human code samples")
    return collected


def generate_ai_code_samples(
    languages: List[str] = ['python', 'javascript', 'java'],
    samples_per_lang: int = 200,
    use_openai: bool = True,
    use_claude: bool = False
) -> int:
    """
    Generate AI code samples using various AI models.
    """
    collected = 0
    
    # Code generation prompts
    prompts = {
        'python': [
            "Write a Python function to calculate the factorial of a number with proper error handling.",
            "Create a Python class for a binary search tree with insert, search, and delete methods.",
            "Write a Python function to parse a CSV file and return a dictionary of column data.",
            "Create a Python decorator that measures execution time of a function.",
            "Write a Python function to find the longest common subsequence between two strings.",
        ],
        'javascript': [
            "Write a JavaScript function to implement a debounce mechanism for event handlers.",
            "Create a JavaScript class for a priority queue with enqueue and dequeue methods.",
            "Write a JavaScript function to deep clone a nested object.",
            "Create a JavaScript async function to fetch data from multiple APIs in parallel.",
            "Write a JavaScript function to validate an email address using regex.",
        ],
        'java': [
            "Write a Java class to implement a thread-safe singleton pattern.",
            "Create a Java method to sort a list of custom objects using Comparator.",
            "Write a Java class for a generic stack data structure.",
            "Create a Java method to read and parse a JSON file.",
            "Write a Java class to implement a simple HTTP client.",
        ]
    }
    
    if use_openai:
        openai_key = os.getenv('OPENAI_API_KEY', '')
        if openai_key:
            # Try to import OpenAI - handle both missing package and API versions
            client = None
            openai_module = None
            use_new_api = None
            
            try:
                # Try new OpenAI API (v1.0+)
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=openai_key)
                use_new_api = True
            except ImportError:
                # Fall back to old API style
                try:
                    import openai  # type: ignore
                    openai.api_key = openai_key
                    openai_module = openai
                    use_new_api = False
                except ImportError:
                    logger.warning("openai package not installed. Install with: pip install openai")
                    use_new_api = None
            
            if use_new_api is not None:
                for lang in languages:
                    lang_prompts = prompts.get(lang, prompts['python'])
                    ext_map = {
                        'python': '.py',
                        'javascript': '.js',
                        'java': '.java'
                    }
                    
                    for i, prompt in enumerate(lang_prompts * (samples_per_lang // len(lang_prompts) + 1)):
                        if collected >= samples_per_lang * len(languages):
                            break
                        
                        try:
                            if use_new_api and client is not None:
                                # New API style
                                response = client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": f"You are a helpful coding assistant. Write clean, well-documented {lang} code."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000
                                )
                                code = response.choices[0].message.content
                            elif not use_new_api and openai_module is not None:
                                # Old API style
                                response = openai_module.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": f"You are a helpful coding assistant. Write clean, well-documented {lang} code."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000
                                )
                                code = response.choices[0].message.content
                            else:
                                break
                                
                                # Extract code block if present
                                if '```' in code:
                                    lines = code.split('\n')
                                    start_idx = next((i for i, line in enumerate(lines) if '```' in line), 0) + 1
                                    end_idx = next((i for i, line in enumerate(lines[start_idx:], start_idx) if '```' in line), len(lines))
                                    code = '\n'.join(lines[start_idx:end_idx])
                                
                                filename = f"{lang}_ai_{collected:04d}.{ext_map[lang]}"
                                filepath = AI_DIR / filename
                                filepath.write_text(code, encoding='utf-8')
                                collected += 1
                                
                                time.sleep(0.5)  # Rate limiting
                            
                        except Exception as e:
                            logger.debug(f"Error generating AI code: {e}")
                            continue
    
    # Fallback: Generate synthetic AI-like code with clear AI patterns
    if collected < samples_per_lang * len(languages):
        logger.info("Generating synthetic AI-like code samples...")
        synthetic_count = generate_synthetic_ai_code(languages, samples_per_lang)
        collected += synthetic_count
    
    logger.info(f"Collected {collected} AI code samples")
    return collected


def generate_synthetic_ai_code(
    languages: List[str],
    samples_per_lang: int
) -> int:
    """Generate synthetic AI-like code with clear distinguishing features."""
    import random
    import textwrap
    
    collected = 0
    ext_map = {
        'python': '.py',
        'javascript': '.js',
        'java': '.java'
    }
    
    # AI-like patterns: verbose names, excessive comments, type hints, docstrings
    ai_templates = {
        'python': [
            lambda i: textwrap.dedent('''\
                """
                This function implements a highly efficient algorithm for computing
                the factorial of a given positive integer using iterative methodology.
                
                Args:
                    input_number: A positive integer for which factorial is computed
                
                Returns:
                    The factorial value as an integer
                
                Raises:
                    ValueError: If input_number is negative
                """
                from typing import Optional
                
                def compute_factorial_of_number(input_number: int) -> Optional[int]:
                    """Compute factorial using iterative approach."""
                    if input_number < 0:
                        raise ValueError("Input must be non-negative")
                    if input_number == 0 or input_number == 1:
                        return 1
                    result_value: int = 1
                    for current_index in range(2, input_number + 1):
                        result_value = result_value * current_index
                    return result_value
            '''),
            lambda i: textwrap.dedent('''\
                """
                A comprehensive implementation of binary search tree operations
                including insertion, search, and deletion with proper error handling.
                """
                from typing import Optional, Any
                
                class BinarySearchTreeNode:
                    """Node class for binary search tree."""
                    
                    def __init__(self, key_value: Any, data_value: Any = None):
                        self.key: Any = key_value
                        self.data: Any = data_value
                        self.left_child: Optional['BinarySearchTreeNode'] = None
                        self.right_child: Optional['BinarySearchTreeNode'] = None
                
                class BinarySearchTree:
                    """Binary search tree implementation."""
                    
                    def __init__(self):
                        self.root_node: Optional[BinarySearchTreeNode] = None
                    
                    def insert_key(self, key: Any, data: Any = None) -> None:
                        """Insert a new key into the tree."""
                        self.root_node = self._insert_recursive(self.root_node, key, data)
                    
                    def _insert_recursive(self, node: Optional[BinarySearchTreeNode], 
                                        key: Any, data: Any) -> BinarySearchTreeNode:
                        if node is None:
                            return BinarySearchTreeNode(key, data)
                        if key < node.key:
                            node.left_child = self._insert_recursive(node.left_child, key, data)
                        elif key > node.key:
                            node.right_child = self._insert_recursive(node.right_child, key, data)
                        return node
            '''),
        ],
        'javascript': [
            lambda i: textwrap.dedent('''\
                /**
                 * Implements a debounce mechanism to limit the rate at which
                 * a function can be invoked, useful for optimizing event handlers.
                 * 
                 * @param {Function} callbackFunction - The function to debounce
                 * @param {number} delayMilliseconds - Delay in milliseconds
                 * @returns {Function} Debounced function
                 */
                function createDebouncedFunction(callbackFunction, delayMilliseconds) {
                    let timeoutIdentifier = null;
                    return function debouncedFunction(...functionArguments) {
                        clearTimeout(timeoutIdentifier);
                        timeoutIdentifier = setTimeout(() => {
                            callbackFunction.apply(this, functionArguments);
                        }, delayMilliseconds);
                    };
                }
            '''),
        ],
        'java': [
            lambda i: textwrap.dedent('''\
                /**
                 * Thread-safe singleton pattern implementation using double-checked locking.
                 * Ensures only one instance exists throughout the application lifecycle.
                 */
                public class ThreadSafeSingleton {
                    private static volatile ThreadSafeSingleton uniqueInstance;
                    
                    private ThreadSafeSingleton() {
                        // Private constructor to prevent instantiation
                    }
                    
                    public static ThreadSafeSingleton getInstance() {
                        if (uniqueInstance == null) {
                            synchronized (ThreadSafeSingleton.class) {
                                if (uniqueInstance == null) {
                                    uniqueInstance = new ThreadSafeSingleton();
                                }
                            }
                        }
                        return uniqueInstance;
                    }
                }
            '''),
        ]
    }
    
    for lang in languages:
        templates = ai_templates.get(lang, ai_templates['python'])
        for i in range(samples_per_lang):
            template = random.choice(templates)
            code = template(i)
            
            filename = f"{lang}_ai_synthetic_{collected:04d}.{ext_map[lang]}"
            filepath = AI_DIR / filename
            filepath.write_text(code, encoding='utf-8')
            collected += 1
    
    return collected


def main():
    """Main function to collect real training data."""
    logger.info("Starting real data collection...")
    
    languages = ['python', 'javascript', 'java']
    samples_per_lang = 200
    
    # Collect human code
    logger.info("=" * 60)
    logger.info("Collecting HUMAN code samples...")
    logger.info("=" * 60)
    human_count = collect_human_code_from_github(
        languages=languages,
        samples_per_lang=samples_per_lang
    )
    
    # Generate AI code
    logger.info("=" * 60)
    logger.info("Collecting AI code samples...")
    logger.info("=" * 60)
    ai_count = generate_ai_code_samples(
        languages=languages,
        samples_per_lang=samples_per_lang
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("Data Collection Summary")
    logger.info("=" * 60)
    logger.info(f"Human code samples: {human_count}")
    logger.info(f"AI code samples: {ai_count}")
    logger.info(f"Total samples: {human_count + ai_count}")
    logger.info(f"Human samples saved to: {HUMAN_DIR}")
    logger.info(f"AI samples saved to: {AI_DIR}")


if __name__ == '__main__':
    main()
