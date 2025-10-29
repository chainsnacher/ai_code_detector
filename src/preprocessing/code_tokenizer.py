"""
Advanced code tokenization with multi-language support.
Provides sophisticated tokenization for various programming languages.
"""

import re
import ast
import tokenize
import io
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from collections import Counter, defaultdict
import logging

logger = logging.getLogger("ai_code_detector")

class AdvancedCodeTokenizer:
    """Advanced tokenizer for multiple programming languages."""
    
    def __init__(self):
        self.language_patterns = {
            'python': self._get_python_patterns(),
            'java': self._get_java_patterns(),
            'javascript': self._get_javascript_patterns(),
            'cpp': self._get_cpp_patterns(),
            'csharp': self._get_csharp_patterns(),
        }
    
    def _get_python_patterns(self) -> Dict[str, str]:
        """Get Python-specific tokenization patterns."""
        return {
            'keywords': r'\b(and|as|assert|break|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|not|or|pass|print|raise|return|try|while|with|yield)\b',
            'operators': r'[+\-*/%=<>!&|^~]',
            'delimiters': r'[(){}[\].,;:]',
            'strings': r'["\'](?:[^"\\]|\\.)*["\']',
            'numbers': r'\b\d+\.?\d*\b',
            'identifiers': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            'comments': r'#.*$',
            'whitespace': r'\s+',
        }
    
    def _get_java_patterns(self) -> Dict[str, str]:
        """Get Java-specific tokenization patterns."""
        return {
            'keywords': r'\b(abstract|assert|boolean|break|byte|case|catch|char|class|const|continue|default|do|double|else|enum|extends|final|finally|float|for|goto|if|implements|import|instanceof|int|interface|long|native|new|package|private|protected|public|return|short|static|strictfp|super|switch|synchronized|this|throw|throws|transient|try|void|volatile|while)\b',
            'operators': r'[+\-*/%=<>!&|^~]',
            'delimiters': r'[(){}[\].,;:]',
            'strings': r'"(?:[^"\\]|\\.)*"',
            'numbers': r'\b\d+\.?\d*\b',
            'identifiers': r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b',
            'comments': r'//.*$|/\*.*?\*/',
            'whitespace': r'\s+',
        }
    
    def _get_javascript_patterns(self) -> Dict[str, str]:
        """Get JavaScript-specific tokenization patterns."""
        return {
            'keywords': r'\b(break|case|catch|class|const|continue|debugger|default|delete|do|else|export|extends|finally|for|function|if|import|in|instanceof|let|new|return|super|switch|this|throw|try|typeof|var|void|while|with|yield)\b',
            'operators': r'[+\-*/%=<>!&|^~]',
            'delimiters': r'[(){}[\].,;:]',
            'strings': r'["\'](?:[^"\\]|\\.)*["\']',
            'numbers': r'\b\d+\.?\d*\b',
            'identifiers': r'\b[a-zA-Z_$][a-zA-Z0-9_$]*\b',
            'comments': r'//.*$|/\*.*?\*/',
            'whitespace': r'\s+',
        }
    
    def _get_cpp_patterns(self) -> Dict[str, str]:
        """Get C++-specific tokenization patterns."""
        return {
            'keywords': r'\b(alignas|alignof|and|and_eq|asm|auto|bitand|bitor|bool|break|case|catch|char|char16_t|char32_t|class|compl|const|constexpr|const_cast|continue|decltype|default|delete|do|double|dynamic_cast|else|enum|explicit|export|extern|false|float|for|friend|goto|if|inline|int|long|mutable|namespace|new|noexcept|not|not_eq|nullptr|operator|or|or_eq|private|protected|public|register|reinterpret_cast|return|short|signed|sizeof|static|static_assert|static_cast|struct|switch|template|this|thread_local|throw|true|try|typedef|typeid|typename|union|unsigned|using|virtual|void|volatile|wchar_t|while|xor|xor_eq)\b',
            'operators': r'[+\-*/%=<>!&|^~]',
            'delimiters': r'[(){}[\].,;:]',
            'strings': r'"(?:[^"\\]|\\.)*"',
            'numbers': r'\b\d+\.?\d*\b',
            'identifiers': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            'comments': r'//.*$|/\*.*?\*/',
            'whitespace': r'\s+',
        }
    
    def _get_csharp_patterns(self) -> Dict[str, str]:
        """Get C#-specific tokenization patterns."""
        return {
            'keywords': r'\b(abstract|as|base|bool|break|byte|case|catch|char|checked|class|const|continue|decimal|default|delegate|do|double|else|enum|event|explicit|extern|false|finally|fixed|float|for|foreach|goto|if|implicit|in|int|interface|internal|is|lock|long|namespace|new|null|object|operator|out|override|params|private|protected|public|readonly|ref|return|sbyte|sealed|short|sizeof|stackalloc|static|string|struct|switch|this|throw|true|try|typeof|uint|ulong|unchecked|unsafe|ushort|using|virtual|void|volatile|while)\b',
            'operators': r'[+\-*/%=<>!&|^~]',
            'delimiters': r'[(){}[\].,;:]',
            'strings': r'"(?:[^"\\]|\\.)*"',
            'numbers': r'\b\d+\.?\d*\b',
            'identifiers': r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            'comments': r'//.*$|/\*.*?\*/',
            'whitespace': r'\s+',
        }
    
    def tokenize(self, code: str, language: str = 'python') -> Dict[str, List[str]]:
        """Tokenize code and return categorized tokens."""
        if language not in self.language_patterns:
            logger.warning(f"Language {language} not supported, using Python patterns")
            language = 'python'
        
        patterns = self.language_patterns[language]
        tokens = {}
        
        for token_type, pattern in patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE | re.DOTALL)
            tokens[token_type] = matches
        
        return tokens
    
    def get_token_statistics(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Get comprehensive token statistics."""
        tokens = self.tokenize(code, language)
        
        stats = {}
        
        # Basic counts
        for token_type, token_list in tokens.items():
            stats[f'{token_type}_count'] = len(token_list)
            stats[f'{token_type}_unique_count'] = len(set(token_list))
        
        # Token diversity
        all_tokens = []
        for token_list in tokens.values():
            all_tokens.extend(token_list)
        
        stats['total_tokens'] = len(all_tokens)
        stats['unique_tokens'] = len(set(all_tokens))
        stats['token_diversity'] = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        
        # Token length statistics
        token_lengths = [len(token) for token in all_tokens]
        if token_lengths:
            stats['avg_token_length'] = np.mean(token_lengths)
            stats['max_token_length'] = np.max(token_lengths)
            stats['min_token_length'] = np.min(token_lengths)
            stats['token_length_std'] = np.std(token_lengths)
        else:
            stats.update({
                'avg_token_length': 0,
                'max_token_length': 0,
                'min_token_length': 0,
                'token_length_std': 0
            })
        
        # Language-specific features
        if language == 'python':
            stats.update(self._get_python_specific_stats(code, tokens))
        elif language == 'java':
            stats.update(self._get_java_specific_stats(code, tokens))
        elif language == 'javascript':
            stats.update(self._get_javascript_specific_stats(code, tokens))
        
        return stats
    
    def _get_python_specific_stats(self, code: str, tokens: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get Python-specific token statistics."""
        stats = {}
        
        # Indentation analysis
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        if indent_levels:
            stats['avg_indent_level'] = np.mean(indent_levels)
            stats['max_indent_level'] = np.max(indent_levels)
            stats['indent_consistency'] = 1.0 if all(indent % 4 == 0 for indent in indent_levels) else 0.0
        else:
            stats.update({
                'avg_indent_level': 0,
                'max_indent_level': 0,
                'indent_consistency': 0
            })
        
        # String analysis
        string_tokens = tokens.get('strings', [])
        stats['string_count'] = len(string_tokens)
        stats['avg_string_length'] = np.mean([len(s) for s in string_tokens]) if string_tokens else 0
        
        # Comment analysis
        comment_tokens = tokens.get('comments', [])
        stats['comment_count'] = len(comment_tokens)
        stats['comment_ratio'] = len(comment_tokens) / len(tokens.get('identifiers', [])) if tokens.get('identifiers') else 0
        
        return stats
    
    def _get_java_specific_stats(self, code: str, tokens: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get Java-specific token statistics."""
        stats = {}
        
        # Brace analysis
        brace_tokens = [token for token in tokens.get('delimiters', []) if token in '{}']
        stats['brace_count'] = len(brace_tokens)
        stats['brace_balance'] = 1.0 if brace_tokens.count('{') == brace_tokens.count('}') else 0.0
        
        # Semicolon analysis
        semicolon_count = tokens.get('delimiters', []).count(';')
        stats['semicolon_count'] = semicolon_count
        
        # Access modifier analysis
        access_modifiers = ['public', 'private', 'protected']
        modifier_count = sum(tokens.get('keywords', []).count(mod) for mod in access_modifiers)
        stats['access_modifier_count'] = modifier_count
        
        return stats
    
    def _get_javascript_specific_stats(self, code: str, tokens: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get JavaScript-specific token statistics."""
        stats = {}
        
        # Arrow function analysis
        arrow_functions = len(re.findall(r'=>', code))
        stats['arrow_function_count'] = arrow_functions
        
        # Template literal analysis
        template_literals = len(re.findall(r'`[^`]*`', code))
        stats['template_literal_count'] = template_literals
        
        # Destructuring analysis
        destructuring = len(re.findall(r'\{[^}]*\}', code))
        stats['destructuring_count'] = destructuring
        
        return stats
    
    def get_semantic_tokens(self, code: str, language: str = 'python') -> Dict[str, List[str]]:
        """Get semantically grouped tokens."""
        tokens = self.tokenize(code, language)
        
        semantic_tokens = {
            'control_flow': [],
            'data_types': [],
            'functions': [],
            'variables': [],
            'operators': [],
            'literals': [],
            'comments': [],
            'whitespace': []
        }
        
        # Categorize tokens
        for token_type, token_list in tokens.items():
            if token_type == 'keywords':
                # Categorize keywords
                for token in token_list:
                    if token in ['if', 'else', 'elif', 'for', 'while', 'break', 'continue', 'return']:
                        semantic_tokens['control_flow'].append(token)
                    elif token in ['def', 'class', 'lambda']:
                        semantic_tokens['functions'].append(token)
                    elif token in ['import', 'from']:
                        semantic_tokens['functions'].append(token)
            elif token_type == 'identifiers':
                # Try to categorize identifiers
                for token in token_list:
                    if token.startswith('def_') or token.endswith('_func'):
                        semantic_tokens['functions'].append(token)
                    elif token.isupper():
                        semantic_tokens['data_types'].append(token)
                    else:
                        semantic_tokens['variables'].append(token)
            elif token_type == 'operators':
                semantic_tokens['operators'].extend(token_list)
            elif token_type == 'strings':
                semantic_tokens['literals'].extend(token_list)
            elif token_type == 'numbers':
                semantic_tokens['literals'].extend(token_list)
            elif token_type == 'comments':
                semantic_tokens['comments'].extend(token_list)
            elif token_type == 'whitespace':
                semantic_tokens['whitespace'].extend(token_list)
        
        return semantic_tokens
    
    def get_token_ngrams(self, code: str, language: str = 'python', n: int = 2) -> Dict[str, int]:
        """Get n-gram statistics for tokens."""
        tokens = self.tokenize(code, language)
        all_tokens = []
        for token_list in tokens.values():
            all_tokens.extend(token_list)
        
        ngrams = []
        for i in range(len(all_tokens) - n + 1):
            ngram = ' '.join(all_tokens[i:i+n])
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def get_syntax_errors(self, code: str, language: str = 'python') -> List[str]:
        """Detect syntax errors in code."""
        errors = []
        
        if language == 'python':
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"Parse error: {str(e)}")
        
        return errors
    
    def get_code_metrics(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Get comprehensive code metrics."""
        metrics = {}
        
        # Basic metrics
        lines = code.split('\n')
        metrics['total_lines'] = len(lines)
        metrics['non_empty_lines'] = len([line for line in lines if line.strip()])
        metrics['total_characters'] = len(code)
        metrics['total_words'] = len(code.split())
        
        # Token metrics
        token_stats = self.get_token_statistics(code, language)
        metrics.update(token_stats)
        
        # Semantic metrics
        semantic_tokens = self.get_semantic_tokens(code, language)
        for category, tokens in semantic_tokens.items():
            metrics[f'{category}_count'] = len(tokens)
            metrics[f'{category}_ratio'] = len(tokens) / len(code.split()) if code.split() else 0
        
        # Complexity metrics
        metrics['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(code, language)
        metrics['nesting_depth'] = self._calculate_nesting_depth(code, language)
        
        return metrics
    
    def _calculate_cyclomatic_complexity(self, code: str, language: str) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        if language == 'python':
            complexity += len(re.findall(r'\b(if|elif|else|for|while|except|and|or)\b', code))
        elif language == 'java':
            complexity += len(re.findall(r'\b(if|else|for|while|catch|&&|\|\|)\b', code))
        elif language == 'javascript':
            complexity += len(re.findall(r'\b(if|else|for|while|catch|&&|\|\|)\b', code))
        
        return complexity
    
    def _calculate_nesting_depth(self, code: str, language: str) -> int:
        """Calculate maximum nesting depth."""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char in '({[':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def compare_tokens(self, code1: str, code2: str, language: str = 'python') -> Dict[str, float]:
        """Compare token patterns between two code samples."""
        tokens1 = self.tokenize(code1, language)
        tokens2 = self.tokenize(code2, language)
        
        similarities = {}
        
        for token_type in tokens1.keys():
            if token_type in tokens2:
                set1 = set(tokens1[token_type])
                set2 = set(tokens2[token_type])
                
                if set1 or set2:
                    jaccard_similarity = len(set1 & set2) / len(set1 | set2)
                    similarities[f'{token_type}_similarity'] = jaccard_similarity
        
        return similarities
