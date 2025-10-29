"""
Advanced statistical feature extraction for code analysis.
Extracts 50+ features including stylometric, lexical, and semantic patterns.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import textstat
import logging
from pathlib import Path

logger = logging.getLogger("ai_code_detector")

class StatisticalFeatureExtractor:
    """Extracts comprehensive statistical features from code."""
    
    def __init__(self):
        self.python_keywords = {
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del',
            'elif', 'else', 'except', 'exec', 'finally', 'for', 'from', 'global',
            'if', 'import', 'in', 'is', 'lambda', 'not', 'or', 'pass', 'print',
            'raise', 'return', 'try', 'while', 'with', 'yield'
        }
        
        self.common_patterns = {
            'list_comprehension': r'\[.*for.*in.*\]',
            'generator_expression': r'\(.*for.*in.*\)',
            'lambda_function': r'lambda\s+\w+.*:',
            'ternary_operator': r'.*\s+if\s+.*\s+else\s+.*',
            'f_string': r'f["\'].*["\']',
            'docstring': r'["\']{3}.*["\']{3}',
            'type_hint': r':\s*\w+',
            'assert_statement': r'assert\s+',
            'with_statement': r'with\s+',
            'try_except': r'try\s*:',
        }
    
    def extract_features(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Extract comprehensive statistical features from code."""
        features = {}
        
        # Basic statistics
        features.update(self._extract_basic_statistics(code))
        
        # Stylometric features
        features.update(self._extract_stylometric_features(code))
        
        # Lexical features
        features.update(self._extract_lexical_features(code))
        
        # Semantic features
        features.update(self._extract_semantic_features(code))
        
        # Pattern-based features
        features.update(self._extract_pattern_features(code))
        
        # Complexity features
        features.update(self._extract_complexity_features(code))
        
        # Style consistency features
        features.update(self._extract_style_consistency_features(code))
        
        return features
    
    def _extract_basic_statistics(self, code: str) -> Dict[str, Any]:
        """Extract basic statistical features."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        features = {
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'total_characters': len(code),
            'total_words': len(code.split()),
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'max_line_length': max([len(line) for line in lines]) if lines else 0,
            'min_line_length': min([len(line) for line in lines]) if lines else 0,
            'line_length_std': np.std([len(line) for line in lines]) if lines else 0,
            'blank_line_ratio': (len(lines) - len(non_empty_lines)) / len(lines) if lines else 0,
            'avg_words_per_line': len(code.split()) / len(lines) if lines else 0,
        }
        
        return features
    
    def _extract_stylometric_features(self, code: str) -> Dict[str, Any]:
        """Extract stylometric features (coding style patterns)."""
        lines = code.split('\n')
        
        # Indentation analysis
        indentation_stats = self._analyze_indentation(code)
        
        # Naming convention analysis
        naming_stats = self._analyze_naming_conventions(code)
        
        # Comment analysis
        comment_stats = self._analyze_comments(code)
        
        # Spacing and formatting
        spacing_stats = self._analyze_spacing(code)
        
        features = {
            **indentation_stats,
            **naming_stats,
            **comment_stats,
            **spacing_stats,
        }
        
        return features
    
    def _extract_lexical_features(self, code: str) -> Dict[str, Any]:
        """Extract lexical features (token-level patterns)."""
        # Token analysis
        tokens = self._tokenize_code(code)
        token_stats = self._analyze_tokens(tokens)
        
        # Keyword analysis
        keyword_stats = self._analyze_keywords(code)
        
        # Character analysis
        char_stats = self._analyze_characters(code)
        
        # String and number analysis
        literal_stats = self._analyze_literals(code)
        
        features = {
            **token_stats,
            **keyword_stats,
            **char_stats,
            **literal_stats,
        }
        
        return features
    
    def _extract_semantic_features(self, code: str) -> Dict[str, Any]:
        """Extract semantic features (meaning-related patterns)."""
        # Function and class analysis
        structure_stats = self._analyze_code_structure(code)
        
        # Control flow analysis
        control_flow_stats = self._analyze_control_flow(code)
        
        # Error handling analysis
        error_handling_stats = self._analyze_error_handling(code)
        
        # Import analysis
        import_stats = self._analyze_imports(code)
        
        features = {
            **structure_stats,
            **control_flow_stats,
            **error_handling_stats,
            **import_stats,
        }
        
        return features
    
    def _extract_pattern_features(self, code: str) -> Dict[str, Any]:
        """Extract pattern-based features."""
        pattern_counts = {}
        
        for pattern_name, pattern_regex in self.common_patterns.items():
            matches = re.findall(pattern_regex, code, re.MULTILINE | re.DOTALL)
            pattern_counts[f'pattern_{pattern_name}'] = len(matches)
            pattern_counts[f'pattern_{pattern_name}_ratio'] = len(matches) / len(code.split('\n')) if code else 0
        
        return pattern_counts
    
    def _extract_complexity_features(self, code: str) -> Dict[str, Any]:
        """Extract complexity-related features."""
        lines = code.split('\n')
        
        # Readability metrics
        readability_score = textstat.flesch_reading_ease(code) if code else 0
        complexity_score = textstat.flesch_kincaid_grade(code) if code else 0
        
        # Nesting analysis
        nesting_stats = self._analyze_nesting(code)
        
        # Cyclomatic complexity (simplified)
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(code)
        
        features = {
            'readability_score': readability_score,
            'complexity_score': complexity_score,
            'cyclomatic_complexity': cyclomatic_complexity,
            **nesting_stats,
        }
        
        return features
    
    def _extract_style_consistency_features(self, code: str) -> Dict[str, Any]:
        """Extract style consistency features."""
        lines = code.split('\n')
        
        # Consistency in indentation
        indentation_consistency = self._calculate_indentation_consistency(code)
        
        # Consistency in spacing
        spacing_consistency = self._calculate_spacing_consistency(code)
        
        # Consistency in naming
        naming_consistency = self._calculate_naming_consistency(code)
        
        features = {
            'indentation_consistency': indentation_consistency,
            'spacing_consistency': spacing_consistency,
            'naming_consistency': naming_consistency,
        }
        
        return features
    
    def _analyze_indentation(self, code: str) -> Dict[str, Any]:
        """Analyze indentation patterns."""
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            return {
                'indentation_spaces': 0,
                'indentation_tabs': 0,
                'indentation_mixed': 0,
                'avg_indentation': 0,
                'indentation_consistency': 0,
            }
        
        space_indents = 0
        tab_indents = 0
        mixed_indents = 0
        indent_levels = []
        
        for line in lines:
            if line.startswith(' '):
                space_indents += 1
                indent_level = len(line) - len(line.lstrip())
                indent_levels.append(indent_level)
            elif line.startswith('\t'):
                tab_indents += 1
                indent_level = len(line) - len(line.lstrip())
                indent_levels.append(indent_level)
            elif line.startswith((' ', '\t')):
                mixed_indents += 1
        
        total_indented = space_indents + tab_indents + mixed_indents
        
        return {
            'indentation_spaces': space_indents / total_indented if total_indented > 0 else 0,
            'indentation_tabs': tab_indents / total_indented if total_indented > 0 else 0,
            'indentation_mixed': mixed_indents / total_indented if total_indented > 0 else 0,
            'avg_indentation': np.mean(indent_levels) if indent_levels else 0,
            'indentation_consistency': 1.0 if (space_indents > 0 and tab_indents == 0) or (tab_indents > 0 and space_indents == 0) else 0.0,
        }
    
    def _analyze_naming_conventions(self, code: str) -> Dict[str, Any]:
        """Analyze naming convention adherence."""
        # Extract variable names, function names, class names
        variable_names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
        function_names = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        class_names = re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', code)
        
        # Analyze naming patterns
        snake_case_vars = sum(1 for name in variable_names if '_' in name and name.islower())
        snake_case_funcs = sum(1 for name in function_names if '_' in name and name.islower())
        pascal_case_classes = sum(1 for name in class_names if name[0].isupper())
        
        total_vars = len(variable_names)
        total_funcs = len(function_names)
        total_classes = len(class_names)
        
        return {
            'snake_case_variables': snake_case_vars / total_vars if total_vars > 0 else 0,
            'snake_case_functions': snake_case_funcs / total_funcs if total_funcs > 0 else 0,
            'pascal_case_classes': pascal_case_classes / total_classes if total_classes > 0 else 0,
            'naming_convention_score': (snake_case_vars + snake_case_funcs + pascal_case_classes) / (total_vars + total_funcs + total_classes) if (total_vars + total_funcs + total_classes) > 0 else 0,
        }
    
    def _analyze_comments(self, code: str) -> Dict[str, Any]:
        """Analyze comment patterns."""
        lines = code.split('\n')
        comment_lines = 0
        docstring_lines = 0
        inline_comments = 0
        
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                docstring_lines += 1
            elif in_docstring:
                docstring_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            elif '#' in line and not stripped.startswith('#'):
                inline_comments += 1
        
        return {
            'comment_lines': comment_lines,
            'docstring_lines': docstring_lines,
            'inline_comments': inline_comments,
            'comment_ratio': comment_lines / len(lines) if lines else 0,
            'docstring_ratio': docstring_lines / len(lines) if lines else 0,
            'inline_comment_ratio': inline_comments / len(lines) if lines else 0,
        }
    
    def _analyze_spacing(self, code: str) -> Dict[str, Any]:
        """Analyze spacing and formatting patterns."""
        lines = code.split('\n')
        
        # Analyze spacing around operators
        operator_spacing = self._analyze_operator_spacing(code)
        
        # Analyze line spacing
        line_spacing = self._analyze_line_spacing(lines)
        
        # Analyze bracket spacing
        bracket_spacing = self._analyze_bracket_spacing(code)
        
        return {
            **operator_spacing,
            **line_spacing,
            **bracket_spacing,
        }
    
    def _analyze_operator_spacing(self, code: str) -> Dict[str, Any]:
        """Analyze spacing around operators."""
        operators = ['=', '+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=']
        
        consistent_spacing = 0
        total_operators = 0
        
        for op in operators:
            # Check for consistent spacing around operators
            pattern = re.escape(op)
            matches = re.findall(f'\\S{pattern}\\S|\\S{pattern}\\s|\\s{pattern}\\S', code)
            total_operators += len(matches)
            
            # Check for proper spacing
            proper_spacing = re.findall(f'\\s{pattern}\\s', code)
            consistent_spacing += len(proper_spacing)
        
        return {
            'operator_spacing_consistency': consistent_spacing / total_operators if total_operators > 0 else 0,
        }
    
    def _analyze_line_spacing(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze line spacing patterns."""
        empty_lines = sum(1 for line in lines if not line.strip())
        consecutive_empty = 0
        max_consecutive_empty = 0
        current_consecutive = 0
        
        for line in lines:
            if not line.strip():
                current_consecutive += 1
            else:
                max_consecutive_empty = max(max_consecutive_empty, current_consecutive)
                current_consecutive = 0
        
        max_consecutive_empty = max(max_consecutive_empty, current_consecutive)
        
        return {
            'empty_line_ratio': empty_lines / len(lines) if lines else 0,
            'max_consecutive_empty_lines': max_consecutive_empty,
            'avg_consecutive_empty_lines': empty_lines / (len(lines) - empty_lines) if (len(lines) - empty_lines) > 0 else 0,
        }
    
    def _analyze_bracket_spacing(self, code: str) -> Dict[str, Any]:
        """Analyze spacing around brackets."""
        # Check spacing around parentheses, brackets, braces
        patterns = {
            'paren_spacing': r'\(\s+|\s+\)',
            'bracket_spacing': r'\[\s+|\s+\]',
            'brace_spacing': r'{\s+|\s+}',
        }
        
        spacing_stats = {}
        for name, pattern in patterns.items():
            matches = len(re.findall(pattern, code))
            spacing_stats[f'{name}_count'] = matches
        
        return spacing_stats
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Simple tokenization of code."""
        # Split by whitespace and common delimiters
        tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
        return tokens
    
    def _analyze_tokens(self, tokens: List[str]) -> Dict[str, Any]:
        """Analyze token patterns."""
        if not tokens:
            return {
                'unique_tokens': 0,
                'token_diversity': 0,
                'avg_token_length': 0,
                'token_frequency_std': 0,
            }
        
        token_counts = Counter(tokens)
        unique_tokens = len(token_counts)
        total_tokens = len(tokens)
        
        return {
            'unique_tokens': unique_tokens,
            'token_diversity': unique_tokens / total_tokens,
            'avg_token_length': np.mean([len(token) for token in tokens]),
            'token_frequency_std': np.std(list(token_counts.values())),
        }
    
    def _analyze_keywords(self, code: str) -> Dict[str, Any]:
        """Analyze keyword usage patterns."""
        words = re.findall(r'\b\w+\b', code)
        keyword_counts = Counter(word for word in words if word in self.python_keywords)
        
        total_keywords = sum(keyword_counts.values())
        unique_keywords = len(keyword_counts)
        
        return {
            'keyword_count': total_keywords,
            'unique_keywords': unique_keywords,
            'keyword_density': total_keywords / len(words) if words else 0,
            'keyword_diversity': unique_keywords / len(self.python_keywords),
        }
    
    def _analyze_characters(self, code: str) -> Dict[str, Any]:
        """Analyze character-level patterns."""
        if not code:
            return {
                'char_diversity': 0,
                'digit_ratio': 0,
                'alpha_ratio': 0,
                'special_char_ratio': 0,
                'uppercase_ratio': 0,
                'lowercase_ratio': 0,
            }
        
        total_chars = len(code)
        digits = sum(1 for c in code if c.isdigit())
        alphas = sum(1 for c in code if c.isalpha())
        specials = sum(1 for c in code if not c.isalnum() and not c.isspace())
        uppers = sum(1 for c in code if c.isupper())
        lowers = sum(1 for c in code if c.islower())
        
        return {
            'char_diversity': len(set(code)) / total_chars,
            'digit_ratio': digits / total_chars,
            'alpha_ratio': alphas / total_chars,
            'special_char_ratio': specials / total_chars,
            'uppercase_ratio': uppers / total_chars,
            'lowercase_ratio': lowers / total_chars,
        }
    
    def _analyze_literals(self, code: str) -> Dict[str, Any]:
        """Analyze string and number literals."""
        string_literals = re.findall(r'["\']([^"\']*)["\']', code)
        number_literals = re.findall(r'\b\d+\.?\d*\b', code)
        
        return {
            'string_literal_count': len(string_literals),
            'number_literal_count': len(number_literals),
            'avg_string_length': np.mean([len(s) for s in string_literals]) if string_literals else 0,
            'avg_number_value': np.mean([float(n) for n in number_literals]) if number_literals else 0,
        }
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure patterns."""
        functions = len(re.findall(r'def\s+\w+', code))
        classes = len(re.findall(r'class\s+\w+', code))
        imports = len(re.findall(r'import\s+|from\s+\w+\s+import', code))
        
        return {
            'function_count': functions,
            'class_count': classes,
            'import_count': imports,
            'function_class_ratio': functions / classes if classes > 0 else float('inf'),
            'import_density': imports / len(code.split('\n')) if code else 0,
        }
    
    def _analyze_control_flow(self, code: str) -> Dict[str, Any]:
        """Analyze control flow patterns."""
        if_statements = len(re.findall(r'\bif\b', code))
        for_loops = len(re.findall(r'\bfor\b', code))
        while_loops = len(re.findall(r'\bwhile\b', code))
        try_blocks = len(re.findall(r'\btry\b', code))
        
        return {
            'if_statements': if_statements,
            'for_loops': for_loops,
            'while_loops': while_loops,
            'try_blocks': try_blocks,
            'control_flow_density': (if_statements + for_loops + while_loops + try_blocks) / len(code.split('\n')) if code else 0,
        }
    
    def _analyze_error_handling(self, code: str) -> Dict[str, Any]:
        """Analyze error handling patterns."""
        try_blocks = len(re.findall(r'\btry\b', code))
        except_blocks = len(re.findall(r'\bexcept\b', code))
        finally_blocks = len(re.findall(r'\bfinally\b', code))
        raise_statements = len(re.findall(r'\braise\b', code))
        
        return {
            'try_blocks': try_blocks,
            'except_blocks': except_blocks,
            'finally_blocks': finally_blocks,
            'raise_statements': raise_statements,
            'error_handling_ratio': except_blocks / try_blocks if try_blocks > 0 else 0,
        }
    
    def _analyze_imports(self, code: str) -> Dict[str, Any]:
        """Analyze import patterns."""
        import_statements = re.findall(r'import\s+([^\n]+)', code)
        from_imports = re.findall(r'from\s+(\w+)\s+import\s+([^\n]+)', code)
        
        # Analyze import organization
        lines = code.split('\n')
        import_lines = [i for i, line in enumerate(lines) if line.strip().startswith(('import ', 'from '))]
        
        return {
            'import_statements': len(import_statements),
            'from_imports': len(from_imports),
            'import_organization_score': 1.0 if import_lines and max(import_lines) < len(lines) * 0.2 else 0.0,
        }
    
    def _analyze_nesting(self, code: str) -> Dict[str, Any]:
        """Analyze nesting patterns."""
        lines = code.split('\n')
        max_indent = 0
        indent_levels = []
        
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
                max_indent = max(max_indent, indent)
        
        return {
            'max_nesting_depth': max_indent // 4,  # Assuming 4-space indentation
            'avg_nesting_depth': np.mean(indent_levels) / 4 if indent_levels else 0,
            'nesting_consistency': 1.0 if all(indent % 4 == 0 for indent in indent_levels) else 0.0,
        }
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'except', 'and', 'or']
        
        for keyword in decision_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code))
        
        return complexity
    
    def _calculate_indentation_consistency(self, code: str) -> float:
        """Calculate indentation consistency score."""
        lines = [line for line in code.split('\n') if line.strip()]
        
        if not lines:
            return 0.0
        
        indent_levels = []
        for line in lines:
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                indent_levels.append(indent)
        
        if not indent_levels:
            return 1.0
        
        # Check if all indents are multiples of a common base
        base_indent = min(indent_levels)
        if base_indent == 0:
            return 0.0
        
        consistent = all(indent % base_indent == 0 for indent in indent_levels)
        return 1.0 if consistent else 0.0
    
    def _calculate_spacing_consistency(self, code: str) -> float:
        """Calculate spacing consistency score."""
        # This is a simplified version - in practice, you'd analyze more spacing patterns
        lines = code.split('\n')
        
        if not lines:
            return 0.0
        
        consistent_lines = 0
        for line in lines:
            if line.strip():
                # Check for consistent spacing around operators
                if re.search(r'[=+\-*/]', line):
                    if re.search(r'\s[=+\-*/]\s', line):
                        consistent_lines += 1
        
        return consistent_lines / len([line for line in lines if line.strip() and re.search(r'[=+\-*/]', line)]) if any(re.search(r'[=+\-*/]', line) for line in lines) else 1.0
    
    def _calculate_naming_consistency(self, code: str) -> float:
        """Calculate naming consistency score."""
        variables = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
        functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
        classes = re.findall(r'class\s+([A-Z][a-zA-Z0-9_]*)', code)
        
        total_items = len(variables) + len(functions) + len(classes)
        if total_items == 0:
            return 1.0
        
        consistent_items = 0
        
        # Check variable naming (snake_case)
        for var in variables:
            if '_' in var and var.islower():
                consistent_items += 1
        
        # Check function naming (snake_case)
        for func in functions:
            if '_' in func and func.islower():
                consistent_items += 1
        
        # Check class naming (PascalCase)
        for cls in classes:
            if cls[0].isupper():
                consistent_items += 1
        
        return consistent_items / total_items
