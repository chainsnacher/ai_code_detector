"""
Advanced AST (Abstract Syntax Tree) feature extraction for code analysis.
Extracts structural, complexity, and pattern-based features from code.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import logging

logger = logging.getLogger("ai_code_detector")

class ASTFeatureExtractor:
    """Extracts comprehensive features from code AST."""
    
    def __init__(self):
        self.supported_languages = ["python"]
        self.complexity_weights = {
            'if': 1, 'elif': 1, 'else': 1, 'for': 1, 'while': 1,
            'try': 1, 'except': 1, 'finally': 1, 'with': 1,
            'and': 1, 'or': 1, 'not': 1, 'assert': 1
        }
    
    def extract_features(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Extract comprehensive AST features from code."""
        if language not in self.supported_languages:
            logger.warning(f"Language {language} not supported for AST analysis")
            return self._get_empty_features()
        
        try:
            tree = ast.parse(code)
            features = {}
            
            # Basic structural features
            features.update(self._extract_structural_features(tree))
            
            # Complexity features
            features.update(self._extract_complexity_features(tree))
            
            # Pattern features
            features.update(self._extract_pattern_features(tree))
            
            # Semantic features
            features.update(self._extract_semantic_features(tree))
            
            # Style features
            features.update(self._extract_style_features(code, tree))
            
            return features
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in code: {e}")
            return self._get_empty_features()
        except Exception as e:
            logger.error(f"Error extracting AST features: {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """Return empty feature set for error cases."""
        return {
            "ast_depth": 0,
            "ast_nodes": 0,
            "ast_branching_factor": 0.0,
            "cyclomatic_complexity": 0,
            "max_nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
            "import_count": 0,
            "comment_ratio": 0.0,
            "docstring_ratio": 0.0,
            "variable_naming_score": 0.0,
            "code_organization_score": 0.0
        }
    
    def _extract_structural_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract basic structural features from AST."""
        features = {}
        
        # Tree depth and size
        features["ast_depth"] = self._calculate_tree_depth(tree)
        features["ast_nodes"] = self._count_nodes(tree)
        features["ast_branching_factor"] = self._calculate_branching_factor(tree)
        
        # Node type distribution
        node_types = self._get_node_type_distribution(tree)
        features.update({
            f"node_type_{node_type}": count 
            for node_type, count in node_types.items()
        })
        
        return features
    
    def _extract_complexity_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract complexity-related features."""
        features = {}
        
        # Cyclomatic complexity
        features["cyclomatic_complexity"] = self._calculate_cyclomatic_complexity(tree)
        
        # Nesting depth
        features["max_nesting_depth"] = self._calculate_max_nesting_depth(tree)
        
        # Control flow complexity
        features["control_flow_complexity"] = self._calculate_control_flow_complexity(tree)
        
        # Cognitive complexity (simplified)
        features["cognitive_complexity"] = self._calculate_cognitive_complexity(tree)
        
        return features
    
    def _extract_pattern_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract coding pattern features."""
        features = {}
        
        # Function and class patterns
        features["function_count"] = self._count_functions(tree)
        features["class_count"] = self._count_classes(tree)
        features["method_count"] = self._count_methods(tree)
        features["import_count"] = self._count_imports(tree)
        
        # Function complexity patterns
        function_complexities = self._get_function_complexities(tree)
        if function_complexities:
            features.update({
                "avg_function_complexity": np.mean(function_complexities),
                "max_function_complexity": np.max(function_complexities),
                "function_complexity_std": np.std(function_complexities)
            })
        else:
            features.update({
                "avg_function_complexity": 0,
                "max_function_complexity": 0,
                "function_complexity_std": 0
            })
        
        # Code organization patterns
        features["function_class_ratio"] = self._calculate_function_class_ratio(tree)
        features["import_organization_score"] = self._calculate_import_organization_score(tree)
        
        return features
    
    def _extract_semantic_features(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract semantic features from AST."""
        features = {}
        
        # Variable usage patterns
        variable_usage = self._analyze_variable_usage(tree)
        features.update(variable_usage)
        
        # Exception handling patterns
        exception_patterns = self._analyze_exception_handling(tree)
        features.update(exception_patterns)
        
        # Type hint usage
        type_hint_usage = self._analyze_type_hints(tree)
        features.update(type_hint_usage)
        
        return features
    
    def _extract_style_features(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Extract style-related features."""
        features = {}
        
        # Comment analysis
        comment_stats = self._analyze_comments(code)
        features.update(comment_stats)
        
        # Docstring analysis
        docstring_stats = self._analyze_docstrings(tree)
        features.update(docstring_stats)
        
        # Naming conventions
        naming_score = self._analyze_naming_conventions(tree)
        features["variable_naming_score"] = naming_score
        
        # Code organization
        organization_score = self._analyze_code_organization(tree)
        features["code_organization_score"] = organization_score
        
        return features
    
    def _calculate_tree_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum depth of AST."""
        if not hasattr(node, '__iter__'):
            return current_depth
        
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            depth = self._calculate_tree_depth(child, current_depth + 1)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _count_nodes(self, node: ast.AST) -> int:
        """Count total number of nodes in AST."""
        count = 1
        for child in ast.iter_child_nodes(node):
            count += self._count_nodes(child)
        return count
    
    def _calculate_branching_factor(self, node: ast.AST) -> float:
        """Calculate average branching factor of AST."""
        if not hasattr(node, '__iter__'):
            return 0.0
        
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 0.0
        
        total_branches = len(children)
        child_branches = sum(self._calculate_branching_factor(child) for child in children)
        
        return (total_branches + child_branches) / len(children) if children else 0.0
    
    def _get_node_type_distribution(self, node: ast.AST) -> Dict[str, int]:
        """Get distribution of node types in AST."""
        node_types = Counter()
        
        def count_node_types(n):
            node_types[type(n).__name__] += 1
            for child in ast.iter_child_nodes(n):
                count_node_types(child)
        
        count_node_types(node)
        return dict(node_types)
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in code."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.If, ast.While, 
                                ast.For, ast.AsyncFor, ast.With, ast.Try)):
                depth = self._calculate_max_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, depth)
            else:
                depth = self._calculate_max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_control_flow_complexity(self, node: ast.AST) -> int:
        """Calculate control flow complexity."""
        complexity = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Break):
                complexity += 1
            elif isinstance(child, ast.Continue):
                complexity += 1
            elif isinstance(child, ast.Return):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (simplified version)."""
        complexity = 0
        nesting_level = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                nesting_level = 0  # Reset for new function/class
            
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + nesting_level
                nesting_level += 1
            elif isinstance(child, (ast.ExceptHandler, ast.With)):
                complexity += 1 + nesting_level
            elif isinstance(child, (ast.BoolOp, ast.Compare)):
                complexity += 1
        
        return complexity
    
    def _count_functions(self, node: ast.AST) -> int:
        """Count number of function definitions."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                count += 1
        return count
    
    def _count_classes(self, node: ast.AST) -> int:
        """Count number of class definitions."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.ClassDef):
                count += 1
        return count
    
    def _count_methods(self, node: ast.AST) -> int:
        """Count number of methods in classes."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and any(
                isinstance(parent, ast.ClassDef) for parent in ast.walk(node)
                if child in ast.iter_child_nodes(parent)
            ):
                count += 1
        return count
    
    def _count_imports(self, node: ast.AST) -> int:
        """Count number of import statements."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                count += 1
        return count
    
    def _get_function_complexities(self, node: ast.AST) -> List[int]:
        """Get complexity scores for all functions."""
        complexities = []
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(child)
                complexities.append(complexity)
        return complexities
    
    def _calculate_function_class_ratio(self, node: ast.AST) -> float:
        """Calculate ratio of functions to classes."""
        function_count = self._count_functions(node)
        class_count = self._count_classes(node)
        
        if class_count == 0:
            return float('inf') if function_count > 0 else 0.0
        
        return function_count / class_count
    
    def _calculate_import_organization_score(self, node: ast.AST) -> float:
        """Calculate score for import organization."""
        imports = []
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                imports.append(child.lineno)
        
        if len(imports) < 2:
            return 1.0
        
        # Check if imports are at the top
        sorted_imports = sorted(imports)
        is_organized = all(lineno <= 20 for lineno in sorted_imports)  # Within first 20 lines
        
        return 1.0 if is_organized else 0.0
    
    def _analyze_variable_usage(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze variable usage patterns."""
        variables = defaultdict(int)
        assignments = 0
        usages = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                assignments += 1
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        variables[target.id] += 1
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                usages += 1
                variables[child.id] += 1
        
        return {
            "unique_variables": len(variables),
            "variable_assignments": assignments,
            "variable_usages": usages,
            "avg_variable_usage": usages / len(variables) if variables else 0
        }
    
    def _analyze_exception_handling(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze exception handling patterns."""
        try_blocks = 0
        except_blocks = 0
        finally_blocks = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                try_blocks += 1
                except_blocks += len(child.handlers)
                if child.finalbody:
                    finally_blocks += 1
        
        return {
            "try_blocks": try_blocks,
            "except_blocks": except_blocks,
            "finally_blocks": finally_blocks,
            "exception_handling_ratio": except_blocks / try_blocks if try_blocks > 0 else 0
        }
    
    def _analyze_type_hints(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze type hint usage."""
        functions_with_hints = 0
        total_functions = 0
        variables_with_hints = 0
        total_variables = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                total_functions += 1
                if child.returns is not None:
                    functions_with_hints += 1
                
                for arg in child.args.args:
                    total_variables += 1
                    if arg.annotation is not None:
                        variables_with_hints += 1
        
        return {
            "functions_with_type_hints": functions_with_hints,
            "type_hint_ratio": functions_with_hints / total_functions if total_functions > 0 else 0,
            "variables_with_type_hints": variables_with_hints,
            "variable_type_hint_ratio": variables_with_hints / total_variables if total_variables > 0 else 0
        }
    
    def _analyze_comments(self, code: str) -> Dict[str, Any]:
        """Analyze comment patterns in code."""
        lines = code.split('\n')
        total_lines = len(lines)
        comment_lines = 0
        docstring_lines = 0
        
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
        
        return {
            "comment_lines": comment_lines,
            "docstring_lines": docstring_lines,
            "comment_ratio": comment_lines / total_lines if total_lines > 0 else 0,
            "docstring_ratio": docstring_lines / total_lines if total_lines > 0 else 0
        }
    
    def _analyze_docstrings(self, node: ast.AST) -> Dict[str, Any]:
        """Analyze docstring usage."""
        docstring_count = 0
        function_count = 0
        class_count = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                function_count += 1
                if ast.get_docstring(child):
                    docstring_count += 1
            elif isinstance(child, ast.ClassDef):
                class_count += 1
                if ast.get_docstring(child):
                    docstring_count += 1
        
        return {
            "docstring_count": docstring_count,
            "docstring_ratio": docstring_count / (function_count + class_count) if (function_count + class_count) > 0 else 0
        }
    
    def _analyze_naming_conventions(self, node: ast.AST) -> float:
        """Analyze adherence to naming conventions."""
        score = 0.0
        total_items = 0
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                total_items += 1
                if child.name.islower() and '_' in child.name:
                    score += 1
            elif isinstance(child, ast.ClassDef):
                total_items += 1
                if child.name[0].isupper():
                    score += 1
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                total_items += 1
                if child.id.islower() and ('_' in child.id or child.id.islower()):
                    score += 1
        
        return score / total_items if total_items > 0 else 0.0
    
    def _analyze_code_organization(self, node: ast.AST) -> float:
        """Analyze code organization quality."""
        score = 0.0
        checks = 0
        
        # Check if imports are at the top
        imports = []
        other_statements = []
        
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                imports.append(child.lineno)
            else:
                other_statements.append(child.lineno)
        
        if imports and other_statements:
            checks += 1
            if max(imports) < min(other_statements):
                score += 1
        
        # Check function/class organization
        functions = []
        classes = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                functions.append(child.lineno)
            elif isinstance(child, ast.ClassDef):
                classes.append(child.lineno)
        
        if functions and classes:
            checks += 1
            # Check if classes are defined before their methods
            score += 0.5  # Simplified check
        
        return score / checks if checks > 0 else 0.0
