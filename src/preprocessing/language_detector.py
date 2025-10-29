"""
Language detection for source code.
Detects and classifies programming languages from code samples.
"""

import re
from typing import Dict, Optional, List, Tuple, Any
import logging

logger = logging.getLogger("ai_code_detector")

class LanguageDetector:
    """Detects programming language from code samples."""
    
    def __init__(self):
        """Initialize language detector with language-specific patterns."""
        
        # Language-specific keywords and patterns
        self.language_patterns = {
            'python': {
                'keywords': {'def', 'class', 'import', 'from', 'if', 'elif', 'else', 'for', 'while', 
                            'try', 'except', 'finally', 'with', 'lambda', 'yield', 'async', 'await',
                            'print', 'return', 'pass', 'continue', 'break', 'raise', 'assert', 'del',
                            'and', 'or', 'not', 'in', 'is', 'as'},
                'pattern': r'def\s+\w+.*:|class\s+\w+.*:|import\s+\w+|from\s+\w+\s+import',
                'file_extensions': ['.py', '.pyw', '.pyc'],
                'import_patterns': [r'import\s+\w+', r'from\s+\w+\s+import']
            },
            'java': {
                'keywords': {'public', 'private', 'protected', 'class', 'interface', 'enum', 'extends', 'implements',
                            'static', 'final', 'void', 'return', 'if', 'else', 'for', 'while', 'switch', 'case',
                            'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super', 'import', 'package'},
                'pattern': r'public\s+class\s+\w+|private\s+\w+|public\s+static\s+void\s+main',
                'file_extensions': ['.java'],
                'import_patterns': [r'import\s+[\w.]+\*?;', r'package\s+[\w.]+;']
            },
            'javascript': {
                'keywords': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'switch', 'case',
                            'try', 'catch', 'finally', 'throw', 'return', 'async', 'await', 'export', 'import',
                            'class', 'extends', 'super', 'this', 'new', 'undefined', 'null', 'typeof', 'instanceof'},
                'pattern': r'function\s+\w+.*\{|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|class\s+\w+',
                'file_extensions': ['.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx'],
                'import_patterns': [r'import\s+.*from|require\(|module\.exports\s*=|\bexport\b']
            },
            'cpp': {
                'keywords': {'#include', 'using', 'namespace', 'class', 'struct', 'public', 'private', 'protected',
                            'virtual', 'override', 'const', 'static', 'template', 'typename', 'auto', 'decltype',
                            'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'try', 'catch', 'throw',
                            'new', 'delete', 'this', 'nullptr', 'true', 'false'},
                'pattern': r'#include\s*<.*>|using\s+namespace|class\s+\w+|struct\s+\w+|template\s*<.*>',
                'file_extensions': ['.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx', '.hh'],
                'import_patterns': [r'#include\s*[<"]']
            },
            'csharp': {
                'keywords': {'public', 'private', 'protected', 'internal', 'class', 'struct', 'interface', 'enum',
                            'namespace', 'using', 'static', 'readonly', 'virtual', 'override', 'abstract', 'sealed',
                            'if', 'else', 'for', 'foreach', 'while', 'switch', 'case', 'return', 'try', 'catch',
                            'finally', 'throw', 'throw', 'new', 'this', 'base', 'null', 'true', 'false', 'var', 'async', 'await'},
                'pattern': r'namespace\s+\w+|using\s+\w+;|public\s+class\s+\w+|private\s+\w+',
                'file_extensions': ['.cs'],
                'import_patterns': [r'using\s+[\w.]+;']
            },
            'go': {
                'keywords': {'package', 'import', 'func', 'var', 'const', 'type', 'struct', 'interface', 'map',
                            'chan', 'go', 'defer', 'range', 'select', 'if', 'else', 'for', 'switch', 'case',
                            'return', 'break', 'continue', 'fallthrough', 'default', 'goto', 'nil', 'make', 'len', 'cap'},
                'pattern': r'package\s+\w+|func\s+\w+.*\{|var\s+\w+|const\s+\w+=|type\s+\w+',
                'file_extensions': ['.go'],
                'import_patterns': [r'import\s*\(', r'import\s+[\w.]+']
            },
            'rust': {
                'keywords': {'fn', 'let', 'mut', 'const', 'static', 'struct', 'enum', 'impl', 'trait', 'pub',
                            'use', 'mod', 'crate', 'self', 'super', 'as', 'match', 'if', 'else', 'for', 'while',
                            'loop', 'return', 'break', 'continue', 'unsafe', 'move', 'async', 'await', 'yield',
                            'dyn', 'box', 'ref', 'static'},
                'pattern': r'fn\s+\w+.*\{|struct\s+\w+|enum\s+\w+|impl\s+\w+|trait\s+\w+|pub\s+fn|let\s+mut',
                'file_extensions': ['.rs'],
                'import_patterns': [r'use\s+[\w::]+']
            }
        }
    
    def detect_language(self, code: str, filename: Optional[str] = None) -> Tuple[str, float]:
        """
        Detect programming language from code sample.
        
        Args:
            code: Source code sample
            filename: Optional filename to help with detection
            
        Returns:
            Tuple of (language_name, confidence_score)
        """
        if not code or not code.strip():
            return 'unknown', 0.0
        
        # Check file extension first
        if filename:
            ext = self._get_extension(filename)
            for lang, patterns in self.language_patterns.items():
                if ext in patterns['file_extensions']:
                    # File extension matches, verify with code patterns
                    confidence = self._calculate_language_confidence(code, lang)
                    if confidence > 0.3:  # Threshold
                        return lang, confidence
        
        # Language detection based on code content
        scores = {}
        for lang, patterns in self.language_patterns.items():
            score = self._calculate_language_confidence(code, lang)
            scores[lang] = score
        
        if not scores or max(scores.values()) < 0.2:
            return 'unknown', 0.0
        
        # Return language with highest score
        detected_lang = max(scores, key=scores.get)
        confidence = scores[detected_lang]
        
        return detected_lang, confidence
    
    def _calculate_language_confidence(self, code: str, language: str) -> float:
        """Calculate confidence score for a specific language."""
        if language not in self.language_patterns:
            return 0.0
        
        patterns = self.language_patterns[language]
        score = 0.0
        total_checks = 0
        
        # Check for keywords
        if 'keywords' in patterns:
            keywords_found = sum(1 for keyword in patterns['keywords'] 
                                if re.search(rf'\b{keyword}\b', code, re.IGNORECASE))
            total_keywords = len(patterns['keywords'])
            if total_keywords > 0:
                keyword_score = keywords_found / total_keywords
                score += keyword_score * 0.5
                total_checks += 1
        
        # Check for patterns
        if 'pattern' in patterns:
            pattern_matches = len(re.findall(patterns['pattern'], code, re.MULTILINE | re.IGNORECASE))
            if pattern_matches > 0:
                pattern_score = min(pattern_matches / 10.0, 1.0)  # Cap at 1.0
                score += pattern_score * 0.3
                total_checks += 1
        
        # Check for imports
        if 'import_patterns' in patterns:
            import_matches = sum(1 for pattern in patterns['import_patterns'] 
                               if re.search(pattern, code, re.MULTILINE))
            if import_matches > 0:
                import_score = min(import_matches / 5.0, 1.0)  # Cap at 1.0
                score += import_score * 0.2
                total_checks += 1
        
        # Normalize score
        if total_checks > 0:
            score = min(score, 1.0)
        
        return score
    
    def _get_extension(self, filename: str) -> str:
        """Get file extension from filename."""
        if '.' in filename:
            return '.' + filename.split('.')[-1].lower()
        return ''
    
    def get_all_supported_languages(self) -> List[str]:
        """Get list of all supported languages."""
        return list(self.language_patterns.keys())
    
    def get_language_statistics(self, code: str) -> Dict[str, Any]:
        """Get detailed statistics for all languages."""
        stats = {}
        
        for lang in self.language_patterns.keys():
            confidence = self._calculate_language_confidence(code, lang)
            
            # Count occurrences
            patterns = self.language_patterns[lang]
            keywords_found = 0
            if 'keywords' in patterns:
                keywords_found = sum(1 for keyword in patterns['keywords'] 
                                    if re.search(rf'\b{keyword}\b', code, re.IGNORECASE))
            
            pattern_matches = 0
            if 'pattern' in patterns:
                pattern_matches = len(re.findall(patterns['pattern'], code, re.MULTILINE))
            
            stats[lang] = {
                'confidence': confidence,
                'keywords_found': keywords_found,
                'pattern_matches': pattern_matches
            }
        
        return stats

