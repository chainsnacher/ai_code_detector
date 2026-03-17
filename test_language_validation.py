"""
Test script for language validation functionality.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'src'))

from src.preprocessing.language_detector import LanguageDetector

def test_language_validation():
    """Test language detection and validation."""
    
    lang_detector = LanguageDetector()
    
    # Test cases
    test_cases = [
        {
            'name': 'Python code - correct selection',
            'code': '''
def hello_world():
    print("Hello, World!")
    return True
''',
            'selected_language': 'python',
            'expected_detection': 'python'
        },
        {
            'name': 'JavaScript code - incorrect selection (Python)',
            'code': '''
function helloWorld() {
    console.log("Hello, World!");
    return true;
}
''',
            'selected_language': 'python',
            'expected_detection': 'javascript'
        },
        {
            'name': 'Java code - incorrect selection (Python)',
            'code': '''
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
''',
            'selected_language': 'python',
            'expected_detection': 'java'
        },
        {
            'name': 'C++ code - incorrect selection (Python)',
            'code': '''
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
''',
            'selected_language': 'python',
            'expected_detection': 'cpp'
        },
    ]
    
    print("=" * 70)
    print("Language Validation Test Results")
    print("=" * 70)
    
    for test in test_cases:
        detected_lang, confidence = lang_detector.detect_language(test['code'])
        selected_lang = test['selected_language']
        
        print(f"\n✓ Test: {test['name']}")
        print(f"  Selected Language: {selected_lang}")
        print(f"  Detected Language: {detected_lang}")
        print(f"  Detection Confidence: {confidence:.1%}")
        
        if detected_lang.lower() != selected_lang.lower():
            print(f"  ⚠️  MISMATCH: Code is {detected_lang.upper()}, but user selected {selected_lang.upper()}")
            if confidence > 0.5:
                print(f"  ❌ Would trigger ERROR: High confidence mismatch")
            else:
                print(f"  ⚠️  Would trigger WARNING: Unclear detection")
        else:
            print(f"  ✅ MATCH: Language selection is correct")
        
        # Show language statistics
        lang_stats = lang_detector.get_language_statistics(test['code'])
        print(f"\n  Language Scores:")
        sorted_scores = sorted(
            lang_stats.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )[:3]
        for lang, stats in sorted_scores:
            print(f"    - {lang.title()}: {stats['confidence']:.1%} confidence")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == '__main__':
    test_language_validation()
