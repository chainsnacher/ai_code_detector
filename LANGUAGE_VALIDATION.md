# Language Validation for AI Code Detection System

## Overview

The language validation feature prevents incorrect code analysis by detecting and alerting users when the selected programming language doesn't match the actual code being analyzed.

## How It Works

### Detection Process

1. **Language Selection**: User selects the programming language from the dropdown
2. **Language Analysis**: The system analyzes the code and calculates confidence scores for all supported languages
3. **Validation Check**: The system compares the user-selected language with the detected language
4. **Results**:
   - ✅ **Match**: If the selected language matches the detected language, analysis proceeds normally
   - ❌ **Mismatch**: If there's a significant mismatch, an error is displayed and analysis stops
   - ⚠️ **Unclear**: If the language detection is uncertain, a warning is shown

## Error Messages

### Language Mismatch Error
```
❌ Language mismatch detected! You selected 'Python' but the code appears to be 'JAVA' 
(Confidence: 13.1%). Please verify your language selection.
```
When the code's language differs from the user's selection with reasonable confidence.

### Unclear Detection Warning
```
⚠️ Could not reliably detect the programming language. You selected 'Python', 
but the code signature is unclear. Possible languages: java, cpp, csharp. 
Please verify your selection.
```
When the language detection cannot clearly identify the language.

## Supported Languages

- Python (.py)
- Java (.java)
- JavaScript (.js, .jsx, .ts, .tsx)
- C++ (.cpp, .h, .hpp)
- C# (.cs)
- Go (.go)
- Rust (.rs)

## Implementation Details

### Validation Function
Located in `web_app/app.py`

```python
def _validate_language_selection(code: str, user_selected_lang: str, lang_detector: Any) -> Dict[str, Any]:
    """
    Validate if the code matches the user-selected language.
    """
```

### Detection Thresholds

- **Mismatch Detection**: Triggered when the highest-scoring language has confidence > 8% 
  and doesn't match the selected language
- **Valid**: When selected language matches the top-scoring language
- **Unclear**: When multiple languages have similar low scores

### Language Mapping

The system normalizes language names to handle common variations:
- `py` → `python`
- `js`, `ts`, `tsx`, `jsx` → `javascript`
- `c++`, `cpp` → `cpp`
- `cs` → `csharp`

## Usage Examples

### Example 1: Correct Language Selection
```
User selects: Python
Code: def hello(): print("Hi")
Result: ✅ Analysis proceeds - Language validated as Python
```

### Example 2: Wrong Language Selection
```
User selects: Python
Code: public class Hello { public static void main(String[] args) { ... } }
Result: ❌ Error - Code is Java, not Python
         Analysis stops - User must select correct language
```

### Example 3: Ambiguous Code
```
User selects: Python
Code: var x = 5; // simple variable with few identifiers
Result: ⚠️ Warning - Could not reliably detect language
         Shows possible alternatives: javascript, cpp, csharp
```

## Benefits

1. **Accuracy**: Ensures models are trained on the correct language patterns
2. **User Guidance**: Helps users understand what language their code is in
3. **Quality Assurance**: Prevents false positives/negatives from language mismatches
4. **Better Insights**: Shows language detection confidence and alternatives

## Technical Details

### Language Scoring Algorithm

The detection uses multiple criteria:
1. **Keyword Matching**: Checks for language-specific keywords (50% weight)
2. **Pattern Matching**: Looks for language-specific syntax patterns (30% weight)
3. **Import Statements**: Checks import/include patterns (20% weight)

### Integration Points

- Called in `analyze_single_code()` after code preprocessing
- Results displayed in analysis output before model predictions
- Language scores shown in expandable details section

## Configuration

To adjust validation sensitivity, modify the threshold in `_validate_language_selection`:

```python
# Current threshold: 0.08 (8% confidence)
if best_confidence > 0.08:  
    return {is_valid: False, ...}
```

Lower values = more strict validation
Higher values = more lenient validation

## Future Enhancements

- [ ] Support for more languages (Kotlin, Swift, Ruby, PHP, etc.)
- [ ] Machine learning-based language detection
- [ ] Auto-correction of language selection
- [ ] Language-specific analysis tuning
- [ ] Dialect detection (Python 2 vs 3, etc.)
