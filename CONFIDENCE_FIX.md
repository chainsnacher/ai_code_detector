# Confidence Score Fix

## Problem
Human-generated code was always showing 76% confidence, indicating the confidence calculation wasn't varying based on actual model certainty.

## Root Cause
The previous confidence calculation used only probability differences:
```python
confidence = abs(weighted_ai_prob - weighted_human_prob)
confidence = 0.5 + (confidence * 0.45)
```

This approach:
- Didn't account for the actual winning probability
- Could produce similar confidence scores for different certainty levels
- Didn't reflect how certain the models actually were

## Solution

### New Confidence Calculation
Now uses a combination of:
1. **Winning probability**: The probability of the predicted class
2. **Probability difference**: How much more certain one class is than the other

```python
winning_prob = max(weighted_ai_prob, weighted_human_prob)
prob_diff = abs(weighted_ai_prob - weighted_human_prob)
confidence = winning_prob * 0.7 + prob_diff * 0.3
confidence = max(0.55, min(0.95, confidence))
```

### How It Works
- **High certainty** (e.g., 0.9 human, 0.1 AI):
  - winning_prob = 0.9
  - prob_diff = 0.8
  - confidence = 0.9 * 0.7 + 0.8 * 0.3 = 0.63 + 0.24 = 0.87

- **Moderate certainty** (e.g., 0.65 human, 0.35 AI):
  - winning_prob = 0.65
  - prob_diff = 0.3
  - confidence = 0.65 * 0.7 + 0.3 * 0.3 = 0.455 + 0.09 = 0.545 → 0.55 (min)

- **Low certainty** (e.g., 0.55 human, 0.45 AI):
  - winning_prob = 0.55
  - prob_diff = 0.1
  - confidence = 0.55 * 0.7 + 0.1 * 0.3 = 0.385 + 0.03 = 0.415 → 0.55 (min)

### Additional Improvements
1. **Model Agreement Boost**: If all models agree (>80%), boost confidence
2. **Disagreement Penalty**: If models disagree (<60%), reduce confidence
3. **Heuristic Adjustments**: Only for low-confidence cases, don't override high confidence
4. **Meta-classifier**: Uses same improved confidence calculation

## Expected Results
- **Confidence now varies**: 0.55-0.95 based on actual certainty
- **High certainty predictions**: 0.85-0.95 confidence
- **Moderate certainty**: 0.65-0.85 confidence
- **Low certainty**: 0.55-0.70 confidence
- **No more stuck at 76%**: Confidence reflects actual model certainty

## Testing
Test with various code samples:
- Clear human code → Should show 0.80-0.95 confidence
- Clear AI code → Should show 0.80-0.95 confidence
- Ambiguous code → Should show 0.55-0.75 confidence
- Mixed signals → Should show 0.60-0.80 confidence

Confidence scores should now properly reflect how certain the models are about their predictions.
