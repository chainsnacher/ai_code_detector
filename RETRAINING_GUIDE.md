# Retraining Guide - AI Code Detector

This guide explains how to retrain the full model pipeline with real AI and human code data.

## Quick Start

### Option 1: Full Automated Retraining

```bash
python retrain_full_pipeline.py --collect-data
```

This will:
1. Collect real training data (AI and human code)
2. Extract comprehensive features
3. Train models with proper class balancing
4. Evaluate and save models

### Option 2: Use Existing Data

If you already have training data in `data/train/human/` and `data/train/ai/`:

```bash
python retrain_full_pipeline.py --skip-data-collection
```

### Option 3: Step-by-Step

1. **Collect Training Data**:
   ```bash
   python scripts/collect_real_data.py
   ```
   
   This script:
   - Collects human code from GitHub (requires `GITHUB_TOKEN` env var, optional)
   - Generates AI code using OpenAI API (requires `OPENAI_API_KEY` env var, optional)
   - Falls back to synthetic AI-like code if APIs are not available
   - Saves files to `data/train/human/` and `data/train/ai/`

2. **Train Models**:
   ```bash
   python scripts/improved_train_pipeline.py
   ```
   
   This script:
   - Loads training data
   - Extracts comprehensive features (statistical, AST, token-based)
   - Balances classes (undersampling)
   - Trains multiple models (Random Forest, SVM, Logistic Regression, Gradient Boosting)
   - Creates ensemble models
   - Evaluates on validation and test sets
   - Saves models to `models/baseline/` and `models/ensemble/`

## What's Improved

### 1. Real Data Collection
- Collects actual human code from GitHub repositories
- Generates AI code using real AI models (OpenAI GPT-4, Claude, etc.)
- Falls back to synthetic data with clear AI patterns if APIs unavailable

### 2. Better Class Balancing
- Uses undersampling to balance classes
- Prevents model from always predicting the majority class
- Ensures proper distinction between AI and human code

### 3. Enhanced Feature Engineering
- Statistical features (50+ features)
- AST-based features (structure, complexity)
- Token-based features (lexical patterns)
- All features properly normalized and sanitized

### 4. Improved Model Configuration
- Random Forest: 300 estimators, balanced class weights
- SVM: Higher C value, balanced classes
- Logistic Regression: Higher C value, balanced classes
- Gradient Boosting: Improved subsampling

### 5. Fixed Prediction Logic
- No longer defaults to "human" for uncertain cases
- Uses heuristics to refine borderline predictions
- Properly distinguishes AI vs human code
- Confidence scores reflect actual model certainty

## Model Performance

After training, check `models/training_results.json` for detailed metrics:
- Accuracy
- F1-Score
- ROC-AUC
- Confusion matrices

Expected performance:
- **Random Forest**: ~85-90% accuracy, ~0.85-0.90 F1
- **SVM**: ~80-85% accuracy, ~0.80-0.85 F1
- **Logistic Regression**: ~80-85% accuracy, ~0.80-0.85 F1
- **Ensemble**: ~90-95% accuracy, ~0.90-0.95 F1

## Troubleshooting

### Issue: "No training files found"
**Solution**: Run `python scripts/collect_real_data.py` first to generate training data.

### Issue: "Models always predict human"
**Solution**: 
1. Check class distribution in training data (should be balanced)
2. Ensure models are trained with `class_weight='balanced'`
3. Verify features are being extracted correctly

### Issue: "Low accuracy/F1 score"
**Solution**:
1. Collect more diverse training data
2. Ensure balanced classes
3. Check feature extraction is working
4. Try adjusting model hyperparameters

### Issue: "API errors during data collection"
**Solution**: 
- The script will automatically fall back to synthetic data
- Set `GITHUB_TOKEN` and `OPENAI_API_KEY` environment variables for real data
- Synthetic data still has clear AI vs human patterns

## Testing the Models

After training, test your models:

```bash
# Quick test
python quick_test.py

# Detailed test
python test_model_predictions.py

# Web app
streamlit run web_app/app.py
```

## File Structure

```
data/
  train/
    human/     # Human-written code samples
    ai/        # AI-generated code samples
  processed/
    features.csv  # Extracted features

models/
  baseline/    # Individual model files
  ensemble/    # Ensemble model files
  feature_columns.json  # Feature column names
  training_results.json  # Training metrics

scripts/
  collect_real_data.py      # Data collection
  improved_train_pipeline.py  # Training pipeline
```

## Environment Variables (Optional)

For better data collection:

```bash
export GITHUB_TOKEN="your_github_token"      # For GitHub API
export OPENAI_API_KEY="your_openai_key"      # For OpenAI API
```

## Next Steps

1. **Monitor Performance**: Check `models/training_results.json` after training
2. **Fine-tune**: Adjust hyperparameters in `src/models/baseline_models.py`
3. **Add More Data**: Collect more diverse samples for better generalization
4. **Evaluate**: Test on real-world code samples
5. **Deploy**: Use trained models in production

## Notes

- Training time: ~5-15 minutes depending on data size
- Minimum recommended data: 200 samples per class
- Best results: 500+ samples per class
- Models are saved automatically after training
- Old models are overwritten (backup if needed)
