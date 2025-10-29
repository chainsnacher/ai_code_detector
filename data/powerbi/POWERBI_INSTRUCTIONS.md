
# Power BI Dashboard Setup Instructions

## Data Sources
1. Navigate to: data/powerbi/
2. Load the following CSV files into Power BI:
   - dashboard_predictions.csv - Prediction results
   - dashboard_performance.csv - Model performance metrics
   - dashboard_features.csv - Feature importance data
   - dashboard_stats.json - Training statistics

## Recommended Visualizations

### 1. Prediction Distribution
- Chart Type: Donut Chart
- Values: Count by prediction type (AI Generated vs Human Written)
- Colors: Red for AI, Green for Human

### 2. Confidence Distribution
- Chart Type: Histogram
- X-Axis: Confidence levels (0.0 - 1.0)
- Y-Axis: Count of predictions
- Color by: Prediction type

### 3. Model Performance Comparison
- Chart Type: Bar Chart
- X-Axis: Model names
- Y-Axis: Accuracy, F1-Score, Precision, Recall

### 4. Language Distribution
- Chart Type: Pie Chart
- Values: Count by language
- Tooltip: Show percentage

### 5. Feature Importance
- Chart Type: Horizontal Bar Chart
- X-Axis: Importance scores
- Y-Axis: Feature names
- Color by: Feature category

### 6. Time Series Analysis
- Chart Type: Line Chart
- X-Axis: Timestamp
- Y-Axis: Prediction count
- Series: Split by prediction type

## DAX Measures (Load from measures.json)
- Total Samples
- AI Percentage
- Average Confidence
- High Confidence Samples
- Accuracy by Language

## Filters
- Add slicers for:
  - Language
  - Date range
  - Confidence threshold
  - Model name
  - Prediction type

## Dashboard Layout
1. Top Section: Overall metrics (KPIs)
2. Left: Prediction distribution
3. Center: Confidence histogram
4. Right: Model performance comparison
5. Bottom: Language distribution and feature importance
