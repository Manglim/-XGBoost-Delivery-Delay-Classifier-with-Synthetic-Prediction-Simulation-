# 📦 Delivery Delay Prediction using XGBoost

This project builds and evaluates an XGBoost classifier to predict whether a shipment will be delivered on time, using historical logistics data. The model is trained with engineered features and standardized inputs, followed by synthetic data generation to simulate future delivery scenarios and assess predicted delays.

---

## 🔍 Overview

- Trains an XGBoost classifier on a labeled dataset of shipment records.
- Evaluates model performance with F1 score, ROC AUC, and a confusion matrix.
- Generates two sets of synthetic delivery orders (200 + 50) to simulate new predictions.
- Predicts delivery delay probabilities and binary delay outcomes for all synthetic records.
- Outputs results with order dates and predicted probabilities.

---

## 📊 Features Used

- `Cost_of_the_Product`, `Prior_purchases`, `Customer_rating`, `Weight_in_gms`, `Discount_offered`
- `Warehouse_block`, `Mode_of_Shipment`, `Product_importance`, `Gender`
- `Order_Date`, `DayOfWeek`, `IsWeekend`

---

## 🧠 Model Configuration

- **Algorithm:** `XGBClassifier`
- **Key Hyperparameters:**
  - `n_estimators=80`
  - `max_depth=4`
  - `learning_rate=0.03`
  - `scale_pos_weight=0.675`
  - `eval_metric='logloss'`

---

## 🧪 Evaluation Metrics

- `F1 Score`
- `ROC AUC Score`
- `Classification Report`
- `Confusion Matrix`

---

## 📈 Visualizations

- Confusion matrix heatmap
- Top 15 feature importances

---

## 📁 Output

The final output consists of **250 synthetic delivery records**, each with:

- `Order_Date`
- `Predicted_Probability_Late`
- `Predicted_Late` (1 = late, 0 = on-time)

---

## 🧰 Requirements

```bash
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
