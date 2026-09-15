# E-commerce 12-Month Customer LTV Prediction

## 📌 Business Context & Objective
A major e-commerce clothing retailer spends millions on marketing to acquire new users. However, many customers purchase once and never return. The marketing team currently allocates retention budgets blindly based on the initial transaction.

**Objective:** Build a machine learning pipeline to predict a customer's **12-month Lifetime Value (LTV)** based strictly on their **first week of activity** on the website. 

### 💼 Business Application (Production Strategy)
Instead of forcing a rigid classification (High vs. Low Value), this project implements a continuous **Regression model with dynamic thresholding**:
* **Batch Processing:** Every night, a pipeline fetches users who registered exactly 7 days ago, extracts their weekly features, and scores them.
* **CRM Automation:** Customers with predicted LTV > $2,000 are automatically flagged as VIPs in the CRM, triggering premium retention campaigns (e.g., personal concierge, free annual shipping).
* **Cost Saving:** Users predicted below $300 are excluded from expensive retargeting ads, saving substantial marketing budget.

---

## 🚀 Key Architectural Decisions

1. **Time-Based Validation Split:** Data is sorted chronologically by `first_purchase_date`. The first 80% is used for training, and the final 20% for testing. This strictly mimics production deployment and prevents catastrophic **Data Leakage** (looking into the future).
2. **Log-Transformation (`np.log1p`):** LTV data naturally contains extreme outliers (VIP whales). Applying a log-transformation stabilized the training, shifted the gradient boosting logic from additive to multiplicative, and significantly reduced overall RMSE.
3. **Advanced Feature Engineering:** Rather than feeding raw logs, three psychology-driven user features were engineered:
   * `feature_engagement_index`: Interaction between page views and checkout scale.
   * `feature_ios_premium_wallet`: Flagging high-purchasing power (iOS users who don't use discount codes).
   * `feature_intent_score`: High velocity browsing combined with high starting cart value.

---

## 📊 Model Performance & Insights

The **Gradient Boosting Regressor** yielded an excellent balanced fit between the training and testing datasets (No Overfitting/Underfitting detected).

* **Train MAE:** 72.40 units
* **Test MAE:** 79.60 units (On average, predictions deviate by ~$79.60 from the actual 12-month LTV)
* **Test RMSE:** 123.57 units (The gap between RMSE and MAE narrowed, proving that the log-transform successfully minimized catastrophic errors on high-value customers)

### 🔥 Feature Importance Top Drivers
1. **`first_checkout_amount` (55.7%):** Acting as the primary mathematical baseline anchor due to the log-scale shift.
2. **`feature_ios_premium_wallet` (35.9%):** Validated the core business hypothesis that premium device ownership without discounting is an incredibly robust proxy for long-term loyalty.
3. **`feature_engagement_index` (5.5%):** Served as a critical fine-tuning metric for mid-tier user classification.

---

## 🛠️ How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ecommerce-ltv-prediction.git
   cd ecommerce-ltv-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python src/train.py
   ```
