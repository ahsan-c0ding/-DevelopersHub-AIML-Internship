# DevelopersHub Corporation - AI/ML Engineering Internship
**Intern Name:** Ahsan Haris  
**Phase:** Phase 2  
**Due Date:** 14th July, 2026
 
## Tasks Completed
 
### Task 1: Multimodal Housing Price Prediction (CNN + Tabular Fusion)
- **Objective:** Build a model that fuses CNN-extracted image features with tabular features to predict housing prices, and evaluate it against single-modality baselines.
- **Dataset:** Houses Dataset (535 houses, each with 4 photos—bedroom, bathroom, kitchen, frontal, and tabular records like bedrooms, bathrooms, area, and zip code).
- **Models:** Tabular-only (small MLP), Image-only (small CNN on a 2x2 photo montage), and Fusion (parallel branches concatenated into dense layers).
- **Results:**
  | Model | MAE | RMSE |
  |---|---|---|
  | Tabular-only | $149,683 | $228,619 |
  | Image-only | $272,664 | $388,596 |
  | Fusion (image + tabular) | $180,482 | $261,171 |
- **Key Findings:**
  - **Tabular data is the strongest single input**, as features like square footage and zip code are low-noise predictors that real estate appraisers rely on.
  - **Image-only was the weakest model.** With only 535 houses, training a CNN from scratch lacked the data needed to learn robust visual features without a pretrained backbone.
  - **Fusion underperformed the tabular-only baseline.** Concatenating a noisy image-feature vector onto strong tabular features actually hurt performance. This highlights that multimodal fusion only pays off when every modality is actually pulling its weight.

### Task 2: News Topic Classifier (BERT Fine-Tuning)
- **Objective:** Fine-tune a transformer-based text classifier to automatically route news headlines into four topical categories, and deploy it as an interactive Streamlit app for live predictions.
- **Dataset:** AG News dataset (120,000 training and 7,600 test examples, evenly split across World, Sports, Business, and Sci/Tech).
- **Tech Stack:** Hugging Face Transformers (`bert-base-uncased`), `BertTokenizerFast`, Streamlit.
- **Results:**
  | Metric | Score |
  |---|---|
  | Test Accuracy | 94.53% |
  | Macro F1 | 94.53% |
  | Macro Precision | 94.55% |
  | Macro Recall | 94.53% |
- **Key Findings:**
  - **Sports is the strongest class** (98.66% F1) due to distinctive, low-ambiguity vocabulary. 
  - **Business and Sci/Tech are the weakest classes** (~91-92% F1), confirming expected real-world topical overlap (e.g., tech-company earnings, product launches).
  - **Dynamic padding** (`DataCollatorWithPadding`) significantly reduced wasted compute compared to fixed-length padding, as headline lengths varied greatly.
  - **Deployment:** Successfully built a lightweight Streamlit app that loads the fine-tuned checkpoint and classifies user-entered text in real time with per-class confidence bars.

### Task 3: Telco Customer Churn (End-to-End sklearn Pipeline)
- **Objective:** Build a reusable, production-ready machine learning pipeline that predicts customer churn, bundling preprocessing and modeling into a single exportable object.
- **Dataset:** Telco Customer Churn (7,043 customers, 20 features including demographics, services, and billing info; ~26.5% churn rate).
- **Models:** Logistic Regression and Random Forest, each wrapped with a `ColumnTransformer` (scaling numerics, one-hot encoding categoricals) inside a single `sklearn` `Pipeline`.
- **Results:**
  | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
  |---|---|---|---|---|---|
  | Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 | 0.841 |
  | Random Forest | 0.774 | 0.557 | 0.717 | 0.627 | 0.838 |
- **Key Findings:**
  - **Logistic Regression was selected as the final exported model.** It achieved the best ROC-AUC and accuracy, and is simpler and faster to serve in production than Random Forest.
  - **Random Forest trades precision for recall** (catches more actual churners but generates more false positives), which might be preferable if the business cost of a missed churner is higher than a retention call.
  - **Biggest churn drivers** identified during EDA: month-to-month contract type, short tenure (new customers), and high monthly charges. Add-on services like tech support correlate with lower churn.