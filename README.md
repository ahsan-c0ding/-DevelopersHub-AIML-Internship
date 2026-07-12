# DevelopersHub Corporation - AI/ML Internship
**Intern:** Ahsan Haris  

## Overview
This repository documents the end-to-end machine learning and AI projects completed during Phase 1 and Phase 2 of the internship. The work progresses from foundational data exploration and classical predictive modeling to advanced deep learning, Natural Language Processing (NLP), and the deployment of production-ready ML pipelines.

## Repository Structure
```text
.
├── .gitignore
├── README.md
├── Phase 1 Tasks/
│   ├── Task1_Dataset_Visualization/   # Dataset Visualization (Iris EDA)
│   ├── Task2_Stock_Prediction/   # Stock Price Prediction (AAPL)
│   ├── Task4_Health_Chatbot/   # General Health Query Chatbot (API & Prompting)
│   └── Task5_Mental_Health_Chatbot/   # Mental Health Support Chatbot (LLM Fine-Tuning)
└── Phase 2 Tasks/
    ├── Task1_News_Topic_Classifier/   # Multimodal Housing Price Prediction (CNN + Tabular)
    ├── Task2_End_to_End_ML_Pipeline/   # News Topic Classifier (BERT + Streamlit)
    └── Task3_Multimodal_ML_Housing_Price_Prediction/   # Telco Customer Churn (End-to-End sklearn Pipeline)
```




## Phase 1: Foundations, Classical ML & LLM Integration
Phase 1 focused on mastering data visualization, baseline regression modeling, and interacting with/fine-tuning Large Language Models (LLMs).

*   **Task 1: Dataset Visualization**
    *   **Focus:** Exploratory Data Analysis (EDA).
    *   **Summary:** Conducted a comprehensive visual exploration of the Iris dataset using Seaborn. Identified key feature distributions, class separability (Setosa vs. Versicolor/Virginica), and confirmed data cleanliness via box plots.
*   **Task 2: Stock Price Prediction**
    *   **Focus:** Time-series regression & baseline modeling.
    *   **Summary:** Predicted next-day AAPL closing prices using historical OHLCV data. Compared Linear Regression and Random Forest Regressor, achieving an excellent fit ($R^2 > 0.90$) and demonstrating that the short-term price relationship is largely linear.
*   **Task 4: General Health Query Chatbot**
    *   **Focus:** LLM API integration, prompt engineering, and AI safety.
    *   **Summary:** Built an ethical health chatbot using the Mistral-7B model via Hugging Face Inference API. Implemented strict system prompts to prevent medical diagnosis and engineered a pre-processing safety filter to intercept high-risk keywords before they reach the LLM.
*   **Task 5: Mental Health Support Chatbot (Fine-Tuned)**
    *   **Focus:** End-to-end LLM fine-tuning.
    *   **Summary:** Fine-tuned a lightweight DistilGPT-2 model on the Empathetic Dialogues dataset to act as an empathetic listener. Handled hardware constraints via mixed-precision training and optimized inference parameters to generate context-aware, supportive conversational responses.

---

## Phase 2: Deep Learning, NLP & Production MLOps
Phase 2 advanced into complex neural network architectures, transformer-based NLP, and building robust, exportable machine learning pipelines.

*   **Task 1: Multimodal Housing Price Prediction**
    *   **Focus:** Multimodal deep learning (Vision + Tabular).
    *   **Summary:** Developed a fusion model combining a CNN (for house photos) and an MLP (for tabular features like square footage) to predict real estate prices. Evaluated single-modality baselines against the fused model, highlighting the challenges of training CNNs from scratch on small datasets and the importance of modality signal quality.
*   **Task 2: News Topic Classifier**
    *   **Focus:** Transformer fine-tuning & model deployment.
    *   **Summary:** Fine-tuned `bert-base-uncased` on the AG News dataset to classify headlines into 4 categories, achieving ~94.5% accuracy. Deployed the model as an interactive, real-time web application using Streamlit, complete with dynamic padding and confidence scoring.
*   **Task 3: Telco Customer Churn**
    *   **Focus:** Production-ready MLOps & `scikit-learn` pipelines.
    *   **Summary:** Built a reusable, end-to-end ML pipeline for predicting customer churn. Bundled data preprocessing (scaling, one-hot encoding) and modeling (Logistic Regression vs. Random Forest) into a single exportable `joblib` object, ensuring seamless transition from training to inference without data leakage or sync issues.

---

## Key Skills Demonstrated
*   **Data Science & EDA:** Statistical analysis, data visualization, and feature engineering.
*   **Classical Machine Learning:** Regression, classification, hyperparameter tuning, and pipeline construction (`scikit-learn`).
*   **Deep Learning & Computer Vision:** CNNs, multimodal fusion, and transfer learning concepts.
*   **NLP & Generative AI:** Transformer fine-tuning (BERT, GPT-2), LLM API integration, prompt engineering, and safety guardrails.
*   **MLOps & Deployment:** Model serialization, end-to-end pipeline packaging, and lightweight web app deployment (Streamlit).