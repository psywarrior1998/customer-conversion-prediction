[![Hugging Face Model](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg)](https://huggingface.co/psyrishi/marketing-conversion-predictor)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-conversion-prediction.streamlit.app/)
# Customer Conversion Prediction for Digital Marketing

An end-to-end machine learning project to predict customer conversion in digital marketing campaigns, aimed at optimizing Return on Ad Spend (ROAS).

---

## 📋 Table of Contents
- [Business Objective](#-business-objective)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Technology Stack](#-technology-stack)
- [Model Performance](#-model-performance)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [Repository Structure](#-repository-structure)

---

## 🎯 Business Objective

The primary goal of this project is to develop a robust machine learning model that accurately predicts which customers are most likely to convert during a digital marketing campaign. By identifying these potential converters, a business can:

-   🎯 **Improve Campaign Targeting**: Focus marketing efforts and ad spend on the highest-potential customers.
-   💰 **Optimize Costs**: Reduce wasted ad spend on audiences that are unlikely to convert.
-   🚀 **Enhance Campaign Strategy**: Gain data-driven insights into the key factors that drive conversions, leading to more effective marketing strategies.

---

## 📊 Dataset

The dataset used for this project contains anonymized customer data, including demographics, online behavior, and campaign interaction metrics.

-   **Source**: [Click here for Dataset Link](https://www.kaggle.com/datasets/rabieelkharoua/predict-conversion-in-digital-marketing-dataset)
-   **Size**: 8,000 initial records, 17 features.
-   **Features Include**: `CustomerID`, `Age`, `Gender`, `Income`, `CampaignChannel`, `CampaignType`, `AdSpend`, `ClickThroughRate`, `ConversionRate`, `AdvertisingPlatform`, `AdvertisingTool`, `WebsiteVisits`, `PagesPerVisit`, `TimeOnSite`, `SocialShares`, `EmailOpens`, `EmailClicks`, `PreviousPurchases`, `LoyaltyPoints`, and the target variable `Conversion`.

---

## ⚙️ Project Workflow

The project followed a systematic, end-to-end machine learning pipeline:

1.  **Data Cleaning & Preprocessing**:
    -   Removed non-predictive identifier columns.
    -   Utilized an **Isolation Forest** algorithm to detect and remove 40 anomalous data points, ensuring a high-quality dataset for training.

2.  **Exploratory Data Analysis (EDA)**:
    -   Visualized feature distributions and relationships to understand the underlying patterns in the data.

3.  **Feature Engineering**:
    -   Created high-impact features to capture complex user behaviors:
        -   `EngagementScore`: `TimeOnSite` * `PagesPerVisit`
        -   `CostPerVisit`: `AdSpend` / `WebsiteVisits`
    -   Binned `Age` and `Income` into categorical groups (`AgeGroup`, `IncomeTier`) to help the model capture non-linear trends.

4.  **Model Selection & Training**:
    -   Evaluated a wide range of classification algorithms.
    -   Selected the top 4 performing models: **CatBoost, XGBoost, LightGBM, and Gradient Boosting**.
    -   Conducted intensive hyperparameter tuning for each model using **Optuna**.
    -   Constructed a final **Voting Classifier Ensemble** to leverage the strengths of all four models, maximizing accuracy and robustness.

5.  **Model Evaluation**:
    -   Assessed the final model on an unseen test set using a comprehensive classification report, ROC Curve, and Precision-Recall Curve.
    -   Optimized the decision threshold to maximize the F1-score, balancing precision and recall for the best business outcome.

6.  **Feature Importance**:
    -   Used **SHAP (SHapley Additive exPlanations)** to interpret the model's predictions and identify the key drivers of customer conversion.

---

## 🛠️ Technology Stack

-   **Programming Language**: Python 3.x
-   **Web Framework**: Streamlit (for model deployment and user interface)
-   **Model Hosting**: Hugging Face Hub
-   **Core Libraries**: Pandas, NumPy, Scikit-learn
-   **Gradient Boosting**: XGBoost, LightGBM, CatBoost
-   **Hyperparameter Tuning**: Optuna
-   **Visualization**: Matplotlib, Seaborn
-   **Model Persistence**: Joblib

---

## 🏆 Model Performance

The final ensemble model achieved excellent performance on the test set:

-   **Accuracy**: **92.21%**
-   **F1-Score (for Converters)**: **0.9569**
-   **ROC-AUC Score**: High performance in distinguishing between converting and non-converting customers.

---

## 💡 Key Findings

The SHAP analysis revealed the most influential factors driving customer conversions:

1.  **TimeOnSite**: The single most important predictor. The more time a user spends on the site, the higher their likelihood of conversion.
2.  **PreviousPurchases**: Past purchasing behavior is a very strong indicator of future conversions.
3.  **LoyaltyPoints**: Customers with more loyalty points are significantly more likely to convert.
4.  **EngagementScore**: Our engineered feature proved to be a top predictor, validating its effectiveness.

---

## 🌐 Live Web Application

Explore the model interactively through the deployed Streamlit web application:
**[Launch App](https://customer-conversion-prediction.streamlit.app/)**

The web app allows marketers to:
* **Input Customer Data**: Manually enter demographic details (Age, Income) and campaign engagement metrics (Time on Site, Pages Per Visit).
* **Get Real-Time Predictions**: Instantly see if a customer is likely to convert based on the final ensemble model.
* **View Confidence Scores**: Understand the model's certainty regarding the prediction.

---

## 🚀 Getting Started

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/psywarrior1998/customer-conversion-prediction.git
    cd customer-conversion-prediction
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook Capstone_Digital_Marketing_Campaign_Prediction.ipynb
    ```

---

## 📁 Repository Structure

```
.
├── Digital_Marketing_Campaign_Prediction.ipynb     \# The main notebook with all analysis and code
├── requirements.txt                                \# A file listing all the necessary Python libraries
└── README.md                                       \# You are here!
```
