# 📊 Telco Customer Churn Analysis

Exploratory data analysis and machine learning project to predict customer churn in a telecommunications company, using the IBM Telco Customer Churn dataset.

---

## 🎯 Objective

Identify key factors that drive customer churn and build predictive models (Logistic Regression and Random Forest) to classify customers at risk of leaving the service.

---

## 📁 Project Structure

```
telco-churn-analysis/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   └── analysis.py
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

- **Source:** [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 21 columns (demographics, services contracted, billing info, and churn label)

---

## 🔍 Key Findings (EDA)

- Customers on **month-to-month contracts** show the highest churn rate.
- **Short-tenure customers (0–1 year)** are significantly more likely to churn.
- Churned customers tend to have **higher monthly charges**.
- `TotalCharges` has a strong positive correlation with `tenure`.

---

## 🤖 Models

Both models use a **scikit-learn Pipeline** that handles preprocessing automatically:
- `StandardScaler` for numeric features
- `OneHotEncoder` for categorical features

| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~79–81% |

The **Random Forest** model also provides **Feature Importance**, identifying the top predictors of churn (e.g., contract type, tenure, monthly charges).

---

## 🛠️ Tech Stack

- Python 3.10+
- pandas · NumPy
- scikit-learn
- Matplotlib · Seaborn
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/telco-churn-analysis.git
cd telco-churn-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the dataset inside the `data/` folder.

4. Run the full pipeline:
```bash
python src/analysis.py
```

5. Or open the notebook for interactive exploration:
```bash
jupyter notebook notebooks/eda.ipynb
```

---

## 📌 Status

🟡 In progress — Random Forest added. Next steps: hyperparameter tuning, ROC-AUC evaluation.

---

## 👤 Author

**Sebastián Aguilar**  
Data Analytics enthusiast | Python · SQL · Tableau  
📧 sebasaguilargimenez@gmail.com
