# 🏠 House Price Predictor — Linear Regression
## Maincrafts Technology | AI/ML Task 1

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## 📌 Objective

Build and evaluate a Linear Regression model on the California Housing dataset following the complete ML workflow — data loading, EDA, preprocessing, training, evaluation, and reporting.

---

## 🗂️ Dataset

**California Housing Dataset** (built into scikit-learn)
- **20,640** samples
- **8 features** — MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- **Target** — Median House Value (in $100,000s)

---

## 📁 Project Structure

```
house-price-prediction-ml/
│
├──  task1_ml_linear_regression.ipynb
├── images/
│ ├── geographic_visualization.png
│ ├── coefficients.png
│ ├── correlation_heatmap.png
│ ├── feature_distribution.png
│ ├── model_evaluation.png
│ ├── residual_analysis.png
│ ├── target_variable.png
│ ├── feature_vs_target.png
│ └── train_vs_test_metrics.png
├── report/
│   └── House_Price_Report.pdf
├── predict_ui.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

```bash
# Step 1 — Clone repo
git clone https://github.com/Rosesharma13/house-price-predictor.git
cd house-price-predictor

# Step 2 — Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Step 3 — Install dependencies
pip install -r requirements.txt

# Step 4 — Launch Jupyter Notebook
jupyter notebook task1_ml_linear_regression.ipynb
```

---

## 🚀 How to Run

### Option A — Jupyter Notebook (Main Deliverable)
```bash
jupyter notebook task1_ml_linear_regression.ipynb
```
Run all cells in order — top to bottom.

### Option B — Streamlit Prediction UI (Optional)
```bash
# First run the notebook to generate the saved model
# Then:
streamlit run predict_ui.py
```
Open **http://localhost:8501** 🎉

---

## 📊 ML Workflow Covered

| Step | Action |
|---|---|
| 1 | Import libraries |
| 2 | Load California Housing dataset |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Feature selection + Train/Test split (80/20) |
| 5 | Feature scaling — StandardScaler |
| 6 | Train LinearRegression model |
| 7 | Evaluate — MAE, RMSE, R² |
| 8 | Visualize — Actual vs Predicted, Residuals |
| 9 | Save model with pickle |
| 10 | Predict on new input |

---

## 📈 Model Results

| Metric | Train | Test |
|---|---|---|
| **MAE** | ~0.53 | ~0.53 |
| **RMSE** | ~0.73 | ~0.74 |
| **R² Score** | ~0.61 | ~0.60 |

> Model explains ~60% of variance in California house prices.
> Average prediction error of ~$53,000.

---

## 📸 Key Visualizations

- Target variable distribution (histogram + boxplot)
- Feature distributions (all 8 features)
- Correlation heatmap
- Scatter plots — top features vs target
- Geographic price map of California
- Feature coefficients bar chart
- Actual vs Predicted scatter plot
- Residual plot
- Residual distribution + Q-Q plot
- Train vs Test metrics comparison

---

## 💡 Key Findings

1. **MedInc** (Median Income) is the strongest predictor of house prices
2. Coastal California (Los Angeles, San Francisco) commands premium prices
3. AveBedrms has a negative coefficient — more bedrooms relative to rooms = lower value
4. Model is slightly limited by linear assumptions — non-linear models would improve accuracy

---

## 🔮 Improvement Ideas

- Polynomial feature engineering for non-linear relationships
- Log transformation of skewed features
- Ridge / Lasso regularization
- Random Forest or XGBoost for better accuracy
- Remove price-capped outliers (value = 5.0)

---

## 📦 Deliverables

- [x] Jupyter Notebook with code, plots, and comments
- [x] Saved model (pickle)
- [x] Optional Streamlit prediction UI
- [ ] PDF report (2-4 pages) — summarize EDA findings manually

---

## 👩‍💻 Author

**Rose Sharma**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rose-sharma13)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=flat&logo=gmail&logoColor=white)](mailto:rosesharmaa132003@gmail.com)

---

## 📄 License
This project is licensed under the MIT License.
