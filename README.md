# 🏠 House Price Predictor — Linear Regression

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

🔗 **Live Demo:** [house-price-predictor-rose.streamlit.app](https://house-price-predictor-rose.streamlit.app)

## 📌 Overview

End-to-end ML pipeline for house price prediction using Linear Regression on the California Housing dataset. Covers the complete ML workflow — data loading, EDA, preprocessing, training, evaluation, and live deployment via Streamlit.

## 📊 Model Results

| Metric | Train | Test |
|---|---|---|
| MAE | ~0.53 | ~0.53 |
| RMSE | ~0.73 | ~0.74 |
| R² Score | ~0.61 | **0.60** |

Model explains ~60% of variance in California house prices with an average prediction error of ~$53,000.

## ✨ Features

- 🔍 **Complete EDA** — distributions, correlations, geographic visualization
- ⚙️ **Full ML Pipeline** — preprocessing, scaling, training, evaluation
- 📈 **10+ Visualizations** — residuals, coefficients, actual vs predicted
- 🌐 **Live Streamlit App** — interactive prediction UI deployed online
- 📄 **PDF Report** — summarized findings and analysis

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **scikit-learn** | Linear Regression, StandardScaler, metrics |
| **Pandas + NumPy** | Data manipulation and processing |
| **Matplotlib + Seaborn** | Data visualization |
| **Streamlit** | Interactive web UI and deployment |

## 🗂️ Dataset

- **California Housing Dataset** — built into scikit-learn
- 20,640 samples · 8 features
- Features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
- Target: Median House Value (in $100,000s)

## 📁 Project Structure

```
house-price-prediction-ml/
├── task1_ml_linear_regression.ipynb   # Main notebook
├── predict_ui.py                       # Streamlit prediction app
├── requirements.txt                    # Dependencies
├── image/                              # Visualization outputs
├── report/
│   └── House_Price_Report.pdf         # Project report
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 How to Run

### Option A — Streamlit App (Live Demo)
Visit directly: **[house-price-predictor-rose.streamlit.app](https://house-price-predictor-rose.streamlit.app)**

### Option B — Run Locally

```bash
# Clone the repo
git clone https://github.com/Rosesharma13/house-price-prediction-ml.git
cd house-price-prediction-ml

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run predict_ui.py
```

### Option C — Jupyter Notebook
```bash
jupyter notebook task1_ml_linear_regression.ipynb
```

## 📊 ML Workflow

1. Load California Housing dataset
2. Exploratory Data Analysis (EDA)
3. Feature selection + Train/Test split (80/20)
4. Feature scaling — StandardScaler
5. Train LinearRegression model
6. Evaluate — MAE, RMSE, R²
7. Visualize — Actual vs Predicted, Residuals
8. Deploy via Streamlit

## 💡 Key Findings

- **MedInc** (Median Income) is the strongest predictor of house prices
- Coastal California commands premium prices — location matters significantly
- AveBedrms has a negative coefficient — more bedrooms relative to rooms lowers value
- Linear model is interpretable but non-linear models would improve R² further

## 🔮 Improvement Ideas

- Polynomial feature engineering for non-linear relationships
- Ridge / Lasso regularization
- Random Forest or XGBoost for better accuracy
- Log transformation of skewed features
- Remove price-capped outliers (value = 5.0)

## 👩‍💻 Author

**Rose Sharma**
- 🌐 Portfolio: [rosesharma13.github.io](https://rosesharma13.github.io)
- 💼 LinkedIn: [linkedin.com/in/rose-sharma13](https://www.linkedin.com/in/rose-sharma13)
- 📧 Email: rosesharmaa132003@gmail.com
- 💻 GitHub: [github.com/Rosesharma13](https://github.com/Rosesharma13)

## 📄 License

This project is licensed under the MIT License.
