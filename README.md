# 📈 Counterfactual Policy Simulator

Simulate the impact of macroeconomic policy decisions (e.g., interest rate changes) on future economic outcomes using causal inference and ML-based time series forecasting. This project uses real data from the FRED API and includes deployment via Streamlit Cloud and Hugging Face Spaces.

---

## 🔍 Key Features
- Pulls live macroeconomic data (GDP, CPI, unemployment, etc.)
- Forecasts future GDP using 5 ML models
- Simulates counterfactual policy effects using 2 causal models
- Deployed Streamlit app with real-time interaction
- Logs results for experiment tracking and interpretability

---

## 🧱 Project Directory Structure
```
counterfactual-policy-simulator/
├── data/
│   ├── raw_macro.csv              # Raw data from FRED
│   ├── clean_macro.csv            # Cleaned time series data
│   └── final_macro.pkl            # Preprocessed DataFrame for modeling
│
├── models/
│   ├── xgb_forecast.pkl           # XGBoost GDP forecast
│   ├── ridge_forecast.pkl         # Ridge forecast
│   ├── lgbm_forecast.pkl          # LightGBM forecast
│   ├── catboost_forecast.pkl      # CatBoost forecast
│   ├── prophet_forecast.pkl       # Prophet model (not pickled)
│   ├── linear_dml.pkl             # LinearDML causal model
│   └── causal_forest.pkl          # CausalForestDML causal model
│
├── notebooks/
│   └── model_evaluation.ipynb     # Evaluation plots and metric comparison
│
├── streamlit_app/
│   └── app.py                     # Main UI entry point
│
├── scripts/
│   └── model_training.py          # Model training pipeline
│
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml                # Streamlit Cloud configuration
```

---

## 🤖 Models and Justification

### Forecasting Models
- **XGBoost**: Strong performance with tabular data and non-linear dynamics. Easy to tune and interpret.
- **Ridge Regression**: Baseline linear model to understand overfitting and provide benchmark.
- **LightGBM**: Efficient, scalable gradient boosting for time series. Often faster and more accurate than XGBoost.
- **CatBoost**: Handles categorical variables and outperforms other gradient boosters in some macroeconomic setups.
- **Prophet**: Time-aware forecasting model developed by Meta. Useful for seasonality-aware GDP projection.

### Causal Inference Models
- **LinearDML**: Doubly robust method for estimating average treatment effects. Good baseline.
- **CausalForestDML**: Non-parametric model capturing heterogeneous treatment effects — great for policy heterogeneity.

---

## 📊 Model Evaluation
Each model is evaluated on the test split using:
- **MSE (Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**

You can run the notebook in `notebooks/model_evaluation.ipynb` to see:
- Error distribution
- Prediction vs. true GDP
- Residual plots

---

## 🚀 Deployment
### Streamlit Cloud
1. Push this repo to GitHub
2. Go to https://streamlit.io/cloud and deploy the repo
3. Make sure to set the API key as a secret:
   ```bash
   FRED_API_KEY = "your_api_key"
   ```

### Hugging Face Spaces
1. Create a new Space: **Streamlit + Python**
2. Upload this entire directory
3. Add a `README.md` and `requirements.txt`
4. Set up `app.py` as entrypoint

---

## ✅ TODO
- [ ] Add Bayesian modeling option (PyMC or Bambi)
- [ ] Enable user-uploaded policy timelines
- [ ] Add multi-country panel simulation

---

## 📬 Contact
Made by [James Burrell](mailto:jamesburrell999@gmail.com).
