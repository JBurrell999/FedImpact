# Counterfactual Policy Simulator: Causal Inference in Macroeconomic Forecasting

This project is a deployable Streamlit-based application that models the **causal effect of macroeconomic policy interventions**—like changes in the federal funds rate—on economic output using machine learning and causal inference. The app enables **interactive simulation of counterfactual economic scenarios** with multiple predictive models and interpretable visualizations.

## 🔍 Motivation
Understanding how macroeconomic variables influence GDP is essential for data-driven economic policy. This tool allows users to **simulate the impact of monetary levers**, estimate causal treatment effects, and evaluate how different models behave in response to changes in inputs such as interest rates, inflation, and unemployment.

## ⚙️ How It Works
### 1. **Data Collection**
We use the FRED API to fetch:
- Gross Domestic Product (GDP)
- Consumer Price Index (CPI)
- Unemployment rate
- Federal Funds Rate (Fed Rate)
- Money supply (M2)
- Employment

These are merged into a time-indexed macroeconomic dataset.

### 2. **Forecasting Models**
We train multiple forecasting models to predict `log(GDP)` using the full set of economic variables:
- 📈 **Ridge Regression** (baseline linear)
- 🌲 **XGBoost**, **LightGBM**, **CatBoost** (tree ensembles)

> 🔹 These 4 models are available for use in the deployed **Streamlit UI**. Prophet and PyTorch models are trained offline but not exposed in the app due to performance or interface constraints.

### 3. **Causal Estimation (T-Learner)**
We simulate the treatment effect of adjusting `fed_rate` using a simple T-learner-style logic:
- Predict `log(GDP)` for current data
- Predict again after perturbing `fed_rate`
- Compute the **difference as treatment effect**

This gives an interpretable time series of **estimated marginal effects** from changing Fed policy.

### 4. **Streamlit Interface**
The user can:
- Choose from 4 forecast models
- Adjust the **Fed rate** using a slider
- View **counterfactual forecasts** of GDP
- View **treatment effect curves**

> ⚠️ Currently only `fed_rate` is user-adjustable. All other features remain fixed, though they are used by the models.

---

## 🧪 Why Not Just Linear Regression?
While Ridge regression offers interpretability, many real-world relationships between macroeconomic factors and GDP are nonlinear or interaction-heavy. That’s why we include multiple model families and causal estimators:

| Model        | Type            | Purpose                               |
|--------------|------------------|---------------------------------------|
| Ridge        | Linear           | Transparent baseline                   |
| XGBoost      | Tree ensemble    | Nonlinear + interaction effects        |
| CatBoost     | Tree ensemble    | Handles heterogeneity robustly        |
| LightGBM     | Tree ensemble    | Fast + high accuracy                   |
| Prophet (offline) | Time series | Captures seasonal trends              |
| PyTorch NN (offline) | Neural Net | General function approximation       |
| T-Learner    | Causal Inference | Measures isolated impact of Fed rate  |

---

## 📊 Project Structure
```text
counterfactual-policy-simulator/
├── data/
│   ├── raw_macro.csv              # Raw data from FRED
│   ├── clean_macro.csv            # Cleaned time series data
│   └── final_macro.pkl            # Preprocessed DataFrame
│
├── models/
│   ├── *.pkl, .pt, .npy           # Trained ML + causal models
│
├── streamlit_app/
│   └── app.py                     # Dashboard entry point
│
├── notebooks/
│   └── model_eval.ipynb          # Forecast comparison + metrics
│
├── scripts/
│   ├── fetch_fred_data.py        # Automated FRED pipeline
│   └── model_training.py         # Full training pipeline
│
├── requirements.txt
└── .streamlit/
    └── config.toml                # Deployment config
```

---

## 🚀 Usage
### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Set FRED API key
```bash
echo "YOUR_KEY" > ~/.fred_api_key
```

### 3. Fetch & preprocess data
```bash
python scripts/fetch_fred_data.py
jupyter notebook models/models.ipynb
```

### 4. Launch dashboard
```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 Deploy It
### Option A: Streamlit Cloud
- Push this repo to GitHub
- Go to https://streamlit.io/cloud
- Set `streamlit_app/app.py` as the entry point

### Option B: Hugging Face Spaces (coming soon)
- Add `app.py`, `requirements.txt`
- Configure as a Gradio or Streamlit app

---

## 🙋 FAQ
**Q: Are features like CPI and unemployment used?**  
✅ Yes. All models are trained on them. The **only variable adjusted in simulation** is the Fed rate.

**Q: Can I simulate inflation shocks or recessions?**  
🛠 You can extend the UI to allow sliders for all variables. We’ll add this soon.

**Q: Where is the causal forest model?**  
❌ Removed due to package issues. A simplified T-learner approach is used instead.

---

## 👤 Author
**James Burrell**  
AI Policy Fellow @ TPI | ML Researcher @ USF  
[GitHub](https://github.com/JBurrell999)

---

## 📜 License
MIT