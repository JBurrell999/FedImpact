import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="CausalSim: Counterfactual Simulator", layout="centered")
st.title("📈 CausalSim: Counterfactual Policy Simulator")

macro_df = pd.read_pickle("../data/final_macro.pkl")
causal_model = joblib.load("../models/causal_forest.pkl")

user_input = st.slider("Simulate FED Rate Policy (bps)", min_value=0.0, max_value=10.0,
                       value=float(macro_df["fed_rate"].mean()), step=0.25)

X_causal = macro_df[['unemployment', 'cpi', 'money_supply', 'employment']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_causal)

avg_effect = causal_model.const_marginal_effect(X_scaled).mean()
user_diff = user_input - macro_df["fed_rate"].mean()
impact = avg_effect * user_diff

st.metric(label="Estimated GDP Change (log units)", value=f"{impact:.4f}")

# Forecast comparison
forecast_model = joblib.load("../models/xgb_forecast.pkl")
x_latest = macro_df.drop(columns=["log_gdp"]).iloc[-1:].values
forecasted_gdp = forecast_model.predict(x_latest)[0]

st.subheader("📊 Forecasted vs. Counterfactual GDP")
st.write(f"**Forecasted log GDP (XGBoost)**: {forecasted_gdp:.4f}")
st.write(f"**Counterfactual log GDP**: {(forecasted_gdp + impact):.4f}")
