

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
macro_df = pd.read_pickle("data/final_macro.pkl")
treatment_effect = np.load("models/treatment_effect.npy")


model_names = [
    "ridge", "xgb", "lgbm", "catboost"
]
models = {}
for name in model_names:
    models[name] = joblib.load(f"models/{name}_forecast.pkl")


st.set_page_config(page_title="Causal Inference Simulator", layout="wide")
st.title("📈 Counterfactual Policy Simulator for GDP Forecasting")

st.sidebar.header("Policy Inputs")
fed_rate_adj = st.sidebar.slider("Adjust Fed Rate (percentage points)", -2.0, 2.0, 0.0, step=0.1)
selected_model = st.sidebar.selectbox("Select Forecast Model", model_names)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with 💡 by James Burrell")


x_cols = macro_df.drop(columns=["log_gdp"]).copy()
x_cols["fed_rate"] += fed_rate_adj

model = models[selected_model]
pred = model.predict(x_cols)


fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(macro_df.index, macro_df["log_gdp"], label="Actual log(GDP)", linewidth=2)
ax.plot(macro_df.index, pred, label=f"Predicted ({selected_model})", linestyle="--")
ax.set_title("GDP Forecast Under Adjusted Fed Rate")
ax.set_ylabel("log(GDP)")
ax.legend()

st.subheader("Forecast Results")
st.pyplot(fig)


st.subheader("Estimated Treatment Effect (Fed Rate → GDP)")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(macro_df.index, treatment_effect, label="Treatment Effect", color="orange")
ax2.axhline(0, linestyle="--", color="black")
ax2.set_title("Estimated Impact of Fed Rate Changes on log(GDP)")
ax2.set_ylabel("Estimated Effect")
ax2.legend()

st.pyplot(fig2)