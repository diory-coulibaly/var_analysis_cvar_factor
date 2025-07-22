
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import norm
import math

st.set_page_config(page_title="VaR & CVaR Risk Analyzer", layout="wide")
st.title("📊 VaR & CVaR Risk Simulation from Historical Prices")

# 1️⃣ Upload and Parse CSV
st.sidebar.header("Upload CombinedPrices.csv")
uploaded_file = st.sidebar.file_uploader("CSV file with Date, Stocks, FVX, SP500", type="csv")
confLevel = st.sidebar.slider("Confidence Level", min_value=0.90, max_value=0.99, value=0.99, step=0.01)
simulate_mc = st.sidebar.checkbox("Include Monte Carlo Simulation", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    # Standardize all float values
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "").str.replace(u'\xa0', '').astype(float)

    # 2️⃣ User selection of factorNames from sidebar
    available_columns = [col for col in df.columns if col != "Date"]
    st.sidebar.header("Select Factor Columns")
    selected_factors = st.sidebar.multiselect("Choose factor columns (others are treated as stocks):",
                                              options=available_columns,
                                              default=["FVX", "SP500"])
    factorNames = selected_factors + ["Intercept"]
    stockNames = [col for col in available_columns if col not in selected_factors]

    # 3️⃣ Calculate Returns
    returns = df.drop(columns=["Date"]).pct_change().dropna()
    returns["Intercept"] = 1
    stockReturns = returns[stockNames]
    factorReturns = returns[factorNames]

    weights = np.array([1.0 / len(stockNames)] * len(stockNames))

    def calculateVaR(risk, confLevel, principal=1, numMonths=1):
        vol = math.sqrt(risk)
        return abs(principal * norm.ppf(1 - confLevel) * vol * math.sqrt(numMonths))

    def calculateCVaR(port_returns, confLevel):
        var_threshold = np.percentile(port_returns, (1 - confLevel) * 100)
        return abs(port_returns[port_returns <= var_threshold].mean())

    # 4️⃣ Historical VaR
    port_returns = stockReturns.dot(weights)
    hist_var = abs(np.percentile(port_returns, (1 - confLevel) * 100))
    hist_cvar = calculateCVaR(port_returns, confLevel)

    # 5️⃣ Parametric (Variance-Covariance) VaR
    historicalTotalRisk = np.dot(np.dot(weights, stockReturns.cov()), weights.T)
    vc_var = calculateVaR(historicalTotalRisk, confLevel)

    # 6️⃣ Monte Carlo Simulation
    if simulate_mc:
        mean_returns = stockReturns.mean()
        cov_matrix = stockReturns.cov()
        sim_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=10000)
        sim_port_returns = np.dot(sim_returns, weights)
        mc_var = abs(np.percentile(sim_port_returns, (1 - confLevel) * 100))
        mc_cvar = calculateCVaR(sim_port_returns, confLevel)
    else:
        mc_var = "-"
        mc_cvar = "-"

    # 7️⃣ Factor Model Risk
    xData = factorReturns
    modelCoeffs = []
    for oneStock in stockNames:
        yData = stockReturns[oneStock]
        model = sm.OLS(yData, xData).fit()
        coeffs = list(model.params)
        coeffs.append(np.std(model.resid, ddof=1))
        modelCoeffs.append(coeffs)

    modelCoeffs = pd.DataFrame(modelCoeffs, columns=factorNames + ["ResidVol"])
    modelCoeffs["Names"] = stockNames

    factorCov = factorReturns[selected_factors].cov()
    B_factors = modelCoeffs[selected_factors]
    reconstructedCov = np.dot(np.dot(B_factors, factorCov), B_factors.T)
    systemicRisk = np.dot(np.dot(weights, reconstructedCov), weights.T)
    idiosyncraticRisk = sum(modelCoeffs["ResidVol"] ** 2 * weights ** 2)
    factor_risk = systemicRisk + idiosyncraticRisk
    factor_var = calculateVaR(factor_risk, confLevel)

    # 8️⃣ Display Results
    results_df = pd.DataFrame([
        ["Historical", hist_var, hist_cvar],
        ["Variance-Covariance", vc_var, "-"],
        ["Monte Carlo", mc_var, mc_cvar],
        ["Factor Model", factor_var, "-"]
    ], columns=["Method", f"VaR ({int(confLevel*100)}%)", f"CVaR ({int(confLevel*100)}%)"])

    st.subheader("📊 Risk Summary Table")
    st.dataframe(results_df)

    st.download_button("⬇️ Download Results", data=results_df.to_csv(index=False).encode("utf-8"),
                       file_name="var_cvar_summary.csv")

    st.subheader("📉 Comparison Chart")
    chart_df = results_df.copy()
    chart_df = chart_df.replace("-", np.nan)
    chart_df.set_index("Method")[chart_df.columns[1:]].plot(kind="bar", figsize=(8, 4))
    st.pyplot(plt.gcf())
else:
    st.info("⬅️ Upload your CombinedPrices.csv file to begin analysis.")
