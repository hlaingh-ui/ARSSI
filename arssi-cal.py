import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
from sklearn.linear_model import LinearRegression

st.title("ARSSI Calculator")

# 1️⃣ Upload Excel File
uploaded_file = st.file_uploader("Upload the Excel data file", type=["xls", "xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("✅ File uploaded successfully.")
    st.dataframe(df.head())

    # 2️⃣ Identify numeric columns between q2 and q39 only
    numeric_cols = [col for col in df.columns if col.startswith("q") and col[1:].isdigit()]
    numeric_cols = [col for col in numeric_cols if 2 <= int(col[1:]) <= 39]

    # Keep only numeric values for SEI calculation
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # 3️⃣ Compute ATR (Ability to Recover)
    df["ATR"] = df["q40"] + df["q41"]

    # 4️⃣ Compute SEI (Shock Exposure Index)
    df["SEI"] = df[numeric_cols].sum(axis=1)

    # 5️⃣ Compute regression coefficient b (ATR ~ SEI)
    model = LinearRegression().fit(df[["SEI"]], df["ATR"])
    b = model.coef_[0]

    # 6️⃣ Compute mean SEI (Y)
    Y = df["SEI"].mean()

    # 7️⃣ Compute ARSSI
    df["ARSSI"] = df["ATR"] + b * (Y - df["SEI"])

    # 8️⃣ Display results
    st.write(f"**Regression Coefficient (b):** {b:.4f}")
    st.write(f"**Mean SEI (Y):** {Y:.4f}")
    st.dataframe(df[["ATR", "SEI", "ARSSI"]].head())

    # 9️⃣ Export to CSV
    output = BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)

    st.download_button(
        label="⬇️ Download ARSSI Results as CSV",
        data=output,
        file_name="ARSSI_results.csv",
        mime="text/csv"
    )
