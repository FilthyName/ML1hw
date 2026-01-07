import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Predictor", layout="wide")
@st.cache_resource

def load_artifact(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)
    
artifact = load_artifact("model.pkl")
pipe = artifact["pipeline"]
cat_cols = artifact["cat_cols"]
num_cols = artifact["num_cols"]
median_num = artifact["median_num"]
st.title("Car Price Prediction (Ridge + OneHot)")
st.write("Загрузите данные для EDA и/или CSV для предсказаний, либо введите признаки вручную.")
st.header("EDA")
eda_file = st.file_uploader("Загрузите CSV для EDA (например, train/test)", type=["csv"], key="eda")

if eda_file is not None:
    df_eda = pd.read_csv(eda_file)
    st.subheader("Превью данных")
    st.dataframe(df_eda.head(20), use_container_width=True)
    st.subheader("Гистограммы числовых признаков")
    num_in_eda = df_eda.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols = st.columns(2)
    for i, col in enumerate(num_in_eda[:10]):
        fig, ax = plt.subplots()
        ax.hist(df_eda[col].dropna().values, bins=30)
        ax.set_title(col)
        cols[i % 2].pyplot(fig)
    if "selling_price" in df_eda.columns and "year" in df_eda.columns:
        st.subheader("Зависимость цены от года")
        fig, ax = plt.subplots()
        ax.scatter(df_eda["year"], df_eda["selling_price"], alpha=0.3)
        ax.set_xlabel("year")
        ax.set_ylabel("selling_price")
        cols = st.columns(1)
        cols[0].pyplot(fig)

st.header("Предсказание цены")
mode = st.radio("Режим ввода", ["CSV", "Ручной ввод"], horizontal=True)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(median_num[c])

    if "seats" in df.columns:
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce")
        df["seats"] = df["seats"].fillna(df["seats"].median())

    needed = set(cat_cols + num_cols)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"В входном файле не хватает колонок: {missing}")

    return df[cat_cols + num_cols]

if mode == "CSV":
    pred_file = st.file_uploader("Загрузите CSV с признаками авто (без selling_price)", type=["csv"], key="pred")
    if pred_file is not None:
        df_in = pd.read_csv(pred_file)
        st.write("Входные данные (превью):")
        st.dataframe(df_in.head(20), use_container_width=True)

        try:
            X_in = prepare_features(df_in)
            preds = pipe.predict(X_in)
            out = df_in.copy()
            out["predicted_price"] = preds
            st.subheader("Результаты")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button(
                "Скачать результаты CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(str(e))

else:
    st.subheader("Ручной ввод признаков (1 авто)")
    cols2 = st.columns(2)
    manual = {}

    for i, c in enumerate(num_cols):
        if c == "seats":
            manual[c] = cols2[i % 2].number_input(c, min_value=2, max_value=10, value=5)
        else:
            manual[c] = cols2[i % 2].number_input(c, value=float(median_num.get(c, 0.0)))

    st.markdown("**Категориальные признаки**")
    cols3 = st.columns(3)
    cat_only = [c for c in cat_cols if c != "seats"]
    for i, c in enumerate(cat_only):
        manual[c] = cols3[i % 3].text_input(c, value="")

    if st.button("Предсказать"):
        df_one = pd.DataFrame([manual])
        try:
            X_one = prepare_features(df_one)
            pred = float(pipe.predict(X_one)[0])
            st.success(f"Предсказанная цена: {pred:,.0f}")
        except Exception as e:
            st.error(str(e))

st.header("Веса (коэффициенты) модели")

model = pipe.named_steps["model"]
pre = pipe.named_steps["preprocess"]

try:
    feat_names = pre.get_feature_names_out()
    coefs = model.coef_
    coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    top_n = st.slider("Сколько коэффициентов показать", min_value=10, max_value=100, value=30, step=5)
    st.dataframe(coef_df.head(top_n), use_container_width=True)
    fig, ax = plt.subplots()
    show = coef_df.head(top_n).iloc[::-1]
    ax.barh(show["feature"], show["coef"])
    ax.set_title(f"Top-{top_n} коэффициентов Ridge")
    st.pyplot(fig)

except Exception as e:
    st.warning("Не удалось получить имена фичей/коэффициенты. Проверьте, что сохранён именно Pipeline.")
    st.text(str(e))