import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Challenge 02 - DSS", layout="wide")
st.title("Challenge 02 — DSS Auditable (RAW / CLEAN / ANOMALÍAS)")
st.caption("Inicio: Inventario Central. Luego replicamos patrón para Transacciones y Feedback.")

# ==========
# Helpers
# ==========
def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def to_datetime(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return low, high

def percentile_bounds(series, p_low=0.01, p_high=0.99):
    low = series.quantile(p_low)
    high = series.quantile(p_high)
    return low, high

def audit_summary(df_raw, df_clean, df_anom, damage_cols, flag_cols):
    return pd.DataFrame([{
        "Filas RAW": len(df_raw),
        "Filas CLEAN": len(df_clean),
        "Filas ANOMALÍAS": len(df_anom),
        "% ANOMALÍAS": (len(df_anom) / len(df_raw) * 100) if len(df_raw) else 0,
        "Promedio % nulos (CLEAN)": float(df_clean.isna().mean().mean() * 100) if len(df_clean.columns) else 0,
        "Columnas daño": len(damage_cols),
        "Columnas flags": len(flag_cols),
    }])


# ==========
# Sidebar: Selección de archivo (por ahora inventario)
# ==========
st.sidebar.header("1) Archivo")
inventory_path = st.sidebar.text_input(
    "Ruta inventario",
    value="/mnt/data/inventario_central_v2.csv"
)

# ==========
# Sidebar: Checklists (Decisiones)
# ==========
st.sidebar.header("2) Decisiones del cliente (Inventario)")

damage_threshold = st.sidebar.number_input(
    "Enviar fila a ANOMALÍAS si columnas dañadas ≥",
    min_value=1, max_value=10, value=2
)

st.sidebar.subheader("Outliers")
outlier_method = st.sidebar.selectbox("Método", ["IQR", "Percentiles"], index=0)
if outlier_method == "IQR":
    iqr_k = st.sidebar.slider("IQR k", 1.0, 3.0, 1.5, 0.1)
    p_low, p_high = None, None
else:
    p_low = st.sidebar.slider("Percentil bajo", 0.0, 0.1, 0.01, 0.005)
    p_high = st.sidebar.slider("Percentil alto", 0.9, 1.0, 0.99, 0.005)
    iqr_k = None

st.sidebar.subheader("Imputación (solo CLEAN)")
impute_lead = st.sidebar.selectbox(
    "Lead_Time_Dias nulo →",
    ["No imputar", "Mediana global", "Mediana por categoría"],
    index=2
)
impute_reorder = st.sidebar.selectbox(
    "Punto_Reorden nulo →",
    ["No imputar", "Mediana global", "Mediana por categoría"],
    index=2
)

st.sidebar.subheader("¿Qué manda a ANOMALÍAS?")
send_flags_to_anom = st.sidebar.checkbox(
    "Enviar a ANOMALÍAS si tiene cualquier flag de riesgo (recomendado)",
    value=True
)

# ==========
# Load data
# ==========
@st.cache_data(show_spinner=False)
def load_inventory(path):
    return pd.read_csv(path)

try:
    inv_raw = load_inventory(inventory_path)
except Exception as e:
    st.error(f"No pude cargar el archivo. Revisa la ruta y formato. Error: {e}")
    st.stop()

# ==========
# Step A: Tipificación mínima (sin borrar filas)
# ==========
inv = inv_raw.copy()

# Intenta tipificar si existen las columnas esperadas
if "Stock_Actual" in inv.columns:
    inv["Stock_Actual"] = to_numeric(inv["Stock_Actual"])
if "Costo_Unitario_USD" in inv.columns:
    inv["Costo_Unitario_USD"] = to_numeric(inv["Costo_Unitario_USD"])
if "Lead_Time_Dias" in inv.columns:
    inv["Lead_Time_Dias"] = to_numeric(inv["Lead_Time_Dias"])
if "Punto_Reorden" in inv.columns:
    inv["Punto_Reorden"] = to_numeric(inv["Punto_Reorden"])
if "Ultima_Revision" in inv.columns:
    inv["Ultima_Revision"] = to_datetime(inv["Ultima_Revision"])

# ==========
# Step B: Generación de "damage__" (daños por columna)
# ==========
# Damage = cosas que impiden usar el dato (nulo/ilegible)
# Flag = cosas "raras" o de riesgo (negativo, outlier, futuro)
damage_cols = []
flag_cols = []

def add_damage(colname, mask):
    cname = f"damage__{colname}"
    inv[cname] = mask.astype(int)
    damage_cols.append(cname)

def add_flag(flagname, mask):
    cname = f"flag__{flagname}"
    inv[cname] = mask.astype(bool)
    flag_cols.append(cname)

# SKU_ID
if "SKU_ID" in inv.columns:
    add_damage("SKU_ID", inv["SKU_ID"].isna())

# Stock_Actual
if "Stock_Actual" in inv.columns:
    add_damage("Stock_Actual", inv["Stock_Actual"].isna())
    add_flag("stock_negativo", inv["Stock_Actual"] < 0)

# Costo_Unitario_USD
if "Costo_Unitario_USD" in inv.columns:
    add_damage("Costo_Unitario_USD", inv["Costo_Unitario_USD"].isna())
    add_flag("costo_no_positivo", inv["Costo_Unitario_USD"] <= 0)

# Lead_Time_Dias
if "Lead_Time_Dias" in inv.columns:
    add_damage("Lead_Time_Dias", inv["Lead_Time_Dias"].isna())
    add_flag("leadtime_negativo", inv["Lead_Time_Dias"] < 0)

# Punto_Reorden
if "Punto_Reorden" in inv.columns:
    add_damage("Punto_Reorden", inv["Punto_Reorden"].isna())
    add_flag("punto_reorden_negativo", inv["Punto_Reorden"] < 0)

# Ultima_Revision
if "Ultima_Revision" in inv.columns:
    add_damage("Ultima_Revision", inv["Ultima_Revision"].isna())
    today = pd.Timestamp.today().normalize()
    add_flag("fecha_revision_futura", inv["Ultima_Revision"] > today)

# ==========
# Step C: Outliers (Costo y LeadTime) como flags
# ==========
def compute_outlier_flag(col, flagname):
    if col not in inv.columns:
        return
    s = inv[col].dropna()
    if len(s) < 20:
        add_flag(flagname, pd.Series(False, index=inv.index))
        return

    if outlier_method == "IQR":
        low, high = iqr_bounds(s, k=iqr_k)
    else:
        low, high = percentile_bounds(s, p_low=p_low, p_high=p_high)

    add_flag(flagname, (inv[col] < low) | (inv[col] > high))

compute_outlier_flag("Costo_Unitario_USD", "costo_outlier")
compute_outlier_flag("Lead_Time_Dias", "leadtime_outlier")

# ==========
# Step D: Score de daño por fila
# ==========
if damage_cols:
    inv["damaged_cols_count"] = inv[damage_cols].sum(axis=1)
else:
    inv["damaged_cols_count"] = 0

if flag_cols:
    inv["any_flag"] = inv[flag_cols].any(axis=1)
else:
    inv["any_flag"] = False

# ==========
# Step E: Split RAW / ANOMALIES / CLEAN
# ==========
base_anom_mask = inv["damaged_cols_count"] >= int(damage_threshold)
if send_flags_to_anom:
    anom_mask = base_anom_mask | inv["any_flag"]
else:
    anom_mask = base_anom_mask

inv_anom = inv[anom_mask].copy()
inv_clean = inv[~anom_mask].copy()

# ==========
# Step F: Imputación SOLO en CLEAN (opcional)
# ==========
def group_median_impute(df, target_col, group_col):
    if target_col not in df.columns or group_col not in df.columns:
        return df
    med = df.groupby(group_col)[target_col].transform("median")
    was_null = df[target_col].isna()
    df.loc[was_null, target_col] = med[was_null]
    df[f"imputed__{target_col}"] = was_null.astype(bool)
    return df

def global_median_impute(df, target_col):
    if target_col not in df.columns:
        return df
    m = df[target_col].median()
    was_null = df[target_col].isna()
    df.loc[was_null, target_col] = m
    df[f"imputed__{target_col}"] = was_null.astype(bool)
    return df

# Lead_Time_Dias
if "Lead_Time_Dias" in inv_clean.columns:
    if impute_lead == "Mediana global":
        inv_clean = global_median_impute(inv_clean, "Lead_Time_Dias")
    elif impute_lead == "Mediana por categoría" and "Categoria" in inv_clean.columns:
        inv_clean = group_median_impute(inv_clean, "Lead_Time_Dias", "Categoria")

# Punto_Reorden
if "Punto_Reorden" in inv_clean.columns:
    if impute_reorder == "Mediana global":
        inv_clean = global_median_impute(inv_clean, "Punto_Reorden")
    elif impute_reorder == "Mediana por categoría" and "Categoria" in inv_clean.columns:
        inv_clean = group_median_impute(inv_clean, "Punto_Reorden", "Categoria")

# ==========
# UI: Tabs (Inventario)
# ==========
tab1, tab2, tab3 = st.tabs(["📋 Auditoría", "✅ CLEAN", "⚠️ ANOMALÍAS"])

with tab1:
    st.subheader("Auditoría de Inventario")
    st.dataframe(audit_summary(inv, inv_clean, inv_anom, damage_cols, flag_cols), use_container_width=True)

    st.markdown("### Top columnas con más nulos (RAW tipificado)")
    st.dataframe(inv.isna().mean().sort_values(ascending=False).head(15).to_frame("% nulos").mul(100), use_container_width=True)

    st.markdown("### Conteo de flags (riesgos)")
    if flag_cols:
        flag_counts = inv[flag_cols].sum().sort_values(ascending=False).to_frame("conteo")
        st.dataframe(flag_counts, use_container_width=True)
    else:
        st.info("No se detectaron flags (o no existen columnas esperadas).")

    st.markdown("### Ejemplos de registros marcados (primeros 50)")
    cols_show = [c for c in inv.columns if not c.startswith("damage__")]
    st.dataframe(inv.loc[inv["any_flag"], cols_show].head(50), use_container_width=True)

with tab2:
    st.subheader("Inventario CLEAN (para KPIs)")
    st.caption("Aquí viven los datos “seguros” para análisis. Si imputaste algo, queda marcado.")
    st.dataframe(inv_clean.head(200), use_container_width=True)

with tab3:
    st.subheader("Inventario ANOMALÍAS (riesgos / outliers / daños)")
    st.caption("Aquí viven los registros excluidos de KPIs, pero útiles para diagnóstico y storytelling.")
    st.dataframe(inv_anom.head(200), use_container_width=True)

