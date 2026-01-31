import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# =========================
# Fuzzy matching (automático si está disponible)
# =========================
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


st.set_page_config(page_title="Challenge 02 - DSS (Inventario)", layout="wide")
st.title("Challenge 02 — DSS Auditable (Inventario)")
st.caption("Inventario: normalización automática + RAW/CLEAN/ANOMALÍAS. Outliers automáticos (IQR).")


# =========================
# Helpers: texto
# =========================
UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null",
    "sin categoria", "sincategoria", "unknown", "sin categoría"
}

def normalize_text_keep_unknown(x: str) -> str:
    """
    Normalización básica automática:
    - trim
    - lower
    - detectar ??? / NA / etc. antes de limpiar símbolos
    - quitar tildes
    - -/_ -> espacio
    - quitar símbolos raros
    - colapsar espacios
    """
    if pd.isna(x):
        return np.nan
    raw = str(x).strip()
    if raw == "":
        return np.nan

    raw_lower = raw.lower().strip()
    if raw_lower in UNKNOWN_TOKENS or (len(raw_lower) > 0 and set(raw_lower) <= {"?"}):
        return "unknown"

    x = raw_lower
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8")
    x = re.sub(r"[-_/]+", " ", x)
    x = re.sub(r"[^a-z0-9\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()

    if x in UNKNOWN_TOKENS:
        return "unknown"
    if x == "":
        return np.nan
    return x


def apply_manual_map(series_norm: pd.Series, manual_map: dict) -> pd.Series:
    return series_norm.map(lambda v: manual_map.get(v, v))


def build_canonical_values(series_after_manual: pd.Series) -> list:
    vals = series_after_manual.dropna().astype(str)
    vals = vals[vals != "unknown"]
    return sorted(set(vals.tolist()))


def fuzzy_map_unique(series_vals: pd.Series, canonical: list, threshold: float = 0.92, delta: float = 0.03):
    """
    Fuzzy matching automático (si rapidfuzz disponible), solo si match es único.
    Devuelve serie mapeada + tabla de sugerencias.
    """
    cols = ["from", "to", "score", "applied"]

    if (not RAPIDFUZZ_AVAILABLE) or (len(canonical) == 0):
        return series_vals, pd.DataFrame(columns=cols)

    thr = threshold * 100
    dlt = delta * 100

    mapped = series_vals.copy()
    changes = []

    unique_vals = sorted(set(series_vals.dropna().astype(str).tolist()))
    unique_vals = [v for v in unique_vals if v != "unknown"]

    for v in unique_vals:
        if v in canonical:
            continue

        matches = process.extract(v, canonical, scorer=fuzz.WRatio, limit=2)
        if not matches:
            continue

        best_match, best_score, _ = matches[0]
        second_score = matches[1][1] if len(matches) > 1 else 0

        is_unique = (best_score >= thr) and (((best_score - second_score) >= dlt) or (second_score < thr))

        if is_unique:
            mapped = mapped.replace(v, best_match)
            changes.append({"from": v, "to": best_match, "score": best_score, "applied": True})
        else:
            changes.append({"from": v, "to": best_match, "score": best_score, "applied": False})

    if not changes:
        return mapped, pd.DataFrame(columns=cols)

    changes_df = pd.DataFrame(changes).sort_values(["applied", "score"], ascending=[False, False])
    return mapped, changes_df


def changes_report(original: pd.Series, final: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"antes": original.astype("string"), "despues": final.astype("string")})
    df = df.dropna(subset=["antes", "despues"])
    if df.empty:
        return pd.DataFrame(columns=["antes", "despues", "conteo"])
    return df.value_counts().reset_index(name="conteo").sort_values("conteo", ascending=False)


# =========================
# Helpers: numéricos/outliers
# =========================
def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def to_datetime(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

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


# =========================
# Sidebar: Upload obligatorio
# =========================
st.sidebar.header("📁 Cargar archivo")
uploaded_inv = st.sidebar.file_uploader("Sube inventario_central_v2.csv", type=["csv"])
if uploaded_inv is None:
    st.info("👈 Sube el archivo de inventario para iniciar. (No se carga nada por defecto).")
    st.stop()

@st.cache_data(show_spinner=False)
def load_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

inv_raw = load_from_upload(uploaded_inv)
st.success("Inventario cargado ✅")

# =========================
# Sidebar: Limpieza Inventario (solo controles necesarios)
# =========================
st.sidebar.header("🧹 Limpieza Inventario")

with st.sidebar.expander("Reglas de ANOMALÍAS / Daños", expanded=True):
    damage_threshold = st.number_input("Enviar fila a ANOMALÍAS si columnas dañadas ≥", 1, 10, 2)
    send_flags_to_anom = st.checkbox("Enviar a ANOMALÍAS si tiene cualquier flag de riesgo", value=True)

with st.sidebar.expander("Regla opcional: Stock negativo", expanded=True):
    fix_negative_stock = st.checkbox("Convertir stock negativo a positivo en CLEAN (abs)", value=False)
    st.caption("RAW y ANOMALÍAS NO se modifican. Solo afecta el dataset CLEAN.")

with st.sidebar.expander("Imputación (solo CLEAN)", expanded=False):
    impute_lead = st.selectbox("Lead_Time_Dias nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2)
    impute_reorder = st.selectbox("Punto_Reorden nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2)


# =========================
# 1) Normalización automática (antes de flags/outliers/split)
# =========================
inv = inv_raw.copy()

CATEGORY_MAP = {
    # laptops
    "laptop": "laptops",
    "laptops": "laptops",
    "notebook": "laptops",
    "notebooks": "laptops",

    # smartphones
    "smartphone": "smartphones",
    "smartphones": "smartphones",
    "smart phone": "smartphones",
    "smart phones": "smartphones",

    # tablets
    "tablet": "tablets",
    "tablets": "tablets",

    # accesorios/monitores
    "accesorio": "accesorios",
    "accesorios": "accesorios",
    "monitor": "monitores",
    "monitores": "monitores",

    # unknown
    "unknown": "unknown",
}

BODEGA_MAP = {
    "med": "medellin",
    "mde": "medellin",
    "medellin": "medellin",
    "bog": "bogota",
    "bogota": "bogota",
    "unknown": "unknown",
}

cat_fuzzy_suggestions = pd.DataFrame(columns=["from", "to", "score", "applied"])
bod_fuzzy_suggestions = pd.DataFrame(columns=["from", "to", "score", "applied"])

if "Categoria" in inv.columns:
    inv["Categoria_original"] = inv["Categoria"].astype("string")
    inv["Categoria_clean"] = inv["Categoria"].apply(normalize_text_keep_unknown)
    inv["Categoria_clean"] = apply_manual_map(inv["Categoria_clean"], CATEGORY_MAP)

    # Fuzzy automático (si está disponible)
    canonical_cat = build_canonical_values(inv["Categoria_clean"])
    inv["Categoria_clean"], cat_fuzzy_suggestions = fuzzy_map_unique(
        inv["Categoria_clean"], canonical_cat, threshold=0.92, delta=0.03
    )

if "Bodega_Origen" in inv.columns:
    inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")
    inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
    inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)

    canonical_bod = build_canonical_values(inv["Bodega_Origen_clean"])
    inv["Bodega_Origen_clean"], bod_fuzzy_suggestions = fuzzy_map_unique(
        inv["Bodega_Origen_clean"], canonical_bod, threshold=0.92, delta=0.03
    )


# =========================
# 2) Tipificación numérica/fechas (antes de outliers)
# =========================
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


# =========================
# 3) damages + flags (incluye outliers automáticos IQR)
# =========================
damage_cols, flag_cols = [], []

def add_damage(colname, mask):
    cname = f"damage__{colname}"
    inv[cname] = mask.astype(int)
    damage_cols.append(cname)

def add_flag(flagname, mask):
    cname = f"flag__{flagname}"
    inv[cname] = mask.astype(bool)
    flag_cols.append(cname)

# Identidad
if "SKU_ID" in inv.columns:
    add_damage("SKU_ID", inv["SKU_ID"].isna())

# Stock
if "Stock_Actual" in inv.columns:
    add_damage("Stock_Actual", inv["Stock_Actual"].isna())
    add_flag("stock_negativo", inv["Stock_Actual"] < 0)

# Costo
if "Costo_Unitario_USD" in inv.columns:
    add_damage("Costo_Unitario_USD", inv["Costo_Unitario_USD"].isna())
    add_flag("costo_no_positivo", inv["Costo_Unitario_USD"] <= 0)

# Lead time
if "Lead_Time_Dias" in inv.columns:
    add_damage("Lead_Time_Dias", inv["Lead_Time_Dias"].isna())
    add_flag("leadtime_negativo", inv["Lead_Time_Dias"] < 0)

# Punto reorden
if "Punto_Reorden" in inv.columns:
    add_damage("Punto_Reorden", inv["Punto_Reorden"].isna())
    add_flag("punto_reorden_negativo", inv["Punto_Reorden"] < 0)

# Ultima revision
if "Ultima_Revision" in inv.columns:
    add_damage("Ultima_Revision", inv["Ultima_Revision"].isna())
    today = pd.Timestamp.today().normalize()
    add_flag("fecha_revision_futura", inv["Ultima_Revision"] > today)

# Categoria/Bodega
if "Categoria_clean" in inv.columns:
    add_damage("Categoria_clean", inv["Categoria_clean"].isna())
    add_flag("categoria_unknown", inv["Categoria_clean"] == "unknown")

if "Bodega_Origen_clean" in inv.columns:
    add_damage("Bodega_Origen_clean", inv["Bodega_Origen_clean"].isna())
    add_flag("bodega_unknown", inv["Bodega_Origen_clean"] == "unknown")

# Outliers automáticos (IQR k=1.5) — SIN CONTROLES, SOLO INFORMAMOS
def compute_outlier_flag_iqr(col, flagname, k=1.5):
    if col not in inv.columns:
        return
    s = inv[col].dropna()
    if len(s) < 20:
        add_flag(flagname, pd.Series(False, index=inv.index))
        return
    low, high = iqr_bounds(s, k=k)
    add_flag(flagname, (inv[col] < low) | (inv[col] > high))

compute_outlier_flag_iqr("Costo_Unitario_USD", "costo_outlier_iqr", k=1.5)
compute_outlier_flag_iqr("Lead_Time_Dias", "leadtime_outlier_iqr", k=1.5)

# Conteos
inv["damaged_cols_count"] = inv[damage_cols].sum(axis=1) if damage_cols else 0
inv["any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False


# =========================
# 4) Split RAW / CLEAN / ANOMALÍAS
# =========================
base_anom_mask = inv["damaged_cols_count"] >= int(damage_threshold)
anom_mask = (base_anom_mask | inv["any_flag"]) if send_flags_to_anom else base_anom_mask

inv_anom = inv[anom_mask].copy()
inv_clean = inv[~anom_mask].copy()


# =========================
# 5) Ajuste opcional: stock negativo a positivo (SOLO CLEAN)
# =========================
if fix_negative_stock and "Stock_Actual" in inv_clean.columns:
    neg_mask = inv_clean["Stock_Actual"] < 0
    inv_clean["Stock_Actual"] = inv_clean["Stock_Actual"].abs()
    inv_clean["imputed__Stock_Actual_abs"] = neg_mask.astype(bool)


# =========================
# 6) Imputación SOLO en CLEAN (usa Categoria_clean)
# =========================
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

group_cat_col = "Categoria_clean" if "Categoria_clean" in inv_clean.columns else ("Categoria" if "Categoria" in inv_clean.columns else None)

if "Lead_Time_Dias" in inv_clean.columns:
    if impute_lead == "Mediana global":
        inv_clean = global_median_impute(inv_clean, "Lead_Time_Dias")
    elif impute_lead == "Mediana por categoría" and group_cat_col is not None:
        inv_clean = group_median_impute(inv_clean, "Lead_Time_Dias", group_cat_col)

if "Punto_Reorden" in inv_clean.columns:
    if impute_reorder == "Mediana global":
        inv_clean = global_median_impute(inv_clean, "Punto_Reorden")
    elif impute_reorder == "Mediana por categoría" and group_cat_col is not None:
        inv_clean = group_median_impute(inv_clean, "Punto_Reorden", group_cat_col)


# =========================
# UI
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Auditoría", "🔁 Cambios (texto)", "✅ CLEAN", "⚠️ ANOMALÍAS"])

with tab1:
    st.subheader("Auditoría de Inventario")
    st.dataframe(audit_summary(inv, inv_clean, inv_anom, damage_cols, flag_cols), use_container_width=True)

    st.info("Outliers detectados automáticamente con IQR (k=1.5) para Costo_Unitario_USD y Lead_Time_Dias.")

    st.markdown("### Valores únicos (Categoría limpia)")
    if "Categoria_clean" in inv.columns:
        st.write(inv["Categoria_clean"].value_counts(dropna=False).head(30))

    st.markdown("### Conteo de flags")
    if flag_cols:
        st.dataframe(inv[flag_cols].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)

with tab2:
    st.subheader("Reporte de cambios (antes → después) + conteo")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Categoria")
        if "Categoria_original" in inv.columns and "Categoria_clean" in inv.columns:
            st.dataframe(changes_report(inv["Categoria_original"], inv["Categoria_clean"]).head(80), use_container_width=True)
            if RAPIDFUZZ_AVAILABLE:
                st.caption("Sugerencias fuzzy (aplicadas / no aplicadas)")
                st.dataframe(cat_fuzzy_suggestions.head(80), use_container_width=True)
        else:
            st.info("No existe columna Categoria en el archivo.")

    with c2:
        st.markdown("### Bodega_Origen")
        if "Bodega_Origen_original" in inv.columns and "Bodega_Origen_clean" in inv.columns:
            st.dataframe(changes_report(inv["Bodega_Origen_original"], inv["Bodega_Origen_clean"]).head(80), use_container_width=True)
            if RAPIDFUZZ_AVAILABLE:
                st.caption("Sugerencias fuzzy (aplicadas / no aplicadas)")
                st.dataframe(bod_fuzzy_suggestions.head(80), use_container_width=True)
        else:
            st.info("No existe columna Bodega_Origen en el archivo.")

    if not RAPIDFUZZ_AVAILABLE:
        st.warning("Para fuzzy matching automático instala rapidfuzz: `pip install rapidfuzz` y agrégalo a requirements.txt.")

with tab3:
    st.subheader("Inventario CLEAN (para KPIs)")
    st.caption("Datos seguros para análisis. Imputaciones marcadas con columnas imputed__*.")
    preferred = [c for c in ["SKU_ID", "Categoria_clean", "Stock_Actual", "Costo_Unitario_USD",
                            "Punto_Reorden", "Lead_Time_Dias", "Bodega_Origen_clean", "Ultima_Revision"]
                 if c in inv_clean.columns]
    st.dataframe(inv_clean[preferred + [c for c in inv_clean.columns if c not in preferred]].head(200), use_container_width=True)

with tab4:
    st.subheader("Inventario ANOMALÍAS (riesgo / outliers / daños)")
    st.caption("Registros excluidos de KPIs, útiles para diagnóstico y storytelling.")
    preferred = [c for c in ["SKU_ID", "Categoria_clean", "Stock_Actual", "Costo_Unitario_USD",
                            "Punto_Reorden", "Lead_Time_Dias", "Bodega_Origen_clean", "Ultima_Revision"]
                 if c in inv_anom.columns]
    st.dataframe(inv_anom[preferred + [c for c in inv_anom.columns if c not in preferred]].head(200), use_container_width=True)
