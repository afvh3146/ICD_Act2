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

# =======================================================================================================================================================================
# =======================================================================================================================================================================
# =======================================================================================================================================================================

# ======================================================
# ================= TRANSACCIONES LOGÍSTICAS ===========
# ======================================================

st.markdown("---")
st.header("🚚 Limpieza de Transacciones Logísticas")

st.sidebar.header("🚚 Limpieza Transacciones")
uploaded_tx = st.sidebar.file_uploader(
    "Sube transacciones_logistica_v2.csv",
    type=["csv"],
    key="uploader_transacciones_logistica"
)

with st.sidebar.expander("Reglas de ANOMALÍAS (Transacciones)", expanded=True):
    tx_damage_threshold = st.number_input(
        "Enviar fila a ANOMALÍAS si columnas dañadas ≥",
        min_value=1, max_value=10, value=2,
        key="tx_damage_threshold_v2"
    )
    tx_send_flags_to_anom = st.checkbox(
        "Enviar a ANOMALÍAS si tiene cualquier flag",
        value=True,
        key="tx_send_flags_v2"
    )

if uploaded_tx is None:
    st.info("👈 Sube el archivo **transacciones_logistica_v2.csv** en la barra lateral para ver auditoría y flags.")
else:
    @st.cache_data(show_spinner=False)
    def load_tx(uploaded_file):
        return pd.read_csv(uploaded_file)

    try:
        tx_raw = load_tx(uploaded_tx)
    except Exception as e:
        st.error(f"No pude leer el CSV de transacciones. Error: {e}")
        st.stop()

    st.success(f"Transacciones cargadas ✅  | Filas: {len(tx_raw):,}  | Columnas: {len(tx_raw.columns)}")

    with st.expander("👀 Vista previa (RAW)", expanded=False):
        st.dataframe(tx_raw.head(50), use_container_width=True)

    # -------------------------
    # Helpers locales
    # -------------------------
    def to_numeric_loose(s: pd.Series) -> pd.Series:
        """Convierte números tolerando símbolos y comas."""
        ss = s.astype("string")
        ss = ss.str.replace(r"[\$€£]", "", regex=True)
        ss = ss.str.replace(r"(usd|cop|eur|mxn|clp)", "", regex=True, case=False)
        ss = ss.str.replace(r"[^\d\-\.\,]", "", regex=True)

        has_dot = ss.str.contains(r"\.", na=False)
        has_comma = ss.str.contains(r",", na=False)
        ss = ss.where(~(has_dot & has_comma), ss.str.replace(",", "", regex=False))
        ss = ss.where(~(has_comma & ~has_dot), ss.str.replace(",", ".", regex=False))
        return pd.to_numeric(ss, errors="coerce")

    def to_datetime_venta(s: pd.Series) -> pd.Series:
        """Fecha_Venta viene como dd/mm/yyyy."""
        return pd.to_datetime(s, errors="coerce", dayfirst=True)

    # -------------------------
    # Copia base
    # -------------------------
    tx = tx_raw.copy()

    # -------------------------
    # Columnas reales (tu header)
    # -------------------------
    col_tx_id = "Transaccion_ID" if "Transaccion_ID" in tx.columns else None
    col_sku = "SKU_ID" if "SKU_ID" in tx.columns else None
    col_fecha_venta = "Fecha_Venta" if "Fecha_Venta" in tx.columns else None
    col_cantidad = "Cantidad_Vendida" if "Cantidad_Vendida" in tx.columns else None
    col_precio = "Precio_Venta_Final" if "Precio_Venta_Final" in tx.columns else None
    col_costo_envio = "Costo_Envio" if "Costo_Envio" in tx.columns else None
    col_tiempo = "Tiempo_Entrega_Real" if "Tiempo_Entrega_Real" in tx.columns else None
    col_estado = "Estado_Envio" if "Estado_Envio" in tx.columns else None
    col_ciudad = "Ciudad_Destino" if "Ciudad_Destino" in tx.columns else None
    col_canal = "Canal_Venta" if "Canal_Venta" in tx.columns else None

    # -------------------------
    # Normalización de texto (automática)
    # Reusa funciones de tu app:
    # normalize_text_keep_unknown, apply_manual_map, build_canonical_values, fuzzy_map_unique
    # -------------------------
    CITY_MAP = {
        "bogota": "bogota", "bog": "bogota",
        "medellin": "medellin", "med": "medellin", "mde": "medellin",
        "cali": "cali",
        "cartagena": "cartagena",
        "barranquilla": "barranquilla",
        "unknown": "unknown",
    }

    STATUS_MAP = {
        "entregado": "entregado",
        "entregada": "entregado",
        "delivered": "entregado",

        "pendiente": "pendiente",
        "pending": "pendiente",

        "en transito": "en_transito",
        "entransito": "en_transito",
        "transito": "en_transito",
        "in transit": "en_transito",

        "cancelado": "cancelado",
        "cancelled": "cancelado",
        "canceled": "cancelado",

        "devuelto": "devuelto",
        "returned": "devuelto",

        "unknown": "unknown",
    }

    CANAL_MAP = {
        "web": "web",
        "pagina web": "web",
        "online": "web",
        "ecommerce": "web",

        "tienda": "tienda",
        "tienda fisica": "tienda",
        "tienda física": "tienda",
        "fisico": "tienda",
        "físico": "tienda",

        "whatsapp": "whatsapp",
        "call center": "call_center",
        "callcenter": "call_center",

        "unknown": "unknown",
    }

    tx_city_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])
    tx_status_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])
    tx_canal_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])

    if col_ciudad:
        tx["Ciudad_Destino_original"] = tx[col_ciudad].astype("string")
        tx["Ciudad_Destino_clean"] = tx[col_ciudad].apply(normalize_text_keep_unknown)
        tx["Ciudad_Destino_clean"] = apply_manual_map(tx["Ciudad_Destino_clean"], CITY_MAP)
        canonical = build_canonical_values(tx["Ciudad_Destino_clean"])
        tx["Ciudad_Destino_clean"], tx_city_fuzzy = fuzzy_map_unique(tx["Ciudad_Destino_clean"], canonical, 0.92, 0.03)

    if col_estado:
        tx["Estado_Envio_original"] = tx[col_estado].astype("string")
        tx["Estado_Envio_clean"] = tx[col_estado].apply(normalize_text_keep_unknown)
        tx["Estado_Envio_clean"] = apply_manual_map(tx["Estado_Envio_clean"], STATUS_MAP)
        canonical = build_canonical_values(tx["Estado_Envio_clean"])
        tx["Estado_Envio_clean"], tx_status_fuzzy = fuzzy_map_unique(tx["Estado_Envio_clean"], canonical, 0.92, 0.03)

    if col_canal:
        tx["Canal_Venta_original"] = tx[col_canal].astype("string")
        tx["Canal_Venta_clean"] = tx[col_canal].apply(normalize_text_keep_unknown)
        # ojo: como normalize_text_keep_unknown quita tildes, "tienda física" ya llega como "tienda fisica"
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], {
            # normaliza llaves posibles tras normalize
            "tienda fisica": "tienda",
            **{normalize_text_keep_unknown(k): v for k, v in CANAL_MAP.items()}
        })
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], tx_canal_fuzzy = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)

    # -------------------------
    # Parseo de fecha y numéricos
    # -------------------------
    if col_fecha_venta:
        tx["Fecha_Venta_dt"] = to_datetime_venta(tx[col_fecha_venta])
    else:
        tx["Fecha_Venta_dt"] = pd.NaT

    if col_costo_envio:
        tx["Costo_Envio_num"] = to_numeric_loose(tx[col_costo_envio])
    else:
        tx["Costo_Envio_num"] = np.nan

    if col_tiempo:
        tx["Tiempo_Entrega_num"] = pd.to_numeric(tx[col_tiempo], errors="coerce")
    else:
        tx["Tiempo_Entrega_num"] = np.nan

    # -------------------------
    # Daños y flags
    # -------------------------
    tx_damage_cols, tx_flag_cols = [], []

    def tx_add_damage(name, mask):
        cname = f"damage__{name}"
        tx[cname] = mask.astype(int)
        tx_damage_cols.append(cname)

    def tx_add_flag(name, mask):
        cname = f"flag__{name}"
        tx[cname] = mask.astype(bool)
        tx_flag_cols.append(cname)

    # daños básicos
    if col_tx_id:
        tx_add_damage("Transaccion_ID", tx[col_tx_id].isna())
    if col_sku:
        tx_add_damage("SKU_ID", tx[col_sku].isna())

    # Fecha venta: nulo + invalido (texto pero NaT)
    if col_fecha_venta:
        tx_add_damage("Fecha_Venta", tx[col_fecha_venta].isna())
        tx_add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx[col_fecha_venta].notna())
    else:
        tx_add_damage("Fecha_Venta", pd.Series(True, index=tx.index))

    # Venta futura (la regla de este dataset)
    today = pd.Timestamp.today().normalize()
    tx_add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

    # Costo: nulo + invalido + <=0
    if col_costo_envio:
        tx_add_damage("Costo_Envio", tx[col_costo_envio].isna())
        tx_add_flag("costo_invalido", tx["Costo_Envio_num"].isna() & tx[col_costo_envio].notna())
        tx_add_flag("costo_no_positivo", tx["Costo_Envio_num"].notna() & (tx["Costo_Envio_num"] <= 0))
    else:
        tx_add_damage("Costo_Envio", pd.Series(True, index=tx.index))

    # Tiempo entrega: nulo + invalido + negativo
    if col_tiempo:
        tx_add_damage("Tiempo_Entrega_Real", tx[col_tiempo].isna())
        tx_add_flag("tiempo_invalido", tx["Tiempo_Entrega_num"].isna() & tx[col_tiempo].notna())
        tx_add_flag("tiempo_negativo", tx["Tiempo_Entrega_num"].notna() & (tx["Tiempo_Entrega_num"] < 0))
    else:
        tx_add_damage("Tiempo_Entrega_Real", pd.Series(True, index=tx.index))

    # Texto unknown
    if "Ciudad_Destino_clean" in tx.columns:
        tx_add_flag("ciudad_unknown", tx["Ciudad_Destino_clean"] == "unknown")
    if "Estado_Envio_clean" in tx.columns:
        tx_add_flag("estado_unknown", tx["Estado_Envio_clean"] == "unknown")
    if "Canal_Venta_clean" in tx.columns:
        tx_add_flag("canal_unknown", tx["Canal_Venta_clean"] == "unknown")

    # Conteos
    tx["damaged_cols_count"] = tx[tx_damage_cols].sum(axis=1) if tx_damage_cols else 0
    tx["any_flag"] = tx[tx_flag_cols].any(axis=1) if tx_flag_cols else False

    # Split
    tx_base_anom = tx["damaged_cols_count"] >= int(tx_damage_threshold)
    tx_anom_mask = (tx_base_anom | tx["any_flag"]) if tx_send_flags_to_anom else tx_base_anom
    tx_anom = tx[tx_anom_mask].copy()
    tx_clean = tx[~tx_anom_mask].copy()

    # -------------------------
    # Diagnóstico
    # -------------------------
    with st.expander("🧪 Diagnóstico (columnas detectadas)", expanded=True):
        st.json({
            "Transaccion_ID": col_tx_id,
            "SKU_ID": col_sku,
            "Fecha_Venta": col_fecha_venta,
            "Costo_Envio": col_costo_envio,
            "Tiempo_Entrega_Real": col_tiempo,
            "Ciudad_Destino": col_ciudad,
            "Estado_Envio": col_estado,
            "Canal_Venta": col_canal
        })

        st.write("Ejemplos Fecha_Venta (raw → parsed):")
        sample = tx[[col_fecha_venta, "Fecha_Venta_dt"]].head(10) if col_fecha_venta else pd.DataFrame()
        if not sample.empty:
            st.dataframe(sample, use_container_width=True)

    # -------------------------
    # UI
    # -------------------------
    tx_tab1, tx_tab2, tx_tab3, tx_tab4 = st.tabs(["📋 Auditoría", "🚩 Flags (ejemplos)", "✅ CLEAN", "⚠️ ANOMALÍAS"])

    with tx_tab1:
        st.subheader("Auditoría de Transacciones")
        st.dataframe(
            pd.DataFrame([{
                "Filas RAW": len(tx),
                "Filas CLEAN": len(tx_clean),
                "Filas ANOMALÍAS": len(tx_anom),
                "% ANOMALÍAS": (len(tx_anom) / len(tx) * 100) if len(tx) else 0,
            }]),
            use_container_width=True
        )

        st.markdown("### Conteo de flags")
        st.dataframe(tx[tx_flag_cols].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)

        st.markdown("### Cambios de texto (antes → después)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if "Ciudad_Destino_original" in tx.columns:
                st.write("Ciudad")
                st.dataframe(changes_report(tx["Ciudad_Destino_original"], tx["Ciudad_Destino_clean"]).head(25),
                             use_container_width=True)
        with c2:
            if "Estado_Envio_original" in tx.columns:
                st.write("Estado")
                st.dataframe(changes_report(tx["Estado_Envio_original"], tx["Estado_Envio_clean"]).head(25),
                             use_container_width=True)
        with c3:
            if "Canal_Venta_original" in tx.columns:
                st.write("Canal")
                st.dataframe(changes_report(tx["Canal_Venta_original"], tx["Canal_Venta_clean"]).head(25),
                             use_container_width=True)

    with tx_tab2:
        st.subheader("Ejemplos por flag")
        selected_flag = st.selectbox("Selecciona un flag:", tx_flag_cols, key="tx_flag_picker_v2")
        flagged = tx[tx[selected_flag] == True].copy()
        st.write(f"Filas con **{selected_flag}**: {len(flagged):,}")
        st.dataframe(flagged.head(200), use_container_width=True)

    with tx_tab3:
        st.subheader("Transacciones CLEAN")
        st.dataframe(tx_clean.head(200), use_container_width=True)

    with tx_tab4:
        st.subheader("Transacciones ANOMALÍAS")
        st.dataframe(tx_anom.head(200), use_container_width=True)

    with st.expander("🔎 Fuzzy matching (auditoría)", expanded=False):
        if RAPIDFUZZ_AVAILABLE:
            st.write("Ciudad – sugerencias:")
            st.dataframe(tx_city_fuzzy.head(50), use_container_width=True)
            st.write("Estado – sugerencias:")
            st.dataframe(tx_status_fuzzy.head(50), use_container_width=True)
            st.write("Canal – sugerencias:")
            st.dataframe(tx_canal_fuzzy.head(50), use_container_width=True)
        else:
            st.caption("rapidfuzz no está instalado; fuzzy matching no se aplicó.")
