import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# Fuzzy matching (opcional)
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False


st.set_page_config(page_title="Challenge 02 - DSS (Inventario)", layout="wide")
st.title("Challenge 02 — DSS Auditable (Inventario)")
st.caption("Inventario: RAW / CLEAN / ANOMALÍAS + normalización automática de Categoria y Bodega_Origen.")


# ======================================================
# 1) Normalización + Mapeo manual + Fuzzy (si aplica)
# ======================================================
def normalize_text(x: str) -> str:
    """Normalización básica: trim, lower, quitar tildes, -/_ -> espacio, colapsar espacios."""
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "":
        return np.nan
    x = x.lower()

    # quitar tildes/acentos
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8")

    # separadores comunes -> espacio
    x = re.sub(r"[-_/]+", " ", x)

    # quitar caracteres raros (dejamos letras/numeros/espacios)
    x = re.sub(r"[^a-z0-9\s]", "", x)

    # colapsar espacios
    x = re.sub(r"\s+", " ", x).strip()
    return x


def apply_manual_map(series_norm: pd.Series, manual_map: dict) -> pd.Series:
    """Mapeo manual sobre el texto ya normalizado."""
    return series_norm.map(lambda v: manual_map.get(v, v))


def normalize_unknowns(series_norm: pd.Series) -> pd.Series:
    """Convierte patrones tipo ???, unknown, n/a, etc. a 'unknown'."""
    unknown_tokens = {"???", "??", "?", "n a", "na", "n a", "none", "null", "sin categoria", "sin categoria", "unknown"}
    def _fix(v):
        if pd.isna(v):
            return np.nan
        vv = str(v).strip().lower()
        if vv in unknown_tokens:
            return "unknown"
        # si eran solo signos de pregunta (por si quedaron tras limpieza)
        if set(vv) <= {"?"}:
            return "unknown"
        return vv
    return series_norm.map(_fix)


def build_canonical_values(series_after_manual: pd.Series) -> list:
    """Define el conjunto de valores canónicos a los que se permite “pegarse” con fuzzy."""
    # Excluir unknown y nulos
    vals = series_after_manual.dropna().astype(str)
    vals = vals[vals != "unknown"]
    # solo valores que aparecen al menos una vez
    canonical = sorted(set(vals.tolist()))
    return canonical


def fuzzy_map_unique(series_vals: pd.Series, canonical: list, threshold: float = 0.92, delta: float = 0.03):
    """
    Aplica fuzzy matching solo si:
    - score >= threshold
    - y match es “único”: la diferencia contra el segundo mejor >= delta (o el 2do no alcanza threshold)
    Retorna la serie mapeada + un dataframe de sugerencias/aplicaciones.
    """
    if not RAPIDFUZZ_AVAILABLE or len(canonical) == 0:
        return series_vals, pd.DataFrame(columns=["from", "to", "score", "applied"])

    # RapidFuzz trabaja en 0-100 (por comodidad convertimos threshold/delta)
    thr = threshold * 100
    dlt = delta * 100

    changes = []
    mapped = series_vals.copy()

    # Solo intentamos mapear valores no nulos, no unknown
    unique_vals = sorted(set(series_vals.dropna().astype(str).tolist()))
    unique_vals = [v for v in unique_vals if v != "unknown"]

    for v in unique_vals:
        # si ya es canónico exacto, no tocamos
        if v in canonical:
            continue

        # Top2 matches
        # extractor returns list of tuples: (match, score, index)
        matches = process.extract(v, canonical, scorer=fuzz.WRatio, limit=2)
        if not matches:
            continue

        best_match, best_score, _ = matches[0]
        second_score = matches[1][1] if len(matches) > 1 else 0

        # Criterio de “único”
        is_unique = (best_score >= thr) and ((best_score - second_score) >= dlt or second_score < thr)

        if is_unique:
            mapped = mapped.replace(v, best_match)
            changes.append({"from": v, "to": best_match, "score": best_score, "applied": True})
        else:
            changes.append({"from": v, "to": best_match, "score": best_score, "applied": False})

    changes_df = pd.DataFrame(changes).sort_values(["applied", "score"], ascending=[False, False])
    return mapped, changes_df


def changes_report(original: pd.Series, final: pd.Series) -> pd.DataFrame:
    """Tabla antes -> después con conteo."""
    df = pd.DataFrame({
        "antes": original.astype("string"),
        "despues": final.astype("string"),
    })
    df = df.dropna(subset=["antes", "despues"])
    rep = (
        df.value_counts()
        .reset_index(name="conteo")
        .sort_values("conteo", ascending=False)
    )
    return rep


# ======================================================
# 2) Helpers numéricos / auditoría
# ======================================================
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


# ======================================================
# Sidebar: carga de archivo
# ======================================================
st.sidebar.header("1) Cargar inventario")
uploaded_inv = st.sidebar.file_uploader("Sube inventario_central_v2.csv", type=["csv"])
use_default = st.sidebar.checkbox("Si no subo archivo, usar ruta por defecto", value=True)
default_path = st.sidebar.text_input("Ruta por defecto", value="/mnt/data/inventario_central_v2.csv")

# ======================================================
# Sidebar: decisiones / checklists inventario
# ======================================================
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

st.sidebar.subheader("Normalización de texto (automática)")
enable_fuzzy = st.sidebar.checkbox(
    "Fuzzy matching (modo avanzado, solo match único)",
    value=True
)
fuzzy_threshold = st.sidebar.slider(
    "Umbral fuzzy (alto recomendado)",
    0.85, 0.99, 0.92, 0.01
)
fuzzy_delta = st.sidebar.slider(
    "Diferencia mínima vs 2do mejor (unicidad)",
    0.01, 0.10, 0.03, 0.01
)

if enable_fuzzy and not RAPIDFUZZ_AVAILABLE:
    st.sidebar.warning("⚠️ rapidfuzz no está instalado. Fuzzy se desactiva automáticamente.")


# ======================================================
# Load inventory
# ======================================================
@st.cache_data(show_spinner=False)
def load_from_upload(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=False)
def load_from_path(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

inv_raw = None
source_label = ""

try:
    if uploaded_inv is not None:
        inv_raw = load_from_upload(uploaded_inv)
        source_label = "Archivo subido"
    elif use_default:
        inv_raw = load_from_path(default_path)
        source_label = f"Ruta por defecto: {default_path}"
except Exception as e:
    st.error(f"No pude cargar el archivo. Error: {e}")
    st.stop()

if inv_raw is None:
    st.warning("Sube el archivo de inventario o activa la ruta por defecto.")
    st.stop()

st.success(f"Inventario cargado ✅ ({source_label})")


# ======================================================
# Tipificación mínima (sin borrar filas)
# ======================================================
inv = inv_raw.copy()

# Normalización automática de texto (Categoria y Bodega_Origen)
# --- Diccionarios manuales (ajústalos con tus casos reales) ---
CATEGORY_MAP = {
    # Después de normalize_text:
    "smart phone": "smartphones",
    "smartphones": "smartphones",
    "smartphone": "smartphones",
    "smart phones": "smartphones",

    # Ejemplos típicos
    "laptop": "laptops",
    "laptops": "laptops",
    "notebook": "laptops",
}

BODEGA_MAP = {
    # después de normalize_text, ejemplos
    "med": "medellin",
    "medellin": "medellin",
    "mde": "medellin",
    "bog": "bogota",
    "bogota": "bogota",
}

# Guardar originales para auditoría
if "Categoria" in inv.columns:
    inv["Categoria_original"] = inv["Categoria"].astype("string")
    inv["Categoria_norm"] = inv["Categoria"].apply(normalize_text)
    inv["Categoria_norm"] = normalize_unknowns(inv["Categoria_norm"])
    inv["Categoria_norm"] = apply_manual_map(inv["Categoria_norm"], CATEGORY_MAP)

if "Bodega_Origen" in inv.columns:
    inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")
    inv["Bodega_Origen_norm"] = inv["Bodega_Origen"].apply(normalize_text)
    inv["Bodega_Origen_norm"] = normalize_unknowns(inv["Bodega_Origen_norm"])
    inv["Bodega_Origen_norm"] = apply_manual_map(inv["Bodega_Origen_norm"], BODEGA_MAP)

# Fuzzy matching (solo match único)
cat_fuzzy_suggestions = pd.DataFrame()
bod_fuzzy_suggestions = pd.DataFrame()

if enable_fuzzy and RAPIDFUZZ_AVAILABLE:
    if "Categoria_norm" in inv.columns:
        canonical_cat = build_canonical_values(inv["Categoria_norm"])
        inv["Categoria_norm"], cat_fuzzy_suggestions = fuzzy_map_unique(
            inv["Categoria_norm"],
            canonical_cat,
            threshold=fuzzy_threshold,
            delta=fuzzy_delta
        )
    if "Bodega_Origen_norm" in inv.columns:
        canonical_bod = build_canonical_values(inv["Bodega_Origen_norm"])
        inv["Bodega_Origen_norm"], bod_fuzzy_suggestions = fuzzy_map_unique(
            inv["Bodega_Origen_norm"],
            canonical_bod,
            threshold=fuzzy_threshold,
            delta=fuzzy_delta
        )

# Tipificación numérica/fechas
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


# ======================================================
# damages + flags (Inventario)
# ======================================================
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

# Categoria unknown (solo para transparencia)
if "Categoria_norm" in inv.columns:
    add_flag("categoria_unknown", inv["Categoria_norm"] == "unknown")

# Outliers como flags
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

# Damage count y flags
inv["damaged_cols_count"] = inv[damage_cols].sum(axis=1) if damage_cols else 0
inv["any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False


# ======================================================
# Split RAW / CLEAN / ANOMALÍAS
# ======================================================
base_anom_mask = inv["damaged_cols_count"] >= int(damage_threshold)
anom_mask = (base_anom_mask | inv["any_flag"]) if send_flags_to_anom else base_anom_mask

inv_anom = inv[anom_mask].copy()
inv_clean = inv[~anom_mask].copy()


# ======================================================
# Imputación SOLO en CLEAN
# (usa Categoria_norm si existe; si no, usa Categoria)
# ======================================================
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

group_cat_col = "Categoria_norm" if "Categoria_norm" in inv_clean.columns else ("Categoria" if "Categoria" in inv_clean.columns else None)

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


# ======================================================
# UI
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs(["📋 Auditoría", "🔁 Cambios (texto)", "✅ CLEAN", "⚠️ ANOMALÍAS"])

with tab1:
    st.subheader("Auditoría de Inventario")
    st.dataframe(audit_summary(inv, inv_clean, inv_anom, damage_cols, flag_cols), use_container_width=True)

    st.markdown("### Top columnas con más nulos (RAW tipificado)")
    st.dataframe(
        inv.isna().mean().sort_values(ascending=False).head(15).to_frame("% nulos").mul(100),
        use_container_width=True
    )

    st.markdown("### Conteo de flags (riesgos)")
    if flag_cols:
        flag_counts = inv[flag_cols].sum().sort_values(ascending=False).to_frame("conteo")
        st.dataframe(flag_counts, use_container_width=True)
    else:
        st.info("No se detectaron flags (o no existen columnas esperadas).")

    st.markdown("### Ejemplos de registros marcados (primeros 50)")
    cols_show = [c for c in inv.columns if not c.startswith("damage__")]
    st.dataframe(inv.loc[inv["any_flag"], cols_show].head(50), use_container_width=True)


with tab4:
    st.subheader("Inventario ANOMALÍAS (riesgo / outliers / daños)")
    st.caption("Registros excluidos de KPIs, pero útiles para diagnóstico y storytelling.")
    st.dataframe(inv_anom.head(200), use_container_width=True)


with tab2:
    st.subheader("Reporte de cambios (antes → después) + conteo")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Categoria")
        if "Categoria_original" in inv.columns and "Categoria_norm" in inv.columns:
            rep = changes_report(inv["Categoria_original"], inv["Categoria_norm"])
            st.dataframe(rep.head(50), use_container_width=True)

            st.markdown("**Sugerencias fuzzy (aplicadas y no aplicadas)**")
            if enable_fuzzy and RAPIDFUZZ_AVAILABLE:
                st.dataframe(cat_fuzzy_suggestions.head(50), use_container_width=True)
            else:
                st.caption("Fuzzy desactivado o rapidfuzz no disponible.")
        else:
            st.info("No existe columna Categoria en el archivo.")

    with colB:
        st.markdown("### Bodega_Origen")
        if "Bodega_Origen_original" in inv.columns and "Bodega_Origen_norm" in inv.columns:
            repb = changes_report(inv["Bodega_Origen_original"], inv["Bodega_Origen_norm"])
            st.dataframe(repb.head(50), use_container_width=True)

            st.markdown("**Sugerencias fuzzy (aplicadas y no aplicadas)**")
            if enable_fuzzy and RAPIDFUZZ_AVAILABLE:
                st.dataframe(bod_fuzzy_suggestions.head(50), use_container_width=True)
            else:
                st.caption("Fuzzy desactivado o rapidfuzz no disponible.")
        else:
            st.info("No existe columna Bodega_Origen en el archivo.")

    if enable_fuzzy and not RAPIDFUZZ_AVAILABLE:
        st.warning("Instala rapidfuzz para habilitar fuzzy matching. Ej: `pip install rapidfuzz`.")


with tab3:
    st.subheader("Inventario CLEAN (para KPIs)")
    st.caption("Datos seguros para análisis. Imputaciones marcadas con columnas imputed__*.")
    st.dataframe(inv_clean.head(200), use_container_width=True)
