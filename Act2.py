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


# =========================
# App config
# =========================
st.set_page_config(page_title="Challenge 02 — DSS Auditable", layout="wide")
st.title("Challenge 02 — DSS Auditable (Inventario + Transacciones + Join)")
st.caption("Estructura única: RAW / CLEAN / ANOMALÍAS + auditoría + ejemplos por flag. Join por SKU preparado para análisis.")

# =========================
# Helpers: texto
# =========================
UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null",
    "sin categoria", "sincategoria", "unknown", "sin categoría"
}

def normalize_text_keep_unknown(x: str) -> str:
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
# Helpers: numéricos / fechas / outliers
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

def compute_outlier_flag_iqr(df, col, k=1.5):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].dropna()
    if len(s) < 20:
        return pd.Series(False, index=df.index)
    low, high = iqr_bounds(s, k=k)
    return (df[col] < low) | (df[col] > high)

def safe_bool_series(df, default=False):
    return pd.Series(default, index=df.index)

# =========================
# IO
# =========================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

# =========================
# Sidebar: upload gate (pedir ambos antes de seguir)
# =========================
st.sidebar.header("📁 Cargar archivos (obligatorio)")
uploaded_inv = st.sidebar.file_uploader("1) inventario_central_v2.csv", type=["csv"], key="up_inv")
uploaded_tx  = st.sidebar.file_uploader("2) transacciones_logistica_v2.csv", type=["csv"], key="up_tx")

st.sidebar.header("📁 Cargar archivo (siguiente paso)")
uploaded_fb  = st.sidebar.file_uploader("3) feedback_clientes_v2.csv (aún no se procesa)", type=["csv"], key="up_fb")

if uploaded_inv is None or uploaded_tx is None:
    st.info("👈 Para continuar, sube **inventario** y **transacciones**. (El feedback queda preparado para el siguiente paso).")
    st.stop()

inv_raw = load_csv(uploaded_inv)
tx_raw  = load_csv(uploaded_tx)

st.success(f"Inventario cargado ✅ | Filas: {len(inv_raw):,} | Columnas: {len(inv_raw.columns)}")
st.success(f"Transacciones cargadas ✅ | Filas: {len(tx_raw):,} | Columnas: {len(tx_raw.columns)}")

# =========================
# Sidebar: reglas globales (misma lógica para datasets)
# =========================
st.sidebar.header("⚙️ Reglas comunes")
send_flags_to_anom_global = st.sidebar.checkbox("Enviar a ANOMALÍAS si tiene cualquier flag (global)", value=True, key="send_flags_global")

# =========================
# Procesamiento: INVENTARIO
# =========================
def process_inventario(inv_raw: pd.DataFrame):
    st.sidebar.header("🧹 Limpieza Inventario")

    with st.sidebar.expander("Reglas de ANOMALÍAS / Daños (Inventario)", expanded=True):
        damage_threshold = st.number_input("Inventario: columnas dañadas ≥", 1, 10, 2, key="inv_damage_thr")

    with st.sidebar.expander("Regla opcional: Stock negativo", expanded=True):
        fix_negative_stock = st.checkbox("Inventario: convertir stock negativo a positivo en CLEAN (abs)", value=False, key="inv_fix_stock")
        st.caption("RAW y ANOMALÍAS NO se modifican. Solo afecta CLEAN.")

    with st.sidebar.expander("Imputación (solo CLEAN)", expanded=False):
        impute_lead = st.selectbox("Inventario: Lead_Time_Dias nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2, key="inv_imp_lead")
        impute_reorder = st.selectbox("Inventario: Punto_Reorden nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2, key="inv_imp_reorder")

    inv = inv_raw.copy()

    CATEGORY_MAP = {
        "laptop": "laptops", "laptops": "laptops", "notebook": "laptops", "notebooks": "laptops",
        "smartphone": "smartphones", "smartphones": "smartphones", "smart phone": "smartphones", "smart phones": "smartphones",
        "tablet": "tablets", "tablets": "tablets",
        "accesorio": "accesorios", "accesorios": "accesorios",
        "monitor": "monitores", "monitores": "monitores",
        "unknown": "unknown",
    }

    BODEGA_MAP = {
        "med": "medellin", "mde": "medellin", "medellin": "medellin",
        "bog": "bogota", "bogota": "bogota",
        "unknown": "unknown",
    }

    cat_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])
    bod_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])

    # Normalización texto
    if "Categoria" in inv.columns:
        inv["Categoria_original"] = inv["Categoria"].astype("string")
        inv["Categoria_clean"] = inv["Categoria"].apply(normalize_text_keep_unknown)
        inv["Categoria_clean"] = apply_manual_map(inv["Categoria_clean"], CATEGORY_MAP)
        canonical = build_canonical_values(inv["Categoria_clean"])
        inv["Categoria_clean"], cat_fuzzy = fuzzy_map_unique(inv["Categoria_clean"], canonical, 0.92, 0.03)

    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")
        inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
        inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)
        canonical = build_canonical_values(inv["Bodega_Origen_clean"])
        inv["Bodega_Origen_clean"], bod_fuzzy = fuzzy_map_unique(inv["Bodega_Origen_clean"], canonical, 0.92, 0.03)

    # Tipificación
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

    damage_cols, flag_cols = [], []

    def add_damage(colname, mask):
        cname = f"damage__{colname}"
        inv[cname] = mask.astype(int)
        damage_cols.append(cname)

    def add_flag(flagname, mask):
        cname = f"flag__{flagname}"
        inv[cname] = mask.astype(bool)
        flag_cols.append(cname)

    # daños + flags básicos
    if "SKU_ID" in inv.columns:
        add_damage("SKU_ID", inv["SKU_ID"].isna())

    if "Stock_Actual" in inv.columns:
        add_damage("Stock_Actual", inv["Stock_Actual"].isna())
        add_flag("stock_negativo", inv["Stock_Actual"] < 0)

    if "Costo_Unitario_USD" in inv.columns:
        add_damage("Costo_Unitario_USD", inv["Costo_Unitario_USD"].isna())
        add_flag("costo_no_positivo", inv["Costo_Unitario_USD"] <= 0)
        add_flag("costo_outlier_iqr", compute_outlier_flag_iqr(inv, "Costo_Unitario_USD", k=1.5))

    if "Lead_Time_Dias" in inv.columns:
        add_damage("Lead_Time_Dias", inv["Lead_Time_Dias"].isna())
        add_flag("leadtime_negativo", inv["Lead_Time_Dias"] < 0)
        add_flag("leadtime_outlier_iqr", compute_outlier_flag_iqr(inv, "Lead_Time_Dias", k=1.5))

    if "Punto_Reorden" in inv.columns:
        add_damage("Punto_Reorden", inv["Punto_Reorden"].isna())
        add_flag("punto_reorden_negativo", inv["Punto_Reorden"] < 0)

    if "Ultima_Revision" in inv.columns:
        add_damage("Ultima_Revision", inv["Ultima_Revision"].isna())
        today = pd.Timestamp.today().normalize()
        add_flag("fecha_revision_futura", inv["Ultima_Revision"] > today)

    if "Categoria_clean" in inv.columns:
        add_damage("Categoria_clean", inv["Categoria_clean"].isna())
        add_flag("categoria_unknown", inv["Categoria_clean"] == "unknown")

    if "Bodega_Origen_clean" in inv.columns:
        add_damage("Bodega_Origen_clean", inv["Bodega_Origen_clean"].isna())
        add_flag("bodega_unknown", inv["Bodega_Origen_clean"] == "unknown")

    inv["damaged_cols_count"] = inv[damage_cols].sum(axis=1) if damage_cols else 0
    inv["any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False

    base_anom = inv["damaged_cols_count"] >= int(damage_threshold)
    anom_mask = (base_anom | inv["any_flag"]) if send_flags_to_anom_global else base_anom

    inv_anom = inv[anom_mask].copy()
    inv_clean = inv[~anom_mask].copy()

    # stock abs solo en clean
    if fix_negative_stock and "Stock_Actual" in inv_clean.columns:
        neg = inv_clean["Stock_Actual"] < 0
        inv_clean["Stock_Actual"] = inv_clean["Stock_Actual"].abs()
        inv_clean["imputed__Stock_Actual_abs"] = neg.astype(bool)

    # imputación en clean
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

    # cambios de texto (para UI)
    text_changes = {
        "Categoria": (inv.get("Categoria_original"), inv.get("Categoria_clean"), cat_fuzzy),
        "Bodega_Origen": (inv.get("Bodega_Origen_original"), inv.get("Bodega_Origen_clean"), bod_fuzzy),
    }

    return inv, inv_clean, inv_anom, damage_cols, flag_cols, text_changes


# =========================
# Procesamiento: TRANSACCIONES
# =========================
def process_transacciones(tx_raw: pd.DataFrame):
    st.sidebar.header("🧹 Limpieza Transacciones")

    with st.sidebar.expander("Reglas de ANOMALÍAS / Daños (Transacciones)", expanded=True):
        damage_threshold = st.number_input("Transacciones: columnas dañadas ≥", 1, 10, 2, key="tx_damage_thr")

    tx = tx_raw.copy()

    # Mapas texto
    CITY_MAP = {
        "bog": "bogota", "bogota": "bogota",
        "med": "medellin", "mde": "medellin", "medellin": "medellin",
        "cali": "cali",
        "barranquilla": "barranquilla",
        "bucaramanga": "bucaramanga",
        "cartagena": "cartagena",
        "unknown": "unknown",
    }

    STATUS_MAP = {
        "entregado": "entregado",
        "devuelto": "devuelto",
        "retrasado": "retrasado",
        "en camino": "en_transito",
        "encamino": "en_transito",
        "en transito": "en_transito",
        "transito": "en_transito",
        "perdido": "perdido",
        "pending": "pendiente",
        "pendiente": "pendiente",
        "unknown": "unknown",
    }

    CANAL_MAP = {
        "fisico": "tienda", "fisco": "tienda", "tienda": "tienda", "tienda fisica": "tienda",
        "online": "web", "web": "web", "ecommerce": "web",
        "whatsapp": "whatsapp",
        "app": "app",
        "unknown": "unknown",
    }

    SUSPICIOUS_CITY_TOKENS = {"ventas", "web", "online", "app", "whatsapp", "canal"}

    tx_city_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])
    tx_status_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])
    tx_canal_fuzzy = pd.DataFrame(columns=["from", "to", "score", "applied"])

    # Normalización texto + regla ciudad sospechosa
    if "Ciudad_Destino" in tx.columns:
        tx["Ciudad_Destino_original"] = tx["Ciudad_Destino"].astype("string")
        tx["Ciudad_Destino_norm"] = tx["Ciudad_Destino"].apply(normalize_text_keep_unknown)

        def _is_city_suspicious(v):
            if pd.isna(v) or v == "unknown":
                return False
            parts = set(str(v).split())
            return len(parts.intersection(SUSPICIOUS_CITY_TOKENS)) > 0

        tx["flag__ciudad_sospechosa"] = tx["Ciudad_Destino_norm"].map(_is_city_suspicious).astype(bool)
        tx.loc[tx["flag__ciudad_sospechosa"], "Ciudad_Destino_norm"] = "unknown"

        tx["Ciudad_Destino_clean"] = apply_manual_map(tx["Ciudad_Destino_norm"], CITY_MAP)
        canonical = build_canonical_values(tx["Ciudad_Destino_clean"])
        tx["Ciudad_Destino_clean"], tx_city_fuzzy = fuzzy_map_unique(tx["Ciudad_Destino_clean"], canonical, 0.92, 0.03)

    if "Estado_Envio" in tx.columns:
        tx["Estado_Envio_original"] = tx["Estado_Envio"].astype("string")
        tx["Estado_Envio_clean"] = tx["Estado_Envio"].apply(normalize_text_keep_unknown)
        tx["Estado_Envio_clean"] = apply_manual_map(tx["Estado_Envio_clean"], STATUS_MAP)
        canonical = build_canonical_values(tx["Estado_Envio_clean"])
        tx["Estado_Envio_clean"], tx_status_fuzzy = fuzzy_map_unique(tx["Estado_Envio_clean"], canonical, 0.92, 0.03)

    if "Canal_Venta" in tx.columns:
        tx["Canal_Venta_original"] = tx["Canal_Venta"].astype("string")
        tx["Canal_Venta_clean"] = tx["Canal_Venta"].apply(normalize_text_keep_unknown)
        # normaliza posibles llaves post-normalize
        extra = {"tienda fisica": "tienda"}
        norm_map = {normalize_text_keep_unknown(k): v for k, v in CANAL_MAP.items()}
        norm_map.update(extra)
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], norm_map)
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], tx_canal_fuzzy = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)

    # Parseo fecha venta dd/mm/yyyy
    if "Fecha_Venta" in tx.columns:
        tx["Fecha_Venta_dt"] = pd.to_datetime(tx["Fecha_Venta"], errors="coerce", dayfirst=True)
    else:
        tx["Fecha_Venta_dt"] = pd.NaT

    # Tipos numéricos básicos
    for c in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real"]:
        if c in tx.columns:
            tx[c] = to_numeric(tx[c])

    damage_cols, flag_cols = [], []

    def add_damage(colname, mask):
        cname = f"damage__{colname}"
        tx[cname] = mask.astype(int)
        damage_cols.append(cname)

    def add_flag(flagname, mask):
        cname = f"flag__{flagname}"
        tx[cname] = mask.astype(bool)
        flag_cols.append(cname)

    # Daños
    if "Transaccion_ID" in tx.columns:
        add_damage("Transaccion_ID", tx["Transaccion_ID"].isna())
    if "SKU_ID" in tx.columns:
        add_damage("SKU_ID", tx["SKU_ID"].isna())
    if "Fecha_Venta" in tx.columns:
        add_damage("Fecha_Venta", tx["Fecha_Venta"].isna())
    if "Costo_Envio" in tx.columns:
        add_damage("Costo_Envio", tx["Costo_Envio"].isna())
    if "Tiempo_Entrega_Real" in tx.columns:
        add_damage("Tiempo_Entrega_Real", tx["Tiempo_Entrega_Real"].isna())

    # Flags
    # Fecha inválida + venta futura
    if "Fecha_Venta" in tx.columns:
        add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())
        today = pd.Timestamp.today().normalize()
        add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

    # Cantidad / precio
    if "Cantidad_Vendida" in tx.columns:
        add_flag("cantidad_no_positiva", tx["Cantidad_Vendida"].notna() & (tx["Cantidad_Vendida"] <= 0))
    if "Precio_Venta_Final" in tx.columns:
        add_flag("precio_no_positivo", tx["Precio_Venta_Final"].notna() & (tx["Precio_Venta_Final"] <= 0))

    # costo
    if "Costo_Envio" in tx.columns:
        add_flag("costo_nulo", tx["Costo_Envio"].isna())
        add_flag("costo_no_positivo", tx["Costo_Envio"].notna() & (tx["Costo_Envio"] <= 0))

    # tiempo
    if "Tiempo_Entrega_Real" in tx.columns:
        add_flag("tiempo_negativo", tx["Tiempo_Entrega_Real"].notna() & (tx["Tiempo_Entrega_Real"] < 0))
        add_flag("tiempo_outlier_iqr", compute_outlier_flag_iqr(tx, "Tiempo_Entrega_Real", k=1.5))

    # unknowns
    if "Ciudad_Destino_clean" in tx.columns:
        add_flag("ciudad_unknown", tx["Ciudad_Destino_clean"] == "unknown")
    if "Estado_Envio_clean" in tx.columns:
        add_flag("estado_unknown", tx["Estado_Envio_clean"] == "unknown")
    if "Canal_Venta_clean" in tx.columns:
        add_flag("canal_unknown", tx["Canal_Venta_clean"] == "unknown")

    # ciudad sospechosa ya está
    if "flag__ciudad_sospechosa" in tx.columns:
        # solo agrega si no existe con el prefijo "flag__" ya
        # (aquí está sin prefijo extra porque la columna ya se llama flag__ciudad_sospechosa)
        flag_cols.append("flag__ciudad_sospechosa")

    tx["damaged_cols_count"] = tx[damage_cols].sum(axis=1) if damage_cols else 0
    tx["any_flag"] = tx[flag_cols].any(axis=1) if flag_cols else False

    base_anom = tx["damaged_cols_count"] >= int(damage_threshold)
    anom_mask = (base_anom | tx["any_flag"]) if send_flags_to_anom_global else base_anom

    tx_anom = tx[anom_mask].copy()
    tx_clean = tx[~anom_mask].copy()

    text_changes = {
        "Ciudad_Destino": (tx.get("Ciudad_Destino_original"), tx.get("Ciudad_Destino_clean"), tx_city_fuzzy),
        "Estado_Envio": (tx.get("Estado_Envio_original"), tx.get("Estado_Envio_clean"), tx_status_fuzzy),
        "Canal_Venta": (tx.get("Canal_Venta_original"), tx.get("Canal_Venta_clean"), tx_canal_fuzzy),
    }

    return tx, tx_clean, tx_anom, damage_cols, flag_cols, text_changes


# =========================
# UI renderer: misma estructura para ambos datasets
# =========================
def render_dataset_section(title, df_raw, df_clean, df_anom, damage_cols, flag_cols, text_changes, key_prefix):
    st.subheader(title)

    t1, t2, t3, t4, t5 = st.tabs(["📋 Auditoría", "🔁 Cambios (texto)", "🚩 Flags (ejemplos)", "✅ CLEAN", "⚠️ ANOMALÍAS"])

    with t1:
        st.dataframe(audit_summary(df_raw, df_clean, df_anom, damage_cols, flag_cols), use_container_width=True)

        st.markdown("### Conteo de flags")
        if flag_cols:
            st.dataframe(df_raw[flag_cols].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)
        else:
            st.info("No hay flags generados para este dataset.")

        with st.expander("👀 Vista previa (RAW)", expanded=False):
            st.dataframe(df_raw.head(50), use_container_width=True)

    with t2:
        st.markdown("### Reporte de cambios (antes → después) + conteo")
        # muestra cambios solo si existen
        for name, (orig, clean, fuzzy_df) in text_changes.items():
            st.markdown(f"#### {name}")
            if orig is None or clean is None:
                st.caption("No aplica / no existe la columna.")
                continue
            st.dataframe(changes_report(orig, clean).head(80), use_container_width=True)
            if RAPIDFUZZ_AVAILABLE and isinstance(fuzzy_df, pd.DataFrame) and len(fuzzy_df) > 0:
                st.caption("Sugerencias fuzzy (aplicadas / no aplicadas)")
                st.dataframe(fuzzy_df.head(80), use_container_width=True)

        if not RAPIDFUZZ_AVAILABLE:
            st.warning("Fuzzy matching no está activo. Instala rapidfuzz: `pip install rapidfuzz`.")

    with t3:
        if not flag_cols:
            st.info("No hay flags para inspeccionar.")
        else:
            selected_flag = st.selectbox(
                "Selecciona un flag para ver ejemplos:",
                flag_cols,
                key=f"{key_prefix}_flag_picker"
            )
            flagged = df_raw[df_raw[selected_flag] == True].copy()
            st.write(f"Filas con **{selected_flag}**: {len(flagged):,}")
            st.dataframe(flagged.head(200), use_container_width=True)

    with t4:
        st.caption("Dataset CLEAN (ideal para KPIs/Modelos).")
        st.dataframe(df_clean.head(200), use_container_width=True)

    with t5:
        st.caption("Dataset ANOMALÍAS (útil para diagnóstico / storytelling).")
        st.dataframe(df_anom.head(200), use_container_width=True)


# =========================
# Ejecutar procesos
# =========================
inv, inv_clean, inv_anom, inv_damage_cols, inv_flag_cols, inv_text_changes = process_inventario(inv_raw)
tx, tx_clean, tx_anom, tx_damage_cols, tx_flag_cols, tx_text_changes = process_transacciones(tx_raw)


# =========================
# Secciones principales
# =========================
st.markdown("## 1) Inventario")
render_dataset_section("Inventario — revisión completa", inv, inv_clean, inv_anom,
                      inv_damage_cols, inv_flag_cols, inv_text_changes, key_prefix="inv")

st.markdown("## 2) Transacciones")
render_dataset_section("Transacciones — revisión completa", tx, tx_clean, tx_anom,
                      tx_damage_cols, tx_flag_cols, tx_text_changes, key_prefix="tx")


# =========================
# 3) JOIN (Inventario_clean ↔ Transacciones_clean)
# =========================
st.markdown("## 3) Join (CLEAN ↔ CLEAN) por SKU_ID")

if "SKU_ID" not in inv_clean.columns or "SKU_ID" not in tx_clean.columns:
    st.error("No puedo hacer el join: falta la columna SKU_ID en Inventario_clean o Transacciones_clean.")
else:
    # normaliza SKU_ID para evitar mismatch por tipo
    inv_join = inv_clean.copy()
    tx_join = tx_clean.copy()
    inv_join["SKU_ID"] = inv_join["SKU_ID"].astype("string").str.strip()
    tx_join["SKU_ID"] = tx_join["SKU_ID"].astype("string").str.strip()

    # left join desde transacciones (para auditar ventas sin inventario)
    joined = tx_join.merge(
        inv_join,
        on="SKU_ID",
        how="left",
        suffixes=("_tx", "_inv"),
        indicator=True
    )

    # flags join
    joined["flag__sku_no_existe_en_inventario"] = (joined["_merge"] == "left_only")
    joined["flag__sku_existente_en_inventario"] = (joined["_merge"] == "both")

    # ejemplo: inconsistencia de categoría desconocida en inventario (si existe)
    if "Categoria_clean" in joined.columns:
        joined["flag__categoria_unknown_en_inventario"] = (joined["Categoria_clean"] == "unknown")
    else:
        joined["flag__categoria_unknown_en_inventario"] = False

    join_flags = [
        "flag__sku_no_existe_en_inventario",
        "flag__sku_existente_en_inventario",
        "flag__categoria_unknown_en_inventario",
    ]

    j1, j2, j3 = st.tabs(["📋 Auditoría Join", "🚩 Flags (ejemplos)", "🧩 Dataset Join"])

    with j1:
        st.dataframe(pd.DataFrame([{
            "Filas tx_clean": len(tx_clean),
            "Filas inv_clean": len(inv_clean),
            "Filas joined": len(joined),
            "SKUs tx_clean únicos": tx_join["SKU_ID"].nunique(dropna=True),
            "SKUs inv_clean únicos": inv_join["SKU_ID"].nunique(dropna=True),
            "Tx sin inventario (left_only)": int(joined["flag__sku_no_existe_en_inventario"].sum()),
        }]), use_container_width=True)

        st.markdown("### Conteo de flags join")
        st.dataframe(joined[join_flags].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)

        with st.expander("👀 Vista previa (join)", expanded=False):
            st.dataframe(joined.head(50), use_container_width=True)

    with j2:
        selected = st.selectbox("Selecciona flag join:", join_flags, key="join_flag_picker")
        flagged = joined[joined[selected] == True].copy()
        st.write(f"Filas con **{selected}**: {len(flagged):,}")
        st.dataframe(flagged.head(200), use_container_width=True)

    with j3:
        st.caption("Join listo para KPIs y análisis cruzado (ventas vs inventario).")
        st.dataframe(joined.head(200), use_container_width=True)


# =========================
# 4) Preparación tercer archivo (feedback)
# =========================
st.markdown("## 4) Preparación para Feedback (siguiente paso)")
if uploaded_fb is None:
    st.info("Cuando quieras, sube el archivo **feedback_clientes_v2.csv** en la barra lateral. Ya está preparado para el siguiente bloque.")
else:
    fb_raw = load_csv(uploaded_fb)
    st.success(f"Feedback cargado ✅ | Filas: {len(fb_raw):,} | Columnas: {len(fb_raw.columns)}")
    with st.expander("👀 Vista previa (Feedback RAW)", expanded=False):
        st.dataframe(fb_raw.head(50), use_container_width=True)
    st.warning("Aún no estamos procesando Feedback. En el siguiente paso lo integramos con la misma estructura (Auditoría / Cambios / Flags / CLEAN / ANOM) y luego hacemos su join.")
