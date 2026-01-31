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
st.title("Challenge 02 — DSS Auditable (Inventario + Transacciones + Feedback + Join)")
st.caption("Estructura uniforme: Auditoría | Cambios texto | Flags (ejemplos) | CLEAN | ANOMALÍAS + Join final.")

# =========================
# Helpers: texto
# =========================
UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null",
    "sin categoria", "sincategoria", "unknown", "sin categoría",
    "---"
}

def normalize_text_keep_unknown(x: str) -> str:
    """
    Normalización automática:
    - trim, lower
    - detecta unknown/???/--- antes de limpiar
    - quita tildes
    - -/_ -> espacio
    - quita símbolos raros
    - colapsa espacios
    """
    if pd.isna(x):
        return np.nan
    raw = str(x).strip()
    if raw == "":
        return np.nan

    raw_lower = raw.lower().strip()

    # placeholders / unknown
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
    Fuzzy automático (si rapidfuzz). Solo aplica si:
    - score >= threshold
    - match es "único" (gap vs 2do score)
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
# Helpers: numéricos/fechas/outliers
# =========================
def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def compute_outlier_flag_iqr(df, col, k=1.5):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].dropna()
    if len(s) < 20:
        return pd.Series(False, index=df.index)
    low, high = iqr_bounds(s, k=k)
    return (df[col] < low) | (df[col] > high)

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
# IO
# =========================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

# =========================
# Sidebar: upload gate
# =========================
st.sidebar.header("📁 Cargar archivos (obligatorio para JOIN final)")
uploaded_inv = st.sidebar.file_uploader("1) inventario_central_v2.csv", type=["csv"], key="up_inv_all")
uploaded_tx  = st.sidebar.file_uploader("2) transacciones_logistica_v2.csv", type=["csv"], key="up_tx_all")
uploaded_fb  = st.sidebar.file_uploader("3) feedback_clientes_v2.csv", type=["csv"], key="up_fb_all")

st.sidebar.divider()
st.sidebar.header("⚙️ Reglas comunes")
send_flags_to_anom_global = st.sidebar.checkbox(
    "Enviar a ANOMALÍAS si tiene cualquier flag (aplica a todo)",
    value=True,
    key="send_flags_global"
)

# =========================
# Si faltan inventario o transacciones, no se hace nada
# =========================
if uploaded_inv is None or uploaded_tx is None:
    st.info("👈 Para empezar, sube **inventario** y **transacciones**. (El feedback es requerido para el JOIN final).")
    st.stop()

inv_raw = load_csv(uploaded_inv)
tx_raw  = load_csv(uploaded_tx)

st.success(f"Inventario cargado ✅ | Filas: {len(inv_raw):,} | Columnas: {len(inv_raw.columns)}")
st.success(f"Transacciones cargadas ✅ | Filas: {len(tx_raw):,} | Columnas: {len(tx_raw.columns)}")

if uploaded_fb is not None:
    fb_raw = load_csv(uploaded_fb)
    st.success(f"Feedback cargado ✅ | Filas: {len(fb_raw):,} | Columnas: {len(fb_raw.columns)}")
else:
    fb_raw = None
    st.warning("Feedback NO cargado aún. Podrás ver Inventario y Transacciones, pero el JOIN final requiere Feedback.")


# =========================
# Render uniforme de secciones
# =========================
def render_section(title, df_raw, df_clean, df_anom, damage_cols, flag_cols, text_changes, key_prefix):
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
        any_shown = False
        for name, (orig, clean, fuzzy_df) in text_changes.items():
            st.markdown(f"#### {name}")
            if orig is None or clean is None:
                st.caption("No aplica / no existe la columna.")
                continue
            any_shown = True
            st.dataframe(changes_report(orig, clean).head(80), use_container_width=True)
            if RAPIDFUZZ_AVAILABLE and isinstance(fuzzy_df, pd.DataFrame) and len(fuzzy_df) > 0:
                st.caption("Sugerencias fuzzy (aplicadas / no aplicadas)")
                st.dataframe(fuzzy_df.head(80), use_container_width=True)
        if not any_shown:
            st.info("No hay columnas de texto normalizadas para mostrar en esta sección.")
        if not RAPIDFUZZ_AVAILABLE:
            st.warning("Fuzzy matching no está activo. Instala rapidfuzz: `pip install rapidfuzz` y agrégalo a requirements.txt.")

    with t3:
        if not flag_cols:
            st.info("No hay flags para inspeccionar.")
        else:
            selected_flag = st.selectbox("Selecciona un flag para ver ejemplos:", flag_cols, key=f"{key_prefix}_flag_picker")
            flagged = df_raw[df_raw[selected_flag] == True].copy()
            st.write(f"Filas con **{selected_flag}**: {len(flagged):,}")
            st.dataframe(flagged.head(200), use_container_width=True)

    with t4:
        st.caption("Dataset CLEAN (ideal para KPIs/modelos).")
        st.dataframe(df_clean.head(200), use_container_width=True)

    with t5:
        st.caption("Dataset ANOMALÍAS (útil para diagnóstico / storytelling).")
        st.dataframe(df_anom.head(200), use_container_width=True)


# =========================
# 1) INVENTARIO
# =========================
def process_inventario(inv_raw: pd.DataFrame):
    st.sidebar.header("🧹 Inventario — controles")

    with st.sidebar.expander("Reglas de ANOMALÍAS / Daños (Inventario)", expanded=True):
        inv_damage_threshold = st.number_input("Inventario: columnas dañadas ≥", 1, 10, 2, key="inv_damage_thr")

    with st.sidebar.expander("Opcional: Stock negativo", expanded=True):
        inv_fix_negative_stock = st.checkbox("Convertir stock negativo a positivo en CLEAN (abs)", value=False, key="inv_fix_stock")
        st.caption("RAW y ANOMALÍAS NO se modifican. Solo CLEAN.")

    with st.sidebar.expander("Imputación (solo CLEAN)", expanded=False):
        inv_impute_lead = st.selectbox("Lead_Time_Dias nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2, key="inv_imp_lead")
        inv_impute_reorder = st.selectbox("Punto_Reorden nulo →", ["No imputar", "Mediana global", "Mediana por categoría"], index=2, key="inv_imp_reorder")

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

    # texto
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

    # tipificación
    for c in ["Stock_Actual", "Costo_Unitario_USD", "Lead_Time_Dias", "Punto_Reorden"]:
        if c in inv.columns:
            inv[c] = to_numeric(inv[c])
    if "Ultima_Revision" in inv.columns:
        inv["Ultima_Revision"] = pd.to_datetime(inv["Ultima_Revision"], errors="coerce")

    damage_cols, flag_cols = [], []

    def add_damage(colname, mask):
        cname = f"damage__{colname}"
        inv[cname] = mask.astype(int)
        damage_cols.append(cname)

    def add_flag(flagname, mask):
        cname = f"flag__{flagname}"
        inv[cname] = mask.astype(bool)
        flag_cols.append(cname)

    # daños + flags
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

    base_anom = inv["damaged_cols_count"] >= int(inv_damage_threshold)
    anom_mask = (base_anom | inv["any_flag"]) if send_flags_to_anom_global else base_anom

    inv_anom = inv[anom_mask].copy()
    inv_clean = inv[~anom_mask].copy()

    # opcional: abs stock
    if inv_fix_negative_stock and "Stock_Actual" in inv_clean.columns:
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
        if inv_impute_lead == "Mediana global":
            inv_clean = global_median_impute(inv_clean, "Lead_Time_Dias")
        elif inv_impute_lead == "Mediana por categoría" and group_cat_col is not None:
            inv_clean = group_median_impute(inv_clean, "Lead_Time_Dias", group_cat_col)

    if "Punto_Reorden" in inv_clean.columns:
        if inv_impute_reorder == "Mediana global":
            inv_clean = global_median_impute(inv_clean, "Punto_Reorden")
        elif inv_impute_reorder == "Mediana por categoría" and group_cat_col is not None:
            inv_clean = group_median_impute(inv_clean, "Punto_Reorden", group_cat_col)

    text_changes = {
        "Categoria": (inv.get("Categoria_original"), inv.get("Categoria_clean"), cat_fuzzy),
        "Bodega_Origen": (inv.get("Bodega_Origen_original"), inv.get("Bodega_Origen_clean"), bod_fuzzy),
    }

    sidebar_desc = [
        "• Normalización texto (trim, lower, sin tildes, guiones→espacio, colapsa espacios).",
        "• `Categoria` y `Bodega_Origen`: diccionario + fuzzy (si rapidfuzz) con umbral alto (0.92) y match único.",
        "• Tipificación numérica: Stock/Costo/Lead/Punto_Reorden.",
        "• Outliers automáticos (IQR k=1.5) en Costo_Unitario_USD y Lead_Time_Dias.",
        f"• ANOMALÍAS si columnas dañadas ≥ {int(inv_damage_threshold)}" + (" o si tiene flags." if send_flags_to_anom_global else "."),
        ("• Opcional: stock negativo → abs() solo en CLEAN." if inv_fix_negative_stock else "• Stock negativo se conserva (solo se marca flag)."),
        f"• Imputación en CLEAN: Lead={inv_impute_lead}; Punto_Reorden={inv_impute_reorder} (marcado con `imputed__*`)."
    ]

    return inv, inv_clean, inv_anom, damage_cols, flag_cols, text_changes, sidebar_desc


# =========================
# 2) TRANSACCIONES
# =========================
def process_transacciones(tx_raw: pd.DataFrame):
    st.sidebar.header("🧹 Transacciones — controles")

    with st.sidebar.expander("Reglas de ANOMALÍAS / Daños (Transacciones)", expanded=True):
        tx_damage_threshold = st.number_input("Transacciones: columnas dañadas ≥", 1, 10, 2, key="tx_damage_thr")

    with st.sidebar.expander("Opcional: ciudad sospechosa", expanded=True):
        tx_strict_city = st.checkbox(
            "Enviar ciudad sospechosa a unknown (recomendado)",
            value=True,
            key="tx_strict_city"
        )
        st.caption("Ej: 'ventas web' en Ciudad_Destino. Si se activa, NO se fuzzy-mapea a una ciudad.")

    tx = tx_raw.copy()

    # tipificación numérica
    for c in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real"]:
        if c in tx.columns:
            tx[c] = to_numeric(tx[c])

    # fecha venta dd/mm/yyyy
    if "Fecha_Venta" in tx.columns:
        tx["Fecha_Venta_dt"] = pd.to_datetime(tx["Fecha_Venta"], errors="coerce", dayfirst=True)
    else:
        tx["Fecha_Venta_dt"] = pd.NaT

    # mapas texto
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

    # ciudad
    if "Ciudad_Destino" in tx.columns:
        tx["Ciudad_Destino_original"] = tx["Ciudad_Destino"].astype("string")
        tx["Ciudad_Destino_norm"] = tx["Ciudad_Destino"].apply(normalize_text_keep_unknown)

        def _is_city_suspicious(v):
            if pd.isna(v) or v == "unknown":
                return False
            parts = set(str(v).split())
            return len(parts.intersection(SUSPICIOUS_CITY_TOKENS)) > 0

        tx["flag__ciudad_sospechosa"] = tx["Ciudad_Destino_norm"].map(_is_city_suspicious).astype(bool)
        if tx_strict_city:
            tx.loc[tx["flag__ciudad_sospechosa"], "Ciudad_Destino_norm"] = "unknown"

        tx["Ciudad_Destino_clean"] = apply_manual_map(tx["Ciudad_Destino_norm"], CITY_MAP)
        canonical = build_canonical_values(tx["Ciudad_Destino_clean"])
        tx["Ciudad_Destino_clean"], tx_city_fuzzy = fuzzy_map_unique(tx["Ciudad_Destino_clean"], canonical, 0.92, 0.03)

    # estado
    if "Estado_Envio" in tx.columns:
        tx["Estado_Envio_original"] = tx["Estado_Envio"].astype("string")
        tx["Estado_Envio_clean"] = tx["Estado_Envio"].apply(normalize_text_keep_unknown)
        tx["Estado_Envio_clean"] = apply_manual_map(tx["Estado_Envio_clean"], STATUS_MAP)
        canonical = build_canonical_values(tx["Estado_Envio_clean"])
        tx["Estado_Envio_clean"], tx_status_fuzzy = fuzzy_map_unique(tx["Estado_Envio_clean"], canonical, 0.92, 0.03)

    # canal
    if "Canal_Venta" in tx.columns:
        tx["Canal_Venta_original"] = tx["Canal_Venta"].astype("string")
        tx["Canal_Venta_clean"] = tx["Canal_Venta"].apply(normalize_text_keep_unknown)
        norm_map = {normalize_text_keep_unknown(k): v for k, v in CANAL_MAP.items()}
        norm_map.update({"tienda fisica": "tienda"})
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], norm_map)
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], tx_canal_fuzzy = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)

    damage_cols, flag_cols = [], []

    def add_damage(colname, mask):
        cname = f"damage__{colname}"
        tx[cname] = mask.astype(int)
        damage_cols.append(cname)

    def add_flag(flagname, mask):
        cname = f"flag__{flagname}"
        tx[cname] = mask.astype(bool)
        flag_cols.append(cname)

    # daños (campos clave)
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

    # flags
    if "Fecha_Venta" in tx.columns:
        add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())
        today = pd.Timestamp.today().normalize()
        add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

    if "Cantidad_Vendida" in tx.columns:
        add_flag("cantidad_no_positiva", tx["Cantidad_Vendida"].notna() & (tx["Cantidad_Vendida"] <= 0))

    if "Precio_Venta_Final" in tx.columns:
        add_flag("precio_no_positivo", tx["Precio_Venta_Final"].notna() & (tx["Precio_Venta_Final"] <= 0))

    if "Costo_Envio" in tx.columns:
        add_flag("costo_nulo", tx["Costo_Envio"].isna())
        add_flag("costo_no_positivo", tx["Costo_Envio"].notna() & (tx["Costo_Envio"] <= 0))

    if "Tiempo_Entrega_Real" in tx.columns:
        add_flag("tiempo_negativo", tx["Tiempo_Entrega_Real"].notna() & (tx["Tiempo_Entrega_Real"] < 0))
        add_flag("tiempo_outlier_iqr", compute_outlier_flag_iqr(tx, "Tiempo_Entrega_Real", k=1.5))

    if "Ciudad_Destino_clean" in tx.columns:
        add_flag("ciudad_unknown", tx["Ciudad_Destino_clean"] == "unknown")
    if "Estado_Envio_clean" in tx.columns:
        add_flag("estado_unknown", tx["Estado_Envio_clean"] == "unknown")
    if "Canal_Venta_clean" in tx.columns:
        add_flag("canal_unknown", tx["Canal_Venta_clean"] == "unknown")

    # ciudad sospechosa ya existe
    if "flag__ciudad_sospechosa" in tx.columns:
        flag_cols.append("flag__ciudad_sospechosa")

    tx["damaged_cols_count"] = tx[damage_cols].sum(axis=1) if damage_cols else 0
    tx["any_flag"] = tx[flag_cols].any(axis=1) if flag_cols else False

    base_anom = tx["damaged_cols_count"] >= int(tx_damage_threshold)
    anom_mask = (base_anom | tx["any_flag"]) if send_flags_to_anom_global else base_anom

    tx_anom = tx[anom_mask].copy()
    tx_clean = tx[~anom_mask].copy()

    text_changes = {
        "Ciudad_Destino": (tx.get("Ciudad_Destino_original"), tx.get("Ciudad_Destino_clean"), tx_city_fuzzy),
        "Estado_Envio": (tx.get("Estado_Envio_original"), tx.get("Estado_Envio_clean"), tx_status_fuzzy),
        "Canal_Venta": (tx.get("Canal_Venta_original"), tx.get("Canal_Venta_clean"), tx_canal_fuzzy),
    }

    sidebar_desc = [
        "• Parseo fecha: `Fecha_Venta` como dd/mm/yyyy (dayfirst=True).",
        "• Tipificación numérica: Cantidad/Precio/Costo/Tiempo.",
        "• Normalización texto (trim, lower, sin tildes, guiones→espacio).",
        "• Diccionario + fuzzy (umbral 0.92, match único) en Ciudad/Estado/Canal.",
        ("• Regla ciudad sospechosa activa: si parece canal ('ventas web', etc.) → unknown (y se marca flag)." if tx_strict_city
         else "• Regla ciudad sospechosa desactivada: se permite fuzzy-mapping incluso si parece canal (no recomendado)."),
        "• Outliers automáticos IQR (k=1.5) en Tiempo_Entrega_Real.",
        f"• ANOMALÍAS si columnas dañadas ≥ {int(tx_damage_threshold)}" + (" o si tiene flags." if send_flags_to_anom_global else "."),
    ]

    return tx, tx_clean, tx_anom, damage_cols, flag_cols, text_changes, sidebar_desc


# =========================
# 3) FEEDBACK
# =========================
def process_feedback(fb_raw: pd.DataFrame):
    st.sidebar.header("🧹 Feedback — controles")

    with st.sidebar.expander("Decisión clave: duplicados / granularidad", expanded=True):
        fb_strategy = st.selectbox(
            "¿Cómo quieres manejar múltiples feedbacks por Transaccion_ID?",
            [
                "Agregar por Transaccion_ID (recomendado para JOIN 1:1)",
                "Mantener 1:N (no recomendado para KPIs)",
                "Deduplicar por Feedback_ID (si existe)",
            ],
            index=0,
            key="fb_strategy"
        )

    with st.sidebar.expander("Opciones de corrección (checklist)", expanded=True):
        fb_fix_rating = st.checkbox("Rating fuera de rango → NaN (marcar flag)", value=True, key="fb_fix_rating")
        fb_fix_age = st.checkbox("Edad fuera de rango → NaN (marcar flag)", value=True, key="fb_fix_age")
        fb_round_nps = st.checkbox("NPS float → redondear a entero", value=True, key="fb_round_nps")
        fb_placeholder_comment = st.checkbox("Comentario placeholder ('---') → NaN", value=True, key="fb_placeholder_comment")
        fb_normalize_recom = st.checkbox("Normalizar Recomienda_Marca (sí/no/maybe/unknown)", value=True, key="fb_norm_recom")
        fb_normalize_ticket = st.checkbox("Normalizar Ticket_Soporte_Abierto a booleano", value=True, key="fb_norm_ticket")

    with st.sidebar.expander("Imputación opcional (avanzado)", expanded=False):
        fb_impute_recom = st.checkbox(
            "Imputar Recomienda_Marca basado en reglas (NPS/ratings)",
            value=False,
            key="fb_imp_recom"
        )
        st.caption("Si se activa, se marca `imputed__Recomienda_Marca=True`.")

    fb = fb_raw.copy()

    # Detectar columnas (robusto)
    # (usamos nombres esperados; si cambian, se pueden ampliar)
    col_fid = "Feedback_ID" if "Feedback_ID" in fb.columns else None
    col_tid = "Transaccion_ID" if "Transaccion_ID" in fb.columns else None
    col_rating_prod = "Rating_Producto" if "Rating_Producto" in fb.columns else None
    col_rating_log = "Rating_Logistica" if "Rating_Logistica" in fb.columns else None
    col_nps = "Satisfaccion_NPS" if "Satisfaccion_NPS" in fb.columns else None
    col_age = "Edad_Cliente" if "Edad_Cliente" in fb.columns else None
    col_comment = "Comentario_Texto" if "Comentario_Texto" in fb.columns else None
    col_recom = "Recomienda_Marca" if "Recomienda_Marca" in fb.columns else None
    col_ticket = "Ticket_Soporte_Abierto" if "Ticket_Soporte_Abierto" in fb.columns else None

    # Tipificación numérica
    for c in [col_rating_prod, col_rating_log, col_nps, col_age]:
        if c is not None:
            fb[c] = to_numeric(fb[c])

    # Texto: comentario/recomienda/ticket (normaliza)
    # Comentario: mantener texto, pero limpieza placeholder
    if col_comment is not None:
        fb["Comentario_Texto_original"] = fb[col_comment].astype("string")
        fb["Comentario_Texto_norm"] = fb[col_comment].astype("string").str.strip()

        if fb_placeholder_comment:
            # placeholder '---' o vacío a NaN
            fb.loc[fb["Comentario_Texto_norm"].isin(["---", "—", "-", ""]), "Comentario_Texto_norm"] = np.nan

    # Recomienda
    if col_recom is not None and fb_normalize_recom:
        fb["Recomienda_Marca_original"] = fb[col_recom].astype("string")
        norm = fb[col_recom].apply(normalize_text_keep_unknown)

        # mapeo básico
        REC_MAP = {
            "si": "yes", "sí": "yes", "s": "yes", "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
            "no": "no", "n": "no", "0": "no", "false": "no",
            "maybe": "maybe", "quizas": "maybe", "quizá": "maybe",
            "unknown": "unknown",
        }
        # norm ya quitó tildes, entonces "sí" → "si"
        norm = norm.map(lambda v: REC_MAP.get(v, v))
        # valores vacíos → unknown
        norm = norm.fillna("unknown")
        fb["Recomienda_Marca_clean"] = norm

    # Ticket soporte
    if col_ticket is not None and fb_normalize_ticket:
        fb["Ticket_Soporte_original"] = fb[col_ticket].astype("string")
        tnorm = fb[col_ticket].apply(normalize_text_keep_unknown)

        def _to_bool(v):
            if pd.isna(v) or v == "unknown":
                return np.nan
            if v in {"1", "si", "yes", "true"}:
                return True
            if v in {"0", "no", "false"}:
                return False
            return np.nan

        fb["Ticket_Soporte_bool"] = tnorm.map(_to_bool)

    # Flags y daños
    damage_cols, flag_cols = [], []

    def add_damage(colname, mask):
        cname = f"damage__{colname}"
        fb[cname] = mask.astype(int)
        damage_cols.append(cname)

    def add_flag(flagname, mask):
        cname = f"flag__{flagname}"
        fb[cname] = mask.astype(bool)
        flag_cols.append(cname)

    # daños: IDs nulos
    if col_tid is not None:
        add_damage("Transaccion_ID", fb[col_tid].isna())
    else:
        add_damage("Transaccion_ID", pd.Series(True, index=fb.index))

    if col_fid is not None:
        add_damage("Feedback_ID", fb[col_fid].isna())

    # duplicados
    if col_fid is not None:
        add_flag("dup_feedback_id", fb[col_fid].notna() & fb[col_fid].duplicated(keep=False))
    if col_tid is not None:
        add_flag("dup_transaccion_id", fb[col_tid].notna() & fb[col_tid].duplicated(keep=False))

    # rating fuera de rango (asumimos 1-5)
    if col_rating_prod is not None:
        add_damage("Rating_Producto", fb[col_rating_prod].isna())
        add_flag("rating_producto_fuera_rango", fb[col_rating_prod].notna() & ((fb[col_rating_prod] < 1) | (fb[col_rating_prod] > 5)))
        if fb_fix_rating:
            mask_bad = fb[col_rating_prod].notna() & ((fb[col_rating_prod] < 1) | (fb[col_rating_prod] > 5))
            fb.loc[mask_bad, col_rating_prod] = np.nan

    if col_rating_log is not None:
        add_damage("Rating_Logistica", fb[col_rating_log].isna())
        add_flag("rating_logistica_fuera_rango", fb[col_rating_log].notna() & ((fb[col_rating_log] < 1) | (fb[col_rating_log] > 5)))
        if fb_fix_rating:
            mask_bad = fb[col_rating_log].notna() & ((fb[col_rating_log] < 1) | (fb[col_rating_log] > 5))
            fb.loc[mask_bad, col_rating_log] = np.nan

    # edad fuera de rango (ej: 18-100)
    if col_age is not None:
        add_damage("Edad_Cliente", fb[col_age].isna())
        add_flag("edad_fuera_rango", fb[col_age].notna() & ((fb[col_age] < 18) | (fb[col_age] > 100)))
        if fb_fix_age:
            mask_bad = fb[col_age].notna() & ((fb[col_age] < 18) | (fb[col_age] > 100))
            fb.loc[mask_bad, col_age] = np.nan

    # nps: rango -100 a 100 + redondeo
    if col_nps is not None:
        add_damage("Satisfaccion_NPS", fb[col_nps].isna())
        add_flag("nps_fuera_rango", fb[col_nps].notna() & ((fb[col_nps] < -100) | (fb[col_nps] > 100)))

        if fb_round_nps:
            # redondeo a entero
            was_float = fb[col_nps].notna() & (fb[col_nps] % 1 != 0)
            add_flag("nps_no_entero", was_float)
            fb[col_nps] = fb[col_nps].round(0)

    # comentario faltante / placeholder
    if col_comment is not None:
        add_damage("Comentario_Texto", fb[col_comment].isna())
        if "Comentario_Texto_norm" in fb.columns:
            add_flag("comentario_faltante", fb["Comentario_Texto_norm"].isna())
        else:
            add_flag("comentario_faltante", fb[col_comment].isna())

    # recomienda faltante / maybe
    if col_recom is not None:
        add_damage("Recomienda_Marca", fb[col_recom].isna())
        if "Recomienda_Marca_clean" in fb.columns:
            add_flag("recomienda_faltante", fb["Recomienda_Marca_clean"].isin(["unknown"]))
            add_flag("recomienda_maybe", fb["Recomienda_Marca_clean"].isin(["maybe"]))
        else:
            add_flag("recomienda_faltante", fb[col_recom].isna())

    # ticket inválido
    if col_ticket is not None and "Ticket_Soporte_bool" in fb.columns:
        add_damage("Ticket_Soporte_Abierto", fb[col_ticket].isna())
        add_flag("ticket_invalido", fb[col_ticket].notna() & fb["Ticket_Soporte_bool"].isna())

    # imputación Recomienda_Marca por reglas (opcional)
    if fb_impute_recom and ("Recomienda_Marca_clean" in fb.columns):
        # reglas simples:
        # - si Rating_Producto >=4 y NPS > 0 => yes
        # - si Rating_Producto <=2 o NPS < 0 => no
        # - si no hay info => unknown
        was_unknown = fb["Recomienda_Marca_clean"].isin(["unknown", "maybe"])
        cond_yes = (fb.get(col_rating_prod).notna() & (fb.get(col_rating_prod) >= 4)) & (fb.get(col_nps).notna() & (fb.get(col_nps) > 0))
        cond_no = (fb.get(col_rating_prod).notna() & (fb.get(col_rating_prod) <= 2)) | (fb.get(col_nps).notna() & (fb.get(col_nps) < 0))

        fb["imputed__Recomienda_Marca"] = False
        fb.loc[was_unknown & cond_yes, "Recomienda_Marca_clean"] = "yes"
        fb.loc[was_unknown & cond_no, "Recomienda_Marca_clean"] = "no"
        fb.loc[(was_unknown & (cond_yes | cond_no)), "imputed__Recomienda_Marca"] = True

    # split CLEAN vs ANOM (feedback: usamos daño>=2 como default)
    # (en feedback, lo más importante es consistencia para join)
    with st.sidebar.expander("Reglas de ANOMALÍAS / Daños (Feedback)", expanded=True):
        fb_damage_threshold = st.number_input("Feedback: columnas dañadas ≥", 1, 10, 2, key="fb_damage_thr")

    fb["damaged_cols_count"] = fb[damage_cols].sum(axis=1) if damage_cols else 0
    fb["any_flag"] = fb[flag_cols].any(axis=1) if flag_cols else False

    base_anom = fb["damaged_cols_count"] >= int(fb_damage_threshold)
    anom_mask = (base_anom | fb["any_flag"]) if send_flags_to_anom_global else base_anom

    fb_anom = fb[anom_mask].copy()
    fb_clean = fb[~anom_mask].copy()

    # estrategia duplicados para CLEAN join
    # - Mantener 1:N: no agregamos
    # - Dedup por Feedback_ID: dejar uno (primero)
    # - Agregar por Transaccion_ID: agregamos numéricos y booleanos
    fb_for_join = fb_clean.copy()

    if fb_strategy == "Deduplicar por Feedback_ID (si existe)" and col_fid is not None:
        fb_for_join = fb_for_join.sort_values(by=[col_fid]).drop_duplicates(subset=[col_fid], keep="first")

    if fb_strategy == "Agregar por Transaccion_ID (recomendado para JOIN 1:1)" and col_tid is not None:
        # agregación conservadora:
        # numéricos: mean
        # comenta: conteo de comentarios no nulos
        # recomienda: moda (si existe) si no unknown
        # ticket: any True
        agg = {}

        if col_rating_prod is not None:
            agg[col_rating_prod] = "mean"
        if col_rating_log is not None:
            agg[col_rating_log] = "mean"
        if col_nps is not None:
            agg[col_nps] = "mean"
        if col_age is not None:
            agg[col_age] = "mean"

        # recomienda
        if "Recomienda_Marca_clean" in fb_for_join.columns:
            def mode_or_unknown(x):
                x = x.dropna()
                if len(x) == 0:
                    return "unknown"
                # ignora unknown si hay otras
                x2 = x[x != "unknown"]
                if len(x2) > 0:
                    return x2.mode().iloc[0]
                return x.mode().iloc[0]
            agg["Recomienda_Marca_clean"] = mode_or_unknown

        # ticket
        if "Ticket_Soporte_bool" in fb_for_join.columns:
            agg["Ticket_Soporte_bool"] = lambda x: bool((x == True).any())

        # comentario
        if "Comentario_Texto_norm" in fb_for_join.columns:
            agg["Comentario_no_nulo"] = lambda x: int(x.notna().sum())
            fb_for_join["Comentario_no_nulo"] = fb_for_join["Comentario_Texto_norm"]

        fb_for_join = fb_for_join.groupby(col_tid, dropna=False).agg(agg).reset_index()

        # flags agregados útiles
        if col_nps is not None:
            fb_for_join["segmento_nps"] = pd.cut(
                fb_for_join[col_nps],
                bins=[-np.inf, -0.0001, 0.0001, np.inf],
                labels=["detractor", "pasivo", "promotor"]
            ).astype("string")

    # text changes para UI
    text_changes = {
        "Recomienda_Marca": (fb.get("Recomienda_Marca_original"), fb.get("Recomienda_Marca_clean"), pd.DataFrame()),
        "Ticket_Soporte_Abierto": (fb.get("Ticket_Soporte_original"), fb.get("Ticket_Soporte_bool"), pd.DataFrame()),
        "Comentario_Texto": (fb.get("Comentario_Texto_original"), fb.get("Comentario_Texto_norm"), pd.DataFrame()),
    }

    sidebar_desc = [
        "• Tipificación numérica: ratings, NPS, edad.",
        ("• Ratings fuera de rango (1–5) → NaN (y se marca flag)." if fb_fix_rating else "• Ratings fuera de rango se conservan (solo se marca flag)."),
        ("• Edad fuera de rango (18–100) → NaN (y se marca flag)." if fb_fix_age else "• Edad fuera de rango se conserva (solo se marca flag)."),
        ("• NPS float → redondeo a entero + flag `nps_no_entero`." if fb_round_nps else "• NPS se conserva como float."),
        ("• Comentario placeholder ('---') → NaN." if fb_placeholder_comment else "• Comentario placeholder se conserva."),
        ("• Normaliza Recomienda_Marca a yes/no/maybe/unknown." if fb_normalize_recom else "• Recomienda_Marca no se normaliza."),
        ("• Normaliza Ticket_Soporte_Abierto a booleano." if fb_normalize_ticket else "• Ticket_Soporte_Abierto no se normaliza."),
        (f"• Estrategia para JOIN: {fb_strategy}." if col_tid is not None else "• No existe Transaccion_ID: el JOIN no será posible."),
        f"• ANOMALÍAS si columnas dañadas ≥ {int(fb_damage_threshold)}" + (" o si tiene flags." if send_flags_to_anom_global else "."),
    ]

    return fb, fb_clean, fb_anom, fb_for_join, damage_cols, flag_cols, text_changes, sidebar_desc


# =========================
# Ejecutar procesos
# =========================
inv, inv_clean, inv_anom, inv_damage_cols, inv_flag_cols, inv_text_changes, inv_sidebar_desc = process_inventario(inv_raw)
tx, tx_clean, tx_anom, tx_damage_cols, tx_flag_cols, tx_text_changes, tx_sidebar_desc = process_transacciones(tx_raw)

if fb_raw is not None:
    fb, fb_clean, fb_anom, fb_for_join, fb_damage_cols, fb_flag_cols, fb_text_changes, fb_sidebar_desc = process_feedback(fb_raw)
else:
    fb = fb_clean = fb_anom = fb_for_join = None
    fb_damage_cols = fb_flag_cols = []
    fb_text_changes = {}
    fb_sidebar_desc = ["• Feedback no cargado aún. (Requerido para JOIN final)."]

# =========================
# Sidebar: explicación explícita de limpieza
# =========================
st.sidebar.divider()
st.sidebar.header("🧾 Cómo estamos limpiando (resumen explícito)")

st.sidebar.markdown("### Inventario")
st.sidebar.markdown("\n".join(inv_sidebar_desc))

st.sidebar.markdown("### Transacciones")
st.sidebar.markdown("\n".join(tx_sidebar_desc))

st.sidebar.markdown("### Feedback")
st.sidebar.markdown("\n".join(fb_sidebar_desc))

# =========================
# UI principal: secciones
# =========================
st.markdown("## 1) Inventario")
render_section("Inventario — revisión completa", inv, inv_clean, inv_anom,
              inv_damage_cols, inv_flag_cols, inv_text_changes, key_prefix="inv")

st.markdown("## 2) Transacciones")
render_section("Transacciones — revisión completa", tx, tx_clean, tx_anom,
              tx_damage_cols, tx_flag_cols, tx_text_changes, key_prefix="tx")

st.markdown("## 3) Feedback")
if fb_raw is None:
    st.info("Sube el archivo de **feedback** en la barra izquierda para revisar flags y habilitar el JOIN final.")
else:
    render_section("Feedback — revisión completa", fb, fb_clean, fb_anom,
                  fb_damage_cols, fb_flag_cols, fb_text_changes, key_prefix="fb")

# =========================
# JOIN FINAL
# =========================
st.markdown("## 4) Join final (CLEAN ↔ CLEAN ↔ CLEAN)")

if fb_raw is None:
    st.warning("Para el JOIN final necesitas subir el archivo de Feedback.")
else:
    # Normalizar IDs para join
    invj = inv_clean.copy()
    txj = tx_clean.copy()
    fbj = fb_for_join.copy()  # ya viene en forma compatible con estrategia elegida

    # SKU_ID como string
    if "SKU_ID" in invj.columns:
        invj["SKU_ID"] = invj["SKU_ID"].astype("string").str.strip()
    if "SKU_ID" in txj.columns:
        txj["SKU_ID"] = txj["SKU_ID"].astype("string").str.strip()

    # Transaccion_ID como string
    if "Transaccion_ID" in txj.columns:
        txj["Transaccion_ID"] = txj["Transaccion_ID"].astype("string").str.strip()
    if "Transaccion_ID" in fbj.columns:
        fbj["Transaccion_ID"] = fbj["Transaccion_ID"].astype("string").str.strip()

    if "SKU_ID" not in invj.columns or "SKU_ID" not in txj.columns:
        st.error("No puedo unir Tx↔Inv: falta SKU_ID en Inventario_clean o Transacciones_clean.")
        st.stop()

    if "Transaccion_ID" not in txj.columns or "Transaccion_ID" not in fbj.columns:
        st.error("No puedo unir Tx↔Feedback: falta Transaccion_ID en Transacciones_clean o Feedback_clean.")
        st.stop()

    # 1) Tx CLEAN ↔ Inv CLEAN (left desde tx)
    joined_1 = txj.merge(
        invj,
        on="SKU_ID",
        how="left",
        suffixes=("_tx", "_inv"),
        indicator="merge_tx_inv"
    )
    joined_1["flag__sku_no_existe_en_inventario"] = (joined_1["merge_tx_inv"] == "left_only")

    # 2) (Tx+Inv) ↔ Feedback CLEAN (left desde tx)
    joined_all = joined_1.merge(
        fbj,
        on="Transaccion_ID",
        how="left",
        suffixes=("", "_fb"),
        indicator="merge_tx_fb"
    )
    joined_all["flag__sin_feedback"] = (joined_all["merge_tx_fb"] == "left_only")

    # flags de coherencia adicionales (opcionales pero útiles)
    # ejemplo: venta con costo_envio nulo ya viene de tx flags, pero aquí lo dejamos como resumen:
    if "Costo_Envio" in joined_all.columns:
        joined_all["flag__costo_envio_nulo_join"] = joined_all["Costo_Envio"].isna()

    # lista flags join
    join_flags = [c for c in joined_all.columns if c.startswith("flag__") and c in [
        "flag__sku_no_existe_en_inventario",
        "flag__sin_feedback",
        "flag__costo_envio_nulo_join",
    ]]

    j1, j2, j3 = st.tabs(["📋 Auditoría Join", "🚩 Flags (ejemplos)", "🧩 Dataset Join"])

    with j1:
        st.dataframe(pd.DataFrame([{
            "Filas tx_clean": len(tx_clean),
            "Filas inv_clean": len(inv_clean),
            "Filas fb_clean_join": len(fbj),
            "Filas joined_all": len(joined_all),
            "Tx sin inventario": int(joined_all["flag__sku_no_existe_en_inventario"].sum()),
            "Tx sin feedback": int(joined_all["flag__sin_feedback"].sum()),
        }]), use_container_width=True)

        st.markdown("### Conteo de flags (JOIN)")
        if join_flags:
            st.dataframe(joined_all[join_flags].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)
        else:
            st.info("No se generaron flags de join (revisa condiciones).")

        with st.expander("👀 Vista previa (JOIN)", expanded=False):
            st.dataframe(joined_all.head(50), use_container_width=True)

    with j2:
        if not join_flags:
            st.info("No hay flags en join para mostrar ejemplos.")
        else:
            sel = st.selectbox("Selecciona flag join:", join_flags, key="join_flag_picker")
            flagged = joined_all[joined_all[sel] == True].copy()
            st.write(f"Filas con **{sel}**: {len(flagged):,}")
            st.dataframe(flagged.head(200), use_container_width=True)

    with j3:
        st.caption("Join final listo para KPIs (ventas vs inventario vs satisfacción).")
        st.dataframe(joined_all.head(200), use_container_width=True)
