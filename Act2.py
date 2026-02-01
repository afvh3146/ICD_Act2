import streamlit as st
import pandas as pd
import numpy as np
import re
import unicodedata

# =========================
# Fuzzy matching (si disponible)
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
st.caption("Flags NO eliminan por defecto. Excluyes solo si lo seleccionas y presionas el botón de aplicar.")

# =========================
# Helpers
# =========================
def dedupe_keep_order(cols):
    """Elimina duplicados preservando el primer orden de aparición."""
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def safe_for_streamlit_df(df: pd.DataFrame) -> pd.DataFrame:
    """Última defensa: si hay columnas duplicadas, se queda con la primera."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df

UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null", "unknown",
    "sin categoria", "sincategoria", "sin categoría",
    "---", "—", "-"
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

def to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def iqr_bounds(series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def outlier_flag_iqr(df, col, k=1.5):
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    s = pd.to_numeric(df[col], errors="coerce")
    base = s.dropna()

    if len(base) < 20:
        return pd.Series(False, index=df.index)

    low, high = iqr_bounds(base, k=k)
    mask = s.notna() & ((s < low) | (s > high))
    return mask.fillna(False)

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def apply_exclusions_button(df, flag_cols, default_selected, key_prefix, help_text=None):
    """
    - Checkboxes dentro de expander (colapsado)
    - NO aplica hasta que presiona botón
    - Retorna df_final + flags aplicadas
    """
    state_key = f"{key_prefix}_applied_flags"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    with st.sidebar.expander(f"🧰 {key_prefix}: excluir por flags (opcional)", expanded=False):
        if help_text:
            st.caption(help_text)

        selected = []
        for fc in flag_cols:
            pre = fc in default_selected
            if st.checkbox(f"Excluir si {fc}", value=pre, key=f"{key_prefix}_ex_{fc}"):
                selected.append(fc)

        applied = st.button(f"✅ Aplicar exclusiones — {key_prefix}", key=f"{key_prefix}_apply_btn")

        if applied:
            st.session_state[state_key] = selected

        applied_flags = st.session_state[state_key]

        if applied_flags:
            st.info(
                f"Exclusiones activas ({len(applied_flags)}): "
                + ", ".join(applied_flags[:8])
                + ("..." if len(applied_flags) > 8 else "")
            )
        else:
            st.info("Exclusiones activas: ninguna")

    applied_flags = st.session_state[state_key]
    if applied_flags:
        df_final = df[~df[applied_flags].any(axis=1)].copy()
    else:
        df_final = df.copy()

    return df_final, applied_flags, False

# =========================
# Sidebar: upload
# =========================
st.sidebar.header("📁 Cargar archivos")
uploaded_inv = st.sidebar.file_uploader("1) inventario_central_v2.csv", type=["csv"], key="up_inv")
uploaded_tx  = st.sidebar.file_uploader("2) transacciones_logistica_v2.csv", type=["csv"], key="up_tx")
uploaded_fb  = st.sidebar.file_uploader("3) feedback_clientes_v2.csv", type=["csv"], key="up_fb")

if uploaded_inv is None or uploaded_tx is None or uploaded_fb is None:
    st.info("👈 Sube los 3 archivos para habilitar el JOIN y dejar todo listo para análisis.")
    st.stop()

inv_raw = load_csv(uploaded_inv)
tx_raw  = load_csv(uploaded_tx)
fb_raw  = load_csv(uploaded_fb)

st.success(f"Inventario cargado ✅ | {len(inv_raw):,} filas")
st.success(f"Transacciones cargadas ✅ | {len(tx_raw):,} filas")
st.success(f"Feedback cargado ✅ | {len(fb_raw):,} filas")

# =====================================================
# Configuración agrupada (expander padre)
# =====================================================
st.sidebar.divider()

with st.sidebar.expander("⚙️ Configuración de la limpieza de datos", expanded=False):
    st.caption("Configura opciones por base. Los datasets no se eliminan por flags a menos que lo apliques.")

    # contenedores para ubicar widgets dentro de este expander padre
    inv_cfg = st.container()
    tx_cfg = st.container()
    fb_cfg = st.container()
    join_cfg = st.container()
    doc_cfg = st.container()

# =========================
# INVENTARIO
# =========================
def process_inventario(df_raw: pd.DataFrame, cfg_container):
    with cfg_container:
        st.markdown("#### 📦 Inventario — opciones")
        fix_stock_abs = st.checkbox(
            "Stock: convertir negativo a positivo (abs) — opcional",
            value=False,
            key="inv_fix_abs"
        )

    inv = df_raw.copy()

    if "Categoria" in inv.columns:
        inv["Categoria_original"] = inv["Categoria"].astype("string")
    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")

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

    if "Categoria" in inv.columns:
        inv["Categoria_clean"] = inv["Categoria"].apply(normalize_text_keep_unknown)
        inv["Categoria_clean"] = apply_manual_map(inv["Categoria_clean"], CATEGORY_MAP)
        canonical = build_canonical_values(inv["Categoria_clean"])
        inv["Categoria_clean"], cat_fuzzy = fuzzy_map_unique(inv["Categoria_clean"], canonical, 0.92, 0.03)

    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
        inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)
        canonical = build_canonical_values(inv["Bodega_Origen_clean"])
        inv["Bodega_Origen_clean"], bod_fuzzy = fuzzy_map_unique(inv["Bodega_Origen_clean"], canonical, 0.92, 0.03)

    for c in ["Stock_Actual", "Costo_Unitario_USD", "Lead_Time_Dias", "Punto_Reorden"]:
        if c in inv.columns:
            inv[c] = to_numeric(inv[c])

    if "Ultima_Revision" in inv.columns:
        inv["Ultima_Revision"] = pd.to_datetime(inv["Ultima_Revision"], errors="coerce")

    flag_cols = []

    def add_flag(name, mask):
        cname = f"flag__{name}"
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=inv.index)
        inv[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)

    if "SKU_ID" in inv.columns:
        add_flag("sku_id_nulo", inv["SKU_ID"].isna())

    if "Stock_Actual" in inv.columns:
        add_flag("stock_nulo", inv["Stock_Actual"].isna())
        add_flag("stock_negativo", inv["Stock_Actual"].notna() & (inv["Stock_Actual"] < 0))

    if "Costo_Unitario_USD" in inv.columns:
        add_flag("costo_nulo", inv["Costo_Unitario_USD"].isna())
        add_flag("costo_no_positivo", inv["Costo_Unitario_USD"].notna() & (inv["Costo_Unitario_USD"] <= 0))
        add_flag("costo_outlier_iqr", outlier_flag_iqr(inv, "Costo_Unitario_USD", k=1.5))

    if "Lead_Time_Dias" in inv.columns:
        add_flag("leadtime_nulo", inv["Lead_Time_Dias"].isna())
        add_flag("leadtime_negativo", inv["Lead_Time_Dias"].notna() & (inv["Lead_Time_Dias"] < 0))
        add_flag("leadtime_outlier_iqr", outlier_flag_iqr(inv, "Lead_Time_Dias", k=1.5))

    if "Categoria_clean" in inv.columns:
        add_flag("categoria_nula", inv["Categoria_clean"].isna())
        add_flag("categoria_unknown", (inv["Categoria_clean"].astype("string") == "unknown").fillna(False))

    if "Bodega_Origen_clean" in inv.columns:
        add_flag("bodega_nula", inv["Bodega_Origen_clean"].isna())
        add_flag("bodega_unknown", (inv["Bodega_Origen_clean"].astype("string") == "unknown").fillna(False))

    inv["has_any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False

    if fix_stock_abs and "Stock_Actual" in inv.columns:
        m = inv["Stock_Actual"].notna() & (inv["Stock_Actual"] < 0)
        inv.loc[m, "Stock_Actual"] = inv.loc[m, "Stock_Actual"].abs()
        inv["fix__stock_abs_applied"] = m.fillna(False).astype(bool)
    else:
        inv["fix__stock_abs_applied"] = False

    inv_rare = inv[inv["has_any_flag"]].copy()

    default_exclude = {"flag__costo_outlier_iqr", "flag__leadtime_outlier_iqr"}
    inv_final, _, _ = apply_exclusions_button(
        inv, flag_cols, default_exclude, "Inventario",
        help_text="Por defecto: outliers IQR están preseleccionados para excluir (pero NO se aplica hasta que presionas el botón)."
    )

    desc = [
        "Normalización texto automática (trim/lower/sin tildes/guiones→espacio/colapsa espacios).",
        "Diccionario + fuzzy (0.92, match único) en Categoria y Bodega_Origen (si rapidfuzz).",
        "Tipificación numérica (Stock/Costo/Lead/Punto) y fecha (Ultima_Revision).",
        "Flags se calculan; NO eliminan por defecto.",
        "Stock: opción de abs() si negativo (checkbox).",
        "Outliers IQR (k=1.5): se marcan con flag y vienen preseleccionados para excluir (pero solo al aplicar).",
    ]

    return df_raw, inv, inv_final, inv_rare, flag_cols, desc

# =========================
# TRANSACCIONES
# =========================
def process_transacciones(df_raw: pd.DataFrame, cfg_container):
    with cfg_container:
        st.markdown("#### 🚚 Transacciones — opciones")
        strict_city = st.checkbox(
            "Ciudad desconocida/sospechosa → unknown (recomendado)",
            value=True,
            key="tx_strict_city"
        )
        fix_future_year = st.checkbox(
            "Venta futura: si año==2026 → cambiar a 2025 (checkbox de corrección)",
            value=False,
            key="tx_fix_future_year"
        )

    tx = df_raw.copy()

    for c in ["Ciudad_Destino", "Estado_Envio", "Canal_Venta"]:
        if c in tx.columns:
            tx[f"{c}_original"] = tx[c].astype("string")

    for c in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real"]:
        if c in tx.columns:
            tx[c] = to_numeric(tx[c])

    if "Fecha_Venta" in tx.columns:
        tx["Fecha_Venta_dt"] = pd.to_datetime(tx["Fecha_Venta"], errors="coerce", dayfirst=True)
    else:
        tx["Fecha_Venta_dt"] = pd.NaT

    CITY_MAP = {
        "bog": "bogota", "bogota": "bogota",
        "med": "medellin", "mde": "medellin", "medellin": "medellin",
        "cali": "cali", "cartagena": "cartagena",
        "barranquilla": "barranquilla", "bucaramanga": "bucaramanga",
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

    if "Ciudad_Destino" in tx.columns:
        tx["Ciudad_Destino_norm"] = tx["Ciudad_Destino"].apply(normalize_text_keep_unknown)

        def _is_city_suspicious(v):
            if pd.isna(v) or v == "unknown":
                return False
            parts = set(str(v).split())
            return len(parts.intersection(SUSPICIOUS_CITY_TOKENS)) > 0

        tx["flag__ciudad_sospechosa"] = tx["Ciudad_Destino_norm"].map(_is_city_suspicious).fillna(False).astype(bool)
        if strict_city:
            tx.loc[tx["flag__ciudad_sospechosa"], "Ciudad_Destino_norm"] = "unknown"

        tx["Ciudad_Destino_clean"] = apply_manual_map(tx["Ciudad_Destino_norm"], CITY_MAP)
        canonical = build_canonical_values(tx["Ciudad_Destino_clean"])
        tx["Ciudad_Destino_clean"], _ = fuzzy_map_unique(tx["Ciudad_Destino_clean"], canonical, 0.92, 0.03)

    if "Estado_Envio" in tx.columns:
        tx["Estado_Envio_clean"] = tx["Estado_Envio"].apply(normalize_text_keep_unknown)
        tx["Estado_Envio_clean"] = apply_manual_map(tx["Estado_Envio_clean"], STATUS_MAP)
        canonical = build_canonical_values(tx["Estado_Envio_clean"])
        tx["Estado_Envio_clean"], _ = fuzzy_map_unique(tx["Estado_Envio_clean"], canonical, 0.92, 0.03)

    if "Canal_Venta" in tx.columns:
        tx["Canal_Venta_clean"] = tx["Canal_Venta"].apply(normalize_text_keep_unknown)
        norm_map = {normalize_text_keep_unknown(k): v for k, v in CANAL_MAP.items()}
        norm_map.update({"tienda fisica": "tienda"})
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], norm_map)
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], _ = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)

    flag_cols = []

    def add_flag(name, mask):
        cname = f"flag__{name}"
        if cname in tx.columns:
            if cname not in flag_cols:
                flag_cols.append(cname)
            return
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=tx.index)
        tx[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)

    if "Transaccion_ID" in tx.columns:
        add_flag("transaccion_id_nulo", tx["Transaccion_ID"].isna())
    if "SKU_ID" in tx.columns:
        add_flag("sku_id_nulo", tx["SKU_ID"].isna())

    if "Fecha_Venta" in tx.columns:
        add_flag("fecha_venta_nula", tx["Fecha_Venta"].isna())
        add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())
        today = pd.Timestamp.today().normalize()
        add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

    if "Cantidad_Vendida" in tx.columns:
        add_flag("cantidad_no_positiva", tx["Cantidad_Vendida"].notna() & (tx["Cantidad_Vendida"] <= 0))

    if "Tiempo_Entrega_Real" in tx.columns:
        add_flag("tiempo_negativo", tx["Tiempo_Entrega_Real"].notna() & (tx["Tiempo_Entrega_Real"] < 0))
        add_flag("tiempo_outlier_iqr", outlier_flag_iqr(tx, "Tiempo_Entrega_Real", k=1.5))

    if "Costo_Envio" in tx.columns:
        add_flag("costo_nulo", tx["Costo_Envio"].isna())
        add_flag("costo_no_positivo", tx["Costo_Envio"].notna() & (tx["Costo_Envio"] <= 0))

    if "Ciudad_Destino_clean" in tx.columns:
        add_flag("ciudad_unknown", (tx["Ciudad_Destino_clean"].astype("string") == "unknown").fillna(False))
    if "Estado_Envio_clean" in tx.columns:
        add_flag("estado_unknown", (tx["Estado_Envio_clean"].astype("string") == "unknown").fillna(False))
    if "Canal_Venta_clean" in tx.columns:
        add_flag("canal_unknown", (tx["Canal_Venta_clean"].astype("string") == "unknown").fillna(False))

    if "flag__ciudad_sospechosa" in tx.columns and "flag__ciudad_sospechosa" not in flag_cols:
        flag_cols.append("flag__ciudad_sospechosa")

    tx["has_any_flag"] = tx[flag_cols].any(axis=1) if flag_cols else False

    tx["fix__venta_year_2026_to_2025"] = False
    tx["Fecha_Venta_dt_fixed"] = tx["Fecha_Venta_dt"]

    if fix_future_year and "Fecha_Venta_dt" in tx.columns:
        today = pd.Timestamp.today().normalize()
        if "flag__venta_futura" in tx.columns:
            m = (tx["Fecha_Venta_dt"].notna() &
                 (tx["Fecha_Venta_dt"].dt.year == 2026) &
                 (tx["flag__venta_futura"] == True))
        else:
            m = (tx["Fecha_Venta_dt"].notna() &
                 (tx["Fecha_Venta_dt"].dt.year == 2026) &
                 (tx["Fecha_Venta_dt"] > today))

        def _replace_year(d):
            try:
                return d.replace(year=2025)
            except Exception:
                return d

        tx.loc[m, "Fecha_Venta_dt_fixed"] = tx.loc[m, "Fecha_Venta_dt"].map(_replace_year)
        tx.loc[m, "fix__venta_year_2026_to_2025"] = True

    tx_rare = tx[tx["has_any_flag"]].copy()

    default_exclude = set()
    tx_final, _, _ = apply_exclusions_button(
        tx, flag_cols, default_exclude, "Transacciones",
        help_text="Cantidad negativa y tiempo outlier quedan para revisión (no se excluyen por defecto)."
    )

    desc = [
        "Fecha_Venta se parsea con dayfirst=True (dd/mm/yyyy).",
        "Normalización texto + diccionario + fuzzy (0.92, match único) para Ciudad/Estado/Canal.",
        "Ciudad sospechosa (parece canal) puede forzarse a unknown (checkbox).",
        "Se calculan flags; NO eliminan por defecto.",
        "Venta futura: opción de corregir año 2026→2025 (checkbox), queda trazado en fix__venta_year_2026_to_2025.",
        "Cantidad negativa y tiempo outlier se dejan para revisión (flags).",
    ]

    return df_raw, tx, tx_final, tx_rare, flag_cols, desc

# =========================
# FEEDBACK
# =========================
def process_feedback(df_raw: pd.DataFrame, cfg_container):
    with cfg_container:
        st.markdown("#### 💬 Feedback — opciones")
        fb_strategy = st.selectbox(
            "Estrategia feedback para JOIN por Transaccion_ID",
            ["Agregar por Transaccion_ID (recomendado 1:1)", "Mantener 1:N"],
            index=0,
            key="fb_strategy"
        )
        fb_round_nps = st.checkbox("NPS float → redondear a entero", value=True, key="fb_round_nps")
        fb_placeholder_comment = st.checkbox("Comentario placeholder ('---') → NaN", value=True, key="fb_comment_placeholder")

    fb = df_raw.copy()

    if "Transaccion_ID" in fb.columns:
        fb["Transaccion_ID_original"] = fb["Transaccion_ID"].astype("string")
        fb["Transaccion_ID_clean"] = fb["Transaccion_ID"].astype("string").str.strip()
    else:
        fb["Transaccion_ID_clean"] = pd.Series([np.nan] * len(fb), index=fb.index)

    if "Comentario_Texto" in fb.columns:
        fb["Comentario_Texto_original"] = fb["Comentario_Texto"].astype("string")
    if "Recomienda_Marca" in fb.columns:
        fb["Recomienda_Marca_original"] = fb["Recomienda_Marca"].astype("string")
    if "Ticket_Soporte_Abierto" in fb.columns:
        fb["Ticket_Soporte_original"] = fb["Ticket_Soporte_Abierto"].astype("string")

    for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
        if c in fb.columns:
            fb[c] = to_numeric(fb[c])

    if "Comentario_Texto" in fb.columns:
        fb["Comentario_Texto_clean"] = fb["Comentario_Texto"].astype("string").str.strip()
        if fb_placeholder_comment:
            fb.loc[fb["Comentario_Texto_clean"].isin(["---", "—", "-", ""]), "Comentario_Texto_clean"] = np.nan

    if "Recomienda_Marca" in fb.columns:
        norm = fb["Recomienda_Marca"].apply(normalize_text_keep_unknown)
        REC_MAP = {
            "si": "yes", "s": "yes", "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
            "no": "no", "n": "no", "0": "no", "false": "no",
            "maybe": "maybe", "quizas": "maybe",
            "unknown": "unknown"
        }
        fb["Recomienda_Marca_clean"] = norm.map(lambda v: REC_MAP.get(v, v)).fillna("unknown")

    if "Ticket_Soporte_Abierto" in fb.columns:
        tnorm = fb["Ticket_Soporte_Abierto"].apply(normalize_text_keep_unknown)

        def _to_bool(v):
            if pd.isna(v) or v == "unknown":
                return np.nan
            if v in {"1", "si", "yes", "true"}:
                return True
            if v in {"0", "no", "false"}:
                return False
            return np.nan

        fb["Ticket_Soporte_bool"] = tnorm.map(_to_bool)

    if "Satisfaccion_NPS" in fb.columns:
        if fb_round_nps:
            fb["flag__nps_no_entero"] = (fb["Satisfaccion_NPS"].notna() & (fb["Satisfaccion_NPS"] % 1 != 0)).fillna(False)
            fb["Satisfaccion_NPS"] = fb["Satisfaccion_NPS"].round(0)
        else:
            fb["flag__nps_no_entero"] = False

        def nps_bucket(v):
            if pd.isna(v):
                return np.nan
            if v <= -1:
                return "muy_negativo"
            if v == 0:
                return "neutral"
            if 1 <= v <= 80:
                return "positivo"
            if 81 <= v <= 100:
                return "excelente"
            return "fuera_rango"

        fb["NPS_categoria"] = fb["Satisfaccion_NPS"].map(nps_bucket).astype("string")

    flag_cols = []

    def add_flag(name, mask):
        cname = f"flag__{name}"
        if cname in fb.columns and cname not in flag_cols:
            flag_cols.append(cname)
            return
        if not isinstance(mask, pd.Series):
            mask = pd.Series(mask, index=fb.index)
        fb[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)

    add_flag("transaccion_id_nulo", fb["Transaccion_ID_clean"].isna() | (fb["Transaccion_ID_clean"].astype("string").str.len() == 0))
    add_flag("dup_transaccion_id", fb["Transaccion_ID_clean"].notna() & fb["Transaccion_ID_clean"].duplicated(keep=False))

    if "Feedback_ID" in fb.columns:
        add_flag("dup_feedback_id", fb["Feedback_ID"].notna() & fb["Feedback_ID"].duplicated(keep=False))

    if "Rating_Producto" in fb.columns:
        add_flag("rating_producto_fuera_rango", fb["Rating_Producto"].notna() & ((fb["Rating_Producto"] < 1) | (fb["Rating_Producto"] > 5)))

    if "Rating_Logistica" in fb.columns:
        add_flag("rating_logistica_fuera_rango", fb["Rating_Logistica"].notna() & ((fb["Rating_Logistica"] < 1) | (fb["Rating_Logistica"] > 5)))

    if "Satisfaccion_NPS" in fb.columns:
        add_flag(
            "nps_fuera_rango",
            fb["Satisfaccion_NPS"].notna()
            & ((fb["Satisfaccion_NPS"] < -100) | (fb["Satisfaccion_NPS"] > 100))
        )
        add_flag("nps_categoria_fuera_rango", (fb["NPS_categoria"].astype("string") == "fuera_rango").fillna(False))

    if "Comentario_Texto_clean" in fb.columns:
        add_flag("comentario_faltante", fb["Comentario_Texto_clean"].isna())

    if "Recomienda_Marca_clean" in fb.columns:
        add_flag("recomienda_unknown", fb["Recomienda_Marca_clean"].isin(["unknown"]))
        add_flag("recomienda_maybe", fb["Recomienda_Marca_clean"].isin(["maybe"]))

    if "Ticket_Soporte_bool" in fb.columns and "Ticket_Soporte_Abierto" in fb.columns:
        add_flag("ticket_invalido", fb["Ticket_Soporte_Abierto"].notna() & fb["Ticket_Soporte_bool"].isna())

    fb["has_any_flag"] = fb[flag_cols].any(axis=1) if flag_cols else False
    fb_rare = fb[fb["has_any_flag"]].copy()

    default_exclude = set()
    fb_final, _, _ = apply_exclusions_button(
        fb, flag_cols, default_exclude, "Feedback",
        help_text="Transaccion_ID se preserva completo. No excluimos nada por defecto."
    )

    fb_for_join = fb_final.copy()
    fb_for_join["Transaccion_ID_clean"] = fb_for_join["Transaccion_ID_clean"].astype("string").str.strip()

    if fb_strategy == "Agregar por Transaccion_ID (recomendado 1:1)":
        agg = {}
        for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
            if c in fb_for_join.columns:
                agg[c] = "mean"

        if "NPS_categoria" in fb_for_join.columns:
            def mode_cat(x):
                x = x.dropna()
                return x.mode().iloc[0] if len(x) else np.nan
            agg["NPS_categoria"] = mode_cat

        if "Recomienda_Marca_clean" in fb_for_join.columns:
            def mode_or_unknown(x):
                x = x.dropna()
                if len(x) == 0:
                    return "unknown"
                x2 = x[x != "unknown"]
                if len(x2) > 0:
                    return x2.mode().iloc[0]
                return x.mode().iloc[0]
            agg["Recomienda_Marca_clean"] = mode_or_unknown

        if "Ticket_Soporte_bool" in fb_for_join.columns:
            agg["Ticket_Soporte_bool"] = lambda x: bool((x == True).any())

        if "Comentario_Texto_clean" in fb_for_join.columns:
            fb_for_join["Comentario_no_nulo"] = fb_for_join["Comentario_Texto_clean"]
            agg["Comentario_no_nulo"] = lambda x: int(x.notna().sum())

        fb_for_join = fb_for_join.groupby("Transaccion_ID_clean", dropna=False).agg(agg).reset_index()

    desc = [
        "Transaccion_ID se preserva completo (string + strip), sin recortes.",
        "Normalización de Recomienda_Marca y Ticket_Soporte (boolean).",
        "NPS se redondea opcionalmente y se categoriza en muy_negativo/neutral/positivo/excelente.",
        "Flags se calculan; NO eliminan por defecto. Excluyes solo al aplicar.",
        f"Estrategia de JOIN: {fb_strategy} (1:1 si se agrega).",
    ]

    return df_raw, fb, fb_final, fb_rare, fb_for_join, flag_cols, desc

# =========================
# Ejecutar procesos (con widgets agrupados)
# =========================
inv_raw_out, inv_clean, inv_final, inv_rare, inv_flags, inv_desc = process_inventario(inv_raw, inv_cfg)
tx_raw_out, tx_clean, tx_final, tx_rare, tx_flags, tx_desc = process_transacciones(tx_raw, tx_cfg)
fb_raw_out, fb_clean, fb_final, fb_rare, fb_for_join, fb_flags, fb_desc = process_feedback(fb_raw, fb_cfg)

# =========================
# Config del JOIN dentro del expander padre
# =========================
with join_cfg:
    st.markdown("#### 🔗 Join — opciones post-join")
    enable_city_by_cost = st.checkbox(
        "Después del join: inferir Ciudad por Costo_Envio cuando Ciudad=unknown (si el match es único)",
        value=True,
        key="join_city_by_cost"
    )
    overwrite_unknown_city = st.checkbox(
        "Sobrescribir Ciudad_Destino_clean=unknown con la ciudad inferida",
        value=True,
        key="join_overwrite_city"
    )
    min_support = st.number_input(
        "Soporte mínimo (n) para aceptar inferencia por costo",
        min_value=2, max_value=100, value=5,
        key="join_city_min_support"
    )

# =========================
# Documentación dentro del expander padre (opcional)
# =========================
with doc_cfg:
    st.markdown("#### 🧾 Cómo estamos limpiando (documentación)")
    with st.expander("📦 Inventario — detalle", expanded=False):
        st.markdown("\n".join([f"• {x}" for x in inv_desc]))
    with st.expander("🚚 Transacciones — detalle", expanded=False):
        st.markdown("\n".join([f"• {x}" for x in tx_desc]))
    with st.expander("💬 Feedback — detalle", expanded=False):
        st.markdown("\n".join([f"• {x}" for x in fb_desc]))

# =====================================================
# PANTALLA PRINCIPAL: SOLO JOIN FINAL
# =====================================================
st.header("✅ Dataset final (JOIN) — listo para análisis")

# Normalizar IDs para join
if "SKU_ID" in inv_final.columns:
    invj = inv_final.copy()
    invj["SKU_ID"] = invj["SKU_ID"].astype("string").str.strip()
else:
    st.error("Inventario no tiene SKU_ID.")
    st.stop()

txj = tx_final.copy()
if "SKU_ID" in txj.columns:
    txj["SKU_ID"] = txj["SKU_ID"].astype("string").str.strip()
else:
    st.error("Transacciones no tiene SKU_ID.")
    st.stop()

if "Transaccion_ID" in txj.columns:
    txj["Transaccion_ID"] = txj["Transaccion_ID"].astype("string").str.strip()
else:
    st.error("Transacciones no tiene Transaccion_ID.")
    st.stop()

fbj = fb_for_join.copy()
if "Transaccion_ID_clean" in fbj.columns:
    fbj = fbj.rename(columns={"Transaccion_ID_clean": "Transaccion_ID"})
if "Transaccion_ID" in fbj.columns:
    fbj["Transaccion_ID"] = fbj["Transaccion_ID"].astype("string").str.strip()
else:
    st.error("Feedback join-ready no tiene Transaccion_ID.")
    st.stop()

# Join Tx ↔ Inv
join_tx_inv = txj.merge(invj, on="SKU_ID", how="left", suffixes=("_tx", "_inv"), indicator="merge_tx_inv")
join_tx_inv["flag__sku_no_existe_en_inventario"] = (join_tx_inv["merge_tx_inv"] == "left_only")

# Join (Tx+Inv) ↔ Feedback
joined = join_tx_inv.merge(fbj, on="Transaccion_ID", how="left", indicator="merge_tx_fb")
joined["flag__sin_feedback"] = (joined["merge_tx_fb"] == "left_only")

# Post-join: inferir ciudad por costo_envio
joined["Ciudad_inferida_por_costo"] = np.nan
joined["flag__ciudad_inferida_por_costo"] = False

if enable_city_by_cost and ("Costo_Envio" in joined.columns) and ("Ciudad_Destino_clean" in joined.columns):
    valid = joined[["Costo_Envio", "Ciudad_Destino_clean"]].copy()
    valid = valid[valid["Costo_Envio"].notna()]
    valid = valid[valid["Ciudad_Destino_clean"].notna()]
    valid = valid[valid["Ciudad_Destino_clean"].astype("string") != "unknown"]

    if len(valid) > 0:
        freq = valid.groupby(["Costo_Envio", "Ciudad_Destino_clean"]).size().reset_index(name="n")
        freq = freq.sort_values(["Costo_Envio", "n"], ascending=[True, False])
        top = freq.groupby("Costo_Envio").head(2)

        city_by_cost = {}
        for cost, g in top.groupby("Costo_Envio"):
            g = g.sort_values("n", ascending=False)
            if len(g) == 1:
                if int(g.iloc[0]["n"]) >= int(min_support):
                    city_by_cost[cost] = g.iloc[0]["Ciudad_Destino_clean"]
            else:
                top1 = g.iloc[0]
                top2 = g.iloc[1]
                if int(top1["n"]) >= int(min_support) and int(top1["n"]) > int(top2["n"]):
                    city_by_cost[cost] = top1["Ciudad_Destino_clean"]

        unknown_mask = (joined["Ciudad_Destino_clean"].astype("string") == "unknown").fillna(False)
        cost_mask = joined["Costo_Envio"].notna()
        target_idx = joined[unknown_mask & cost_mask].index

        for idx in target_idx:
            c = joined.at[idx, "Costo_Envio"]
            if c in city_by_cost:
                joined.at[idx, "Ciudad_inferida_por_costo"] = city_by_cost[c]
                joined.at[idx, "flag__ciudad_inferida_por_costo"] = True
                if overwrite_unknown_city:
                    joined.at[idx, "Ciudad_Destino_clean"] = city_by_cost[c]

# UI Join
jtabs = st.tabs(["📋 Auditoría Join", "🚩 Flags Join (ejemplos)", "🧩 Dataset Join", "🧠 Diagnóstico inferencia ciudad"])
join_flags = [c for c in joined.columns if c.startswith("flag__")]

with jtabs[0]:
    st.dataframe(pd.DataFrame([{
        "Tx FINAL filas": len(txj),
        "Inv FINAL filas": len(invj),
        "Fb join filas": len(fbj),
        "JOIN filas": len(joined),
        "Tx sin inventario": int(joined["flag__sku_no_existe_en_inventario"].sum()),
        "Tx sin feedback": int(joined["flag__sin_feedback"].sum()),
        "Ciudad inferida por costo": int(joined["flag__ciudad_inferida_por_costo"].sum()),
    }]), use_container_width=True)

    st.markdown("### Conteo flags Join")
    st.dataframe(joined[join_flags].sum().sort_values(ascending=False).to_frame("conteo"), use_container_width=True)

with jtabs[1]:
    sel = st.selectbox("Selecciona flag del JOIN:", join_flags, key="join_flag_picker")
    flagged = joined[joined[sel] == True].copy()
    st.write(f"Filas con **{sel}**: {len(flagged):,}")
    st.dataframe(flagged.head(200), use_container_width=True)

with jtabs[2]:
    st.caption("Dataset JOIN final listo para KPIs y análisis gráfico.")
    st.dataframe(joined.head(200), use_container_width=True)

with jtabs[3]:
    st.caption("Solo aplica a filas donde Ciudad_Destino_clean era unknown y se intentó inferir por Costo_Envio.")
    cols = [c for c in ["Transaccion_ID", "Costo_Envio", "Ciudad_Destino_clean", "Ciudad_inferida_por_costo", "flag__ciudad_inferida_por_costo"] if c in joined.columns]
    st.dataframe(joined[joined["flag__ciudad_inferida_por_costo"] == True][cols].head(300), use_container_width=True)
