"""
Act2.py -- Streamlit dashboard for Challenge 02 (Data Cleaning, Integration and DSS)
------------------------------------------------------------------------------

This single-file Streamlit application encapsulates the entire data processing
pipeline for the second challenge of your master's programme.  It performs
auditable cleaning for three CSV sources (inventory, transactions and
customer feedback), builds a reproducible join, exposes a health score
before/after view, computes KPIs to answer management questions, and even
offers the option to call an external LLM (Groq) for data-driven
recommendations.  The file is heavily documented so another colleague can
pick it up quickly.  To run locally simply execute:

    streamlit run Act2.py
"""

import os
import re
import unicodedata
import json
from datetime import datetime
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import streamlit as st

# Try to import rapidfuzz for fuzzy matching; fall back gracefully if not
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# Try to import requests for optional Groq API call
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Optional charting lib (Altair) for better visuals (pie/donut/scatter with tooltips)
try:
    import altair as alt  # type: ignore
    ALTAIR_AVAILABLE = True
except Exception:
    ALTAIR_AVAILABLE = False

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null", "unknown",
    "sin categoria", "sincategoria", "sin categoría",
    "---", "—", "-"
}


def dedupe_keep_order(seq: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order."""
    seen, out = set(), []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def safe_for_streamlit_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there are no duplicate columns which confuse Streamlit tables."""
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def normalize_text_keep_unknown(x: Any) -> Any:
    """Normalise text by lowercasing, removing accents and punctuation.

    Unknown tokens (e.g. '???', 'na', etc.) are mapped to the literal string
    ``'unknown'``.  Empty strings and actual NA values propagate as NA.
    """
    if pd.isna(x):
        return np.nan
    raw = str(x).strip()
    if raw == "":
        return np.nan
    raw_lower = raw.lower().strip()
    if raw_lower in UNKNOWN_TOKENS or (len(set(raw_lower)) == 1 and "?" in raw_lower):
        return "unknown"
    x2 = unicodedata.normalize("NFKD", raw_lower).encode("ascii", "ignore").decode("utf-8")
    x2 = re.sub(r"[-_/]+", " ", x2)
    x2 = re.sub(r"[^a-z0-9\s]", "", x2)
    x2 = re.sub(r"\s+", " ", x2).strip()
    if x2 in UNKNOWN_TOKENS:
        return "unknown"
    return x2 or np.nan


def apply_manual_map(series_norm: pd.Series, manual_map: Dict[str, str]) -> pd.Series:
    """Replace values in a normalised series according to a manual mapping."""
    return series_norm.map(lambda v: manual_map.get(v, v))


def build_canonical_values(series_after_manual: pd.Series) -> List[str]:
    """Build a sorted list of canonical values for fuzzy matching."""
    vals = series_after_manual.dropna().astype(str)
    vals = vals[vals != "unknown"]
    return sorted(set(vals.tolist()))


def fuzzy_map_unique(
    series_vals: pd.Series,
    canonical: List[str],
    threshold: float = 0.92,
    delta: float = 0.03,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Apply fuzzy matching to map values to canonical values."""
    cols = ["from", "to", "score", "applied"]
    if (not RAPIDFUZZ_AVAILABLE) or (len(canonical) == 0):
        return series_vals, pd.DataFrame(columns=cols)

    thr, dlt = threshold * 100, delta * 100
    mapped = series_vals.copy()
    changes: List[Dict[str, Any]] = []

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

    changes_df = (
        pd.DataFrame(changes).sort_values(["applied", "score"], ascending=[False, False])
        if changes else pd.DataFrame(columns=cols)
    )
    return mapped, changes_df


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """Compute lower and upper bounds for outlier detection via IQR."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_flag_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    """Return a boolean Series marking outliers in ``col`` according to IQR."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    base = s.dropna()
    if len(base) < 20:
        return pd.Series(False, index=df.index)
    low, high = iqr_bounds(base, k)
    mask = s.notna() & ((s < low) | (s > high))
    return mask.fillna(False)


def compute_health_metrics(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    final_df: pd.DataFrame,
    flags: List[str],
) -> Dict[str, Any]:
    """Compute a health report summarising issues before and after cleaning."""
    report: Dict[str, Any] = {}

    n_raw, n_final = len(raw_df), len(final_df)
    report["rows_raw"] = int(n_raw)
    report["rows_final"] = int(n_final)

    dup_raw = raw_df.duplicated().sum()
    dup_final = final_df.duplicated().sum()
    report["duplicates_raw"] = int(dup_raw)
    report["duplicates_final"] = int(dup_final)

    missing_raw = raw_df.isna().sum().to_dict()
    missing_final = final_df.isna().sum().to_dict()
    report["missing_raw"] = {k: int(v) for k, v in missing_raw.items()}
    report["missing_final"] = {k: int(v) for k, v in missing_final.items()}

    flagged_raw = 0
    flagged_final = 0
    for fc in flags:
        if fc in raw_df.columns:
            flagged_raw += int(raw_df[fc].sum())
        if fc in final_df.columns:
            flagged_final += int(final_df[fc].sum())
    report["flagged_raw"] = int(flagged_raw)
    report["flagged_final"] = int(flagged_final)

    total_missing_raw = sum(missing_raw.values())
    total_missing_final = sum(missing_final.values())
    total_issues_raw = total_missing_raw + flagged_raw
    total_issues_final = total_missing_final + flagged_final

    score_raw = 100 * (1 - (total_issues_raw / (max(1, n_raw) * max(1, raw_df.shape[1]))))
    score_final = 100 * (1 - (total_issues_final / (max(1, n_final) * max(1, final_df.shape[1]))))

    report["health_score_raw"] = round(max(0, score_raw), 2)
    report["health_score_final"] = round(max(0, score_final), 2)
    return report

def _fmt_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return str(x)

def chart_bar(df: pd.DataFrame, x: str, y: str, title: str, height: int = 320):
    st.markdown(f"#### {title}")
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return

    if ALTAIR_AVAILABLE:
        c = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{x}:N", sort="-y", title=x),
                y=alt.Y(f"{y}:Q", title=y),
                tooltip=[x, y],
            )
            .properties(height=height)
        )
        st.altair_chart(c, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])

def chart_scatter(df: pd.DataFrame, x: str, y: str, color: str | None, size: str | None, title: str, height: int = 380):
    st.markdown(f"#### {title}")
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return

    if ALTAIR_AVAILABLE:
        enc = {
            "x": alt.X(f"{x}:Q", title=x),
            "y": alt.Y(f"{y}:Q", title=y),
            "tooltip": list(df.columns),
        }
        if color and color in df.columns:
            enc["color"] = alt.Color(f"{color}:N", title=color)
        if size and size in df.columns:
            enc["size"] = alt.Size(f"{size}:Q", title=size)

        c = alt.Chart(df).mark_circle(opacity=0.7).encode(**enc).properties(height=height)
        st.altair_chart(c, use_container_width=True)
    else:
        # Fallback simple
        st.scatter_chart(df[[x, y]].dropna())

def chart_donut(df: pd.DataFrame, category_col: str, value_col: str, title: str, height: int = 320):
    st.markdown(f"#### {title}")
    if df.empty or category_col not in df.columns or value_col not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return

    if not ALTAIR_AVAILABLE:
        st.info("Altair no está disponible; mostrando tabla en su lugar.")
        st.dataframe(df, use_container_width=True)
        return

    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(f"{category_col}:N", legend=alt.Legend(title=category_col)),
        tooltip=[category_col, value_col],
    )
    donut = base.mark_arc(innerRadius=70).properties(height=height)
    st.altair_chart(donut, use_container_width=True)


# -----------------------------------------------------------------------------
# Data processing functions
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV from an uploaded file.  Caches the result for speed."""
    return pd.read_csv(uploaded_file)


def apply_exclusions_button(
    df: pd.DataFrame,
    flag_cols: List[str],
    default_selected: set,
    key_prefix: str,
    help_text: str | None = None,
) -> Tuple[pd.DataFrame, List[str], bool]:
    """Interactive sidebar for flag exclusions."""
    state_key = f"{key_prefix}_applied_flags"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    with st.sidebar.expander(f"🧰 {key_prefix}: excluir por flags", expanded=False):
        if help_text:
            st.caption(help_text)
        selected: List[str] = []
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
        modified = True
    else:
        df_final = df.copy()
        modified = False

    return df_final, applied_flags, modified


def process_inventario(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    """Clean the inventory dataset and compute flags."""
    with cfg_container:
        st.markdown("#### 📦 Inventario — opciones")
        fix_stock_abs = st.checkbox(
            "Stock: convertir negativo a positivo (abs)", value=False, key="inv_fix_abs"
        )

    inv = df_raw.copy()
    actions: List[str] = []

    if "Categoria" in inv.columns:
        inv["Categoria_original"] = inv["Categoria"].astype("string")
    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")

    CATEGORY_MAP = {
        "laptop": "laptops", "laptops": "laptops", "notebook": "laptops", "notebooks": "laptops",
        "smartphone": "smartphones", "smartphones": "smartphones", "smart phone": "smartphones", "smart phones": "smartphones",
        "tablet": "tablets", "tablets": "tablets",
        "accesorio": "accesorios", "accesorios": "accesorios", "accesories": "accesorios",
        "monitor": "monitores", "monitores": "monitores",
        "unknown": "unknown",
    }
    BODEGA_MAP = {
        "med": "medellin", "mde": "medellin", "medellin": "medellin",
        "bog": "bogota", "bogota": "bogota",
        "norte": "norte", "sur": "sur", "east": "east", "west": "west",
        "unknown": "unknown",
    }

    if "Categoria" in inv.columns:
        inv["Categoria_clean"] = inv["Categoria"].apply(normalize_text_keep_unknown)
        inv["Categoria_clean"] = apply_manual_map(inv["Categoria_clean"], CATEGORY_MAP)
        canonical = build_canonical_values(inv["Categoria_clean"])
        inv["Categoria_clean"], cat_fuzzy = fuzzy_map_unique(inv["Categoria_clean"], canonical, 0.92, 0.03)
        if not cat_fuzzy.empty:
            actions.append("Fuzzy matching aplicado en Categoria (" + str(len(cat_fuzzy[cat_fuzzy["applied"]])) + " reemplazos)")

    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
        inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)
        canonical = build_canonical_values(inv["Bodega_Origen_clean"])
        inv["Bodega_Origen_clean"], bod_fuzzy = fuzzy_map_unique(inv["Bodega_Origen_clean"], canonical, 0.92, 0.03)
        if not bod_fuzzy.empty:
            actions.append("Fuzzy matching aplicado en Bodega_Origen (" + str(len(bod_fuzzy[bod_fuzzy["applied"]])) + " reemplazos)")

    for c in ["Stock_Actual", "Costo_Unitario_USD", "Lead_Time_Dias", "Punto_Reorden"]:
        if c in inv.columns:
            inv[c] = to_numeric(inv[c])

    if "Ultima_Revision" in inv.columns:
        inv["Ultima_Revision_dt"] = pd.to_datetime(inv["Ultima_Revision"], errors="coerce")

    flag_cols: List[str] = []

    def add_flag(name: str, mask: pd.Series):
        cname = f"flag__{name}"
        if cname in inv.columns:
            return
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
        add_flag("categoria_unknown", (inv["Categoria_clean"].astype("string") == "unknown"))
    if "Bodega_Origen_clean" in inv.columns:
        add_flag("bodega_nula", inv["Bodega_Origen_clean"].isna())
        add_flag("bodega_unknown", (inv["Bodega_Origen_clean"].astype("string") == "unknown"))

    inv["has_any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False

    inv["fix__stock_abs_applied"] = False
    if fix_stock_abs and "Stock_Actual" in inv.columns:
        m = inv["Stock_Actual"].notna() & (inv["Stock_Actual"] < 0)
        inv.loc[m, "Stock_Actual"] = inv.loc[m, "Stock_Actual"].abs()
        inv.loc[m, "fix__stock_abs_applied"] = True
        actions.append(f"Stock negativo convertido a valor absoluto en {int(m.sum())} filas")

    inv_rare = inv[inv["has_any_flag"]].copy()

    default_exclude = {"flag__costo_outlier_iqr", "flag__leadtime_outlier_iqr"}
    inv_final, applied_flags, _ = apply_exclusions_button(
        inv, flag_cols, default_exclude, "Inventario",
        help_text="Por defecto se preseleccionan outliers de costo y lead time para excluir (requiere aplicar)."
    )
    if applied_flags:
        actions.append("Exclusiones aplicadas en inventario: " + ", ".join(applied_flags))

    desc = [
        "Normalización de texto (lowercase, sin tildes, reemplazo de guiones).",
        "Mapeo manual + fuzzy matching en Categoria y Bodega_Origen.",
        "Conversión de columnas numéricas y de fechas.",
        "Cálculo de flags para nulos, valores negativos, outliers IQR, unknown, etc.",
        "Opción de convertir stock negativo a valor absoluto (no por defecto).",
        "Outliers IQR preseleccionados para excluir mediante botón.",
    ]
    desc.extend(actions)

    return df_raw, inv, inv_final, inv_rare, flag_cols, desc


def process_transacciones(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    """Clean the transactions dataset and compute flags."""
    with cfg_container:
        st.markdown("#### 🚚 Transacciones — opciones")
        strict_city = st.checkbox(
            "Ciudad desconocida/sospechosa → unknown", value=True, key="tx_strict_city"
        )
        fix_future_year = st.checkbox(
            "Venta futura: si año==2026 → cambiar a 2025", value=False, key="tx_fix_future_year"
        )

    tx = df_raw.copy()
    actions: List[str] = []

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
        "cali": "cali", "cartagena": "cartagena", "barranquilla": "barranquilla", "bucaramanga": "bucaramanga",
        "norte": "norte", "sur": "sur", "east": "east", "west": "west",
        "ventas": "unknown", "web": "unknown", "online": "unknown", "app": "unknown", "whatsapp": "unknown",
        "canal": "unknown",
        "unknown": "unknown",
    }
    STATUS_MAP = {
        "entregado": "entregado", "devuelto": "devuelto", "retrasado": "retrasado",
        "en camino": "en_transito", "encamino": "en_transito", "en transito": "en_transito", "transito": "en_transito",
        "perdido": "perdido", "pending": "pendiente", "pendiente": "pendiente", "unknown": "unknown",
    }
    CANAL_MAP = {
        "fisico": "tienda", "físico": "tienda", "fisco": "tienda",
        "tienda fisica": "tienda", "tienda física": "tienda", "tienda": "tienda",
        "online": "web", "web": "web", "ecommerce": "web", "app": "app", "whatsapp": "whatsapp", "unknown": "unknown",
    }

    SUSPICIOUS_CITY_TOKENS = {"ventas", "web", "online", "app", "whatsapp", "canal"}

    if "Ciudad_Destino" in tx.columns:
        tx["Ciudad_Destino_norm"] = tx["Ciudad_Destino"].apply(normalize_text_keep_unknown)

        def _is_city_suspicious(v: Any) -> bool:
            if pd.isna(v) or v == "unknown":
                return False
            parts = set(str(v).split())
            return len(parts.intersection(SUSPICIOUS_CITY_TOKENS)) > 0

        tx["flag__ciudad_sospechosa"] = tx["Ciudad_Destino_norm"].map(_is_city_suspicious).fillna(False)

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
        norm_map.update({"tienda fisica": "tienda", "tienda física": "tienda"})
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], norm_map)
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], _ = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)

    flag_cols: List[str] = []

    def add_flag(name: str, mask: pd.Series):
        cname = f"flag__{name}"
        if cname in tx.columns:
            return
        tx[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)

    # Missing IDs
    if "Transaccion_ID" in tx.columns:
        add_flag("transaccion_id_nulo", tx["Transaccion_ID"].isna())
    if "SKU_ID" in tx.columns:
        add_flag("sku_id_nulo", tx["SKU_ID"].isna())

    # Date flags
    if "Fecha_Venta" in tx.columns:
        add_flag("fecha_venta_nula", tx["Fecha_Venta"].isna())
        add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())

        today = pd.Timestamp.now().normalize()
        add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

        # ✅ Fix opcional: si es futura y año==2026 → cambiar a 2025
        if fix_future_year:
            m_fix = tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today) & (tx["Fecha_Venta_dt"].dt.year == 2026)
            if m_fix.any():
                tx.loc[m_fix, "Fecha_Venta_dt"] = tx.loc[m_fix, "Fecha_Venta_dt"] - pd.DateOffset(years=1)
                actions.append(f"Fechas futuras 2026→2025 corregidas en {int(m_fix.sum())} filas")

    # Numeric flags
    if "Cantidad_Vendida" in tx.columns:
        add_flag("cantidad_no_positiva", tx["Cantidad_Vendida"].notna() & (tx["Cantidad_Vendida"] <= 0))

    if "Tiempo_Entrega_Real" in tx.columns:
        add_flag("tiempo_negativo", tx["Tiempo_Entrega_Real"].notna() & (tx["Tiempo_Entrega_Real"] < 0))
        add_flag("tiempo_outlier_iqr", outlier_flag_iqr(tx, "Tiempo_Entrega_Real", k=1.5))

    if "Costo_Envio" in tx.columns:
        add_flag("costo_nulo", tx["Costo_Envio"].isna())
        add_flag("costo_no_positivo", tx["Costo_Envio"].notna() & (tx["Costo_Envio"] <= 0))

    # Unknown flags
    if "Ciudad_Destino_clean" in tx.columns:
        add_flag("ciudad_unknown", (tx["Ciudad_Destino_clean"].astype("string") == "unknown"))
    if "Estado_Envio_clean" in tx.columns:
        add_flag("estado_unknown", (tx["Estado_Envio_clean"].astype("string") == "unknown"))
    if "Canal_Venta_clean" in tx.columns:
        add_flag("canal_unknown", (tx["Canal_Venta_clean"].astype("string") == "unknown"))

    if "flag__ciudad_sospechosa" in tx.columns and "flag__ciudad_sospechosa" not in flag_cols:
        flag_cols.append("flag__ciudad_sospechosa")

    tx["has_any_flag"] = tx[flag_cols].any(axis=1) if flag_cols else False
    tx_rare = tx[tx["has_any_flag"]].copy()

    default_exclude: set[str] = set()
    tx_final, applied_flags, _ = apply_exclusions_button(
        tx, flag_cols, default_exclude, "Transacciones",
        help_text="Marca flags para excluir y presiona aplicar."
    )
    if applied_flags:
        actions.append("Exclusiones aplicadas en transacciones: " + ", ".join(applied_flags))

    desc = [
        "Fecha_Venta parseada con dayfirst=True (dd/mm/yyyy).",
        "Normalización de texto + mapeo + fuzzy para Ciudad/Estado/Canal.",
        "Detección de ciudades sospechosas (nombres de canal).",
        "Cálculo de flags para fechas, cantidad, tiempos y costos.",
        "Opción de corregir año 2026 a 2025 para ventas futuras (si se activa).",
    ]
    desc.extend(actions)

    return df_raw, tx, tx_final, tx_rare, flag_cols, desc


def process_feedback(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    """Clean the feedback dataset and compute flags."""
    with cfg_container:
        st.markdown("#### 💬 Feedback — opciones")
        fb_strategy = st.selectbox(
            "Estrategia feedback para JOIN por Transaccion_ID",
            ["Agregar por Transaccion_ID (recomendado 1:1)", "Mantener 1:N"],
            index=0,
            key="fb_strategy"
        )
        fb_round_nps = st.checkbox("NPS float → redondear a entero", value=True, key="fb_round_nps")
        fb_placeholder_comment = st.checkbox(
            "Comentario placeholder ('---') → NaN", value=True, key="fb_comment_placeholder"
        )

    fb = df_raw.copy()
    actions: List[str] = []

    if "Transaccion_ID" in fb.columns:
        fb["Transaccion_ID_original"] = fb["Transaccion_ID"].astype("string")
        fb["Transaccion_ID_clean"] = fb["Transaccion_ID"].astype("string").str.strip()
    else:
        fb["Transaccion_ID_clean"] = pd.Series([np.nan] * len(fb), index=fb.index)

    for c in ["Comentario_Texto", "Recomienda_Marca", "Ticket_Soporte_Abierto"]:
        if c in fb.columns:
            fb[f"{c}_original"] = fb[c].astype("string")

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
            "si": "yes", "sí": "yes", "s": "yes", "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
            "no": "no", "n": "no", "0": "no", "false": "no",
            "maybe": "maybe", "quizas": "maybe", "quizás": "maybe",
            "unknown": "unknown",
        }
        fb["Recomienda_Marca_clean"] = norm.map(lambda v: REC_MAP.get(v, v)).fillna("unknown")

    if "Ticket_Soporte_Abierto" in fb.columns:
        tnorm = fb["Ticket_Soporte_Abierto"].apply(normalize_text_keep_unknown)

        def _to_bool(v: Any) -> Any:
            if pd.isna(v) or v == "unknown":
                return np.nan
            if v in {"1", "si", "sí", "yes", "true", "s"}:
                return True
            if v in {"0", "no", "false", "n"}:
                return False
            return np.nan

        fb["Ticket_Soporte_bool"] = tnorm.map(_to_bool)

    if "Satisfaccion_NPS" in fb.columns:
        if fb_round_nps:
            fb["flag__nps_no_entero"] = (
                fb["Satisfaccion_NPS"].notna() & (fb["Satisfaccion_NPS"] % 1 != 0)
            ).fillna(False)
            fb["Satisfaccion_NPS"] = fb["Satisfaccion_NPS"].round(0)
            actions.append("NPS redondeado a entero")
        else:
            fb["flag__nps_no_entero"] = False

        def nps_bucket(v: Any) -> Any:
            if pd.isna(v):
                return np.nan
            try:
                v2 = float(v)
            except Exception:
                return np.nan
            if v2 < 0:
                return "detractor"
            if v2 == 0:
                return "neutral"
            if v2 > 0:
                return "promoter"
            return np.nan

        fb["NPS_categoria"] = fb["Satisfaccion_NPS"].map(nps_bucket).astype("string")

    flag_cols: List[str] = []

    def add_flag(name: str, mask: pd.Series):
        cname = f"flag__{name}"
        if cname in fb.columns:
            return
        fb[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)

    add_flag(
        "transaccion_id_nulo",
        fb["Transaccion_ID_clean"].isna() | (fb["Transaccion_ID_clean"].astype("string").str.len() == 0)
    )
    if "Feedback_ID" in fb.columns:
        add_flag("dup_feedback_id", fb["Feedback_ID"].notna() & fb["Feedback_ID"].duplicated(keep=False))
    add_flag("dup_transaccion_id", fb["Transaccion_ID_clean"].notna() & fb["Transaccion_ID_clean"].duplicated(keep=False))

    if "Rating_Producto" in fb.columns:
        add_flag("rating_producto_fuera_rango", fb["Rating_Producto"].notna() & ((fb["Rating_Producto"] < 1) | (fb["Rating_Producto"] > 5)))
    if "Rating_Logistica" in fb.columns:
        add_flag("rating_logistica_fuera_rango", fb["Rating_Logistica"].notna() & ((fb["Rating_Logistica"] < 1) | (fb["Rating_Logistica"] > 5)))

    if "Satisfaccion_NPS" in fb.columns:
        add_flag("nps_fuera_rango", fb["Satisfaccion_NPS"].notna() & ((fb["Satisfaccion_NPS"] < -100) | (fb["Satisfaccion_NPS"] > 100)))

    if "Comentario_Texto_clean" in fb.columns:
        add_flag("comentario_faltante", fb["Comentario_Texto_clean"].isna())

    if "Recomienda_Marca_clean" in fb.columns:
        add_flag("recomienda_unknown", fb["Recomienda_Marca_clean"].isin(["unknown"]))
        add_flag("recomienda_maybe", fb["Recomienda_Marca_clean"].isin(["maybe"]))

    if "Ticket_Soporte_bool" in fb.columns and "Ticket_Soporte_Abierto" in fb.columns:
        add_flag("ticket_invalido", fb["Ticket_Soporte_Abierto"].notna() & fb["Ticket_Soporte_bool"].isna())

    fb["has_any_flag"] = fb[flag_cols].any(axis=1) if flag_cols else False
    fb_rare = fb[fb["has_any_flag"]].copy()

    default_exclude: set[str] = set()
    fb_final, applied_flags, _ = apply_exclusions_button(
        fb, flag_cols, default_exclude, "Feedback",
        help_text="Marca flags para excluir. No excluimos nada por defecto."
    )
    if applied_flags:
        actions.append("Exclusiones aplicadas en feedback: " + ", ".join(applied_flags))

    fb_for_join = fb_final.copy()
    fb_for_join["Transaccion_ID_clean"] = fb_for_join["Transaccion_ID_clean"].astype("string").str.strip()

    if fb_strategy == "Agregar por Transaccion_ID (recomendado 1:1)":
        agg: Dict[str, Any] = {}
        for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
            if c in fb_for_join.columns:
                agg[c] = "mean"

        if "NPS_categoria" in fb_for_join.columns:
            def mode_cat(x: pd.Series) -> Any:
                x2 = x.dropna()
                return x2.mode().iloc[0] if len(x2) else np.nan
            agg["NPS_categoria"] = mode_cat

        if "Recomienda_Marca_clean" in fb_for_join.columns:
            def mode_or_unknown(x: pd.Series) -> Any:
                x2 = x.dropna()
                if len(x2) == 0:
                    return "unknown"
                x3 = x2[x2 != "unknown"]
                if len(x3) > 0:
                    return x3.mode().iloc[0]
                return x2.mode().iloc[0]
            agg["Recomienda_Marca_clean"] = mode_or_unknown

        if "Ticket_Soporte_bool" in fb_for_join.columns:
            agg["Ticket_Soporte_bool"] = lambda x: bool((x == True).any())

        if "Comentario_Texto_clean" in fb_for_join.columns:
            fb_for_join["Comentario_no_nulo"] = fb_for_join["Comentario_Texto_clean"]
            agg["Comentario_no_nulo"] = lambda x: int(x.notna().sum())

        fb_for_join = fb_for_join.groupby("Transaccion_ID_clean", dropna=False).agg(agg).reset_index()
        actions.append("Feedback agregado por Transaccion_ID (1:1)")

    desc = [
        "Transaccion_ID preservado (string + strip).",
        "Normalización de Recomienda_Marca y Ticket_Soporte a valores consistentes.",
        "NPS redondeado e interpretado en categorías detractor/neutral/promoter.",
        "Cálculo de flags para duplicados, rangos de rating y NPS, comentarios faltantes, etc.",
        f"Estrategia de JOIN: {fb_strategy}.",
    ]
    desc.extend(actions)

    return df_raw, fb, fb_final, fb_rare, fb_for_join, flag_cols, desc


# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------

def build_kpi_cards(df_joined: pd.DataFrame):
    """Display KPI cards summarising key metrics of the joined dataset."""
    df = df_joined.copy()
    total_tx = len(df)
    margin_neg = int(df["Margen_Bruto"].lt(0).sum()) if "Margen_Bruto" in df.columns else 0
    sku_fantasma = int(df["flag__sku_no_existe_en_inventario"].sum()) if "flag__sku_no_existe_en_inventario" in df.columns else 0
    sin_feedback = int(df["flag__sin_feedback"].sum()) if "flag__sin_feedback" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transacciones", f"{total_tx:,}")
    col2.metric("Margen negativo", f"{margin_neg:,}")
    col3.metric("SKU fantasma", f"{sku_fantasma:,}")
    col4.metric("Sin feedback", f"{sin_feedback:,}")


def compute_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for the five management questions."""
    results: Dict[str, Any] = {}
        # --- P1: Margen negativo (vista por categoría) + priorización por SKU ---
    if "Categoria_clean" in df.columns and "Margen_Bruto" in df.columns:
        by_cat = (
            df.groupby("Categoria_clean", dropna=False)["Margen_Bruto"]
            .sum()
            .reset_index()
            .rename(columns={"Margen_Bruto": "Margen_Total"})
            .sort_values("Margen_Total")
        )
        results["margen_por_categoria"] = by_cat

    if "SKU_ID" in df.columns and "Margen_Bruto" in df.columns and "Cantidad_Vendida" in df.columns:
        by_sku = (
            df.groupby("SKU_ID", dropna=False)
            .agg(
                Margen_Total=("Margen_Bruto", "sum"),
                Cantidad_Total=("Cantidad_Vendida", "sum"),
                Ingreso_Total=("Ingreso", "sum") if "Ingreso" in df.columns else ("Cantidad_Vendida", "sum"),
            )
            .reset_index()
        )
        results["sku_scatter_margen_vs_cantidad"] = by_sku

        # --- P3: SKU fantasma por categoría (impacto en ingresos) ---
    if "flag__sku_no_existe_en_inventario" in df.columns and "Ingreso" in df.columns:
        ghost = df[df["flag__sku_no_existe_en_inventario"]].copy()
        if "Categoria_clean" in ghost.columns:
            ghost_by_cat = (
                ghost.groupby("Categoria_clean", dropna=False)["Ingreso"]
                .sum()
                .reset_index()
                .rename(columns={"Ingreso": "Ingreso_Perdido"})
                .sort_values("Ingreso_Perdido", ascending=False)
            )
            results["sku_fantasma_por_categoria"] = ghost_by_cat

        total_ing = float(df["Ingreso"].sum()) if float(df["Ingreso"].sum()) != 0 else 0.0
        lost_ing = float(ghost["Ingreso"].sum())
        donut_df = pd.DataFrame([
            {"Tipo": "Ingreso normal", "Valor": max(total_ing - lost_ing, 0)},
            {"Tipo": "Ingreso en riesgo (SKU fantasma)", "Valor": max(lost_ing, 0)},
        ])
        results["donut_ingreso_riesgo_fantasma"] = donut_df

        # --- P4: Stock vs NPS por categoría (cuadrantes) ---
    if "Stock_Actual" in df.columns and "Satisfaccion_NPS" in df.columns and "Categoria_clean" in df.columns:
        cat_scatter = (
            df.groupby("Categoria_clean", dropna=False)
            .agg(
                Stock_Prom=("Stock_Actual", "mean"),
                NPS_Prom=("Satisfaccion_NPS", "mean"),
                N=("Satisfaccion_NPS", "count"),
            )
            .reset_index()
        )
        results["stock_vs_nps_scatter"] = cat_scatter

        # Cuadrante: alto stock (>= Q75) + bajo NPS (<= 0)
        st_thr = float(df["Stock_Actual"].quantile(0.75))
        red = cat_scatter[(cat_scatter["Stock_Prom"] >= st_thr) & (cat_scatter["NPS_Prom"] <= 0)].copy()
        red = red.sort_values(["NPS_Prom", "Stock_Prom"], ascending=[True, False])
        results["stock_alto_nps_bajo_alerta"] = red

        # --- P5: Riesgo operativo (revisión vs tickets) + NPS por bodega (plus) ---
    if "Bodega_Origen_clean" in df.columns:
        if "Ticket_Soporte_bool" in df.columns:
            b = df.copy()
            b["Ticket_Soporte_bool"] = b["Ticket_Soporte_bool"].fillna(False)

            agg = {
                "Ticket_Soporte_bool": ["count", lambda x: int((x == True).sum())],
            }
            if "Dias_desde_revision" in b.columns:
                agg["Dias_desde_revision"] = "mean"
            if "Satisfaccion_NPS" in b.columns:
                agg["Satisfaccion_NPS"] = "mean"

            tmp = b.groupby("Bodega_Origen_clean", dropna=False).agg(agg)
            tmp.columns = ["_".join([c for c in col if c]).strip() for col in tmp.columns.values]
            tmp = tmp.reset_index()

            tmp = tmp.rename(columns={
                "Ticket_Soporte_bool_count": "Total",
                "Ticket_Soporte_bool_<lambda_0>": "Tickets_Abiertos",
                "Dias_desde_revision_mean": "Dias_Revision_Prom",
                "Satisfaccion_NPS_mean": "NPS_Prom",
            })

            tmp["Ticket_Rate"] = tmp["Tickets_Abiertos"] / tmp["Total"].replace({0: np.nan})
            results["riesgo_bodega_plus"] = tmp.sort_values("Ticket_Rate", ascending=False)
    
    if df.empty:
        return results

    df = df.copy()
    df["Ingreso"] = df.get("Ingreso", 0)
    df["Margen_Bruto"] = df.get("Margen_Bruto", 0)
    df["Cantidad_Vendida"] = df.get("Cantidad_Vendida", 0)

    if "SKU_ID" in df.columns and "Margen_Bruto" in df.columns:
        q1 = df.groupby("SKU_ID")["Margen_Bruto"].sum().reset_index().rename(columns={"Margen_Bruto": "Margen_Total"})
        q1 = q1[q1["Margen_Total"] < 0].sort_values("Margen_Total")
        results["margen_negativo"] = q1

    if ("Tiempo_Entrega_Real" in df.columns and "Satisfaccion_NPS" in df.columns
            and "Ciudad_Destino_clean" in df.columns and "Bodega_Origen_clean" in df.columns):
        df_corr = df[["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Real", "Satisfaccion_NPS"]].dropna()
        corr_table = (
            df_corr.groupby(["Ciudad_Destino_clean", "Bodega_Origen_clean"])
            .agg({"Tiempo_Entrega_Real": "mean", "Satisfaccion_NPS": "mean"})
            .reset_index()
            .rename(columns={"Tiempo_Entrega_Real": "Tiempo_Entrega_Prom", "Satisfaccion_NPS": "NPS_Prom"})
        )
        results["logistica_vs_nps"] = corr_table

    if "flag__sku_no_existe_en_inventario" in df.columns and "Ingreso" in df.columns:
        lost_sales = df[df["flag__sku_no_existe_en_inventario"]]
        total_lost = float(lost_sales["Ingreso"].sum())
        count_lost = int(len(lost_sales))
        total_ing = float(df["Ingreso"].sum()) if float(df["Ingreso"].sum()) != 0 else 0.0
        results["sku_fantasma"] = {
            "total_perdido": total_lost,
            "num_transacciones": count_lost,
            "porcentaje": float(total_lost / total_ing) if total_ing != 0 else 0.0,
        }

    if ("Stock_Actual" in df.columns and "Satisfaccion_NPS" in df.columns and "Categoria_clean" in df.columns):
        stock_threshold = df["Stock_Actual"].quantile(0.75)
        low_nps_df = df[(df["Stock_Actual"] >= stock_threshold) & (df["Satisfaccion_NPS"] <= 0)]
        quality_df = (
            low_nps_df.groupby("Categoria_clean")
            .agg(
                stock_prom=("Stock_Actual", "mean"),
                nps_prom=("Satisfaccion_NPS", "mean"),
                total_items=("Stock_Actual", "count"),
            )
            .reset_index()
        )
        results["stock_vs_nps"] = quality_df

    if ("Ultima_Revision_dt" in df.columns and "Ticket_Soporte_bool" in df.columns and "Bodega_Origen_clean" in df.columns):
        today = pd.Timestamp(datetime.now().date())
        df_oper = df[["Ultima_Revision_dt", "Ticket_Soporte_bool", "Bodega_Origen_clean"]].dropna().copy()
        df_oper["dias_desde_revision"] = (today - df_oper["Ultima_Revision_dt"].dt.floor("D")).dt.days
        risk_table = (
            df_oper.groupby("Bodega_Origen_clean")
            .agg(
                dias_prom=("dias_desde_revision", "mean"),
                tickets_abiertos=("Ticket_Soporte_bool", lambda x: int((x == True).sum())),
                total=("Ticket_Soporte_bool", "count"),
            )
            .reset_index()
        )
        risk_table["ticket_rate"] = risk_table["tickets_abiertos"] / risk_table["total"]
        results["operational_risk"] = risk_table

    return results


def get_groq_api_key() -> str | None:
    """
    Devuelve la GROQ_API_KEY en este orden:
    1) key pegada en el sidebar (st.session_state)
    2) st.secrets["GROQ_API_KEY"]
    3) variable de entorno GROQ_API_KEY
    """
    k = st.session_state.get("groq_api_key_input")
    if k:
        return k

    if hasattr(st, "secrets"):
        k2 = st.secrets.get("GROQ_API_KEY")
        if k2:
            return k2

    return os.environ.get("GROQ_API_KEY")


def call_groq(messages: List[Dict[str, str]]) -> str:
    """Call Groq's OpenAI-compatible chat completions endpoint."""
    api_key = get_groq_api_key()
    if not api_key:
        return (
            "Error: No se encontró la clave GROQ_API_KEY en st.secrets ni en variables de entorno, "
            "y no se pegó una key en el sidebar. Configúrala para usar IA."
        )
    if not REQUESTS_AVAILABLE:
        return "Error: la librería requests no está disponible en este entorno."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    model_id = st.session_state.get("groq_model_id", "llama-3.3-70b-versatile")

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512,
        "top_p": 0.9,
        "stream": False,
    }

    try:
        res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if res.status_code != 200:
            return f"Error Groq {res.status_code}: {res.text}"
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error al llamar a Groq: {e}"


def build_ai_prompt(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Construct a summarised prompt for Groq based on filtered data."""
    messages: List[Dict[str, str]] = []
    messages.append({
        "role": "system",
        "content": (
            "Eres un analista de datos senior que brinda recomendaciones de negocio "
            "basadas en estadísticas resumidas de ventas, inventario y feedback. "
            "Nunca menciones que te dieron un resumen; simplemente responde con "
            "tres párrafos de insights accionables para la gerencia."
        ),
    })

    total_tx = len(df)
    margen_total = float(df.get("Margen_Bruto", pd.Series(dtype=float)).sum())
    margen_neg_count = int((df.get("Margen_Bruto", pd.Series(dtype=float)) < 0).sum())
    sku_fantasma_count = int(df.get("flag__sku_no_existe_en_inventario", pd.Series(dtype=bool)).sum())
    sin_feedback_count = int(df.get("flag__sin_feedback", pd.Series(dtype=bool)).sum())
    avg_nps = float(df.get("Satisfaccion_NPS", pd.Series(dtype=float)).mean()) if "Satisfaccion_NPS" in df.columns else 0.0
    avg_entrega = float(df.get("Tiempo_Entrega_Real", pd.Series(dtype=float)).mean()) if "Tiempo_Entrega_Real" in df.columns else 0.0

    messages.append({
        "role": "user",
        "content": (
            f"Resumen de datos:\n"
            f"- Total de transacciones: {total_tx}\n"
            f"- Margen total (USD): {margen_total:.2f}\n"
            f"- Transacciones con margen negativo: {margen_neg_count}\n"
            f"- SKUs fantasma: {sku_fantasma_count}\n"
            f"- Transacciones sin feedback: {sin_feedback_count}\n"
            f"- NPS promedio: {avg_nps:.2f}\n"
            f"- Tiempo de entrega promedio: {avg_entrega:.2f}\n"
            "Genera tres recomendaciones estratégicas basadas en estos datos."
        ),
    })
    return messages


def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(page_title="Challenge 02 — DSS Auditable", layout="wide")
    st.title("Challenge 02 — DSS Auditable (Inventario + Transacciones + Feedback + Join)")
    st.caption(
        "Utiliza los controles laterales para limpiar los datos de manera auditable, "
        "realizar el JOIN y analizar KPIs. Las exclusiones no se aplican hasta que "
        "presiones los botones correspondientes."
    )

    # Sidebar uploads
    st.sidebar.header("📁 Cargar archivos")
    uploaded_inv = st.sidebar.file_uploader("1) inventario_central_v2.csv", type=["csv"], key="up_inv")
    uploaded_tx = st.sidebar.file_uploader("2) transacciones_logistica_v2.csv", type=["csv"], key="up_tx")
    uploaded_fb = st.sidebar.file_uploader("3) feedback_clientes_v2.csv", type=["csv"], key="up_fb")

    # -------------------------
    # 🧠 IA (Groq) - Sidebar
    # (va ANTES del return por falta de archivos)
    # -------------------------
    st.sidebar.divider()
    st.sidebar.subheader("🧠 IA (Groq)")

    default_key = None
    if hasattr(st, "secrets"):
        default_key = st.secrets.get("GROQ_API_KEY")
    if not default_key:
        default_key = os.environ.get("GROQ_API_KEY")

    if "groq_api_key_input" not in st.session_state:
        st.session_state["groq_api_key_input"] = ""

    if not default_key:
        st.session_state["groq_api_key_input"] = st.sidebar.text_input(
            "Pega tu GROQ_API_KEY (no se guarda)",
            type="password",
            value=st.session_state["groq_api_key_input"],
            help="Se usa solo en esta sesión. Para producción, configura st.secrets o variables de entorno."
        )
        st.sidebar.caption("✅ La clave se usa solo mientras la app está abierta.")
    else:
        st.sidebar.success("GROQ_API_KEY detectada en secrets o variables de entorno.")

    st.session_state["groq_model_id"] = st.sidebar.selectbox(
        "Modelo",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="Si te da error 400, prueba cambiar de modelo."
    )

    if st.sidebar.button("🔌 Probar conexión Groq"):
        if not REQUESTS_AVAILABLE:
            st.sidebar.error("La librería requests no está disponible en este entorno.")
        else:
            k = default_key or st.session_state.get("groq_api_key_input")
            if not k:
                st.sidebar.error("No hay GROQ_API_KEY aún.")
            else:
                try:
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {k}", "Content-Type": "application/json"}
                    payload = {
                        "model": st.session_state["groq_model_id"],
                        "messages": [{"role": "user", "content": "Di 'ok'."}],
                        "temperature": 0,
                        "max_tokens": 10,
                    }
                    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
                    if r.status_code == 200:
                        st.sidebar.success("✅ Conexión OK")
                    else:
                        st.sidebar.error(f"❌ Error {r.status_code}: {r.text}")
                except Exception as e:
                    st.sidebar.error(f"❌ Excepción: {e}")

    # Si faltan archivos, avisar y salir (pero dejando visible IA en sidebar)
    if uploaded_inv is None or uploaded_tx is None or uploaded_fb is None:
        st.info("👈 Sube los 3 archivos para habilitar el JOIN y dejar todo listo para análisis.")
        return

    # Load CSVs
    inv_raw = load_csv(uploaded_inv)
    tx_raw = load_csv(uploaded_tx)
    fb_raw = load_csv(uploaded_fb)

    st.success(f"Inventario cargado ✅ | {len(inv_raw):,} filas")
    st.success(f"Transacciones cargadas ✅ | {len(tx_raw):,} filas")
    st.success(f"Feedback cargado ✅ | {len(fb_raw):,} filas")

    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Configuración de la limpieza de datos", expanded=False):
        st.caption("Configura opciones por base. Los datasets no se eliminan por flags a menos que lo apliques.")
        inv_cfg = st.container()
        tx_cfg = st.container()
        fb_cfg = st.container()
        join_cfg = st.container()
        doc_cfg = st.container()

    inv_raw_out, inv_clean, inv_final, inv_rare, inv_flags, inv_desc = process_inventario(inv_raw, inv_cfg)
    tx_raw_out, tx_clean, tx_final, tx_rare, tx_flags, tx_desc = process_transacciones(tx_raw, tx_cfg)
    fb_raw_out, fb_clean, fb_final, fb_rare, fb_for_join, fb_flags, fb_desc = process_feedback(fb_raw, fb_cfg)

    inv_health = compute_health_metrics(inv_raw_out, inv_clean, inv_final, inv_flags)
    tx_health = compute_health_metrics(tx_raw_out, tx_clean, tx_final, tx_flags)
    fb_health = compute_health_metrics(fb_raw_out, fb_clean, fb_final, fb_flags)

    with join_cfg:
        st.markdown("#### 🔗 Join — opciones post-join")
        enable_city_by_cost = st.checkbox(
            "Inferir Ciudad por Costo_Envio cuando Ciudad=unknown (si el match es único)",
            value=True,
            key="join_city_by_cost",
        )
        overwrite_unknown_city = st.checkbox(
            "Sobrescribir Ciudad_Destino_clean=unknown con la ciudad inferida",
            value=True,
            key="join_overwrite_city",
        )
        min_support = st.number_input(
            "Soporte mínimo (n) para aceptar inferencia por costo",
            min_value=2,
            max_value=100,
            value=5,
            key="join_city_min_support",
        )

    with doc_cfg:
        st.markdown("#### 🧾 Cómo estamos limpiando (documentación)")
        with st.expander("📦 Inventario — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in inv_desc]))
        with st.expander("🚚 Transacciones — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in tx_desc]))
        with st.expander("💬 Feedback — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in fb_desc]))

    st.header("✅ Dataset final (JOIN) — listo para análisis")

    if "SKU_ID" not in inv_final.columns:
        st.error("Inventario no tiene SKU_ID.")
        return
    invj = inv_final.copy()
    invj["SKU_ID"] = invj["SKU_ID"].astype("string").str.strip()

    txj = tx_final.copy()
    if "SKU_ID" not in txj.columns:
        st.error("Transacciones no tiene SKU_ID.")
        return
    txj["SKU_ID"] = txj["SKU_ID"].astype("string").str.strip()

    if "Transaccion_ID" not in txj.columns:
        st.error("Transacciones no tiene Transaccion_ID.")
        return
    txj["Transaccion_ID"] = txj["Transaccion_ID"].astype("string").str.strip()

    fbj = fb_for_join.copy()
    if "Transaccion_ID_clean" in fbj.columns:
        fbj = fbj.rename(columns={"Transaccion_ID_clean": "Transaccion_ID"})
    if "Transaccion_ID" not in fbj.columns:
        st.error("Feedback join-ready no tiene Transaccion_ID.")
        return
    fbj["Transaccion_ID"] = fbj["Transaccion_ID"].astype("string").str.strip()

    join_tx_inv = txj.merge(invj, on="SKU_ID", how="left", suffixes=("_tx", "_inv"), indicator="merge_tx_inv")
    join_tx_inv["flag__sku_no_existe_en_inventario"] = (join_tx_inv["merge_tx_inv"] == "left_only")

    joined = join_tx_inv.merge(fbj, on="Transaccion_ID", how="left", indicator="merge_tx_fb")
    joined["flag__sin_feedback"] = (joined["merge_tx_fb"] == "left_only")

    joined["Ciudad_inferida_por_costo"] = np.nan
    joined["flag__ciudad_inferida_por_costo"] = False

    if enable_city_by_cost and ("Costo_Envio" in joined.columns) and ("Ciudad_Destino_clean" in joined.columns):
        valid = joined[["Costo_Envio", "Ciudad_Destino_clean"]].copy()
        valid = valid[valid["Costo_Envio"].notna()]
        valid = valid[valid["Ciudad_Destino_clean"].notna()]
        valid = valid[valid["Ciudad_Destino_clean"].astype("string") != "unknown"]

        if len(valid) > 0:
            freq = (
                valid.groupby(["Costo_Envio", "Ciudad_Destino_clean"]).size().reset_index(name="n")
                .sort_values(["Costo_Envio", "n"], ascending=[True, False])
            )
            top = freq.groupby("Costo_Envio").head(2)

            city_by_cost: Dict[float, str] = {}
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
            target_idx = joined.index[unknown_mask & cost_mask]

            for idx in target_idx:
                cost_val = joined.at[idx, "Costo_Envio"]
                if cost_val in city_by_cost:
                    joined.at[idx, "Ciudad_inferida_por_costo"] = city_by_cost[cost_val]
                    joined.at[idx, "flag__ciudad_inferida_por_costo"] = True
                    if overwrite_unknown_city:
                        joined.at[idx, "Ciudad_Destino_clean"] = city_by_cost[cost_val]

    joined["Cantidad_Vendida"] = pd.to_numeric(joined.get("Cantidad_Vendida", 0), errors="coerce")
    joined["Precio_Venta_Final"] = pd.to_numeric(joined.get("Precio_Venta_Final", 0), errors="coerce")
    joined["Costo_Unitario_USD"] = pd.to_numeric(joined.get("Costo_Unitario_USD", 0), errors="coerce")
    joined["Costo_Envio"] = pd.to_numeric(joined.get("Costo_Envio", 0), errors="coerce")
    joined["Stock_Actual"] = pd.to_numeric(joined.get("Stock_Actual", 0), errors="coerce")

    joined["Ingreso"] = joined["Cantidad_Vendida"] * joined["Precio_Venta_Final"]
    joined["Costo_producto"] = joined["Cantidad_Vendida"] * joined["Costo_Unitario_USD"]
    joined["Margen_Bruto"] = joined["Ingreso"] - joined["Costo_producto"]
    joined["Margen_Neto_aprox"] = joined["Margen_Bruto"] - joined["Costo_Envio"]

    if "Ultima_Revision_dt" in joined.columns:
        today_dt = pd.Timestamp(datetime.now().date())
        joined["Dias_desde_revision"] = (today_dt - joined["Ultima_Revision_dt"].dt.floor("D")).dt.days

    build_kpi_cards(joined)
    analysis_results = compute_analysis(joined)

    tabs = st.tabs(["🧮 Auditoría", "📊 Operaciones", "😊 Cliente", "🧠 Insights IA"])

    with tabs[0]:
        st.subheader("Auditoría de datos")
        health_df = pd.DataFrame([
            {
                "Dataset": "Inventario",
                "Filas (raw)": inv_health["rows_raw"],
                "Filas (final)": inv_health["rows_final"],
                "Duplicados (raw)": inv_health["duplicates_raw"],
                "Duplicados (final)": inv_health["duplicates_final"],
                "Flags (raw)": inv_health["flagged_raw"],
                "Flags (final)": inv_health["flagged_final"],
                "Health Score (raw)": inv_health["health_score_raw"],
                "Health Score (final)": inv_health["health_score_final"],
            },
            {
                "Dataset": "Transacciones",
                "Filas (raw)": tx_health["rows_raw"],
                "Filas (final)": tx_health["rows_final"],
                "Duplicados (raw)": tx_health["duplicates_raw"],
                "Duplicados (final)": tx_health["duplicates_final"],
                "Flags (raw)": tx_health["flagged_raw"],
                "Flags (final)": tx_health["flagged_final"],
                "Health Score (raw)": tx_health["health_score_raw"],
                "Health Score (final)": tx_health["health_score_final"],
            },
            {
                "Dataset": "Feedback",
                "Filas (raw)": fb_health["rows_raw"],
                "Filas (final)": fb_health["rows_final"],
                "Duplicados (raw)": fb_health["duplicates_raw"],
                "Duplicados (final)": fb_health["duplicates_final"],
                "Flags (raw)": fb_health["flagged_raw"],
                "Flags (final)": fb_health["flagged_final"],
                "Health Score (raw)": fb_health["health_score_raw"],
                "Health Score (final)": fb_health["health_score_final"],
            },
        ])
        st.dataframe(health_df, use_container_width=True)

        audit_report = {
            "inventario": inv_health,
            "transacciones": tx_health,
            "feedback": fb_health,
            "generated_at": datetime.now().isoformat(),
        }
        st.download_button(
            label="📥 Descargar reporte de auditoría (JSON)",
            data=json.dumps(audit_report, indent=2),
            file_name="audit_report.json",
            mime="application/json",
        )
        st.download_button(
            label="📥 Descargar dataset JOIN (CSV)",
            data=joined.to_csv(index=False).encode("utf-8"),
            file_name="dataset_join.csv",
            mime="text/csv",
        )

        with st.expander("🚩 Filas con flags (muestras)"):
            sel_dataset = st.selectbox(
                "Elige el dataset para mostrar flags", ["Inventario", "Transacciones", "Feedback"], key="aud_sel_dataset"
            )
            if sel_dataset == "Inventario":
                st.write(f"Filas con flags: {len(inv_rare):,}")
                st.dataframe(inv_rare.head(200), use_container_width=True)
            elif sel_dataset == "Transacciones":
                st.write(f"Filas con flags: {len(tx_rare):,}")
                st.dataframe(tx_rare.head(200), use_container_width=True)
            else:
                st.write(f"Filas con flags: {len(fb_rare):,}")
                st.dataframe(fb_rare.head(200), use_container_width=True)

        with tabs[1]:
            st.subheader("📊 Operaciones")
            st.markdown("### 1️⃣ Margen negativo — insights accionables")

    # P1: margen por categoría
            if "margen_por_categoria" in analysis_results:
                df_cat = analysis_results["margen_por_categoria"].copy()
                show_only_neg = st.checkbox(
                    "Mostrar solo categorías con margen total negativo",
                    value=False,
                    key="p1_only_neg_cat"
                )
            if show_only_neg:
                df_cat = df_cat[df_cat["Margen_Total"] < 0]

            chart_bar(
                df_cat,
                "Categoria_clean",
                "Margen_Total",
                "Margen total por categoría"
            )

    # P1: scatter por SKU (priorización)
            if "sku_scatter_margen_vs_cantidad" in analysis_results:
                by_sku = analysis_results["sku_scatter_margen_vs_cantidad"].copy()
                show_only_neg_sku = st.checkbox(
                    "Mostrar solo SKUs con margen total negativo",
                    value=True,
                    key="p1_only_neg_sku"
                )
            if show_only_neg_sku:
                by_sku = by_sku[by_sku["Margen_Total"] < 0]

            topn = st.slider(
                "Top N SKUs por impacto (|margen|)",
                50, 500, 200, step=50, key="p1_topn_sku"
            )
            by_sku["abs_margen"] = by_sku["Margen_Total"].abs()
            by_sku = by_sku.sort_values("abs_margen", ascending=False).head(topn)

            chart_scatter(
                by_sku,
                x="Cantidad_Total",
                y="Margen_Total",
                color=None,
                size="Ingreso_Total" if "Ingreso_Total" in by_sku.columns else None,
                title="Prioriza SKUs: Cantidad total vs Margen total (tamaño = ingreso)",
            )

        st.markdown("---")
        st.markdown("### 3️⃣ Venta invisible (SKU fantasma)")

        if "sku_fantasma_por_categoria" in analysis_results:
            ghost_cat = analysis_results["sku_fantasma_por_categoria"].copy()
            topc = st.slider(
                "Top categorías por ingreso perdido",
                5, 30, 10, key="p3_top_cat"
            )
            chart_bar(
                ghost_cat.head(topc),
                "Categoria_clean",
                "Ingreso_Perdido",
                "Ingresos perdidos por SKU fantasma (Top categorías)"
            )

        if "donut_ingreso_riesgo_fantasma" in analysis_results:
            donut_df = analysis_results["donut_ingreso_riesgo_fantasma"].copy()
            chart_donut(
                donut_df,
                "Tipo",
                "Valor",
                "Proporción del ingreso en riesgo por SKU fantasma"
            )


    with tabs[2]:
        st.subheader("😊 Cliente")
        if "logistica_vs_nps" in analysis_results:
            st.markdown("### 2️⃣ Correlación logística vs NPS")
            corr_df = analysis_results["logistica_vs_nps"]
            st.dataframe(corr_df.head(100), use_container_width=True)

        if "stock_vs_nps" in analysis_results:
            st.markdown("### 4️⃣ Stock alto y NPS bajo por categoría")
            quality_df = analysis_results["stock_vs_nps"]
            st.dataframe(quality_df, use_container_width=True)

        if "operational_risk" in analysis_results:
            st.markdown("### 5️⃣ Riesgo operativo por bodega")
            risk_df = analysis_results["operational_risk"]
            st.dataframe(risk_df, use_container_width=True)
                st.markdown("---")
        st.markdown("### 2️⃣ Logística vs NPS — visual")

        if "logistica_vs_nps" in analysis_results:
            corr_df = analysis_results["logistica_vs_nps"].copy()
            # agregar n si no existe
            if "N" not in corr_df.columns:
                corr_df["N"] = 1
            chart_scatter(
                corr_df,
                x="Tiempo_Entrega_Prom",
                y="NPS_Prom",
                color="Bodega_Origen_clean",
                size=None,
                title="Tiempo de entrega promedio vs NPS promedio (por ciudad y bodega)",
            )

        st.markdown("---")
        st.markdown("### 4️⃣ Stock alto y NPS bajo — cuadrantes")

        if "stock_vs_nps_scatter" in analysis_results:
            s = analysis_results["stock_vs_nps_scatter"].copy()
            chart_scatter(
                s,
                x="Stock_Prom",
                y="NPS_Prom",
                color=None,
                size="N",
                title="Por categoría: Stock promedio vs NPS promedio (tamaño = n)",
            )

        if "stock_alto_nps_bajo_alerta" in analysis_results:
            red = analysis_results["stock_alto_nps_bajo_alerta"]
            st.markdown("**Categorías en alerta: alto stock + NPS bajo (<= 0)**")
            st.dataframe(red, use_container_width=True)

        st.markdown("---")
        st.markdown("### 5️⃣ Riesgo operativo por bodega — (plus: NPS)")

        if "riesgo_bodega_plus" in analysis_results:
            r = analysis_results["riesgo_bodega_plus"].copy()
            # gráfico bar ticket rate
            show = r[["Bodega_Origen_clean", "Ticket_Rate"]].dropna().sort_values("Ticket_Rate", ascending=False)
            chart_bar(show, "Bodega_Origen_clean", "Ticket_Rate", "Ticket rate por bodega (mayor = más riesgo)")

            # scatter días vs ticket rate si existe
            if "Dias_Revision_Prom" in r.columns and r["Dias_Revision_Prom"].notna().any():
                rr = r.dropna(subset=["Dias_Revision_Prom", "Ticket_Rate"])
                chart_scatter(rr, "Dias_Revision_Prom", "Ticket_Rate", None, None, "Días desde revisión (prom) vs Ticket rate")

            # NPS por bodega (conexión cliente-operación)
            if "NPS_Prom" in r.columns and r["NPS_Prom"].notna().any():
                nps_df = r[["Bodega_Origen_clean", "NPS_Prom"]].dropna().sort_values("NPS_Prom")
                chart_bar(nps_df, "Bodega_Origen_clean", "NPS_Prom", "NPS promedio por bodega (contexto)")


    with tabs[3]:
        st.subheader("🧠 Insights IA")
        st.write(
            "Utiliza IA generativa para obtener recomendaciones. El modelo solo recibe "
            "estadísticas resumidas (no datos crudos)."
        )
        if st.button("Generar recomendaciones con IA"):
            with st.spinner("Consultando al modelo..."):
                prompt_messages = build_ai_prompt(joined)
                ai_result = call_groq(prompt_messages)
                st.text_area(
                    "Recomendaciones IA",
                    value=ai_result,
                    height=300,
                    help="Resultado generado por Llama en Groq"
                )


if __name__ == "__main__":
    main()
