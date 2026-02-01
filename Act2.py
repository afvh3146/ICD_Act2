"""
Act2.py -- Streamlit dashboard for Challenge 02 (Data Cleaning, Integration and DSS)
------------------------------------------------------------------------------

This single-file Streamlit application encapsulates the entire data processing
pipeline for the second challenge of your master's programme.  It performs
auditable cleaning for three CSV sources (inventory, transactions and
customer feedback), builds a reproducible join, exposes a health score
before/after view, computes KPIs to answer management questions, and even
offers the option to call an external LLM (Groq) for data‑driven
recommendations.  The file is heavily documented so another colleague can
pick it up quickly.  To run locally simply execute:

    streamlit run Act2.py

Expected inputs
----------------
The app expects three CSV files with the following columns (as delivered
along with the challenge):

* **inventario_central_v2.csv**: ``SKU_ID``, ``Categoria``, ``Stock_Actual``,
  ``Costo_Unitario_USD``, ``Punto_Reorden``, ``Lead_Time_Dias``,
  ``Bodega_Origen``, ``Ultima_Revision``.

* **transacciones_logistica_v2.csv**: ``Transaccion_ID``, ``SKU_ID``,
  ``Fecha_Venta`` (dd/mm/yyyy), ``Cantidad_Vendida``, ``Precio_Venta_Final``,
  ``Costo_Envio``, ``Tiempo_Entrega_Real``, ``Estado_Envio``,
  ``Ciudad_Destino``, ``Canal_Venta``.

* **feedback_clientes_v2.csv**: ``Feedback_ID``, ``Transaccion_ID``,
  ``Rating_Producto``, ``Rating_Logistica``, ``Comentario_Texto``,
  ``Recomienda_Marca``, ``Ticket_Soporte_Abierto``, ``Edad_Cliente``,
  ``Satisfaccion_NPS``.

Overview of features
--------------------
1. **Data loading and caching** with automatic column deduplication and
   Unicode normalisation.
2. **Cleaning helper functions** for text normalisation, fuzzy matching,
   manual remapping and outlier detection.
3. **Per‑dataset processing functions** that compute flags, allow
   user‑driven exclusions, and log every decision.  They return both the
   clean dataframe and metadata necessary for audit.
4. **Health score computation** showing how many issues were present in the
   raw data versus the final dataset, along with a simple scoring formula.
5. **Join orchestration** with explicit flags for orphaned SKUs and missing
   feedback, plus optional inference of destination city from shipping cost.
6. **Feature engineering** creating revenue, cost, margin and age columns
   needed to answer the business questions.
7. **Interactive dashboards** split into four tabs: Audit, Operations,
   Customer and AI Insights.  Each tab contains high‑level KPIs and charts
   relevant to the challenge questions.
8. **Download buttons** for exporting both the cleaning audit report and the
   final joined dataset in CSV/JSON format.
9. **LLM integration** (optional) using Groq's API.  The prompt is built
   from the current filtered dataset and summarised statistics.  API keys
   should be provided via ``st.secrets["GROQ_API_KEY"]`` or an
   environment variable ``GROQ_API_KEY``.

This code is deliberately verbose and modular so that you and your
colleagues can easily modify thresholds, mappings or add new features.
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
    # If raw is unknown or consists solely of question marks, return 'unknown'
    if raw_lower in UNKNOWN_TOKENS or (len(set(raw_lower)) == 1 and "?" in raw_lower):
        return "unknown"
    # Remove accents and punctuation
    x = unicodedata.normalize("NFKD", raw_lower).encode("ascii", "ignore").decode("utf-8")
    x = re.sub(r"[-_/]+", " ", x)
    x = re.sub(r"[^a-z0-9\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    if x in UNKNOWN_TOKENS:
        return "unknown"
    return x or np.nan


def apply_manual_map(series_norm: pd.Series, manual_map: Dict[str, str]) -> pd.Series:
    """Replace values in a normalised series according to a manual mapping.

    The keys in ``manual_map`` should be normalised tokens; the values are
    canonical forms.
    """
    return series_norm.map(lambda v: manual_map.get(v, v))


def build_canonical_values(series_after_manual: pd.Series) -> List[str]:
    """Build a sorted list of canonical values for fuzzy matching.

    Excludes NaNs and the literal string ``'unknown'``.
    """
    vals = series_after_manual.dropna().astype(str)
    vals = vals[vals != "unknown"]
    return sorted(set(vals.tolist()))


def fuzzy_map_unique(
    series_vals: pd.Series,
    canonical: List[str],
    threshold: float = 0.92,
    delta: float = 0.03,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Apply fuzzy matching to map values to canonical values.

    Returns the mapped series and a DataFrame describing which replacements
    were made.  If rapidfuzz is unavailable or there are no canonical values
    to match against, the original series is returned with an empty report.
    """
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
    changes_df = pd.DataFrame(changes).sort_values(["applied", "score"], ascending=[False, False]) if changes else pd.DataFrame(columns=cols)
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
    """Compute a health report summarising issues before and after cleaning.

    The report includes:
    - number of rows in raw vs final
    - total duplicate rows identified in raw
    - missing value counts per column (raw and final)
    - flagged rows (based on provided flag columns)
    - a simple health score (0–100).

    Health score formula: 100 * (1 - (total_issues / (n_raw * n_cols))).
    Total issues counts missing values and flagged rows.  The final dataset
    always scores higher because issues have been removed or imputed.
    """
    report: Dict[str, Any] = {}
    n_raw, n_final = len(raw_df), len(final_df)
    report["rows_raw"] = int(n_raw)
    report["rows_final"] = int(n_final)
    # Duplicate detection on complete rows
    dup_raw = raw_df.duplicated().sum()
    dup_final = final_df.duplicated().sum()
    report["duplicates_raw"] = int(dup_raw)
    report["duplicates_final"] = int(dup_final)
    # Missing values per column
    missing_raw = raw_df.isna().sum().to_dict()
    missing_final = final_df.isna().sum().to_dict()
    report["missing_raw"] = {k: int(v) for k, v in missing_raw.items()}
    report["missing_final"] = {k: int(v) for k, v in missing_final.items()}
    # Flagged rows (issues) count
    flagged_raw = 0
    flagged_final = 0
    for fc in flags:
        if fc in raw_df.columns:
            flagged_raw += int(raw_df[fc].sum())
        if fc in final_df.columns:
            flagged_final += int(final_df[fc].sum())
    report["flagged_raw"] = int(flagged_raw)
    report["flagged_final"] = int(flagged_final)
    # Health score computation
    # Count total issues as sum of missing values across all columns plus flagged rows
    total_missing_raw = sum(missing_raw.values())
    total_missing_final = sum(missing_final.values())
    total_issues_raw = total_missing_raw + flagged_raw
    total_issues_final = total_missing_final + flagged_final
    # Avoid division by zero by adding small epsilon
    epsilon = 1e-9
    score_raw = 100 * (1 - (total_issues_raw / (n_raw * max(1, raw_df.shape[1]))))
    score_final = 100 * (1 - (total_issues_final / (n_final * max(1, final_df.shape[1]))))
    report["health_score_raw"] = round(max(0, score_raw), 2)
    report["health_score_final"] = round(max(0, score_final), 2)
    return report


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
    """Interactive sidebar for flag exclusions.

    Presents checkboxes for each flag column.  The user can select which
    flags should cause row exclusion.  Nothing happens until the user
    presses the "Apply" button.  Returns the filtered dataframe,
    the list of applied flags, and a boolean indicating whether the
    dataframe was modified.
    """
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
        applied = st.button(
            f"✅ Aplicar exclusiones — {key_prefix}", key=f"{key_prefix}_apply_btn"
        )
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
    """Clean the inventory dataset and compute flags.

    Returns a tuple of (raw_df, clean_df, final_df, flagged_df, flag_cols,
    actions).  ``actions`` is a list of strings summarising the cleaning
    operations performed.
    """
    with cfg_container:
        st.markdown("#### 📦 Inventario — opciones")
        fix_stock_abs = st.checkbox(
            "Stock: convertir negativo a positivo (abs)", value=False, key="inv_fix_abs"
        )
    inv = df_raw.copy()
    actions: List[str] = []
    # Keep copies of original text columns for later auditing
    if "Categoria" in inv.columns:
        inv["Categoria_original"] = inv["Categoria"].astype("string")
    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_original"] = inv["Bodega_Origen"].astype("string")
    # Manual mappings for category and bodega normalisation
    CATEGORY_MAP = {
        "laptop": "laptops", "laptops": "laptops", "notebook": "laptops", "notebooks": "laptops",
        "smartphone": "smartphones", "smartphones": "smartphones", "smart phone": "smartphones", "smart phones": "smartphones",
        "smart phone": "smartphones", "smart‑phone": "smartphones", "smart-phone": "smartphones",
        "tablet": "tablets", "tablets": "tablets",
        "accesorio": "accesorios", "accesorios": "accesorios", "accesories": "accesorios",
        "monitor": "monitores", "monitores": "monitores", "monitor": "monitores",
        "unknown": "unknown",
    }
    BODEGA_MAP = {
        "med": "medellin", "mde": "medellin", "medellin": "medellin",
        "bog": "bogota", "bogota": "bogota",
        "norte": "norte", "sur": "sur", "east": "east", "west": "west",
        "unknown": "unknown",
    }
    # Normalise Categoria and Bodega
    if "Categoria" in inv.columns:
        inv["Categoria_clean"] = inv["Categoria"].apply(normalize_text_keep_unknown)
        inv["Categoria_clean"] = apply_manual_map(inv["Categoria_clean"], CATEGORY_MAP)
        canonical = build_canonical_values(inv["Categoria_clean"])
        inv["Categoria_clean"], cat_fuzzy = fuzzy_map_unique(inv["Categoria_clean"], canonical, 0.92, 0.03)
        if not cat_fuzzy.empty:
            actions.append("Fuzzy matching aplicado en Categoria (" + str(len(cat_fuzzy[cat_fuzzy['applied']])) + " reemplazos)")
    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
        inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)
        canonical = build_canonical_values(inv["Bodega_Origen_clean"])
        inv["Bodega_Origen_clean"], bod_fuzzy = fuzzy_map_unique(inv["Bodega_Origen_clean"], canonical, 0.92, 0.03)
        if not bod_fuzzy.empty:
            actions.append("Fuzzy matching aplicado en Bodega_Origen (" + str(len(bod_fuzzy[bod_fuzzy['applied']])) + " reemplazos)")
    # Numeric conversions
    for c in ["Stock_Actual", "Costo_Unitario_USD", "Lead_Time_Dias", "Punto_Reorden"]:
        if c in inv.columns:
            inv[c] = to_numeric(inv[c])
    # Date parsing
    if "Ultima_Revision" in inv.columns:
        inv["Ultima_Revision_dt"] = pd.to_datetime(inv["Ultima_Revision"], errors="coerce")
    # Initialise flag columns list
    flag_cols: List[str] = []
    def add_flag(name: str, mask: pd.Series):
        """Add a boolean flag column to inv and record the flag name."""
        cname = f"flag__{name}"
        if cname in inv.columns:
            return
        inv[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)
    # Compute flags for missing / invalid values
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
    # Flag summarisation
    inv["has_any_flag"] = inv[flag_cols].any(axis=1) if flag_cols else False
    # Optional fix: convert negative stock to absolute
    inv["fix__stock_abs_applied"] = False
    if fix_stock_abs and "Stock_Actual" in inv.columns:
        m = inv["Stock_Actual"].notna() & (inv["Stock_Actual"] < 0)
        inv.loc[m, "Stock_Actual"] = inv.loc[m, "Stock_Actual"].abs()
        inv.loc[m, "fix__stock_abs_applied"] = True
        actions.append(f"Stock negativo convertido a valor absoluto en {int(m.sum())} filas")
    # Determine rows with any flag
    inv_rare = inv[inv["has_any_flag"]].copy()
    # Default exclude outliers but let user apply
    default_exclude = {"flag__costo_outlier_iqr", "flag__leadtime_outlier_iqr"}
    inv_final, applied_flags, _ = apply_exclusions_button(
        inv, flag_cols, default_exclude, "Inventario",
        help_text="Por defecto se preseleccionan outliers de costo y lead time para excluir (requiere aplicar)."
    )
    if applied_flags:
        actions.append("Exclusiones aplicadas en inventario: " + ", ".join(applied_flags))
    # Description summary
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
    """Clean the transactions dataset and compute flags.

    Returns a tuple of (raw_df, clean_df, final_df, flagged_df, flag_cols,
    actions).  ``actions`` is a list summarising the cleaning steps.
    """
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
    # Keep original text columns for auditing
    for c in ["Ciudad_Destino", "Estado_Envio", "Canal_Venta"]:
        if c in tx.columns:
            tx[f"{c}_original"] = tx[c].astype("string")
    # Numeric conversions
    for c in ["Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real"]:
        if c in tx.columns:
            tx[c] = to_numeric(tx[c])
    # Date parsing
    if "Fecha_Venta" in tx.columns:
        # Input format dd/mm/yyyy; dayfirst=True ensures correct parsing
        tx["Fecha_Venta_dt"] = pd.to_datetime(tx["Fecha_Venta"], errors="coerce", dayfirst=True)
    else:
        tx["Fecha_Venta_dt"] = pd.NaT
    # Manual maps for city, status and channel
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
        "fisico": "tienda", "físico": "tienda", "fisco": "tienda", "tienda fisica": "tienda", "tienda física": "tienda", "tienda": "tienda",
        "online": "web", "web": "web", "ecommerce": "web", "app": "app", "whatsapp": "whatsapp", "unknown": "unknown",
    }
    # Suspicious city tokens that indicate city field actually contains a channel
    SUSPICIOUS_CITY_TOKENS = {"ventas", "web", "online", "app", "whatsapp", "canal"}
    # Normalise and map city
    if "Ciudad_Destino" in tx.columns:
        tx["Ciudad_Destino_norm"] = tx["Ciudad_Destino"].apply(normalize_text_keep_unknown)
        # Flag suspicious city names (likely channel names)
        def _is_city_suspicious(v: Any) -> bool:
            if pd.isna(v) or v == "unknown":
                return False
            parts = set(str(v).split())
            return len(parts.intersection(SUSPICIOUS_CITY_TOKENS)) > 0
        tx["flag__ciudad_sospechosa"] = tx["Ciudad_Destino_norm"].map(_is_city_suspicious).fillna(False)
        # If strict, assign unknown to suspicious cities
        if strict_city:
            tx.loc[tx["flag__ciudad_sospechosa"], "Ciudad_Destino_norm"] = "unknown"
        tx["Ciudad_Destino_clean"] = apply_manual_map(tx["Ciudad_Destino_norm"], CITY_MAP)
        canonical = build_canonical_values(tx["Ciudad_Destino_clean"])
        tx["Ciudad_Destino_clean"], _ = fuzzy_map_unique(tx["Ciudad_Destino_clean"], canonical, 0.92, 0.03)
    # Normalize and map Estado_Envio
    if "Estado_Envio" in tx.columns:
        tx["Estado_Envio_clean"] = tx["Estado_Envio"].apply(normalize_text_keep_unknown)
        tx["Estado_Envio_clean"] = apply_manual_map(tx["Estado_Envio_clean"], STATUS_MAP)
        canonical = build_canonical_values(tx["Estado_Envio_clean"])
        tx["Estado_Envio_clean"], _ = fuzzy_map_unique(tx["Estado_Envio_clean"], canonical, 0.92, 0.03)
    # Normalize and map Canal_Venta
    if "Canal_Venta" in tx.columns:
        tx["Canal_Venta_clean"] = tx["Canal_Venta"].apply(normalize_text_keep_unknown)
        # normalise keys first to build correct mapping
        norm_map = {normalize_text_keep_unknown(k): v for k, v in CANAL_MAP.items()}
        norm_map.update({"tienda fisica": "tienda", "tienda física": "tienda"})
        tx["Canal_Venta_clean"] = apply_manual_map(tx["Canal_Venta_clean"], norm_map)
        canonical = build_canonical_values(tx["Canal_Venta_clean"])
        tx["Canal_Venta_clean"], _ = fuzzy_map_unique(tx["Canal_Venta_clean"], canonical, 0.92, 0.03)
    # Compute flags
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
    # Date flags
if "Fecha_Venta" in tx.columns:
    add_flag("fecha_venta_nula", tx["Fecha_Venta"].isna())
    add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())

    # --- robust "today" as Timestamp (NOT datetime.date)
    today = pd.Timestamp.now().normalize()

    # --- ensure Fecha_Venta_dt is datetime64[ns] and tz-naive
    # (re-parse to be safe in case upstream logic changes)
    tx["Fecha_Venta_dt"] = pd.to_datetime(tx["Fecha_Venta"], errors="coerce", dayfirst=True)
    if getattr(tx["Fecha_Venta_dt"].dt, "tz", None) is not None:
        tx["Fecha_Venta_dt"] = tx["Fecha_Venta_dt"].dt.tz_localize(None)

    add_flag(
        "venta_futura",
        tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today)
    )


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
    # Bring suspicious city flag into list if not already
    if "flag__ciudad_sospechosa" in tx.columns and "flag__ciudad_sospechosa" not in flag_cols:
        flag_cols.append("flag__ciudad_sospechosa")
    # Has any flag indicator
    tx["has_any_flag"] = tx[flag_cols].any(axis=1) if flag_cols else False
    # Future year correction if requested
    tx["fix__venta_year_2026_to_2025"] = False
    tx["Fecha_Venta_dt_fixed"] = tx["Fecha_Venta_dt"]
    if fix_future_year and "Fecha_Venta_dt" in tx.columns:
        # Replace year 2026 with 2025 only for future dates
        today = pd.Timestamp.now().normalize()
        if "flag__venta_futura" in tx.columns:
            m = (tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"].dt.year == 2026) & (tx["flag__venta_futura"]))
        else:
            m = (tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"].dt.year == 2026) & (tx["Fecha_Venta_dt"] > today))
        def _replace_year(d: pd.Timestamp) -> pd.Timestamp:
            try:
                return d.replace(year=2025)
            except Exception:
                return d
        tx.loc[m, "Fecha_Venta_dt_fixed"] = tx.loc[m, "Fecha_Venta_dt"].map(_replace_year)
        tx.loc[m, "fix__venta_year_2026_to_2025"] = True
        actions.append(f"Fechas 2026 corregidas a 2025 en {int(m.sum())} filas")
    # Rare rows
    tx_rare = tx[tx["has_any_flag"]].copy()
    # Default exclude none; user decides
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
        "Opción de corregir año 2026 a 2025 para ventas futuras.",
    ]
    desc.extend(actions)
    return df_raw, tx, tx_final, tx_rare, flag_cols, desc


def process_feedback(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    """Clean the feedback dataset and compute flags.

    Returns a tuple of (raw_df, clean_df, final_df, flagged_df, df_for_join,
    flag_cols, actions).  ``df_for_join`` may be aggregated to one row per
    Transaccion_ID depending on strategy.  ``actions`` summarises the
    operations performed.
    """
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
    # Preserve original fields
    if "Transaccion_ID" in fb.columns:
        fb["Transaccion_ID_original"] = fb["Transaccion_ID"].astype("string")
        fb["Transaccion_ID_clean"] = fb["Transaccion_ID"].astype("string").str.strip()
    else:
        fb["Transaccion_ID_clean"] = pd.Series([np.nan] * len(fb), index=fb.index)
    for c in ["Comentario_Texto", "Recomienda_Marca", "Ticket_Soporte_Abierto"]:
        if c in fb.columns:
            fb[f"{c}_original"] = fb[c].astype("string")
    # Numeric conversions
    for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
        if c in fb.columns:
            fb[c] = to_numeric(fb[c])
    # Clean comment
    if "Comentario_Texto" in fb.columns:
        fb["Comentario_Texto_clean"] = fb["Comentario_Texto"].astype("string").str.strip()
        if fb_placeholder_comment:
            fb.loc[fb["Comentario_Texto_clean"].isin(["---", "—", "-", ""]), "Comentario_Texto_clean"] = np.nan
    # Normalize Recomienda_Marca to yes/no/maybe/unknown
    if "Recomienda_Marca" in fb.columns:
        norm = fb["Recomienda_Marca"].apply(normalize_text_keep_unknown)
        REC_MAP = {
            "si": "yes", "sí": "yes", "s": "yes", "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
            "no": "no", "n": "no", "0": "no", "false": "no",
            "maybe": "maybe", "quizas": "maybe", "quizás": "maybe",
            "unknown": "unknown",
        }
        fb["Recomienda_Marca_clean"] = norm.map(lambda v: REC_MAP.get(v, v)).fillna("unknown")
    # Normalize Ticket_Soporte_Abierto to boolean
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
    # NPS rounding and categorisation
    if "Satisfaccion_NPS" in fb.columns:
        if fb_round_nps:
            fb["flag__nps_no_entero"] = (fb["Satisfaccion_NPS"].notna() & (fb["Satisfaccion_NPS"] % 1 != 0)).fillna(False)
            fb["Satisfaccion_NPS"] = fb["Satisfaccion_NPS"].round(0)
            actions.append("NPS redondeado a entero")
        else:
            fb["flag__nps_no_entero"] = False
        # Map NPS to buckets using a simple promoter/detractor scheme
        def nps_bucket(v: Any) -> Any:
            if pd.isna(v):
                return np.nan
            try:
                v = float(v)
            except Exception:
                return np.nan
            if v < 0:
                return "detractor"
            if v == 0:
                return "neutral"
            if v > 0:
                return "promoter"
            return np.nan
        fb["NPS_categoria"] = fb["Satisfaccion_NPS"].map(nps_bucket).astype("string")
    # Build flag columns
    flag_cols: List[str] = []
    def add_flag(name: str, mask: pd.Series):
        cname = f"flag__{name}"
        if cname in fb.columns:
            return
        fb[cname] = mask.fillna(False).astype(bool)
        flag_cols.append(cname)
    # Flag missing or duplicate IDs
    add_flag("transaccion_id_nulo", fb["Transaccion_ID_clean"].isna() | (fb["Transaccion_ID_clean"].astype("string").str.len() == 0))
    if "Feedback_ID" in fb.columns:
        add_flag("dup_feedback_id", fb["Feedback_ID"].notna() & fb["Feedback_ID"].duplicated(keep=False))
    add_flag("dup_transaccion_id", fb["Transaccion_ID_clean"].notna() & fb["Transaccion_ID_clean"].duplicated(keep=False))
    # Rating flags
    if "Rating_Producto" in fb.columns:
        add_flag("rating_producto_fuera_rango", fb["Rating_Producto"].notna() & ((fb["Rating_Producto"] < 1) | (fb["Rating_Producto"] > 5)))
    if "Rating_Logistica" in fb.columns:
        add_flag("rating_logistica_fuera_rango", fb["Rating_Logistica"].notna() & ((fb["Rating_Logistica"] < 1) | (fb["Rating_Logistica"] > 5)))
    # NPS flags
    if "Satisfaccion_NPS" in fb.columns:
        add_flag("nps_fuera_rango", fb["Satisfaccion_NPS"].notna() & ((fb["Satisfaccion_NPS"] < -100) | (fb["Satisfaccion_NPS"] > 100)))
        add_flag("nps_categoria_fuera_rango", (fb["NPS_categoria"].astype("string") == "fuera_rango"))
    # Comment flag
    if "Comentario_Texto_clean" in fb.columns:
        add_flag("comentario_faltante", fb["Comentario_Texto_clean"].isna())
    # Recomienda flags
    if "Recomienda_Marca_clean" in fb.columns:
        add_flag("recomienda_unknown", fb["Recomienda_Marca_clean"].isin(["unknown"]))
        add_flag("recomienda_maybe", fb["Recomienda_Marca_clean"].isin(["maybe"]))
    # Ticket flag
    if "Ticket_Soporte_bool" in fb.columns and "Ticket_Soporte_Abierto" in fb.columns:
        add_flag("ticket_invalido", fb["Ticket_Soporte_Abierto"].notna() & fb["Ticket_Soporte_bool"].isna())
    fb["has_any_flag"] = fb[flag_cols].any(axis=1) if flag_cols else False
    fb_rare = fb[fb["has_any_flag"]].copy()
    # Default no exclusions in feedback
    default_exclude: set[str] = set()
    fb_final, applied_flags, _ = apply_exclusions_button(
        fb, flag_cols, default_exclude, "Feedback",
        help_text="Marca flags para excluir. No excluimos nada por defecto."
    )
    if applied_flags:
        actions.append("Exclusiones aplicadas en feedback: " + ", ".join(applied_flags))
    # Prepare df_for_join based on strategy
    fb_for_join = fb_final.copy()
    fb_for_join["Transaccion_ID_clean"] = fb_for_join["Transaccion_ID_clean"].astype("string").str.strip()
    if fb_strategy == "Agregar por Transaccion_ID (recomendado 1:1)":
        agg: Dict[str, Any] = {}
        for c in ["Rating_Producto", "Rating_Logistica", "Satisfaccion_NPS", "Edad_Cliente"]:
            if c in fb_for_join.columns:
                agg[c] = "mean"
        if "NPS_categoria" in fb_for_join.columns:
            def mode_cat(x: pd.Series) -> Any:
                x = x.dropna()
                return x.mode().iloc[0] if len(x) else np.nan
            agg["NPS_categoria"] = mode_cat
        if "Recomienda_Marca_clean" in fb_for_join.columns:
            def mode_or_unknown(x: pd.Series) -> Any:
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
        fb_for_join = (
            fb_for_join.groupby("Transaccion_ID_clean", dropna=False).agg(agg).reset_index()
        )
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
    # Ensure numeric types
    df = df_joined.copy()
    # KPI: number of rows
    total_tx = len(df)
    margin_neg = df["Margen_Bruto"].lt(0).sum() if "Margen_Bruto" in df.columns else 0
    sku_fantasma = df["flag__sku_no_existe_en_inventario"].sum() if "flag__sku_no_existe_en_inventario" in df.columns else 0
    sin_feedback = df["flag__sin_feedback"].sum() if "flag__sin_feedback" in df.columns else 0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transacciones", f"{total_tx:,}")
    col2.metric("Margen negativo", f"{margin_neg:,}")
    col3.metric("SKU fantasma", f"{sku_fantasma:,}")
    col4.metric("Sin feedback", f"{sin_feedback:,}")


def compute_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for the five management questions.

    Returns a dictionary with keys: 'margen_negativo', 'sku_fantasma',
    'logistica_vs_nps', 'stock_vs_nps', 'operational_risk'.  Each entry
    contains a DataFrame summarising the finding or a metric.
    """
    results: Dict[str, Any] = {}
    if df.empty:
        return results
    df = df.copy()
    # Ensure necessary columns exist
    df["Ingreso"] = df.get("Ingreso", 0)
    df["Margen_Bruto"] = df.get("Margen_Bruto", 0)
    df["Cantidad_Vendida"] = df.get("Cantidad_Vendida", 0)
    # Question 1: Margen negativo por SKU
    if "SKU_ID" in df.columns and "Margen_Bruto" in df.columns:
        q1 = (
            df.groupby("SKU_ID")["Margen_Bruto"].sum().reset_index().rename(columns={"Margen_Bruto": "Margen_Total"})
        )
        q1 = q1[q1["Margen_Total"] < 0].sort_values("Margen_Total")
        results["margen_negativo"] = q1
    # Question 2: Correlación logística vs NPS por ciudad y bodega
    if ("Tiempo_Entrega_Real" in df.columns and "Satisfaccion_NPS" in df.columns
            and "Ciudad_Destino_clean" in df.columns and "Bodega_Origen_clean" in df.columns):
        df_corr = df[["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Real", "Satisfaccion_NPS"]].dropna()
        # Compute average delivery time and average NPS per city/bodega
        corr_table = (
            df_corr.groupby(["Ciudad_Destino_clean", "Bodega_Origen_clean"])
            .agg({"Tiempo_Entrega_Real": "mean", "Satisfaccion_NPS": "mean"})
            .reset_index()
            .rename(columns={"Tiempo_Entrega_Real": "Tiempo_Entrega_Prom", "Satisfaccion_NPS": "NPS_Prom"})
        )
        results["logistica_vs_nps"] = corr_table
    # Question 3: SKU fantasma — impacto financiero
    if "flag__sku_no_existe_en_inventario" in df.columns and "Ingreso" in df.columns:
        lost_sales = df[df["flag__sku_no_existe_en_inventario"]]
        total_lost = lost_sales["Ingreso"].sum()
        count_lost = len(lost_sales)
        results["sku_fantasma"] = {
            "total_perdido": float(total_lost),
            "num_transacciones": int(count_lost),
            "porcentaje": float(total_lost / df["Ingreso"].sum()) if df["Ingreso"].sum() != 0 else 0.0,
        }
    # Question 4: Stock alto + NPS bajo — calidad vs sobrecostos
    if ("Stock_Actual" in df.columns and "Satisfaccion_NPS" in df.columns and "Categoria_clean" in df.columns):
        # Define thresholds heuristically: top quartile of stock and NPS <= 0
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
    # Question 5: Riesgo operativo — última revisión vs tickets de soporte
    if ("Ultima_Revision_dt" in df.columns and "Ticket_Soporte_bool" in df.columns and "Bodega_Origen_clean" in df.columns):
        today = pd.Timestamp(datetime.now().date())
        df_oper = df[["Ultima_Revision_dt", "Ticket_Soporte_bool", "Bodega_Origen_clean"]].dropna()
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


def call_groq(messages: List[Dict[str, str]]) -> str:
    """Call Groq's LLM API with a chat format.  Returns the assistant content.

    This function looks for a key in Streamlit secrets (``GROQ_API_KEY``) or the
    environment.  If neither is present or the requests library is not
    available, it returns a message explaining the situation.  The API URL
    follows Groq's OpenAI‑compatible endpoint.
    """
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Error: No se encontró la clave GROQ_API_KEY en st.secrets ni en variables de entorno. Configúrala para usar IA."
    if not REQUESTS_AVAILABLE:
        return "Error: la librería requests no está disponible en este entorno."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 512,
        "top_p": 0.9,
        "stream": False,
    }
    try:
        res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error al llamar a Groq: {e}"


def build_ai_prompt(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Construct a summarised prompt for Groq based on filtered data.

    Only high‑level statistics (counts, sums, averages) are exposed to
    the model, never raw rows.  The user can customise this function to
    control what information is sent.
    """
    messages: List[Dict[str, str]] = []
    # System role instructs the model to act as a senior data consultant
    messages.append({
        "role": "system",
        "content": (
            "Eres un analista de datos senior que brinda recomendaciones de negocio "
            "basadas en estadísticas resumidas de ventas, inventario y feedback. "
            "Nunca menciones que te dieron un resumen; simplemente responde con "
            "tres párrafos de insights accionables para la gerencia."
        ),
    })
    # Summarise dataset
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
    # Sidebar config container placeholders
    st.sidebar.divider()
    with st.sidebar.expander("⚙️ Configuración de la limpieza de datos", expanded=False):
        st.caption("Configura opciones por base. Los datasets no se eliminan por flags a menos que lo apliques.")
        inv_cfg = st.container()
        tx_cfg = st.container()
        fb_cfg = st.container()
        join_cfg = st.container()
        doc_cfg = st.container()
    # Process each dataset
    inv_raw_out, inv_clean, inv_final, inv_rare, inv_flags, inv_desc = process_inventario(inv_raw, inv_cfg)
    tx_raw_out, tx_clean, tx_final, tx_rare, tx_flags, tx_desc = process_transacciones(tx_raw, tx_cfg)
    fb_raw_out, fb_clean, fb_final, fb_rare, fb_for_join, fb_flags, fb_desc = process_feedback(fb_raw, fb_cfg)
    # Compute health reports
    inv_health = compute_health_metrics(inv_raw_out, inv_clean, inv_final, inv_flags)
    tx_health = compute_health_metrics(tx_raw_out, tx_clean, tx_final, tx_flags)
    fb_health = compute_health_metrics(fb_raw_out, fb_clean, fb_final, fb_flags)
    # Join configuration
    with join_cfg:
        st.markdown("#### 🔗 Join — opciones post‑join")
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
    # Documentation expander
    with doc_cfg:
        st.markdown("#### 🧾 Cómo estamos limpiando (documentación)")
        with st.expander("📦 Inventario — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in inv_desc]))
        with st.expander("🚚 Transacciones — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in tx_desc]))
        with st.expander("💬 Feedback — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in fb_desc]))
    # Build Join
    st.header("✅ Dataset final (JOIN) — listo para análisis")
    # Normalise IDs before join
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
    # Join Tx ↔ Inv
    join_tx_inv = txj.merge(invj, on="SKU_ID", how="left", suffixes=("_tx", "_inv"), indicator="merge_tx_inv")
    join_tx_inv["flag__sku_no_existe_en_inventario"] = (join_tx_inv["merge_tx_inv"] == "left_only")
    # Join (Tx+Inv) ↔ Feedback
    joined = join_tx_inv.merge(fbj, on="Transaccion_ID", how="left", indicator="merge_tx_fb")
    joined["flag__sin_feedback"] = (joined["merge_tx_fb"] == "left_only")
    # Post-join: optional city inference via shipping cost
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
    # Feature engineering (after join) for KPIs
    # Numeric safety conversions
    joined["Cantidad_Vendida"] = pd.to_numeric(joined.get("Cantidad_Vendida", 0), errors="coerce")
    joined["Precio_Venta_Final"] = pd.to_numeric(joined.get("Precio_Venta_Final", 0), errors="coerce")
    joined["Costo_Unitario_USD"] = pd.to_numeric(joined.get("Costo_Unitario_USD", 0), errors="coerce")
    joined["Costo_Envio"] = pd.to_numeric(joined.get("Costo_Envio", 0), errors="coerce")
    joined["Stock_Actual"] = pd.to_numeric(joined.get("Stock_Actual", 0), errors="coerce")
    # Ingreso y costo
    joined["Ingreso"] = joined["Cantidad_Vendida"] * joined["Precio_Venta_Final"]
    joined["Costo_producto"] = joined["Cantidad_Vendida"] * joined["Costo_Unitario_USD"]
    joined["Margen_Bruto"] = joined["Ingreso"] - joined["Costo_producto"]
    joined["Margen_Neto_aprox"] = joined["Margen_Bruto"] - joined["Costo_Envio"]
    # Age of last revision (days)
    if "Ultima_Revision_dt" in joined.columns:
        today_dt = pd.Timestamp(datetime.now().date())
        joined["Dias_desde_revision"] = (today_dt - joined["Ultima_Revision_dt"].dt.floor("D")).dt.days
    # KPI Dashboard
    build_kpi_cards(joined)
    # Create analysis summary for 5 questions
    analysis_results = compute_analysis(joined)
    # UI tabs
    tabs = st.tabs([
        "🧮 Auditoría", "📊 Operaciones", "😊 Cliente", "🧠 Insights IA"
    ])
    # Auditoría tab
    with tabs[0]:
        st.subheader("Auditoría de datos")
        # Show health score summary
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
        # Provide download of audit report
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
        # Provide download of final dataset
        st.download_button(
            label="📥 Descargar dataset JOIN (CSV)",
            data=joined.to_csv(index=False).encode("utf-8"),
            file_name="dataset_join.csv",
            mime="text/csv",
        )
        # Show rare flagged rows for exploration
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
    # Operations tab (Margen negativo, SKU fantasma)
    with tabs[1]:
        st.subheader("📊 Operaciones")
        # Question 1: Margen negativo
        if "margen_negativo" in analysis_results:
            st.markdown("### 1️⃣ Margen negativo por SKU")
            q1_df = analysis_results["margen_negativo"]
            st.write(f"SKUs con margen total negativo: {len(q1_df):,}")
            st.dataframe(q1_df.head(50), use_container_width=True)
        # Question 3: SKU fantasma
        if "sku_fantasma" in analysis_results:
            st.markdown("### 3️⃣ Venta invisible (SKU fantasma)")
            info = analysis_results["sku_fantasma"]
            st.write(
                f"Transacciones con SKU fantasma: {info['num_transacciones']:,}\n"
                f"Ingresos perdidos: USD {info['total_perdido']:,.2f}\n"
                f"% del ingreso total en riesgo: {info['porcentaje']*100:.2f}%"
            )
        # Visualise margin distribution
        if "Margen_Bruto" in joined.columns:
            st.markdown("#### Distribución de margen bruto")
            st.bar_chart(joined["Margen_Bruto"].dropna().clip(-1e4, 1e4))
    # Customer tab (Logística vs NPS, Stock vs NPS)
    with tabs[2]:
        st.subheader("😊 Cliente")
        # Question 2: Logística vs NPS
        if "logistica_vs_nps" in analysis_results:
            st.markdown("### 2️⃣ Correlación logística vs NPS")
            corr_df = analysis_results["logistica_vs_nps"]
            st.dataframe(corr_df.head(100), use_container_width=True)
        # Question 4: Stock alto + NPS bajo
        if "stock_vs_nps" in analysis_results:
            st.markdown("### 4️⃣ Stock alto y NPS bajo por categoría")
            quality_df = analysis_results["stock_vs_nps"]
            st.dataframe(quality_df, use_container_width=True)
        # Question 5: Riesgo operativo
        if "operational_risk" in analysis_results:
            st.markdown("### 5️⃣ Riesgo operativo por bodega")
            risk_df = analysis_results["operational_risk"]
            st.dataframe(risk_df, use_container_width=True)
    # AI Insights tab
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
                    "Recomendaciones IA", value=ai_result, height=300, help="Resultado generado por Llama-3 en Groq"
                )


if __name__ == "__main__":
    main()
