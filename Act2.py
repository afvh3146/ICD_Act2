# Act2.py — Streamlit dashboard for Challenge 02 (Data Cleaning, Integration and DSS)
# ----------------------------------------------------------------------------------
#
# Incluye:
# 1) Limpieza auditable (inventario, transacciones, feedback) con flags y exclusiones por botón.
# 2) Health score (raw vs final) + descarga de reporte JSON.
# 3) JOIN reproducible (Tx↔Inv↔Fb) + flags de SKU fantasma y sin feedback.
# 4) Feature engineering (Ingreso, Costos, Margen, Días desde revisión).
# 5) Visualizaciones alineadas a P1..P5.
# 6) IA opcional (Groq) con prompt basado SOLO en estadísticas agregadas.
# 7) PDF (botón) con narrativa + evidencia visual (gráficas de la app) + plan de acción.
#    - El análisis se muestra en la app exactamente como se imprime.

import os
import re
import unicodedata
import json
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Fuzzy matching (opcional)
try:
    from rapidfuzz import process, fuzz  # type: ignore
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# Peticiones a la API Groq (opcional)
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Visualizaciones con Altair (opcional)
try:
    import altair as alt  # type: ignore
    ALTAIR_AVAILABLE = True
except Exception:
    ALTAIR_AVAILABLE = False

# Exportar Altair a PNG (para PDF)
try:
    import vl_convert as vlc  # type: ignore
    VLCONVERT_AVAILABLE = True
except Exception:
    VLCONVERT_AVAILABLE = False

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

UNKNOWN_TOKENS = {
    "???", "??", "?", "na", "n a", "none", "null", "unknown",
    "sin categoria", "sincategoria", "sin categoría",
    "---", "—", "-"
}


def safe_for_streamlit_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there are no duplicate columns which confuse Streamlit tables."""
    if df is None:
        return pd.DataFrame()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def normalize_text_keep_unknown(x: Any) -> Any:
    """Normalize text; unknown tokens mapped to 'unknown'."""
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
    return series_norm.map(lambda v: manual_map.get(v, v))


def build_canonical_values(series_after_manual: pd.Series) -> List[str]:
    vals = series_after_manual.dropna().astype(str)
    vals = vals[vals != "unknown"]
    return sorted(set(vals.tolist()))


def fuzzy_map_unique(
    series_vals: pd.Series,
    canonical: List[str],
    threshold: float = 0.92,
    delta: float = 0.03,
) -> Tuple[pd.Series, pd.DataFrame]:
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
    return pd.to_numeric(series, errors="coerce")


def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_flag_iqr(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
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
    report: Dict[str, Any] = {}

    n_raw, n_final = len(raw_df), len(final_df)
    report["rows_raw"] = int(n_raw)
    report["rows_final"] = int(n_final)

    dup_raw = int(raw_df.duplicated().sum())
    dup_final = int(final_df.duplicated().sum())
    report["duplicates_raw"] = dup_raw
    report["duplicates_final"] = dup_final

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


def _wrap_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, line_height: float) -> float:
    """Draw wrapped text; returns new y."""
    words = (text or "").split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 10) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= line_height
            line = w
    if line:
        c.drawString(x, y, line)
        y -= line_height
    return y


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _df_top_to_text(df: pd.DataFrame, cols: List[str], n: int = 5) -> str:
    """Convert a top N table into bullet-like lines (for PDF and prompt)."""
    if df is None or df.empty:
        return "No hay datos suficientes."
    use = [c for c in cols if c in df.columns]
    if not use:
        return "No hay columnas suficientes."
    df2 = df[use].head(n).copy()
    lines = []
    for _, r in df2.iterrows():
        parts = []
        for c in use:
            v = r.get(c, "")
            if isinstance(v, (int, np.integer)):
                parts.append(f"{c}={int(v)}")
            elif isinstance(v, (float, np.floating)):
                parts.append(f"{c}={v:.2f}")
            else:
                parts.append(f"{c}={v}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# PDF: build charts as PNG (from app logic) + print-ready narrative preview
# -----------------------------------------------------------------------------

def altair_to_png_bytes(chart: "alt.Chart") -> bytes:
    if not (ALTAIR_AVAILABLE and VLCONVERT_AVAILABLE):
        raise RuntimeError("Altair+vl-convert-python requeridos para exportar charts a PNG.")
    spec = chart.to_dict()
    return vlc.vegalite_to_png(spec, scale=2)


def build_pdf_charts_from_analysis(analysis_results: Dict[str, Any]) -> Dict[str, bytes]:
    """
    Construye las evidencias visuales (mínimo 4) desde analysis_results.
    Retorna dict: titulo -> bytes(PNG)
    """
    charts: Dict[str, bytes] = {}
    if not (ALTAIR_AVAILABLE and VLCONVERT_AVAILABLE):
        return charts

    # P1: Margen por categoría
    if isinstance(analysis_results.get("margen_por_categoria"), pd.DataFrame):
        df = analysis_results["margen_por_categoria"].copy()
        if not df.empty and {"Categoria_clean", "Margen_Total"}.issubset(df.columns):
            ch = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Categoria_clean:N", sort="-y", title="Categoría"),
                    y=alt.Y("Margen_Total:Q", title="Margen total"),
                    tooltip=["Categoria_clean", "Margen_Total"],
                )
                .properties(height=280, title="P1 — Margen total por categoría")
            )
            charts["P1 — Margen total por categoría"] = altair_to_png_bytes(ch)

    # P2: Tiempo entrega vs NPS (ciudad/bodega)
    if isinstance(analysis_results.get("logistica_vs_nps"), pd.DataFrame):
        df = analysis_results["logistica_vs_nps"].copy()
        needed = {"Tiempo_Entrega_Prom", "NPS_Prom", "Ciudad_Destino_clean", "Bodega_Origen_clean"}
        if not df.empty and needed.issubset(df.columns):
            ch = (
                alt.Chart(df)
                .mark_circle(opacity=0.75)
                .encode(
                    x=alt.X("Tiempo_Entrega_Prom:Q", title="Tiempo entrega prom"),
                    y=alt.Y("NPS_Prom:Q", title="NPS prom"),
                    color=alt.Color("Bodega_Origen_clean:N", title="Bodega"),
                    size=alt.Size("N:Q", title="n") if "N" in df.columns else alt.value(60),
                    tooltip=list(df.columns[:25]),
                )
                .properties(height=280, title="P2 — Tiempo de entrega vs NPS (ciudad+bodega)")
            )
            charts["P2 — Entrega vs NPS"] = altair_to_png_bytes(ch)

    # P3: Donut ingreso en riesgo
    if isinstance(analysis_results.get("donut_ingreso_riesgo_fantasma"), pd.DataFrame):
        df = analysis_results["donut_ingreso_riesgo_fantasma"].copy()
        if not df.empty and {"Tipo", "Valor"}.issubset(df.columns):
            ch = (
                alt.Chart(df)
                .mark_arc(innerRadius=70)
                .encode(
                    theta="Valor:Q",
                    color=alt.Color("Tipo:N", title="Tipo"),
                    tooltip=["Tipo", "Valor"],
                )
                .properties(height=280, title="P3 — Ingreso en riesgo por SKU fantasma")
            )
            charts["P3 — Ingreso en riesgo"] = altair_to_png_bytes(ch)

    # P4: Stock vs NPS por categoría
    if isinstance(analysis_results.get("stock_vs_nps_scatter"), pd.DataFrame):
        df = analysis_results["stock_vs_nps_scatter"].copy()
        if not df.empty and {"Stock_Prom", "NPS_Prom"}.issubset(df.columns):
            ch = (
                alt.Chart(df)
                .mark_circle(opacity=0.75)
                .encode(
                    x=alt.X("Stock_Prom:Q", title="Stock prom"),
                    y=alt.Y("NPS_Prom:Q", title="NPS prom"),
                    size=alt.Size("N:Q", title="n") if "N" in df.columns else alt.value(60),
                    tooltip=list(df.columns[:25]),
                )
                .properties(height=280, title="P4 — Stock alto vs NPS (por categoría)")
            )
            charts["P4 — Stock vs NPS"] = altair_to_png_bytes(ch)

    # P5: Ticket rate por bodega
    if isinstance(analysis_results.get("riesgo_bodega_plus"), pd.DataFrame):
        df = analysis_results["riesgo_bodega_plus"].copy()
        if not df.empty and {"Bodega_Origen_clean", "Ticket_Rate"}.issubset(df.columns):
            df2 = df[["Bodega_Origen_clean", "Ticket_Rate"]].dropna().sort_values("Ticket_Rate", ascending=False)
            if not df2.empty:
                ch = (
                    alt.Chart(df2)
                    .mark_bar()
                    .encode(
                        x=alt.X("Bodega_Origen_clean:N", sort="-y", title="Bodega"),
                        y=alt.Y("Ticket_Rate:Q", title="Ticket rate"),
                        tooltip=["Bodega_Origen_clean", "Ticket_Rate"],
                    )
                    .properties(height=280, title="P5 — Ticket rate por bodega")
                )
                charts["P5 — Ticket rate"] = altair_to_png_bytes(ch)

    return charts


def build_print_ready_report_text(
    df_filtered: pd.DataFrame,
    analysis_results: Dict[str, Any],
    ai_text: str,
) -> str:
    """
    Texto que se muestra en la app EXACTAMENTE como se imprime en el PDF.
    Incluye resumen ejecutivo + respuestas P1..P5 + plan acción (IA o fallback).
    """
    d = df_filtered if df_filtered is not None else pd.DataFrame()
    total_tx = int(len(d)) if not d.empty else 0
    total_ing = _safe_float(d["Ingreso"].sum(), 0.0) if (not d.empty and "Ingreso" in d.columns) else 0.0
    total_margen = _safe_float(d["Margen_Bruto"].sum(), 0.0) if (not d.empty and "Margen_Bruto" in d.columns) else 0.0

    # P1
    p1_counts = "Sin datos."
    p1_top = "No hay datos suficientes."
    if isinstance(analysis_results.get("margen_negativo"), pd.DataFrame):
        mneg = analysis_results["margen_negativo"].copy()
        p1_counts = f"SKUs con margen total negativo: {len(mneg):,}"
        p1_top = _df_top_to_text(mneg.sort_values("Margen_Total").head(10), ["SKU_ID", "Margen_Total", "Cantidad_Total", "Ingreso_Total"], n=10)

    # P2
    p2_top = "No hay datos suficientes."
    if isinstance(analysis_results.get("logistica_vs_nps"), pd.DataFrame):
        l = analysis_results["logistica_vs_nps"].copy()
        if "N" in l.columns:
            l = l[l["N"] >= 5].copy()
        if (not l.empty) and {"Tiempo_Entrega_Prom", "NPS_Prom"}.issubset(l.columns):
            l["score_riesgo"] = (l["Tiempo_Entrega_Prom"].rank(pct=True)) + ((-l["NPS_Prom"]).rank(pct=True))
            l = l.sort_values("score_riesgo", ascending=False).head(8)
            p2_top = _df_top_to_text(l, ["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Prom", "NPS_Prom", "N"], n=8)

    # P3
    p3_summary = "Sin datos."
    p3_topcat = "No hay datos suficientes."
    if isinstance(analysis_results.get("sku_fantasma"), dict):
        sf = analysis_results["sku_fantasma"]
        p3_summary = (
            f"Transacciones con SKU fantasma: {int(sf.get('num_transacciones', 0)):,} | "
            f"Ingreso en riesgo: {float(sf.get('total_perdido', 0.0)):.2f} USD | "
            f"% ingreso en riesgo: {float(sf.get('porcentaje', 0.0))*100:.2f}%"
        )
    if isinstance(analysis_results.get("sku_fantasma_por_categoria"), pd.DataFrame):
        gc = analysis_results["sku_fantasma_por_categoria"].copy().sort_values("Ingreso_Perdido", ascending=False).head(10)
        cat_col = "Categoria_fantasma" if "Categoria_fantasma" in gc.columns else ("Categoria_clean" if "Categoria_clean" in gc.columns else None)
        if cat_col:
            p3_topcat = _df_top_to_text(gc, [cat_col, "Ingreso_Perdido"], n=10)

    # P4
    p4_alert = "No hay datos suficientes."
    if isinstance(analysis_results.get("stock_alto_nps_bajo_alerta"), pd.DataFrame):
        red = analysis_results["stock_alto_nps_bajo_alerta"].copy()
        p4_alert = _df_top_to_text(red, ["Categoria_clean", "Stock_Prom", "NPS_Prom", "N"], n=10)

    # P5
    p5_top = "No hay datos suficientes."
    if isinstance(analysis_results.get("riesgo_bodega_plus"), pd.DataFrame):
        r = analysis_results["riesgo_bodega_plus"].copy().sort_values("Ticket_Rate", ascending=False).head(10)
        p5_top = _df_top_to_text(r, ["Bodega_Origen_clean", "Ticket_Rate", "Tickets_Abiertos", "Total", "Dias_Revision_Prom", "NPS_Prom"], n=10)

    if not ai_text or not ai_text.strip():
        ai_text = (
            "1) (Baja) Bloquear ventas con SKU no registrado y auditar integraciones (reduce ingreso en riesgo inmediato).\n"
            "2) (Media) Recalibrar pricing/promos en canal web para SKUs con margen negativo priorizando impacto por volumen.\n"
            "3) (Alta) Intervenir operación en zona Ciudad–Bodega crítica: nuevo SLA, cambio de operador y monitoreo NPS."
        )

    report = f"""
DOCUMENTO DE HALLAZGOS — VISTA PREVIA (se imprime igual)

Contexto general:
- Total de transacciones: {total_tx:,}
- Ingreso total estimado: {total_ing:.2f} USD
- Margen bruto total estimado: {total_margen:.2f} USD

P1) Fuga de capital y rentabilidad:
- {p1_counts}
- Top SKUs con peor margen total:
{p1_top}

P2) Crisis logística y cuellos de botella:
- Top combinaciones Ciudad–Bodega con mayor riesgo (tiempo alto + NPS bajo):
{p2_top}

P3) Análisis de la venta invisible (SKU fantasma):
- {p3_summary}
- Top categorías/segmentos asociados a ventas fantasma:
{p3_topcat}

P4) Diagnóstico de fidelidad (stock alto + NPS bajo):
- Categorías en alerta (alto stock y NPS <= 0):
{p4_alert}

P5) Storytelling de riesgo operativo:
- Ranking bodegas por ticket rate + días desde revisión + NPS:
{p5_top}

PLAN DE ACCIÓN (3 recomendaciones priorizadas):
{ai_text.strip()}
""".strip()
    return report


def generate_findings_pdf_auto(
    report_text: str,
    charts_png: Dict[str, bytes],
) -> bytes:
    """
    Genera PDF final: texto (igual a vista previa) + mínimo 4 evidencias visuales si están disponibles.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.8 * inch

    # Página 1: texto
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, "Documento de Hallazgos — Consultoría (Challenge 02)")
    y -= 0.35 * inch

    c.setFont("Helvetica", 10)
    for block in report_text.split("\n\n"):
        y = _wrap_text(c, block.strip(), margin, y, width - 2 * margin, 14)
        y -= 8
        if y < margin + 60:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)

    # Páginas de evidencias visuales
    if charts_png:
        c.showPage()
        y = height - margin
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Pruebas visuales (Dashboard)")
        y -= 0.25 * inch

        img_max_w = width - 2 * margin
        img_max_h = 2.7 * inch

        i = 1
        for title, png_b in charts_png.items():
            if y < margin + img_max_h + 40:
                c.showPage()
                y = height - margin

            c.setFont("Helvetica-Bold", 10)
            c.drawString(margin, y, f"Figura {i}: {title}")
            y -= 0.18 * inch

            try:
                img = Image.open(BytesIO(png_b))
                img_w, img_h = img.size
                scale = min(img_max_w / img_w, img_max_h / img_h)
                draw_w, draw_h = img_w * scale, img_h * scale
                c.drawImage(ImageReader(img), margin, y - draw_h, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
                y -= (draw_h + 0.25 * inch)
                i += 1
            except Exception:
                c.setFont("Helvetica", 10)
                y = _wrap_text(c, f"[No se pudo renderizar la figura: {title}]", margin, y, width - 2 * margin, 14)
                y -= 10

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# -------------------------
# Charts (Altair preferred)
# -------------------------

def chart_bar(df: pd.DataFrame, x: str, y: str, title: str, height: int = 320):
    st.markdown(f"#### {title}")
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return None

    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        st.info("No hay datos suficientes para este gráfico.")
        return None

    if ALTAIR_AVAILABLE:
        ch = (
            alt.Chart(tmp)
            .mark_bar()
            .encode(
                x=alt.X(f"{x}:N", sort="-y", title=x),
                y=alt.Y(f"{y}:Q", title=y),
                tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q")],
            )
            .properties(height=height)
        )
        st.altair_chart(ch, use_container_width=True)
        return ch
    else:
        st.bar_chart(tmp.set_index(x)[y])
        return None


def chart_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
    size: Optional[str],
    title: str,
    height: int = 380
):
    st.markdown(f"#### {title}")
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return None

    tmp = df.copy()
    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    tooltip_cols = [c for c in [x, y, color, size] if c and c in tmp.columns]
    for extra in ["Categoria_clean", "Bodega_Origen_clean", "Ciudad_Destino_clean", "SKU_ID"]:
        if extra in tmp.columns and extra not in tooltip_cols:
            tooltip_cols.append(extra)

    if ALTAIR_AVAILABLE:
        enc: Dict[str, Any] = {
            "x": alt.X(f"{x}:Q", title=x),
            "y": alt.Y(f"{y}:Q", title=y),
            "tooltip": [alt.Tooltip(f"{c}:N") if tmp[c].dtype == "object" else alt.Tooltip(f"{c}:Q") for c in tooltip_cols],
        }
        if color and color in tmp.columns:
            enc["color"] = alt.Color(f"{color}:N", title=color)
        if size and size in tmp.columns:
            enc["size"] = alt.Size(f"{size}:Q", title=size)

        ch = alt.Chart(tmp).mark_circle(opacity=0.7).encode(**enc).properties(height=height)
        st.altair_chart(ch, use_container_width=True)
        return ch
    else:
        st.scatter_chart(tmp[[x, y]].dropna())
        return None


def chart_donut(df: pd.DataFrame, category_col: str, value_col: str, title: str, height: int = 320):
    st.markdown(f"#### {title}")
    if df is None or df.empty or category_col not in df.columns or value_col not in df.columns:
        st.info("No hay datos suficientes para este gráfico.")
        return None

    if not ALTAIR_AVAILABLE:
        st.info("Altair no está disponible; mostrando tabla en su lugar.")
        st.dataframe(df, use_container_width=True)
        return None

    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(f"{category_col}:N", legend=alt.Legend(title=category_col)),
        tooltip=[alt.Tooltip(f"{category_col}:N"), alt.Tooltip(f"{value_col}:Q")],
    )
    donut = base.mark_arc(innerRadius=70).properties(height=height)
    st.altair_chart(donut, use_container_width=True)
    return donut


# -----------------------------------------------------------------------------
# Data loading & exclusion UI
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def apply_exclusions_button(
    df: pd.DataFrame,
    flag_cols: List[str],
    default_selected: set,
    key_prefix: str,
    help_text: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], bool]:
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


# -----------------------------------------------------------------------------
# Processing — Inventario
# -----------------------------------------------------------------------------

def process_inventario(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    with cfg_container:
        st.markdown("#### 📦 Inventario — opciones")
        fix_stock_abs = st.checkbox("Stock: convertir negativo a positivo (abs)", value=False, key="inv_fix_abs")

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
            actions.append(f"Fuzzy matching en Categoria ({int((cat_fuzzy['applied']==True).sum())} reemplazos)")

    if "Bodega_Origen" in inv.columns:
        inv["Bodega_Origen_clean"] = inv["Bodega_Origen"].apply(normalize_text_keep_unknown)
        inv["Bodega_Origen_clean"] = apply_manual_map(inv["Bodega_Origen_clean"], BODEGA_MAP)
        canonical = build_canonical_values(inv["Bodega_Origen_clean"])
        inv["Bodega_Origen_clean"], bod_fuzzy = fuzzy_map_unique(inv["Bodega_Origen_clean"], canonical, 0.92, 0.03)
        if not bod_fuzzy.empty:
            actions.append(f"Fuzzy matching en Bodega_Origen ({int((bod_fuzzy['applied']==True).sum())} reemplazos)")

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
        actions.append(f"Stock negativo → abs() en {int(m.sum())} filas")

    inv_rare = inv[inv["has_any_flag"]].copy()

    default_exclude = {"flag__costo_outlier_iqr", "flag__leadtime_outlier_iqr"}
    inv_final, applied_flags, _ = apply_exclusions_button(
        inv, flag_cols, default_exclude, "Inventario",
        help_text="Por defecto se preseleccionan outliers de costo y lead time para excluir (requiere aplicar)."
    )
    if applied_flags:
        actions.append("Exclusiones inventario: " + ", ".join(applied_flags))

    desc = [
        "Normalización de texto (lowercase, sin tildes, limpieza de signos).",
        "Mapeo manual + fuzzy matching en Categoria y Bodega_Origen.",
        "Conversión numérica y fechas.",
        "Flags: nulos, negativos, outliers IQR, unknown.",
        "Opcional: stock negativo → abs().",
        "Exclusiones por botón (outliers preseleccionados).",
    ]
    desc.extend(actions)

    return df_raw, inv, inv_final, inv_rare, flag_cols, desc


# -----------------------------------------------------------------------------
# Processing — Transacciones
# -----------------------------------------------------------------------------

def process_transacciones(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
    with cfg_container:
        st.markdown("#### 🚚 Transacciones — opciones")
        strict_city = st.checkbox("Ciudad desconocida/sospechosa → unknown", value=True, key="tx_strict_city")
        fix_future_year = st.checkbox("Venta futura: si año==2026 → cambiar a 2025", value=False, key="tx_fix_future_year")

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
        "fisico": "tienda", "físico": "tienda",
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

    if "Transaccion_ID" in tx.columns:
        add_flag("transaccion_id_nulo", tx["Transaccion_ID"].isna())
    if "SKU_ID" in tx.columns:
        add_flag("sku_id_nulo", tx["SKU_ID"].isna())

    if "Fecha_Venta" in tx.columns:
        add_flag("fecha_venta_nula", tx["Fecha_Venta"].isna())
        add_flag("fecha_venta_invalida", tx["Fecha_Venta_dt"].isna() & tx["Fecha_Venta"].notna())

        today = pd.Timestamp.now().normalize()
        add_flag("venta_futura", tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today))

        if fix_future_year:
            m_fix = tx["Fecha_Venta_dt"].notna() & (tx["Fecha_Venta_dt"] > today) & (tx["Fecha_Venta_dt"].dt.year == 2026)
            if m_fix.any():
                tx.loc[m_fix, "Fecha_Venta_dt"] = tx.loc[m_fix, "Fecha_Venta_dt"] - pd.DateOffset(years=1)
                actions.append(f"Fechas futuras 2026→2025 corregidas en {int(m_fix.sum())} filas")

    if "Cantidad_Vendida" in tx.columns:
        add_flag("cantidad_no_positiva", tx["Cantidad_Vendida"].notna() & (tx["Cantidad_Vendida"] <= 0))

    if "Tiempo_Entrega_Real" in tx.columns:
        add_flag("tiempo_negativo", tx["Tiempo_Entrega_Real"].notna() & (tx["Tiempo_Entrega_Real"] < 0))
        add_flag("tiempo_outlier_iqr", outlier_flag_iqr(tx, "Tiempo_Entrega_Real", k=1.5))

    if "Costo_Envio" in tx.columns:
        add_flag("costo_nulo", tx["Costo_Envio"].isna())
        add_flag("costo_no_positivo", tx["Costo_Envio"].notna() & (tx["Costo_Envio"] <= 0))

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

    default_exclude: set = set()
    tx_final, applied_flags, _ = apply_exclusions_button(
        tx, flag_cols, default_exclude, "Transacciones",
        help_text="Marca flags para excluir y presiona aplicar."
    )
    if applied_flags:
        actions.append("Exclusiones transacciones: " + ", ".join(applied_flags))

    desc = [
        "Fecha_Venta parseada con dayfirst=True (dd/mm/yyyy).",
        "Normalización de texto + mapeo + fuzzy para Ciudad/Estado/Canal.",
        "Detección de ciudades sospechosas (nombres de canal).",
        "Flags: fechas (invalida/futura), cantidad, tiempos, costos, unknown.",
        "Opcional: corregir año 2026→2025 en ventas futuras.",
    ]
    desc.extend(actions)

    return df_raw, tx, tx_final, tx_rare, flag_cols, desc


# -----------------------------------------------------------------------------
# Processing — Feedback
# -----------------------------------------------------------------------------

def process_feedback(df_raw: pd.DataFrame, cfg_container) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]
]:
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

    add_flag("transaccion_id_nulo", fb["Transaccion_ID_clean"].isna() | (fb["Transaccion_ID_clean"].astype("string").str.len() == 0))
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

    default_exclude: set = set()
    fb_final, applied_flags, _ = apply_exclusions_button(
        fb, flag_cols, default_exclude, "Feedback",
        help_text="Marca flags para excluir. No excluimos nada por defecto."
    )
    if applied_flags:
        actions.append("Exclusiones feedback: " + ", ".join(applied_flags))

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
        "Normalización: Recomienda_Marca (yes/no/maybe/unknown) y Ticket_Soporte_bool.",
        "NPS redondeado e interpretado en categorías.",
        "Flags: duplicados, rangos rating/NPS, comentario faltante, ticket inválido, etc.",
        f"Estrategia de JOIN: {fb_strategy}.",
    ]
    desc.extend(actions)

    return df_raw, fb, fb_final, fb_rare, fb_for_join, flag_cols, desc


# -----------------------------------------------------------------------------
# KPIs & Analysis
# -----------------------------------------------------------------------------

def build_kpi_cards(df_joined: pd.DataFrame):
    df = df_joined.copy()
    total_tx = int(len(df))
    margin_neg = int(df["Margen_Bruto"].lt(0).sum()) if "Margen_Bruto" in df.columns else 0
    sku_fantasma = int(df["flag__sku_no_existe_en_inventario"].sum()) if "flag__sku_no_existe_en_inventario" in df.columns else 0
    sin_feedback = int(df["flag__sin_feedback"].sum()) if "flag__sin_feedback" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transacciones", f"{total_tx:,}")
    col2.metric("Margen negativo", f"{margin_neg:,}")
    col3.metric("SKU fantasma", f"{sku_fantasma:,}")
    col4.metric("Sin feedback", f"{sin_feedback:,}")


def compute_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for P1..P5 + visuals-friendly tables."""
    results: Dict[str, Any] = {}
    if df is None or df.empty:
        return results

    d = df.copy()

    # Safety defaults
    if "Ingreso" not in d.columns:
        d["Ingreso"] = 0.0
    if "Margen_Bruto" not in d.columns:
        d["Margen_Bruto"] = 0.0
    if "Cantidad_Vendida" not in d.columns:
        d["Cantidad_Vendida"] = 0.0

    # -------------------------
    # P1) Margen negativo
    # -------------------------
    if "Categoria_clean" in d.columns:
        by_cat = (
            d.groupby("Categoria_clean", dropna=False)["Margen_Bruto"]
            .sum()
            .reset_index()
            .rename(columns={"Margen_Bruto": "Margen_Total"})
            .sort_values("Margen_Total")
        )
        results["margen_por_categoria"] = by_cat

    if "SKU_ID" in d.columns:
        by_sku = (
            d.groupby("SKU_ID", dropna=False)
            .agg(
                Margen_Total=("Margen_Bruto", "sum"),
                Cantidad_Total=("Cantidad_Vendida", "sum"),
                Ingreso_Total=("Ingreso", "sum"),
            )
            .reset_index()
        )
        results["sku_scatter_margen_vs_cantidad"] = by_sku
        results["margen_negativo"] = (
            by_sku[by_sku["Margen_Total"] < 0].sort_values("Margen_Total")
        )

    # -------------------------
    # P2) Logística vs NPS
    # -------------------------
    if (
        "Tiempo_Entrega_Real" in d.columns
        and "Satisfaccion_NPS" in d.columns
        and "Ciudad_Destino_clean" in d.columns
        and "Bodega_Origen_clean" in d.columns
    ):
        df_corr = d[
            ["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Real", "Satisfaccion_NPS"]
        ].dropna()

        corr_table = (
            df_corr.groupby(["Ciudad_Destino_clean", "Bodega_Origen_clean"])
            .agg(
                Tiempo_Entrega_Prom=("Tiempo_Entrega_Real", "mean"),
                NPS_Prom=("Satisfaccion_NPS", "mean"),
                N=("Satisfaccion_NPS", "count"),
            )
            .reset_index()
        )
        results["logistica_vs_nps"] = corr_table

    # -------------------------
    # P3) SKU fantasma
    # -------------------------
    if "flag__sku_no_existe_en_inventario" in d.columns:
        ghost = d[d["flag__sku_no_existe_en_inventario"]].copy()

        total_ing = float(d["Ingreso"].sum()) if float(d["Ingreso"].sum()) != 0 else 0.0
        lost_ing = float(ghost["Ingreso"].sum()) if "Ingreso" in ghost.columns else 0.0

        results["sku_fantasma"] = {
            "total_perdido": lost_ing,
            "num_transacciones": int(len(ghost)),
            "porcentaje": float(lost_ing / total_ing) if total_ing != 0 else 0.0,
        }

        donut_df = pd.DataFrame(
            [
                {"Tipo": "Ingreso normal", "Valor": max(total_ing - lost_ing, 0)},
                {"Tipo": "Ingreso en riesgo (SKU fantasma)", "Valor": max(lost_ing, 0)},
            ]
        )
        results["donut_ingreso_riesgo_fantasma"] = donut_df

        # ✅ FIX tabla vacía: siempre crear una categoría fallback aunque Categoria_clean esté NaN
        if "Categoria_clean" in ghost.columns:
            ghost["Categoria_fantasma"] = ghost["Categoria_clean"].astype("string").fillna("sin categoría (SKU no existe en inventario)")
            ghost.loc[ghost["Categoria_fantasma"].isin(["<NA>", "nan", "None", ""]), "Categoria_fantasma"] = "sin categoría (SKU no existe en inventario)"
        else:
            ghost["Categoria_fantasma"] = "sin categoría (SKU no existe en inventario)"

        if "Ingreso" in ghost.columns and not ghost.empty:
            ghost_by_cat = (
                ghost.groupby("Categoria_fantasma", dropna=False)["Ingreso"]
                .sum()
                .reset_index()
                .rename(columns={"Ingreso": "Ingreso_Perdido"})
                .sort_values("Ingreso_Perdido", ascending=False)
            )
            results["sku_fantasma_por_categoria"] = ghost_by_cat

    # -------------------------
    # P4) Stock vs NPS por categoría
    # -------------------------
    if (
        "Stock_Actual" in d.columns
        and "Satisfaccion_NPS" in d.columns
        and "Categoria_clean" in d.columns
    ):
        cat_scatter = (
            d.groupby("Categoria_clean", dropna=False)
            .agg(
                Stock_Prom=("Stock_Actual", "mean"),
                NPS_Prom=("Satisfaccion_NPS", "mean"),
                N=("Satisfaccion_NPS", "count"),
            )
            .reset_index()
        )
        results["stock_vs_nps_scatter"] = cat_scatter

        stock_thr = float(cat_scatter["Stock_Prom"].quantile(0.75)) if not cat_scatter.empty else np.nan
        red = cat_scatter[
            (cat_scatter["Stock_Prom"] >= stock_thr) & (cat_scatter["NPS_Prom"] <= 0)
        ].copy()
        red = red.sort_values(["NPS_Prom", "Stock_Prom"], ascending=[True, False])
        results["stock_alto_nps_bajo_alerta"] = red

    # -------------------------
    # P5) Riesgo operativo por bodega
    # -------------------------
    if "Bodega_Origen_clean" in d.columns and "Ticket_Soporte_bool" in d.columns:
        b = d.copy()
        b["Ticket_Soporte_bool"] = b["Ticket_Soporte_bool"].fillna(False)

        agg_dict: Dict[str, Any] = {
            "Total": ("Ticket_Soporte_bool", "count"),
            "Tickets_Abiertos": ("Ticket_Soporte_bool", lambda x: int((x == True).sum())),
        }
        if "Dias_desde_revision" in b.columns:
            agg_dict["Dias_Revision_Prom"] = ("Dias_desde_revision", "mean")
        if "Satisfaccion_NPS" in b.columns:
            agg_dict["NPS_Prom"] = ("Satisfaccion_NPS", "mean")

        rows = (
            b.groupby("Bodega_Origen_clean", dropna=False)
            .agg(**agg_dict)
            .reset_index()
        )

        if "Dias_Revision_Prom" not in rows.columns:
            rows["Dias_Revision_Prom"] = np.nan
        if "NPS_Prom" not in rows.columns:
            rows["NPS_Prom"] = np.nan

        rows["Ticket_Rate"] = rows["Tickets_Abiertos"] / rows["Total"].replace({0: np.nan})
        results["riesgo_bodega_plus"] = rows.sort_values("Ticket_Rate", ascending=False)

    return results


# -----------------------------------------------------------------------------
# Groq (optional)
# -----------------------------------------------------------------------------

def get_groq_api_key() -> Optional[str]:
    k = st.session_state.get("groq_api_key_input")
    if k:
        return k
    if hasattr(st, "secrets"):
        k2 = st.secrets.get("GROQ_API_KEY")
        if k2:
            return k2
    return os.environ.get("GROQ_API_KEY")


def call_groq(messages: List[Dict[str, str]]) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        return (
            "Error: No se encontró GROQ_API_KEY en st.secrets/variables de entorno, "
            "y no se pegó una key en el sidebar."
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
        "max_tokens": 650,
        "top_p": 0.9,
        "stream": False,
    }

    try:
        res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=35)
        if res.status_code != 200:
            return f"Error Groq {res.status_code}: {res.text}"
        data = res.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error al llamar a Groq: {e}"


def build_ai_prompt_from_analysis(
    df_filtered: pd.DataFrame,
    analysis_results: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Prompt SOLO con estadísticas agregadas.
    Requisito: 3 párrafos, recomendaciones numeradas y con complejidad (Baja/Media/Alta).
    Debe responder P1..P5 con evidencia.
    """
    messages: List[Dict[str, str]] = []
    messages.append({
        "role": "system",
        "content": (
            "Eres un consultor senior de analítica para retail, inventario y logística. Audiencia: Junta Directiva.\n"
            "Reglas estrictas:\n"
            "- NO expliques código. NO menciones pandas/streamlit/dataset.\n"
            "- NO pongas solamente datos, explícalos en el contexto del problema.\n"
            "- Usa los gráficos y tablas para analizar el problema.\n"
            "- Basarte ÚNICAMENTE en la evidencia numérica provista (USD, %, top SKUs, rankings, zonas).\n"
            "- Entregar EXACTAMENTE 3 párrafos.\n"
            "- Cada párrafo debe incluir recomendaciones numeradas y priorizadas por complejidad: (Baja), (Media), (Alta).\n"
            "- Debes cubrir P1..P5: margen negativo, logística vs NPS, SKU fantasma, stock alto con NPS bajo, revisión vs tickets.\n"
            "- En cada recomendación referencia explícita a la evidencia: cifras USD/%, o un top/ranking mencionado.\n"
            "Si faltan datos en algún punto, dilo como riesgo de medición y propone cómo cerrarlo."
        )
    })

    d = df_filtered.copy() if df_filtered is not None else pd.DataFrame()
    total_tx = int(len(d)) if not d.empty else 0
    total_ing = _safe_float(d["Ingreso"].sum(), 0.0) if (not d.empty and "Ingreso" in d.columns) else 0.0
    total_margen = _safe_float(d["Margen_Bruto"].sum(), 0.0) if (not d.empty and "Margen_Bruto" in d.columns) else 0.0

    # P1
    p1_counts = "Sin datos."
    p1_top_skus_txt = "No hay datos suficientes."
    p1_by_cat_txt = "No hay datos suficientes."
    if isinstance(analysis_results.get("margen_negativo"), pd.DataFrame):
        mneg = analysis_results["margen_negativo"].copy()
        p1_counts = f"SKUs con margen total negativo: {len(mneg):,}"
        p1_top_skus_txt = _df_top_to_text(mneg.sort_values("Margen_Total").head(10), ["SKU_ID", "Margen_Total", "Cantidad_Total", "Ingreso_Total"], n=10)
    if isinstance(analysis_results.get("margen_por_categoria"), pd.DataFrame):
        mc = analysis_results["margen_por_categoria"].copy().sort_values("Margen_Total").head(10)
        p1_by_cat_txt = _df_top_to_text(mc, ["Categoria_clean", "Margen_Total"], n=10)

    online_hint = "No hay desglose por canal disponible."
    if not d.empty and "Canal_Venta_clean" in d.columns and "Margen_Bruto" in d.columns:
        online = d[d["Canal_Venta_clean"].astype("string") == "web"].copy()
        if len(online) > 0:
            online_margen = _safe_float(online["Margen_Bruto"].sum(), 0.0)
            online_ing = _safe_float(online["Ingreso"].sum(), 0.0) if "Ingreso" in online.columns else 0.0
            online_hint = f"Canal web: transacciones={len(online):,}, ingreso={online_ing:.2f} USD, margen_total={online_margen:.2f} USD."

    # P2
    p2_txt = "No hay datos suficientes."
    if isinstance(analysis_results.get("logistica_vs_nps"), pd.DataFrame):
        l = analysis_results["logistica_vs_nps"].copy()
        if "N" in l.columns:
            l = l[l["N"] >= 5].copy()
        if not l.empty and {"Tiempo_Entrega_Prom", "NPS_Prom"}.issubset(l.columns):
            l["score_riesgo"] = (l["Tiempo_Entrega_Prom"].rank(pct=True)) + ((-l["NPS_Prom"]).rank(pct=True))
            l = l.sort_values("score_riesgo", ascending=False).head(8)
            p2_txt = _df_top_to_text(l, ["Ciudad_Destino_clean", "Bodega_Origen_clean", "Tiempo_Entrega_Prom", "NPS_Prom", "N"], n=8)

    # P3
    p3_summary = "No hay datos suficientes."
    p3_top_cat = "No hay datos suficientes."
    if isinstance(analysis_results.get("sku_fantasma"), dict):
        sf = analysis_results["sku_fantasma"]
        p3_summary = (
            f"Transacciones con SKU fantasma: {int(sf.get('num_transacciones', 0)):,}. "
            f"Ingreso en riesgo: {float(sf.get('total_perdido', 0.0)):.2f} USD. "
            f"% del ingreso total en riesgo: {float(sf.get('porcentaje', 0.0)) * 100:.2f}%."
        )
    if isinstance(analysis_results.get("sku_fantasma_por_categoria"), pd.DataFrame):
        gc = analysis_results["sku_fantasma_por_categoria"].copy().sort_values("Ingreso_Perdido", ascending=False).head(10)
        cat_col = "Categoria_fantasma" if "Categoria_fantasma" in gc.columns else ("Categoria_clean" if "Categoria_clean" in gc.columns else None)
        if cat_col:
            p3_top_cat = _df_top_to_text(gc, [cat_col, "Ingreso_Perdido"], n=10)

    # P4
    p4_alert_txt = "No hay datos suficientes."
    if isinstance(analysis_results.get("stock_alto_nps_bajo_alerta"), pd.DataFrame):
        red = analysis_results["stock_alto_nps_bajo_alerta"].copy()
        p4_alert_txt = _df_top_to_text(red, ["Categoria_clean", "Stock_Prom", "NPS_Prom", "N"], n=10)

    # P5
    p5_txt = "No hay datos suficientes."
    if isinstance(analysis_results.get("riesgo_bodega_plus"), pd.DataFrame):
        r = analysis_results["riesgo_bodega_plus"].copy().sort_values("Ticket_Rate", ascending=False).head(10)
        p5_txt = _df_top_to_text(r, ["Bodega_Origen_clean", "Ticket_Rate", "Tickets_Abiertos", "Total", "Dias_Revision_Prom", "NPS_Prom"], n=10)

    user_summary = f"""
RESUMEN EJECUTIVO (estadísticas agregadas)

Contexto general:
- Total de transacciones: {total_tx:,}
- Ingreso total estimado: {total_ing:.2f} USD
- Margen bruto total estimado: {total_margen:.2f} USD

P1) Fuga de capital y rentabilidad:
- {p1_counts}
- Top SKUs con peor margen total:
{p1_top_skus_txt}
- Categorías con peor margen total:
{p1_by_cat_txt}
- Señal por canal:
{online_hint}

P2) Crisis logística y cuellos de botella:
- Top combinaciones Ciudad–Bodega con mayor riesgo (tiempo alto + NPS bajo):
{p2_txt}

P3) Venta invisible (SKU fantasma):
- {p3_summary}
- Top categorías/segmentos asociados a ventas fantasma:
{p3_top_cat}

P4) Diagnóstico de fidelidad (stock alto + NPS bajo):
- Categorías en alerta:
{p4_alert_txt}

P5) Riesgo operativo (revisión vs tickets):
- Ranking bodegas por ticket rate + días desde revisión + NPS:
{p5_txt}

INSTRUCCIÓN DE SALIDA:
Narrativa de Negocio: No explique código; explique por qué la empresa está perdiendo dinero y cómo los datos lo demuestran.
ruebas Visuales: Incluya al menos 4 capturas de pantalla del Dashboard que soporten sus recomendaciones.
Entrega EXACTAMENTE la respuesta a las siguientes preguntas: 
Interrogantes Estratégicos Obligatorios
1. Fuga de Capital y Rentabilidad: Localice los SKUs que se están vendiendo con margen negativo. ¿Representan una pérdida aceptable por volumen o es una falla crítica de precios en el canal Online?
2. Crisis Logística y Cuellos de Botella: ¿En qué ciudades y bodegas la correlación entre Tiempo de Entrega y NPS bajo es más fuerte? Identifique la zona que requiere un cambio inmediato de operador.
3. Análisis de la Venta Invisible: Cuantifique el impacto financiero (en USD) de las ventas cuyos SKUs no están en el maestro de inventario. ¿Qué porcentaje del ingreso total está en riesgo por falta de control de inventario?
4. DiagnósticodeFidelidad:¿Existencategoríasdeproductosconaltadisponibilidad (stock alto) pero con un sentimiento de cliente negativo? Explique la paradoja: ¿Es mala calidad de producto o sobrecosto?
5. Storytelling de Riesgo Operativo: Visualice la relación entre la antigüedad de la Última Revisión del stock y la tasa de Tickets de Soporte. ¿Qué bodegas están operando a ciegas y cómo impacta esto en la satisfacción final?.
Usa las tablas y gráficos para sustentar cada punto
En cada párrafo incluye recomendaciones numeradas con complejidad (Baja/Media/Alta): Plan de Acción: Tres recomendaciones tácticas numeradas y priorizadas (Baja, Media, Alta complejidad)..
Conecta recomendaciones con los hallazgos anteriores y referencia evidencia (USD/%, tops, rankings).
""".strip()

    messages.append({"role": "user", "content": user_summary})
    return messages


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Challenge 02 — DSS Auditable", layout="wide")
    st.title("Challenge 02 — DSS Auditable (Inventario + Transacciones + Feedback + JOIN)")
    st.caption(
        "Controles laterales para limpieza auditable, JOIN y análisis. "
        "Las exclusiones no se aplican hasta presionar 'Aplicar exclusiones'."
    )

    # Sidebar uploads
    st.sidebar.header("📁 Cargar archivos")
    uploaded_inv = st.sidebar.file_uploader("1) inventario_central_v2.csv", type=["csv"], key="up_inv")
    uploaded_tx = st.sidebar.file_uploader("2) transacciones_logistica_v2.csv", type=["csv"], key="up_tx")
    uploaded_fb = st.sidebar.file_uploader("3) feedback_clientes_v2.csv", type=["csv"], key="up_fb")

    # -------------------------
    # 🧠 IA (Groq) - Sidebar
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
            help="Se usa solo en esta sesión. Para producción, usa st.secrets o variables de entorno."
        )
        st.sidebar.caption("✅ La clave se usa solo mientras la app está abierta.")
    else:
        st.sidebar.success("GROQ_API_KEY detectada en secrets/variables de entorno.")

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

    # Si faltan archivos, avisar y salir (pero dejando IA visible en sidebar)
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

    # Config containers
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

    # Health reports
    inv_health = compute_health_metrics(inv_raw_out, inv_clean, inv_final, inv_flags)
    tx_health = compute_health_metrics(tx_raw_out, tx_clean, tx_final, tx_flags)
    fb_health = compute_health_metrics(fb_raw_out, fb_clean, fb_final, fb_flags)

    # Join configuration
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

    # Documentation
    with doc_cfg:
        st.markdown("#### 🧾 Cómo estamos limpiando (documentación)")
        with st.expander("📦 Inventario — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in inv_desc]))
        with st.expander("🚚 Transacciones — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in tx_desc]))
        with st.expander("💬 Feedback — detalle", expanded=False):
            st.markdown("\n".join([f"• {x}" for x in fb_desc]))

    # -------------------------
    # JOIN
    # -------------------------
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

    # Tx ↔ Inv
    join_tx_inv = txj.merge(invj, on="SKU_ID", how="left", suffixes=("_tx", "_inv"), indicator="merge_tx_inv")
    join_tx_inv["flag__sku_no_existe_en_inventario"] = (join_tx_inv["merge_tx_inv"] == "left_only")

    # (Tx+Inv) ↔ Feedback
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

    # -------------------------
    # Feature engineering
    # -------------------------
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

    # KPI cards + analysis
    build_kpi_cards(joined)
    analysis_results = compute_analysis(joined)

    # Session default for IA text
    if "ai_last_text" not in st.session_state:
        st.session_state["ai_last_text"] = ""

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(["🧮 Auditoría", "📊 Operaciones", "😊 Cliente", "🧠 Insights IA", "📄 Documento PDF"])

    # --- Tab 0: Auditoría
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
            sel_dataset = st.selectbox("Elige dataset", ["Inventario", "Transacciones", "Feedback"], key="aud_sel_dataset")
            if sel_dataset == "Inventario":
                st.write(f"Filas con flags: {len(inv_rare):,}")
                st.dataframe(safe_for_streamlit_df(inv_rare.head(200)), use_container_width=True)
            elif sel_dataset == "Transacciones":
                st.write(f"Filas con flags: {len(tx_rare):,}")
                st.dataframe(safe_for_streamlit_df(tx_rare.head(200)), use_container_width=True)
            else:
                st.write(f"Filas con flags: {len(fb_rare):,}")
                st.dataframe(safe_for_streamlit_df(fb_rare.head(200)), use_container_width=True)

    # --- Tab 1: Operaciones (P1 + P3)
    with tabs[1]:
        st.subheader("📊 Operaciones")

        st.markdown("### 1️⃣ Margen negativo — insights accionables")

        if "margen_por_categoria" in analysis_results:
            df_cat = analysis_results["margen_por_categoria"].copy()
            show_only_neg = st.checkbox("Mostrar solo categorías con margen total negativo", value=False, key="p1_only_neg_cat")
            if show_only_neg:
                df_cat = df_cat[df_cat["Margen_Total"] < 0]
            chart_bar(df_cat, "Categoria_clean", "Margen_Total", "Margen total por categoría")

        if "sku_scatter_margen_vs_cantidad" in analysis_results:
            by_sku = analysis_results["sku_scatter_margen_vs_cantidad"].copy()
            show_only_neg_sku = st.checkbox("Mostrar solo SKUs con margen total negativo", value=True, key="p1_only_neg_sku")
            if show_only_neg_sku:
                by_sku = by_sku[by_sku["Margen_Total"] < 0]

            topn = st.slider("Top N SKUs por impacto (|margen|)", 50, 500, 200, step=50, key="p1_topn_sku")
            by_sku["abs_margen"] = by_sku["Margen_Total"].abs()
            by_sku = by_sku.sort_values("abs_margen", ascending=False).head(topn)

            chart_scatter(
                by_sku,
                x="Cantidad_Total",
                y="Margen_Total",
                color=None,
                size="Ingreso_Total" if "Ingreso_Total" in by_sku.columns else None,
                title="Prioriza SKUs: Cantidad total vs Margen total (tamaño = ingreso)"
            )

        st.markdown("---")
        st.markdown("### 3️⃣ Venta invisible (SKU fantasma)")

        if "sku_fantasma" in analysis_results:
            info = analysis_results["sku_fantasma"]
            st.write(
                f"Transacciones con SKU fantasma: **{info['num_transacciones']:,}**  \n"
                f"Ingreso en riesgo: **USD {info['total_perdido']:,.2f}**  \n"
                f"% del ingreso total en riesgo: **{info['porcentaje']*100:.2f}%**"
            )

        if "sku_fantasma_por_categoria" in analysis_results:
            ghost_cat = analysis_results["sku_fantasma_por_categoria"].copy()
            topc = st.slider("Top categorías por ingreso perdido", 5, 30, 10, key="p3_top_cat")
            chart_bar(
                ghost_cat.head(topc),
                "Categoria_fantasma",
                "Ingreso_Perdido",
                "Ingresos perdidos por SKU fantasma (Top categorías)"
            )

        if "donut_ingreso_riesgo_fantasma" in analysis_results:
            donut_df = analysis_results["donut_ingreso_riesgo_fantasma"].copy()
            chart_donut(donut_df, "Tipo", "Valor", "Proporción del ingreso en riesgo por SKU fantasma")

    # --- Tab 2: Cliente (P2 + P4 + P5)
    with tabs[2]:
        st.subheader("😊 Cliente")

        st.markdown("### 2️⃣ Logística vs NPS — visual")
        if "logistica_vs_nps" in analysis_results:
            corr_df = analysis_results["logistica_vs_nps"].copy()
            chart_scatter(
                corr_df,
                x="Tiempo_Entrega_Prom",
                y="NPS_Prom",
                color="Bodega_Origen_clean",
                size="N" if "N" in corr_df.columns else None,
                title="Tiempo de entrega promedio vs NPS promedio (por ciudad y bodega)"
            )
            st.dataframe(corr_df.sort_values("N", ascending=False).head(100), use_container_width=True)

        st.markdown("---")
        st.markdown("### 4️⃣ Stock alto y NPS bajo — cuadrantes")
        if "stock_vs_nps_scatter" in analysis_results:
            s = analysis_results["stock_vs_nps_scatter"].copy()
            chart_scatter(
                s,
                x="Stock_Prom",
                y="NPS_Prom",
                color=None,
                size="N" if "N" in s.columns else None,
                title="Por categoría: Stock promedio vs NPS promedio (tamaño = n)"
            )

        if "stock_alto_nps_bajo_alerta" in analysis_results:
            red = analysis_results["stock_alto_nps_bajo_alerta"].copy()
            st.markdown("**Categorías en alerta: alto stock (>=Q75) + NPS bajo (<=0)**")
            st.dataframe(red, use_container_width=True)

        st.markdown("---")
        st.markdown("### 5️⃣ Riesgo operativo por bodega — (plus: NPS)")
        if "riesgo_bodega_plus" in analysis_results:
            r = analysis_results["riesgo_bodega_plus"].copy()
            show = r[["Bodega_Origen_clean", "Ticket_Rate"]].dropna().sort_values("Ticket_Rate", ascending=False)
            chart_bar(show, "Bodega_Origen_clean", "Ticket_Rate", "Ticket rate por bodega (mayor = más riesgo)")

            if "Dias_Revision_Prom" in r.columns and r["Dias_Revision_Prom"].notna().any():
                rr = r.dropna(subset=["Dias_Revision_Prom", "Ticket_Rate"])
                chart_scatter(rr, "Dias_Revision_Prom", "Ticket_Rate", None, None, "Días desde revisión (prom) vs Ticket rate")

            if "NPS_Prom" in r.columns and r["NPS_Prom"].notna().any():
                nps_df = r[["Bodega_Origen_clean", "NPS_Prom"]].dropna().sort_values("NPS_Prom")
                chart_bar(nps_df, "Bodega_Origen_clean", "NPS_Prom", "NPS promedio por bodega (contexto)")
            st.dataframe(r, use_container_width=True)

    # --- Tab 3: IA
    with tabs[3]:
        st.subheader("🧠 Insights IA")
        st.write(
            "Genera recomendaciones con IA. El modelo recibe **solo estadísticas agregadas** "
            "(no se envían filas crudas)."
        )

        if st.button("Generar recomendaciones con IA", key="btn_ai"):
            with st.spinner("Consultando al modelo..."):
                prompt_messages = build_ai_prompt_from_analysis(joined, analysis_results)
                ai_result = call_groq(prompt_messages)
                st.session_state["ai_last_text"] = ai_result

        st.text_area("Recomendaciones IA (persisten para el PDF)", value=st.session_state["ai_last_text"], height=320)

        with st.expander("Ver prompt enviado (solo agregados)", expanded=False):
            try:
                msgs = build_ai_prompt_from_analysis(joined, analysis_results)
                st.code(msgs[-1]["content"])
            except Exception as e:
                st.error(f"No pude construir el prompt: {e}")

    # --- Tab 4: PDF (AUTO + vista previa)
    with tabs[4]:
        st.subheader("📄 Documento de Hallazgos (PDF)")
        st.write("Este PDF es el informe de consultoría para junta directiva. Incluye narrativa, evidencia y plan de acción.")

        # 1) Vista previa (igual a impresión)
        report_text = build_print_ready_report_text(
            df_filtered=joined,
            analysis_results=analysis_results,
            ai_text=st.session_state.get("ai_last_text", "")
        )

        st.markdown("### Vista previa (se imprime igual)")
        st.text_area("Contenido del informe", value=report_text, height=420)

        # 2) Evidencias visuales (auto desde gráficos)
        st.markdown("### Evidencia visual (auto desde el dashboard)")
        if not (ALTAIR_AVAILABLE and VLCONVERT_AVAILABLE):
            st.warning(
                "Para incluir gráficas en el PDF necesitas instalar `altair` y `vl-convert-python`. "
                "El PDF se generará con texto/tablas si no están."
            )
            charts_png = {}
        else:
            charts_png = build_pdf_charts_from_analysis(analysis_results)
            st.caption(f"Gráficas listas para PDF: {len(charts_png)} (ideal >= 4).")
            if charts_png:
                st.success("✅ Evidencias construidas automáticamente desde analysis_results.")
            else:
                st.warning("No pude construir evidencias visuales (revisa que existan P1..P5 en analysis_results).")

        # 3) Botón PDF
        st.markdown("### Generar PDF")
        if st.button("📄 Generar hallazgos.pdf (auto)", key="btn_make_pdf_auto"):
            # Requisito visual mínimo: intentamos 4; si no, avisamos pero generamos igual (para no bloquear)
            if len(charts_png) < 4:
                st.warning(f"Evidencia visual < 4 (solo {len(charts_png)}). Se generará igual con lo disponible.")

            pdf_bytes = generate_findings_pdf_auto(
                report_text=report_text,
                charts_png=charts_png
            )

            st.success("PDF generado ✅")
            st.download_button(
                "⬇️ Descargar hallazgos.pdf",
                data=pdf_bytes,
                file_name="hallazgos.pdf",
                mime="application/pdf",
                key="dl_pdf"
            )


if __name__ == "__main__":
    main()
