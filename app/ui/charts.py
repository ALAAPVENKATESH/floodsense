"""Plotly charts styled for FloodSense. Matplotlib figures stay in reports/figures."""

from __future__ import annotations

import plotly.graph_objects as go

from app.ui.theme import RISK_COLORS, ChartTheme, display_name

_CLASS_ORDER = ("LOW", "MODERATE", "HIGH")


def _ordered(labels) -> list[str]:
    present = list(labels)
    ordered = [name for name in _CLASS_ORDER if name in present]
    return ordered + [name for name in present if name not in ordered]


def finish(fig: go.Figure, theme: ChartTheme, title: str, height: int = 340) -> go.Figure:
    """Apply the shared transparent layout."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=theme.text), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=theme.text, size=12),
        height=height,
        margin=dict(l=8, r=12, t=52, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)),
        hoverlabel=dict(font=dict(family="Inter, sans-serif", size=12)),
    )
    fig.update_xaxes(gridcolor=theme.grid, zeroline=False, linecolor=theme.grid, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=theme.grid, zeroline=False, linecolor=theme.grid, title_font=dict(size=12))
    return fig


def class_donut(counts: dict, theme: ChartTheme) -> go.Figure:
    """Share of each risk class."""
    labels = _ordered(counts)
    colors = [RISK_COLORS.get(name, theme.brand) for name in labels]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=[counts[name] for name in labels],
            hole=0.62,
            sort=False,
            direction="clockwise",
            marker=dict(colors=colors, line=dict(color=theme.surface, width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value} rows · %{percent}<extra></extra>",
        )
    )
    return finish(fig, theme, "Class distribution", 340)


def confusion_heatmap(matrix, labels: list[str], theme: ChartTheme, normalized: bool = False) -> go.Figure:
    """Annotated confusion matrix. Rows are the true class."""
    grid = [[float(value) for value in row] for row in matrix]
    if normalized:
        shown = []
        for row in grid:
            total = sum(row) or 1.0
            shown.append([value / total for value in row])
        text = [[f"{value:.0%}" for value in row] for row in shown]
        z = shown
        title = "Confusion matrix (row share)"
        hover = "True %{y}<br>Predicted %{x}<br>%{text}<extra></extra>"
    else:
        text = [[str(int(value)) for value in row] for row in grid]
        z = grid
        title = "Confusion matrix (counts)"
        hover = "True %{y}<br>Predicted %{x}<br>%{text} rows<extra></extra>"
    if theme.mode == "dark":
        start, end, ink = "#1A2238", "#4F46E5", "#F8FAFC"
    else:
        start, end, ink = "#F8FAFC", "#C7D2FE", "#0F172A"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(labels),
            y=list(labels),
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=ink, size=13),
            colorscale=[[0, start], [1, end]],
            showscale=False,
            hovertemplate=hover,
            xgap=4,
            ygap=4,
        )
    )
    fig.update_yaxes(autorange="reversed", title="True class")
    fig.update_xaxes(title="Predicted class")
    return finish(fig, theme, title, 360)


def importance_bars(
    rows: list[dict],
    theme: ChartTheme,
    title: str,
    limit: int = 8,
    x_title: str = "Share of importance",
) -> go.Figure:
    """Horizontal importance bars, largest at the top."""
    chosen = list(rows)[:limit]
    names = [display_name(row["feature"]) for row in chosen][::-1]
    values = [float(row["importance"]) for row in chosen][::-1]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(color=theme.brand, cornerradius=6),
            hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title=x_title)
    return finish(fig, theme, title, 340)


def metric_compare(train: dict, test: dict, theme: ChartTheme) -> go.Figure:
    """Grouped bars for the train and test scores."""
    keys = (
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1"),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Train",
            x=[label for _, label in keys],
            y=[float(train[key]) for key, _ in keys],
            marker=dict(color=theme.brand, cornerradius=6),
            hovertemplate="Train %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Test",
            x=[label for _, label in keys],
            y=[float(test[key]) for key, _ in keys],
            marker=dict(color="#94A3B8", cornerradius=6),
            hovertemplate="Test %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Score", range=[0, 1])
    return finish(fig, theme, "Train vs test", 340)


def probability_bars(probabilities: dict, theme: ChartTheme) -> go.Figure:
    """Class probabilities for one prediction."""
    labels = _ordered(probabilities)
    fig = go.Figure(
        go.Bar(
            x=[float(probabilities[name]) for name in labels],
            y=labels,
            orientation="h",
            marker=dict(color=[RISK_COLORS.get(name, theme.brand) for name in labels], cornerradius=6),
            hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Probability", range=[0, 1], tickformat=".0%")
    return finish(fig, theme, "Class probabilities", 240)


def vote_bar(votes: dict, theme: ChartTheme) -> go.Figure:
    """One stacked bar of tree votes."""
    labels = _ordered(votes)
    fig = go.Figure()
    for name in labels:
        fig.add_trace(
            go.Bar(
                name=name,
                x=[int(votes[name])],
                y=["Votes"],
                orientation="h",
                marker=dict(color=RISK_COLORS.get(name, theme.brand), cornerradius=6),
                hovertemplate=f"{name}<br>%{{x}} trees<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Trees")
    return finish(fig, theme, "Tree votes", 180)


def gauge(confidence: float, label: str, theme: ChartTheme) -> go.Figure:
    """Semicircle confidence gauge for the predicted class."""
    color = RISK_COLORS.get(label, theme.brand)
    soft = "#1A2238" if theme.mode == "dark" else "#E2E8F0"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(confidence),
            number={"suffix": "%", "font": {"size": 28, "color": theme.text}},
            title={"text": "Confidence", "font": {"size": 13, "color": theme.muted}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": theme.muted, "tickfont": {"color": theme.muted}},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": soft,
                "borderwidth": 0,
                "shape": "angular",
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=theme.text),
        height=220,
        margin=dict(l=16, r=16, t=36, b=0),
    )
    return fig


def what_if(xs, ys, current: float, feature: str, theme: ChartTheme) -> go.Figure:
    """HIGH-class probability while one input moves and the rest stay fixed."""
    fig = go.Figure(
        go.Scatter(
            x=list(xs),
            y=list(ys),
            mode="lines",
            line=dict(color=RISK_COLORS["HIGH"], width=2.5, shape="spline"),
            hovertemplate=f"{display_name(feature)}=%{{x:.3g}}<br>P(HIGH)=%{{y:.1%}}<extra></extra>",
        )
    )
    fig.add_vline(x=float(current), line_dash="dot", line_color=theme.brand, line_width=1.5)
    fig.update_xaxes(title=display_name(feature))
    fig.update_yaxes(title="P(HIGH)", range=[0, 1], tickformat=".0%")
    return finish(fig, theme, f"What-if · {display_name(feature)}", 300)


def learning_curve(rows: list[dict], theme: ChartTheme) -> go.Figure:
    """Out-of-bag accuracy as trees are added."""
    fig = go.Figure(
        go.Scatter(
            x=[row["n_estimators"] for row in rows],
            y=[row["oob_accuracy"] for row in rows],
            mode="lines+markers",
            line=dict(color=theme.brand, width=2.5),
            marker=dict(size=8, color=theme.brand),
            hovertemplate="%{x} trees<br>OOB accuracy %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Trees")
    fig.update_yaxes(title="Out-of-bag accuracy", range=[0, 1])
    return finish(fig, theme, "Learning curve", 320)


def depth_curve(depths: list[int], scores: list[float], theme: ChartTheme) -> go.Figure:
    """Best validation macro-F1 at each depth in the search log."""
    fig = go.Figure(
        go.Scatter(
            x=depths,
            y=scores,
            mode="lines+markers",
            line=dict(color=theme.brand, width=2.5),
            marker=dict(size=9, color=theme.brand),
            hovertemplate="Depth %{x}<br>Validation macro F1 %{y:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Max depth", dtick=1)
    fig.update_yaxes(title="Validation macro F1", range=[0, 1])
    return finish(fig, theme, "Depth vs validation F1", 320)


def histogram(values, feature: str, theme: ChartTheme) -> go.Figure:
    """Distribution of one numeric column."""
    fig = go.Figure(
        go.Histogram(
            x=list(values),
            nbinsx=32,
            marker=dict(color=theme.brand, cornerradius=4),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    fig.update_xaxes(title=display_name(feature))
    fig.update_yaxes(title="Rows")
    return finish(fig, theme, f"{display_name(feature)} distribution", 320)


def correlation_heatmap(columns: list[str], matrix, theme: ChartTheme) -> go.Figure:
    """Pairwise correlation of numeric columns."""
    names = [display_name(name) for name in columns]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=[[f"{float(value):.2f}" for value in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=10, color=theme.text),
            hovertemplate="%{y} vs %{x}<br>r=%{z:.2f}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return finish(fig, theme, "Feature correlation", 460)


def batch_counts(counts: dict, theme: ChartTheme) -> go.Figure:
    """How many uploaded rows fell in each class."""
    labels = _ordered(counts)
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=[counts.get(name, 0) for name in labels],
            marker=dict(color=[RISK_COLORS.get(name, theme.brand) for name in labels], cornerradius=6),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    fig.update_yaxes(title="Rows")
    return finish(fig, theme, "Predicted class counts", 300)


def location_map(frame, theme: ChartTheme) -> go.Figure:
    """OpenStreetMap points colored by the predicted class. Needs lat and lon."""
    colors = [RISK_COLORS.get(str(label), theme.brand) for label in frame["risk_class"]]
    fig = go.Figure(
        go.Scattermapbox(
            lat=frame["latitude"],
            lon=frame["longitude"],
            mode="markers",
            marker=dict(size=9, color=colors),
            text=frame["risk_class"],
            hovertemplate="Lat %{lat:.4f}<br>Lon %{lon:.4f}<br>%{text}<extra></extra>",
        )
    )
    lat = float(frame["latitude"].mean())
    lon = float(frame["longitude"].mean())
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=8),
        title=dict(text="Locations", font=dict(size=15, color=theme.text), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=theme.text),
        height=420,
        margin=dict(l=0, r=0, t=48, b=0),
    )
    return fig
