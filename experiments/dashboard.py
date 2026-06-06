"""Real-time dashboard for federated maxent-HDC distribution estimation.

Watch the global distribution + 99% VaR assemble as banks join the consortium
(press Play), and drag the privacy budget epsilon to see the privately-recovered
density update live against the DP-histogram baseline. Per-bank sketches are
precomputed once at startup, so every interaction is instant.

    pip install -e ".[dashboard]"        # or: pip install dash plotly
    python -m experiments.dashboard
    # then open http://127.0.0.1:8050

Datasets: 'synthetic' (instant) and 'creditcard' (needs data/creditcard.csv).
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State  # top-level (works on Dash 2.x and 3.x)
import plotly.graph_objs as go

from maxent import (
    sample_frequencies, institution_sketch, merge, privatize, readout, var_es,
    dp_histogram, make_consortium,
)

LEVEL = 0.99
DELTA = 1e-5
EPS_CHOICES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DP_REG = 1e-3

# --- palette ----------------------------------------------------------------
INK, MUTED, LINE = "#0f172a", "#64748b", "#e2e8f0"
BG, CARD = "#f1f5f9", "#ffffff"
ACCENT = "#4f46e5"                 # indigo (brand)
C_TRUTH, C_FED, C_HIST = "#94a3b8", "#e11d48", "#059669"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_CACHE = {}


# --- dataset preparation (sketches precomputed once) ------------------------

def prepare_dataset(name):
    if name == "synthetic":
        rng = np.random.default_rng(0)
        banks, _ = make_consortium(25, 2000, rng)
        grid = np.linspace(-6, 10, 800)
        M, bw, to_usd, unit = 32, 1.0, (lambda z: np.asarray(z)), ""
    elif name == "creditcard":
        from experiments.federated_creditcard import load_amounts, noniid_partition
        z, (mean, std) = load_amounts()
        banks = noniid_partition(z, 30, 10, 0.3, np.random.default_rng(0))
        grid = np.linspace(-3, 6, 1000)
        M, bw = 32, 1.0
        to_usd, unit = (lambda zz: np.expm1(np.asarray(zz) * std + mean)), "$"
    else:
        raise ValueError(name)

    omega = sample_frequencies(M, bw, "gaussian", 11)
    sketches = [institution_sketch(b, omega) for b in banks]
    pooled = np.concatenate(banks)
    truth = np.append(np.histogram(pooled, bins=grid, density=True)[0], 0.0)
    return {
        "banks": banks, "omega": omega, "sketches": sketches, "grid": grid,
        "edges": np.linspace(grid[0], grid[-1], 60), "M": M,
        "truth": truth, "v_emp": float(np.quantile(pooled, LEVEL)),
        "to_usd": to_usd, "unit": unit, "n_banks": len(banks),
    }


def get_dataset(name):
    if name not in _CACHE:
        _CACHE[name] = prepare_dataset(name)
    return _CACHE[name]


# --- pure figure builder (headless-testable) --------------------------------

def build_figure(name, n_banks, eps, show_hist):
    d = get_dataset(name)
    grid, omega, to_usd, unit = d["grid"], d["omega"], d["to_usd"], d["unit"]
    n_banks = int(np.clip(n_banks, 1, d["n_banks"]))

    S, N = merge(d["sketches"][:n_banks])
    if eps is None:
        pdf = readout(S, N, omega, grid, reg=0.0)
    else:
        S = privatize(S, eps, DELTA, np.random.default_rng(0))
        pdf = readout(S, N, omega, grid, reg=DP_REG)
    v_fed = var_es(pdf, grid, LEVEL)[0]

    traces = [
        go.Scatter(x=grid, y=d["truth"], fill="tozeroy", name="True distribution (all data)",
                   line={"color": C_TRUTH, "width": 0}, fillcolor="rgba(148,163,184,0.35)",
                   hoverinfo="skip"),
        go.Scatter(x=grid, y=pdf, name="Federated estimate", mode="lines",
                   line={"color": C_FED, "width": 3}),
    ]
    if show_hist and eps is not None:
        his = dp_histogram(d["banks"][:n_banks], grid, d["edges"], eps, DELTA,
                           np.random.default_rng(1))
        traces.append(go.Scatter(x=grid, y=his, name="DP-histogram baseline", mode="lines",
                                  line={"color": C_HIST, "width": 2, "dash": "dot"}))

    ymax = float(max(d["truth"].max(), pdf.max()) * 1.18)
    money = lambda z: f"{unit}{to_usd(z):,.0f}"
    shapes, annos = [], []
    for x, color, label in [(d["v_emp"], "#0f172a", "True VaR"), (v_fed, C_FED, "Est. VaR")]:
        shapes.append({"type": "line", "x0": x, "x1": x, "y0": 0, "y1": ymax,
                       "line": {"color": color, "dash": "dot", "width": 1.5}})
        annos.append({"x": x, "y": ymax, "text": f"{label}<br>{money(x)}", "showarrow": False,
                      "font": {"size": 11, "color": color}, "yanchor": "top",
                      "bgcolor": "rgba(255,255,255,0.7)"})

    fig = go.Figure(traces, go.Layout(
        template="plotly_white",
        title={"text": f"<b>{n_banks} of {d['n_banks']} banks merged</b>"
                       + ("   ·   no privacy" if eps is None else f"   ·   ε = {eps}"),
               "x": 0.02, "font": {"size": 16, "color": INK}},
        xaxis={"title": "transaction size  (standardized log-amount)", "showgrid": False},
        yaxis={"title": "probability density", "range": [0, ymax], "gridcolor": LINE},
        shapes=shapes, annotations=annos, font={"family": FONT, "color": INK},
        legend={"orientation": "h", "y": -0.22, "x": 0}, margin={"t": 44, "l": 60, "r": 20, "b": 70},
        height=460, paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified"))

    metrics = [
        ("Banks merged", f"{n_banks} / {d['n_banks']}", "contributing their sketch"),
        ("Transactions", f"{N:,}", "summarized so far"),
        ("VaR — estimate", money(v_fed), "our private 99% VaR"),
        ("VaR — truth", money(d["v_emp"]), "ground truth (all data)"),
        ("VaR error", f"{unit}{abs(to_usd(v_fed) - to_usd(d['v_emp'])):,.0f}", "estimate − truth"),
        ("Comm / bank", f"{2 * d['M'] * 8} B", f"vs {N // max(n_banks,1) * 8:,} B of raw rows"),
        ("Privacy", "off" if eps is None else f"ε = {eps}", "smaller ε = more private"),
    ]
    return fig, metrics


# --- presentational helpers -------------------------------------------------

def _card(children, **style):
    base = {"background": CARD, "borderRadius": "14px", "padding": "18px 20px",
            "boxShadow": "0 1px 3px rgba(15,23,42,0.08)", "border": f"1px solid {LINE}"}
    base.update(style)
    return html.Div(children, style=base)


def _step(n, title, desc):
    badge = html.Div(str(n), style={
        "minWidth": "26px", "height": "26px", "borderRadius": "50%", "background": ACCENT,
        "color": "white", "display": "flex", "alignItems": "center",
        "justifyContent": "center", "fontWeight": "700", "fontSize": "13px"})
    body = html.Div([html.Div(title, style={"fontWeight": "600", "color": INK, "fontSize": "14px"}),
                     html.Div(desc, style={"color": MUTED, "fontSize": "12.5px", "lineHeight": "1.45"})])
    return html.Div([badge, body], style={"display": "flex", "gap": "12px", "marginBottom": "14px"})


def _swatch(color, label, dash=False):
    bar = html.Div(style={"width": "22px", "height": "0", "marginTop": "9px",
                          "borderTop": f"3px {'dotted' if dash else 'solid'} {color}"})
    return html.Div([bar, html.Span(label, style={"fontSize": "12.5px", "color": INK})],
                    style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "7px"})


def _metric_cards(metrics):
    out = []
    for label, value, hint in metrics:
        out.append(html.Div([
            html.Div(label.upper(), style={"fontSize": "10.5px", "letterSpacing": ".04em",
                                           "color": MUTED, "fontWeight": "600"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": "700", "color": INK,
                                   "margin": "2px 0"}),
            html.Div(hint, style={"fontSize": "11px", "color": MUTED}),
        ], style={"background": CARD, "border": f"1px solid {LINE}", "borderRadius": "12px",
                  "padding": "12px 14px", "flex": "1", "minWidth": "120px"}))
    return out


def _control(label, hint, component):
    return html.Div([
        html.Div(label, style={"fontWeight": "600", "fontSize": "13px", "color": INK}),
        html.Div(hint, style={"fontSize": "11.5px", "color": MUTED, "marginBottom": "8px"}),
        component,
    ], style={"flex": "1", "minWidth": "200px"})


# --- app --------------------------------------------------------------------

app = dash.Dash(__name__, title="Federated MaxEnt")

SIDEBAR = _card([
    html.Div("HOW IT WORKS", style={"fontSize": "11px", "letterSpacing": ".06em",
                                     "color": ACCENT, "fontWeight": "700", "marginBottom": "14px"}),
    _step(1, "Encode", "Each bank turns every transaction amount into a high-dimensional vector "
                       "(a random-Fourier 'hypervector')."),
    _step(2, "Bundle", "A bank sums its vectors into ONE fixed-size sketch + a count. Raw rows "
                       "never leave the bank."),
    _step(3, "Merge", "The server adds the banks' sketches together — mathematically identical to "
                      "pooling all the data, but it's just vector addition, in one round."),
    _step(4, "Privatize", "Calibrated noise is added to the merged sketch for differential privacy. "
                          "Smaller ε → more noise → more privacy."),
    _step(5, "Recover", "The server fits the maximum-entropy density that matches the sketch, giving "
                        "the global distribution and its 99% VaR (Value-at-Risk)."),
    html.Hr(style={"border": "none", "borderTop": f"1px solid {LINE}", "margin": "8px 0 14px"}),
    html.Div("READING THE CHART", style={"fontSize": "11px", "letterSpacing": ".06em",
                                         "color": ACCENT, "fontWeight": "700", "marginBottom": "12px"}),
    _swatch(C_TRUTH, "True distribution (all data pooled)"),
    _swatch(C_FED, "Federated private estimate"),
    _swatch(C_HIST, "DP-histogram baseline (comparison)", dash=True),
    _swatch("#0f172a", "99% VaR — true vs estimated", dash=True),
], marginBottom="0")

CONTROLS = _card([
    html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "alignItems": "flex-start"},
             children=[
        _control("Dataset", "what we estimate the distribution of",
                 dcc.Dropdown(id="dataset", value="synthetic", clearable=False,
                              options=[{"label": "Synthetic consortium", "value": "synthetic"},
                                       {"label": "Credit-card (real)", "value": "creditcard"}])),
        _control("Privacy budget  ε", "drag toward 0.1 for strong privacy; 'off' = none",
                 dcc.Slider(id="eps", min=0, max=len(EPS_CHOICES), step=1, value=len(EPS_CHOICES),
                            marks={i: str(e) for i, e in enumerate(EPS_CHOICES)}
                                  | {len(EPS_CHOICES): "off"})),
        html.Div([
            html.Div("Baseline", style={"fontWeight": "600", "fontSize": "13px", "color": INK}),
            html.Div("overlay the standard method", style={"fontSize": "11.5px", "color": MUTED, "marginBottom": "8px"}),
            dcc.Checklist(id="show_hist", options=[{"label": " DP histogram", "value": "y"}],
                          value=["y"], style={"fontSize": "13px"}),
        ], style={"minWidth": "150px"}),
    ]),
    html.Div(style={"display": "flex", "gap": "16px", "alignItems": "center", "marginTop": "18px"},
             children=[
        html.Button("▶  Play", id="play", n_clicks=0, style={
            "background": ACCENT, "color": "white", "border": "none", "borderRadius": "10px",
            "padding": "10px 20px", "fontWeight": "600", "cursor": "pointer", "fontSize": "14px"}),
        html.Div(_control("Banks merged", "how many banks have contributed their sketch",
                          dcc.Slider(id="banks", min=1, max=25, step=1, value=1, marks=None,
                                     tooltip={"placement": "bottom", "always_visible": True})),
                 style={"flex": "1"}),
    ]),
])


app.layout = html.Div(style={"background": BG, "minHeight": "100vh", "fontFamily": FONT,
                             "padding": "26px 32px"}, children=[
    html.Div([
        html.H1("Federated Maximum-Entropy Distribution Estimation",
                style={"margin": "0", "fontSize": "26px", "color": INK}),
        html.P("Estimate a global distribution across banks — and its tail risk — without ever "
               "sharing raw data. Private, one round, fixed cost per bank.",
               style={"margin": "6px 0 22px", "color": MUTED, "fontSize": "14.5px"}),
    ]),
    html.Div(style={"display": "flex", "gap": "22px", "alignItems": "flex-start", "flexWrap": "wrap"},
             children=[
        html.Div(SIDEBAR, style={"width": "330px", "flexShrink": "0"}),
        html.Div(style={"flex": "1", "minWidth": "520px", "display": "flex",
                        "flexDirection": "column", "gap": "18px"}, children=[
            CONTROLS,
            _card(dcc.Graph(id="graph", config={"displayModeBar": False}), padding="8px"),
            html.Div(id="metrics", style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),
    ]),
    dcc.Interval(id="timer", interval=650, disabled=True),
])


# --- callbacks (unchanged logic) --------------------------------------------

@app.callback(Output("banks", "max"), Output("banks", "value"), Input("dataset", "value"))
def _on_dataset(name):
    return get_dataset(name)["n_banks"], 1


@app.callback(Output("timer", "disabled"), Output("play", "children"),
              Input("play", "n_clicks"), State("timer", "disabled"))
def _toggle_play(n, disabled):
    if not n:
        return True, "▶  Play"
    return (not disabled), ("⏸  Pause" if disabled else "▶  Play")


@app.callback(Output("banks", "value", allow_duplicate=True),
              Input("timer", "n_intervals"),
              State("banks", "value"), State("banks", "max"), prevent_initial_call=True)
def _advance(_, value, vmax):
    return (value % vmax) + 1


@app.callback(Output("graph", "figure"), Output("metrics", "children"),
              Input("dataset", "value"), Input("banks", "value"),
              Input("eps", "value"), Input("show_hist", "value"))
def _update(name, n_banks, eps_idx, show_hist):
    eps = None if eps_idx == len(EPS_CHOICES) else EPS_CHOICES[int(eps_idx)]
    fig, metrics = build_figure(name, n_banks, eps, bool(show_hist))
    return fig, _metric_cards(metrics)


if __name__ == "__main__":
    app.run(debug=False, port=8050)
