"""Real-time dashboard for federated maxent-HDC distribution estimation.

Watch the global distribution + 99% VaR assemble as banks join the consortium
(press Play), and drag the privacy budget epsilon to see the privately-recovered
density update live against the DP-histogram baseline. Per-bank sketches are
precomputed once at startup, so every interaction is instant.

    pip install dash plotly        # (also in legacy/requirements.txt)
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
from maxent.data import DATA_DIR

LEVEL = 0.99
DELTA = 1e-5
EPS_CHOICES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DP_REG = 1e-3

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
        go.Scatter(x=grid, y=d["truth"], fill="tozeroy", name="empirical (all data)",
                   line={"color": "lightgray"}),
        go.Scatter(x=grid, y=pdf, name="federated maxent", line={"color": "crimson"}),
    ]
    if show_hist and eps is not None:
        his = dp_histogram(d["banks"][:n_banks], grid, d["edges"], eps, DELTA,
                           np.random.default_rng(1))
        traces.append(go.Scatter(x=grid, y=his, name="DP histogram",
                                  line={"color": "seagreen", "dash": "dot"}))

    ymax = float(max(d["truth"].max(), pdf.max()) * 1.1)
    shapes = [
        {"type": "line", "x0": d["v_emp"], "x1": d["v_emp"], "y0": 0, "y1": ymax,
         "line": {"color": "black", "dash": "dot"}},
        {"type": "line", "x0": v_fed, "x1": v_fed, "y0": 0, "y1": ymax,
         "line": {"color": "crimson", "dash": "dash"}},
    ]
    fig = go.Figure(traces, go.Layout(
        title=f"{name}: {n_banks}/{d['n_banks']} banks merged"
              + ("  (no privacy)" if eps is None else f"  ·  ε = {eps}"),
        xaxis={"title": "standardized log-amount"}, yaxis={"title": "density", "range": [0, ymax]},
        shapes=shapes, legend={"orientation": "h"}, margin={"t": 40}))

    fmt = lambda z: f"{unit}{to_usd(z):,.0f}"
    metrics = {
        "banks": f"{n_banks} / {d['n_banks']}",
        "transactions": f"{N:,}",
        "VaR (federated)": fmt(v_fed),
        "VaR (empirical)": fmt(d["v_emp"]),
        "VaR error": f"{unit}{abs(to_usd(v_fed) - to_usd(d['v_emp'])):,.0f}",
        "comm / bank": f"{2 * d['M'] * 8} B  (vs {N // max(n_banks,1) * 8:,} B raw)",
        "privacy": "off" if eps is None else f"ε = {eps}",
    }
    return fig, metrics


# --- Dash app ---------------------------------------------------------------

app = dash.Dash(__name__)


def _metric_cards(metrics):
    return [html.Div([html.Div(k, style={"fontSize": "12px", "color": "#666"}),
                      html.Div(v, style={"fontSize": "18px", "fontWeight": "bold"})],
                     style={"padding": "8px 14px", "background": "#f4f4f7",
                            "borderRadius": "8px", "minWidth": "120px"})
            for k, v in metrics.items()]


app.layout = html.Div([
    html.H2("Federated maximum-entropy distribution estimation"),
    html.Div([
        html.Div([html.Label("Dataset"),
                  dcc.Dropdown(id="dataset", value="synthetic", clearable=False,
                               options=[{"label": "Synthetic consortium", "value": "synthetic"},
                                        {"label": "Credit-card (real)", "value": "creditcard"}],
                               style={"width": "220px"})]),
        html.Div([html.Label("Privacy budget ε"),
                  dcc.Slider(id="eps", min=0, max=len(EPS_CHOICES), step=1,
                             value=len(EPS_CHOICES),
                             marks={i: str(e) for i, e in enumerate(EPS_CHOICES)}
                                   | {len(EPS_CHOICES): "off"})],
                 style={"flex": 1, "padding": "0 20px"}),
        html.Div([dcc.Checklist(id="show_hist", options=[{"label": " DP histogram", "value": "y"}],
                                value=["y"])]),
        html.Button("▶ Play", id="play", n_clicks=0),
    ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),

    html.Div([html.Label("Banks merged"),
              dcc.Slider(id="banks", min=1, max=25, step=1, value=1, marks=None,
                         tooltip={"placement": "bottom", "always_visible": True})],
             style={"padding": "10px 0"}),

    dcc.Graph(id="graph"),
    html.Div(id="metrics", style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    dcc.Interval(id="timer", interval=600, disabled=True),
])


@app.callback(Output("banks", "max"), Output("banks", "value"),
              Input("dataset", "value"))
def _on_dataset(name):
    return get_dataset(name)["n_banks"], 1


@app.callback(Output("timer", "disabled"), Output("play", "children"),
              Input("play", "n_clicks"), State("timer", "disabled"))
def _toggle_play(n, disabled):
    if not n:
        return True, "▶ Play"
    return (not disabled), ("⏸ Pause" if disabled else "▶ Play")


@app.callback(Output("banks", "value", allow_duplicate=True),
              Input("timer", "n_intervals"),
              State("banks", "value"), State("banks", "max"),
              prevent_initial_call=True)
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
