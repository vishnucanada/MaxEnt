"""Interactive Dash app: see how lambda_2, lambda_3 and the center (average)
reshape the max-entropy density. Uses the shared `maxent.core.pdf`.

Run from the repo root:  python -m legacy.visualizer
"""

import numpy as np
import scipy.integrate as integrate
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

from legacy.core import pdf

app = dash.Dash(__name__)


def normalized_pdf(x, lambda_2, lambda_3, average):
    lambdas = [0.0, lambda_2, lambda_3]
    integral, _ = integrate.quad(
        lambda t: pdf(t, lambdas, average), -np.inf, np.inf
    )
    if integral == 0 or np.isnan(integral):
        return np.full_like(x, np.nan)
    return pdf(x, lambdas, average) / integral


def _slider(id_, lo, hi, step, marks):
    return html.Div([html.Label(id_.split("-")[0]),
                     dcc.Slider(id=id_, min=lo, max=hi, step=step, value=0, marks=marks)])


app.layout = html.Div([
    dcc.Graph(id="pdf-plot"),
    _slider("lambda_2-slider", -5, 5, 0.1, {i: f"{i}" for i in range(-5, 6)}),
    _slider("lambda_3-slider", -1, 1, 0.01, {i / 10: f"{i/10}" for i in range(-10, 11)}),
    _slider("average-slider", -10, 10, 0.1, {i: f"{i}" for i in range(-10, 11)}),
    html.Div(id="output-container"),
])


@app.callback(
    [Output("pdf-plot", "figure"), Output("output-container", "children")],
    [Input("lambda_2-slider", "value"),
     Input("lambda_3-slider", "value"),
     Input("average-slider", "value")],
)
def update_plot(lambda_2, lambda_3, average):
    x = np.linspace(-20, 20, 1000)
    y = normalized_pdf(x, lambda_2, lambda_3, average)

    title = "Normalized Custom Probability Density Function"
    if np.isnan(y).any() or np.isinf(y).any():
        msg = "Encountered NaN or inf values in the PDF. Adjust the parameters."
        return {
            "data": [],
            "layout": go.Layout(title=title, xaxis={"title": "x"}, yaxis={"title": "PDF(x)"},
                                annotations=[{"text": msg, "xref": "paper", "yref": "paper",
                                              "showarrow": False, "font": {"size": 16, "color": "red"}}]),
        }, msg

    layout = go.Layout(title=title, xaxis={"title": "x"}, yaxis={"title": "PDF(x)"}, annotations=[])
    output_text = [
        f"Is PDF non-negative? {np.all(y >= 0)}",
        f"Approximate integral of PDF: {np.trapz(y, x):.4f}",
    ]
    return {"data": [go.Scatter(x=x, y=y, mode="lines", name="PDF")], "layout": layout}, output_text


if __name__ == "__main__":
    app.run_server(debug=True)
