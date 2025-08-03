import numpy as np
import scipy.integrate as integrate
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

def unnormalized_pdf(x, lambda_2, lambda_3, average):
    exponential = lambda_2 * x + lambda_3 * ((x - average)**2)
    return np.exp(exponential)

def normalized_pdf(x, lambda_2, lambda_3, average):
    integral, _ = integrate.quad(
        lambda t: unnormalized_pdf(t, lambda_2, lambda_3, average), 
        -np.inf, 
        np.inf
    )
    if integral == 0 or np.isnan(integral):
        return np.full_like(x, np.nan)
    return unnormalized_pdf(x, lambda_2, lambda_3, average) / integral

app.layout = html.Div([
    dcc.Graph(id='pdf-plot'),
    html.Div([
        html.Label('lambda_2'),
        dcc.Slider(
            id='lambda_2-slider',
            min=-5,
            max=5,
            step=0.1,
            value=0,
            marks={i: f'{i}' for i in range(-5, 6)}
        ),
    ]),
    html.Div([
        html.Label('lambda_3'),
        dcc.Slider(
            id='lambda_3-slider',
            min=-1,
            max=1,
            step=0.01,
            value=0,
            marks={i/10: f'{i/10}' for i in range(-10, 11)}
        ),
    ]),
    html.Div([
        html.Label('average'),
        dcc.Slider(
            id='average-slider',
            min=-10,
            max=10,
            step=0.1,
            value=0,
            marks={i: f'{i}' for i in range(-10, 11)}
        ),
    ]),
    html.Div(id='output-container')
])

@app.callback(
    [Output('pdf-plot', 'figure'),
     Output('output-container', 'children')],
    [Input('lambda_2-slider', 'value'),
     Input('lambda_3-slider', 'value'),
     Input('average-slider', 'value')]
)
def update_plot(lambda_2, lambda_3, average):
    x = np.linspace(-20, 20, 1000)
    y = normalized_pdf(x, lambda_2, lambda_3, average)

    if np.isnan(y).any() or np.isinf(y).any():
        return {
            'data': [],
            'layout': go.Layout(
                title='Normalized Custom Probability Density Function',
                xaxis={'title': 'x'},
                yaxis={'title': 'PDF(x)'},
                annotations=[
                    {
                        'text': 'Encountered NaN or inf values in the PDF. Adjust the parameters.',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': 'red'}
                    }
                ]
            )
        }, 'Encountered NaN or inf values in the PDF. Adjust the parameters.'

    is_non_negative = np.all(y >= 0)
    integral = np.trapz(y, x)
    plot_data = go.Scatter(x=x, y=y, mode='lines', name='PDF')

    layout = go.Layout(
        title='Normalized Custom Probability Density Function',
        xaxis={'title': 'x'},
        yaxis={'title': 'PDF(x)'},
        annotations=[]
    )

    output_text = [
        f"Is PDF non-negative? {is_non_negative}",
        f"Approximate integral of PDF: {integral:.4f}"
    ]

    return {'data': [plot_data], 'layout': layout}, output_text

if __name__ == '__main__':
    app.run_server(debug=True)
