import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly
import plotly.graph_objects as go

import pandas as pd
import numpy as np

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")
genes = genes = list(dataset.columns[4:39])

for i in genes:
    genes[genes.index(i)] = genes[genes.index(i)][0:-3]

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])


######################app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash("Ploting", external_stylesheets=external_stylesheets)

app.layout = html.Div(style={"text-align": "center"}, children=[
    html.Div([
        html.Div(style={"padding": "20", "width":1200}, children=[
            html.Div(style={"text-align": "left", "width": "50%", "float": "left"}, children=[
                html.H6("Timepoint:"),
                dcc.Slider(
                    id='timepoint_slider',
                    min=1,
                    max=6,
                    value=1,
                    marks={str(timepoint): str(timepoint) for timepoint in (np.array(range(6)) + 1)},
                    step=None,
                    included=False
                )
            ]),

            html.Div(style={"text-align": "left", "width": "50%", "float": "left"}, children=[
                html.H6("Colorscale contrast:"),
                dcc.RangeSlider(
                    id='contrast_slider',
                    min=0,
                    max=2,
                    step=0.01,
                    value=[0, 1],
                    marks={str(tick): str(tick) for tick in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]}
                )
            ])
        ]),

        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        html.Div(children=[
            dcc.Graph(id="expression_graph")
        ]),

        html.Br(),

        html.Div(style={"text-align": "left"}, children=[
            html.Button("Advanced", id="advanced_options_button", n_clicks=0),

            html.Br(),

            html.Div(id="advanced_options",
                     style={"text-align": "left", "padding": "20", "width": 1200, "display": "none"},
                     children=[
                        html.Div(style={"text-align": "left", "width": "20%", "float": "left"}, children=[
                            html.P("Gene:"),
                            dcc.Dropdown(
                                id='genes_menu',
                                options=[
                                    {'label': i, 'value': i} for i in genes
                                ],
                                value='eve'
                            ),
                        ]),

                        html.Div(style={"text-align": "left", "width": "5%", "float": "left"}, children=[
                            html.Br()
                        ]),

                        html.Div(style={"text-align": "left", "width": "20%", "float": "left"}, children=[
                            html.P("Point size:"),
                            dcc.Input(
                                id='size_menu',
                                type='number',
                                debounce=True,
                                min=1,
                                step=0.1,
                                value=9)
                        ]),
                     ]
            ),
        ])
    ],  style={"display": "inline-block"})
])

###figures###
@app.callback(
    Output("expression_graph", "figure"),
    [Input("timepoint_slider", "value"),
     Input("contrast_slider", "value"),
     Input("genes_menu","value"),
     Input("size_menu", "value")])
def update_graph(timepoint, contrast, gene, size):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=dataset[f"x__{timepoint}"],
            y=dataset[f"y__{timepoint}"],
            z=dataset[f"z__{timepoint}"],
            name="1",
            mode='markers',
            visible=True,
            marker=dict(color=dataset[f"{gene}__{timepoint}"],
                        colorbar=dict(title="Colorscale"),
                        colorscale="blues",
                        cmin=contrast[0],
                        cmax=contrast[1],
                        size=size
                        )
        )
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                      title_text=f"Expression of {gene} at timepoint {timepoint}",
                      height=700,
                      width=1200,
                      scene=dict(
                          xaxis=dict(range=[0, 440]),
                          yaxis=dict(range=[0, 160]),
                          zaxis=dict(range=[0, 160]),
                          xaxis_title="Anterior-Posterior Axis",
                          zaxis_title="Dorso-Ventral Axis",
                          yaxis_title="Left-Right Axis")
                      )
    return fig

##advanced options###
@app.callback(
    Output(component_id="advanced_options", component_property="style"),
    [Input(component_id="advanced_options_button", component_property="n_clicks")]
)
def update_output(n_clicks):
    if (n_clicks % 2) != 0:
        return {"text-align": "left", "padding": "20", "width":1200, "display":"inline-block"}
    else:
        return {"text-align": "left", "padding": "20", "width": 1200, "display": "none"}

###END###
app.run_server(debug=False)
#how to: run the script from the terminal ($ python interactive_plots.py) and go to localhost:8050 in your browser