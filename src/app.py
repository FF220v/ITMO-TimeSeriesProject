import base64
import io
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Output, Input, State
from flask import send_file

from src.backend import get_time_column, predict

external_stylesheets = [dbc.themes.COSMO]


prediction_methods_map = {
    "Good": lambda x: x,
    "Better": lambda x: x,
    "Best": lambda x: x
}


TABLE_MAX_SIZE = 20
TABLE_BUF_FILE = "current_table.csv"
RESULTING_TABLE_FILE = "resulting_table.csv"


class CardsWidth:
    THIRD = "33.33%"
    HALF = "50%"
    QUARTER = "25%"
    THREE_QUARTERS = "75%"
    FULL = "100%"


class ComponentIds:
    SELECTED_FEATURES_WIDGET = "selected_features_widget"
    TIME_FEATURE = "time_feature"
    SELECTED_FEATURES = "data_features"
    SELECTED_TIMESPAN = "selected_timespan"
    UPLOAD_DATA = "upload_data"
    TABLE_WIDGET = "table_widget"
    SELECTED_FEATURES_GRAPH = "selected_features_graph"
    PREDICTION_LENGTH = "prediction_length"
    PREDICTION_LENGTH_SLIDER = "prediction_length_slider"
    METHOD_SELECTOR = "method_selector"
    PREDICT = "predict"
    DOWNLOAD_PREDICTION = "download_prediction"
    PREDICTION_GRAPH = "prediction_graph"


def uploading_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Center(
                    [
                        dcc.Upload(
                            id=ComponentIds.UPLOAD_DATA,
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '75%',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                            },
                            multiple=False
                        ),
                        html.Br(),
                        html.Span(".xls and .csv are supported.")
                    ]
                )
            ],
            style={"margin-top": "10%"}
        ),
        style={"width": width}
    )


def table_widget(width):
    return dbc.Card(
        dbc.CardBody(
            dcc.Loading(
                html.Div(dash_table.DataTable(), id=ComponentIds.TABLE_WIDGET),
            )
        ),
        style={"width": width},
    )


def generate_select_features_widget(columns: list):
    options = [{'label': col, 'value': col} for col in columns]
    return dbc.CardBody(
        [
            html.Span("Timespan columns (Most significant go first. Example: date, hours, minutes, seconds)"),
            dcc.Dropdown(id=ComponentIds.TIME_FEATURE, options=options, multi=True),
            html.Br(),
            html.Span("Feature columns: "),
            dcc.Dropdown(id=ComponentIds.SELECTED_FEATURES, options=options, multi=True),
            html.Br(),
        ],
    )


def select_features_widget(width):
    return dbc.Card(
        dcc.Loading(
            html.Div(generate_select_features_widget([]), id=ComponentIds.SELECTED_FEATURES_WIDGET)
        ),
        style={"width": width}
    )


def features_over_time_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(id=ComponentIds.SELECTED_FEATURES_GRAPH,
                          figure={'layout': {
                              'title': "Here will be a graph of selected features over time."
                          }})
            ],
        ),
        style={"width": width}
    )


def prediction_over_time_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(id=ComponentIds.PREDICTION_GRAPH,
                          figure={'layout': {
                              'title': "Here will be a graph of predicted features over time."
                          }})
            ],
        ),
        style={"width": width}
    )


def generate_prediction_length_slider(step, max):
    marks = [0]
    while marks[-1] < max:
        marks.append(marks[-1] + step)
    return dcc.Slider(
        id=ComponentIds.PREDICTION_LENGTH,
        min=0,
        max=max,
        step=None,
        marks={k: str(v) for k, v in enumerate(marks)},
        value=0
    )


def prediction_config_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Span("Method: "),
                dcc.Dropdown(id=ComponentIds.METHOD_SELECTOR, options=[{'label': key, 'value': key}
                                                                       for key in prediction_methods_map]),
                html.Br(),
                html.Span("Desired prediction length: "),
                html.Div(generate_prediction_length_slider(0, 0), id=ComponentIds.PREDICTION_LENGTH_SLIDER),
                html.Br()
            ],
        ),
        style={"width": width}
    )


def generate_download_link():
    return html.A("Download prediction",
                  href='/downloadResults',
                  target="_blank")


def prediction_start_and_download_widget(width):
    return dbc.Card(
        dbc.CardBody(
            html.Center(
                [
                    html.Br(),
                    dbc.Button(id=ComponentIds.PREDICT,
                               children=html.Span("Predict")),
                    html.Br(),
                    html.Br(),
                    dcc.Loading(
                        html.Div(
                            html.Span("Download link will be here"),
                            id=ComponentIds.DOWNLOAD_PREDICTION
                        )
                    ),
                    html.Br()
                ],
            )
        ),
        style={"width": width}
    )


def create_app():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = dbc.Container(
        [
            html.Br(),
            html.H1(children='Super Mega Pro Time Series Analyser'),
            html.Br(),
            dbc.Row(
                [
                    uploading_widget(CardsWidth.QUARTER), select_features_widget(CardsWidth.THREE_QUARTERS),
                ]
            ),
            dbc.Row(
                [
                    table_widget(CardsWidth.FULL)
                ]
            ),
            dbc.Row(
                [
                    features_over_time_widget(CardsWidth.FULL)
                ]
            ),
            dbc.Row(
                [
                    prediction_config_widget(CardsWidth.HALF), prediction_start_and_download_widget(CardsWidth.HALF)
                ]
            ),
            dbc.Row(
                [
                    prediction_over_time_widget(CardsWidth.FULL)
                ]
            ),
            html.Br(),
            html.Br()
        ], style={'max-width': '90%'}
    )

    @app.callback(Output(ComponentIds.TABLE_WIDGET, 'children'),
                  Output(ComponentIds.SELECTED_FEATURES_WIDGET, 'children'),
                  Input(ComponentIds.UPLOAD_DATA, 'contents'),
                  State(ComponentIds.UPLOAD_DATA, 'filename'),
                  prevent_initial_call=True)
    def update_data(contents, filename):
        if contents:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                elif 'xls' in filename:
                    # Assume that the user uploaded an excel file
                    df = pd.read_excel(io.BytesIO(decoded))
                df.to_csv(TABLE_BUF_FILE)
            except Exception as e:
                print(e)
                return 'There was an error processing this file.', generate_select_features_widget([])
            return dash_table.DataTable(
                data=df.head(TABLE_MAX_SIZE).to_dict('records'),
                columns=[{'name': col, 'id': col} for col in df.columns],
                style_table={
                    'overflowY': 'scroll',
                    'overflowX': 'scroll',
                }
            ), generate_select_features_widget(df.columns)
        return 'Data table will be shown here.', generate_select_features_widget([])

    @app.callback(Output(ComponentIds.SELECTED_FEATURES_GRAPH, 'figure'),
                  Output(ComponentIds.PREDICTION_LENGTH_SLIDER, 'children'),
                  Input(ComponentIds.TIME_FEATURE, 'value'),
                  Input(ComponentIds.SELECTED_FEATURES, 'value'),
                  prevent_initial_call=True)
    def update_features_graph(selected_timespan_keys, selected_feature_keys):
        if selected_timespan_keys and selected_feature_keys:
            try:
                df = pd.read_csv(TABLE_BUF_FILE)
            except Exception as e:
                print(e)
                return
            time_column = get_time_column(df, selected_timespan_keys)
            fig = {
                "data":
                    [
                        {"y": df[key].to_list(),
                         "x": time_column,
                         "name": key}
                        for key in selected_feature_keys
                    ],
                'layout': {
                    'title': "Graph of selected features over time."
                }
            }
            return fig, generate_prediction_length_slider(1, 10)
        return {'layout': {
            'title': "Here will be a graph of selected features over time."
        }}, generate_prediction_length_slider(0, 0)

    @app.callback(Output(ComponentIds.PREDICTION_GRAPH, 'figure'),
                  Output(ComponentIds.DOWNLOAD_PREDICTION, 'children'),
                  Input(ComponentIds.PREDICT, 'n_clicks'),
                  State(ComponentIds.PREDICTION_LENGTH, 'value'),
                  State(ComponentIds.METHOD_SELECTOR, 'value'),
                  State(ComponentIds.SELECTED_FEATURES, 'value'),
                  State(ComponentIds.TIME_FEATURE, 'value'),
                  prevent_initial_call=True)
    def predict_(prediction_button, prediction_length, method, selected_feature_keys, selected_timespan_keys):
        if prediction_length and method and selected_feature_keys and selected_timespan_keys:
            df = predict(pd.read_csv(TABLE_BUF_FILE), method, prediction_length,
                         selected_feature_keys, selected_timespan_keys)
            df.to_csv(RESULTING_TABLE_FILE)
            fig = {
                "data":
                    [
                        {"y": df[key].to_list(),
                         "x": df["time_"].to_list(),
                         "name": key}
                        for key in selected_feature_keys
                    ],
                'layout': {
                    'title': "Graph of predicted features over time."
                }
            }
            return fig, generate_download_link()
        return {
            'layout': {
                    'title': "Here will be a graph of predicted features over time."
                }
            }, html.Span("Please ensure that you filled prediction length, method and features are selected.")

    @app.server.route('/downloadResults')
    def download_csv():
        return send_file(RESULTING_TABLE_FILE, mimetype='text/csv', attachment_filename=RESULTING_TABLE_FILE,
                         as_attachment=True)

    return app
