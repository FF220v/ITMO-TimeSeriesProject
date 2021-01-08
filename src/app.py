import base64
import uuid
import io
import pandas as pd
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Output, Input, State

external_stylesheets = [dbc.themes.COSMO]


prediction_methods_map = {"1": lambda x: x}


TABLE_MAX_SIZE = 50
TABLE_BUF_FILE = "current_table.csv"


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
    METHOD_SELECTOR = "method_selector"
    PREDICT = "predict"
    DOWNLOAD_PREDICTION = "download_prediction"
    PREDICTION_GRAPH = "prediction_graph"


def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div([
        html.Div(session_id, id='session-id', style={'display': 'none'}),
        html.Button('Get data', id='get-data-button'),
        html.Div(id='output-1'),
        html.Div(id='output-2')
    ])


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
            html.Span("Timespan column: "),
            dcc.Dropdown(id=ComponentIds.TIME_FEATURE, options=options),
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
                dcc.Graph(id=ComponentIds.SELECTED_FEATURES_GRAPH)
            ],
        ),
        style={"width": width}
    )


def prediction_over_time_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Graph(id=ComponentIds.PREDICTION_GRAPH)
            ],
        ),
        style={"width": width}
    )


def prediction_config_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Span("Method: "),
                dcc.Dropdown(id=ComponentIds.METHOD_SELECTOR),
                html.Br(),
                html.Span("Desired prediction length: "),
                dcc.Slider(id=ComponentIds.PREDICTION_LENGTH),
                html.Br()
            ],
        ),
        style={"width": width}
    )


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
                    dbc.Button(id=ComponentIds.DOWNLOAD_PREDICTION,
                               children=html.Span("Download prediction")),
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
                  State(ComponentIds.UPLOAD_DATA, 'filename'))
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
                  Input(ComponentIds.TIME_FEATURE, 'value'),
                  Input(ComponentIds.SELECTED_FEATURES, 'value'))
    def update_features_graph(selected_timespan_key, selected_feature_keys):
        if selected_timespan_key and selected_feature_keys:
            selected_timespan_key = selected_timespan_key
            try:
                df = pd.read_csv(TABLE_BUF_FILE)
            except Exception as e:
                print(e)
                return
            fig = {
                "data":
                    [
                        {"y": df[key].to_list(), "x": df[selected_timespan_key].to_list(), "name": key}
                        for key in selected_feature_keys
                    ],
                'layout': {
                    'title': "Graph of selected features over time."
                }

            }
            return fig
        return {'layout': {
            'title': "Here will be a graph of selected feature over time."
        }}

    return app
