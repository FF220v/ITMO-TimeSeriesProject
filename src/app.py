import base64
import datetime
import io
import pandas as pd

import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table
from dash.dependencies import Output, Input, State
from flask import send_file

from src.backend import predict, squash_timespan_to_one_column, prediction_methods_map

external_stylesheets = [dbc.themes.COSMO]

TABLE_MAX_SIZE = 20
TABLE_BUF_FILE = "current_table.csv"
RESULTING_TABLE_FILE = "resulting_table.csv"


STEP_LENGTH_MAP = {
    0: datetime.timedelta(seconds=1),
    1: datetime.timedelta(minutes=1),
    2: datetime.timedelta(hours=1),
    3: datetime.timedelta(days=1),
    4: datetime.timedelta(weeks=1),
    5: datetime.timedelta(days=30),
    6: datetime.timedelta(days=365),
}


class CardsWidth:
    THIRD = "33.33%"
    HALF = "50%"
    QUARTER = "25%"
    THREE_QUARTERS = "75%"
    FULL = "100%"


class ComponentIds:
    SELECTED_FEATURES_WIDGET = "selected_features_widget"
    TIME_FEATURE = "time_feature"
    TIME_FORMAT = "time_format"
    SELECTED_FEATURES = "data_features"
    DELIMITER_INPUT = "delimiter_input"
    FEATURE_SELECTOR_ERROR = "feature_selector_error"
    SELECTED_TIMESPAN = "selected_timespan"
    UPLOAD_DATA = "upload_data"
    TABLE_WIDGET = "table_widget"
    SELECTED_FEATURES_GRAPH = "selected_features_graph"
    PREDICTION_STEPS = "prediction_steps"
    PREDICTION_STEP_LENGTH = "prediction_length"
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
                            className = "uploading-area",
                            multiple=False
                        ),
                        html.Br(),
                        html.Span(".xls and .csv are supported."),
                        html.Br(),
                        html.Br(),
                        html.Span("Delimiter character used in csv file:"),
                        html.Br(),
                        dcc.Input(value=",",
                                  type="text",
                                  maxLength=1,
                                  size="10",
                                  id=ComponentIds.DELIMITER_INPUT)
                    ]
                )
            ],
            className="card-body"
        ),
        className="uploading"
    )


def table_widget(width):
    return dbc.Card(
        dbc.CardBody(
            dcc.Loading(
                  html.Div("Data table will be shown here.", id=ComponentIds.TABLE_WIDGET),
            )
        ),
        style={"width": width},
    )


def generate_select_features_widget(columns: list):
    options = [{'label': col, 'value': col} for col in columns]
    return dbc.CardBody(
        [
            html.Span("Timespan columns. More then one column are concatenated with space (' ') character",
                      className="span-label"),
            dcc.Dropdown(id=ComponentIds.TIME_FEATURE, options=options, multi=True),
            html.Br(),
            html.Span("Timespan format. Mostly it is picked automatically. For peculiar cases use this:", className="span-label"),
            html.Span("%Y - four digit years; %y - two digit years; %m - months; %d - days", className="span-label"),
            html.Span("%H - hours; %M - minutes; %S - seconds", className="span-label"),
            html.Span("Example: to read date like 2020-12-31 00:00:00 use %Y-%m-%d %H:%M:%S", className="span-label"),
            dcc.Input(id=ComponentIds.TIME_FORMAT),
            html.Br(),
            html.Br(),
            html.Span("Feature columns: ", className="span-label"),
            dcc.Dropdown(id=ComponentIds.SELECTED_FEATURES, options=options, multi=True),
            html.Br(),
            html.Span(id=ComponentIds.FEATURE_SELECTOR_ERROR, style={'color': 'red'}, className="span-label"),
        ],
    )


def select_features_widget(width):
    return dbc.Card(
        dcc.Loading(
            html.Div(generate_select_features_widget([]), id=ComponentIds.SELECTED_FEATURES_WIDGET)
        ),
        style={"width": width}
    )


def draw_feature_graphs(df, feature_columns):
    graphs = []
    for col in feature_columns:
        graphs.append(dcc.Graph(figure={
            "data":
                [
                    {
                        "y": df[col].to_list(),
                        "x": df["time_"].to_list(),
                        "name": col
                    }
                ],
            "layout":
                {
                    "title": f"Graph of feature [{col}] over time."
                }
        }
        ))
    if not graphs:
        return [dcc.Graph(figure={'layout': {
            'title': "Here will be a graph of selected features over time."
        }})]
    return graphs


def draw_prediction_graphs(df, prediction_df, feature_columns):
    graphs = []
    for col in feature_columns:
        prediction_data = prediction_df[col].to_list()
        prediction_time = prediction_df["time_"].to_list()
        df_data = df[col].to_list()
        df_time = df["time_"].to_list()
        # while df_time[-1] - df_time[0] > prediction_time[-1] - prediction_time[0]:  # This algorithm sucks macaroni
        #     df_time.pop(0)
        #     df_data.pop(0)
        graphs.append(dcc.Graph(figure={
            "data":
                [
                    {
                        "y": df_data,
                        "x": df_time,
                        "name": col
                    },
                    {
                        "y": prediction_data,
                        "x": prediction_time,
                        "name": col
                    }
                ],
            "layout":
                {
                    "title": f"Graph of predicted feature [{col}] over time."
                }
        }
        ))
    if not graphs:
        return [dcc.Graph(figure={'layout': {'title': "Here will be a graph of predicted features over time."}})]
    return graphs


def features_over_time_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Loading(html.Div(dcc.Graph(figure={'layout': {
                    'title': "Here will be a graph of selected features over time."
                }}),
                    id=ComponentIds.SELECTED_FEATURES_GRAPH))
            ],
        ),
        style={"width": width}
    )


def prediction_over_time_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dcc.Loading(html.Div(dcc.Graph(figure={'layout': {
                    'title': "Here will be a graph of predicted features over time."
                }}),
                    id=ComponentIds.PREDICTION_GRAPH))
            ],
        ),
        style={"width": width}
    )


def prediction_config_widget():
    return dbc.Card(
        dbc.CardBody(
            [
                html.Span("Method: ", className="span-label"),
                dcc.Dropdown(id=ComponentIds.METHOD_SELECTOR, options=[{'label': key, 'value': key}
                                                                       for key in prediction_methods_map]),
                html.Br(),
                html.Span("Prediction step length: ", className="span-label"),
                dcc.Slider(id=ComponentIds.PREDICTION_STEP_LENGTH,
                           min=0,
                           max=6,
                           step=None,
                           value=0,
                           marks={
                               0: "second",
                               1: "minute",
                               2: "hour",
                               3: "day",
                               4: "week",
                               5: "month",
                               6: "year"
                           },
                           ),
                html.Br(),
                html.Span("Prediction steps (integer number 0-100000)", className="span-label"),
                dcc.Input(value=1000, min=0, max=100000, type='number', id=ComponentIds.PREDICTION_STEPS),
                html.Br()
            ],
        ),
        className="method"
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
                    prediction_config_widget(), prediction_start_and_download_widget(CardsWidth.HALF)
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
                  State(ComponentIds.DELIMITER_INPUT, 'value'),
                  prevent_initial_call=True)
    def update_data(contents, filename, delimiter):
        if contents:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            try:
                if 'csv' in filename:
                    # Assume that the user uploaded a CSV file
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter=delimiter)
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

    @app.callback(Output(ComponentIds.SELECTED_FEATURES_GRAPH, 'children'),
                  Output(ComponentIds.FEATURE_SELECTOR_ERROR, 'children'),
                  Input(ComponentIds.TIME_FEATURE, 'value'),
                  Input(ComponentIds.SELECTED_FEATURES, 'value'),
                  Input(ComponentIds.TIME_FORMAT, 'value'),
                  prevent_initial_call=True)
    def update_features_graph(selected_timespan_keys, selected_feature_keys, time_format):
        if selected_timespan_keys and selected_feature_keys:
            try:
                df = pd.read_csv(TABLE_BUF_FILE)
                squash_timespan_to_one_column(df, selected_timespan_keys, time_format)
            except Exception as e:
                print(e)
                return dcc.Graph(figure={'layout': {'title': "Error building a graph."}}), f"Error: {str(e)}"
            return draw_feature_graphs(df, selected_feature_keys), ""
        return dcc.Graph(figure={'layout': {'title': "Here will be a graph of selected features over time."}}), ""

    @app.callback(Output(ComponentIds.PREDICTION_GRAPH, 'children'),
                  Output(ComponentIds.DOWNLOAD_PREDICTION, 'children'),
                  Input(ComponentIds.PREDICT, 'n_clicks'),
                  State(ComponentIds.TIME_FORMAT, 'value'),
                  State(ComponentIds.PREDICTION_STEP_LENGTH, 'value'),
                  State(ComponentIds.PREDICTION_STEPS, 'value'),
                  State(ComponentIds.METHOD_SELECTOR, 'value'),
                  State(ComponentIds.SELECTED_FEATURES, 'value'),
                  State(ComponentIds.TIME_FEATURE, 'value'),
                  prevent_initial_call=True)
    def predict_(prediction_button,
                 time_format,
                 prediction_step_length,
                 prediction_steps,
                 method,
                 selected_feature_keys,
                 selected_timespan_keys):
        if method and selected_feature_keys and selected_timespan_keys:
            try:
                prediction_step_length = STEP_LENGTH_MAP[prediction_step_length]
                df = pd.read_csv(TABLE_BUF_FILE)
                predicted_df = predict(pd.read_csv(TABLE_BUF_FILE),
                                       method,
                                       prediction_steps,
                                       prediction_step_length,
                                       selected_feature_keys,
                                       selected_timespan_keys,
                                       time_format)
                df.to_csv(RESULTING_TABLE_FILE)
                return draw_prediction_graphs(df, predicted_df, selected_feature_keys), generate_download_link()
            except Exception as e:
                raise e
                # TODO uncomment when work is done
                # return dcc.Graph(figure={'layout': {
                #     'title': "Here will be a graph of predicted features over time."
                # }}), html.Span(f"Error occured: {str(e)}", style={'color': 'red'})

        return dcc.Graph(figure={
            'layout': {
                    'title': "Here will be a graph of predicted features over time."
                }
            }), html.Span("Please ensure that you uploaded the file, chosen a method and features are selected.",
                          style={'color': 'red'})

    @app.server.route('/downloadResults')
    def download_csv():
        return send_file(RESULTING_TABLE_FILE, mimetype='text/csv', attachment_filename=RESULTING_TABLE_FILE,
                         as_attachment=True)

    return app
