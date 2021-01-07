import uuid
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_table


external_stylesheets = [dbc.themes.BOOTSTRAP]


class CardsWidth:
    THIRD = "33.33%"
    HALF = "50%"
    QUARTER = "25%"
    THREE_QUARTERS = "75%"
    FULL = "100%"


class ComponentIds:
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
                    dcc.Upload(
                        id=ComponentIds.UPLOAD_DATA,
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '75%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                        },
                        multiple=False
                    ),
                )
            ],
            style={"margin-top": "10%"}
        ),
        style={"width": width}
    )


def table_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                dash_table.DataTable(id=ComponentIds.TABLE_WIDGET)
            ],
        ),
        style={"width": width}
    )


def select_features_widget(width):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Span("Timespan column: "),
                dcc.Dropdown(id=ComponentIds.TIME_FEATURE),
                html.Br(),
                html.Span("Feature columns: "),
                dcc.Dropdown(id=ComponentIds.SELECTED_FEATURES),
                html.Br(),
                html.Span("Timespan to use as a reference: "),
                dcc.RangeSlider(id=ComponentIds.SELECTED_TIMESPAN),
            ],
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
                    dbc.Button(id=ComponentIds.PREDICT, children=html.Span("Predict")),
                    html.Br(),
                    html.Br(),
                    dbc.Button(id=ComponentIds.DOWNLOAD_PREDICTION, children=html.Span("Download prediction")),
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

        ], style={'max-width': '90%'}
    )
    return app
