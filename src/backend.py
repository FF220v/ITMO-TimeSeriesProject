import datetime
import pandas as pd
from pandas import DataFrame


def get_prediction_step_length():
    return 1


def get_time_column(df, timespan_keys):
    return [" ".join((str(elem) for elem in tup))
            for tup in zip(*[df[k].to_list()
                             for k in timespan_keys])]


def squash_timespan_to_one_column(df: pd.DataFrame, timespan_columns, format):
    if not format:
        format = None
    df["time_"] = pd.to_datetime(get_time_column(df, timespan_columns), format=format)
    df.columns.drop(labels=timespan_columns)


def predict(df, method, prediction_steps, prediction_step_length, feature_columns, timespan_columns, time_format):
    prediction_func = prediction_methods_map[method]
    squash_timespan_to_one_column(df, timespan_columns, time_format)
    prepared_df = DataFrame()
    init_time = df['time_'].array[-1]
    step = datetime.timedelta(days=prediction_step_length)  # например
    prepared_df['time_'] = [init_time + i * step for i in range(prediction_steps)]
    result = pd.concat(
        [prediction_func(df, prepared_df.copy(), feature) for feature in feature_columns],
        axis=1
    )
    result['time_'] = prepared_df['time_']
    return result


def avg_forecast(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


def moving_avg_forecast(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


def simple_exp_smoothing(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


def holt_linear(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


def holt_winter(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


def sarima(df: DataFrame, prepared_df: DataFrame, feature_column: list) -> DataFrame:
    pass


prediction_methods_map = {
    "Average forecast": avg_forecast,
    "Moving average forecast": moving_avg_forecast,
    "Simple exponential smoothing": simple_exp_smoothing,
    "Holt linear": holt_linear,
    "Holt-Winter": holt_winter,
    "SARIMA": sarima
}
