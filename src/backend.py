import datetime
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
import statsmodels.api as sm

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
    prepared_df['time_'] = [init_time + i * prediction_step_length for i in range(prediction_steps)]
    result = pd.concat(
        [prediction_func(df, prepared_df.copy(), feature) for feature in feature_columns],
        axis=1
    )
    result['time_'] = prepared_df['time_']
    return result


def avg_forecast(df: DataFrame, prepared_df: DataFrame, feature_column: list):
    prepared_df[feature_column] = df[feature_column].mean()
    return prepared_df


def moving_avg_forecast(df: DataFrame, prepared_df: DataFrame, feature_column: list):
    prepared_df[feature_column] = df[feature_column].rolling(60).mean().iloc[-1]
    return prepared_df


def simple_exp_smoothing(df: DataFrame, prepared_df: DataFrame, feature_column: list):

    fit2 = SimpleExpSmoothing(np.asarray(df[feature_column])).fit(smoothing_level=0.3, optimized=False)
    prepared_df[feature_column] =fit2.forecast(len(prepared_df))
    return prepared_df


def holt_linear(df: DataFrame, prepared_df: DataFrame, feature_column: list):
    fit1 = Holt(np.asarray(df[feature_column])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    prepared_df[feature_column] = fit1.forecast(len(prepared_df))
    return prepared_df


def holt_winter(df: DataFrame, prepared_df: DataFrame, feature_column: list):
    fit1 = ExponentialSmoothing(np.asarray(df[feature_column]), seasonal_periods=7, trend='add', seasonal='add', ).fit()
    prepared_df[feature_column] = fit1.forecast(len(prepared_df))
    return prepared_df


def sarima(df: DataFrame, prepared_df: DataFrame, feature_column: list):
    fit1 = sm.tsa.statespace.SARIMAX(df[feature_column], order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
    prepared_df[feature_column] = fit1.predict(start=df['time_'].array[-1], end=prepared_df['time_'].array[-1], dynamic=True)
    return prepared_df


prediction_methods_map = {
    "Average forecast": avg_forecast,
    "Moving average forecast": moving_avg_forecast,
    "Simple exponential smoothing": simple_exp_smoothing,
    "Holt linear": holt_linear,
    "Holt-Winter": holt_winter,
    "SARIMA": sarima
}
