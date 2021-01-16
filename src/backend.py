from datetime import timedelta
from functools import partial
import pandas as pd
import numpy as np
from pandas import DataFrame
import statsmodels.api as sm
from pandas._libs.tslibs.timedeltas import Timedelta
from xgboost import XGBRegressor


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
    squash_timespan_to_one_column(df, timespan_columns, time_format)

    # prepare copied df
    prepared_df = DataFrame()
    init_time = df['time_'].array[-1]
    prepared_df['time_'] = [init_time + i * prediction_step_length for i in range(prediction_steps)]

    #  You can change prepared_df here. It is basically made for shortness.
    results = [prediction_methods_map[method](df,
                                              prepared_df.copy(),
                                              prediction_step_length,
                                              feature) for feature in feature_columns]
    if len(results) > 1:
        result = results.pop(0)
        for res in results:
            result = result.merge(res)
    else:
        result = results[0]
    return result


def avg_forecast(df: DataFrame, prepared_df: DataFrame, prediction_step_length: timedelta, feature_column: list):
    prepared_df[feature_column] = df[feature_column].mean()
    return prepared_df


def moving_avg_forecast(df: DataFrame, prepared_df: DataFrame, prediction_step_length: timedelta, feature_column: list):
    prepared_df[feature_column] = df[feature_column].rolling(60).mean().iloc[-1]
    return prepared_df


def xgb_model(df: DataFrame, prepared_df: DataFrame, prediction_step_length: timedelta, feature_column: list):
    df_series = pd.Series(df[feature_column].array, df['time_'])
    df_series = df_series.resample(Timedelta(prediction_step_length)).ffill()

    df_time = np.array(range(len(df_series)))
    prediction_time = np.array(range(len(df_series), len(prepared_df['time_']) + len(df_series)))

    model = XGBRegressor(
        max_depth=5,
        n_estimators=350,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.8,
        eta=0.3,
        seed=42
    )

    model.fit(np.array(df_series), df_time)
    prepared_df[feature_column] = model.predict(prediction_time)
    return prepared_df


def statsmodels_worker(model, df: DataFrame, prepared_df: DataFrame,
                       prediction_step_length: timedelta, feature_column: str):
    df_series = pd.Series(df[feature_column].array, df['time_'])
    df_series = df_series.resample(Timedelta(prediction_step_length)).ffill()
    model_ = model(endog=df_series)
    res = model_.fit()

    prepared_df[feature_column] = res.forecast(len(prepared_df['time_'])).array
    # plt.plot(prepared_df['time_'], prepared_df[feature_column])
    # plt.show()
    return prepared_df


prediction_methods_map = {
    "Average forecast": avg_forecast,
    "XGB": xgb_model,
    "Moving average forecast": moving_avg_forecast,
    "Simple exponential smoothing": partial(statsmodels_worker, partial(sm.tsa.SimpleExpSmoothing, )),
    "Holt linear": partial(statsmodels_worker, partial(sm.tsa.Holt, )),
    "Holt-Winter": partial(statsmodels_worker, partial(sm.tsa.ExponentialSmoothing, seasonal_periods=7,
                                                       trend='add', seasonal='add')),
    "SARIMAX": partial(statsmodels_worker, partial(sm.tsa.SARIMAX, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)))
}

if __name__ == "__main__":
    source_data = pd.read_csv("Train.csv")
    result = predict(source_data, "XGB", 100, timedelta(hours=1), ["Count"], ["Datetime"], None)
    print('')
