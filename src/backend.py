from datetime import timedelta
from functools import partial

import os
import random
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
import tensorflow as tf
from pandas import DataFrame
from pandas._libs.tslibs.timedeltas import Timedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


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


def statsmodels_worker(model, df: DataFrame, prepared_df: DataFrame,
                       prediction_step_length: timedelta, feature_column: str):
    df_series = pd.Series(df[feature_column].array, df['time_'])
    df_series = df_series.resample(Timedelta(prediction_step_length)).ffill()
    model_ = model(endog=df_series)
    res = model_.fit()

    prepared_df[feature_column] = res.forecast(len(prepared_df['time_'])).array
    # pyplot.plot(prepared_df['time_'], prepared_df[feature_column])
    # pyplot.show()
    return prepared_df


############# Deep Neural Network #################
class TimeSeriesLoader:
    def __init__(self, windows):
        self.windows = windows

    def get_data(self):
        num_records = len(self.windows.index)

        features = self.windows.drop('y', axis=1).values
        target = self.windows['y'].values

        features_batchmajor = np.array(features).reshape(num_records, -1, 1)
        return features_batchmajor, target


def predict_dnn(df: DataFrame, prepared_df: DataFrame, prediction_step_length: timedelta, feature_column: str):
    df_series = pd.Series(df[feature_column].array, df['time_'])
    df_series = df_series.resample(Timedelta(prediction_step_length)).ffill()
    forecast_interval = len(prepared_df)

    history_length = min(24 * 7 * 4, len(df) / 8)  # in dataset points (=4w if interval=1h)
    validation_delta = 2 * history_length  # validation size
    step_size = 1  # step for sliding window stripe
    target_step = 0  # target shift (0 - predict next data after sliding window)
    batch_size = 128  # minibatch size
    num_epochs = 20  # epochs count

    df_val = df_series[-validation_delta:]
    df_train = df_series[0:-validation_delta]

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_train_scaled = scaler.fit_transform(df_train.values.reshape(-1, 1)).reshape(-1, )
    df_val_scaled = scaler.transform(df_val.values.reshape(-1, 1)).reshape(-1, )

    print('Validation dates: {} to {}'.format(df_val.index.min(), df_val.index.max()))
    print('Train dates: {} to {}'.format(df_train.index.min(), df_train.index.max()))

    def create_window(dataset,
                      start_index,
                      end_index,
                      history_length,
                      step_size,
                      target_step):
        time_lags = sorted(range(target_step + 1, target_step + history_length + 1, step_size), reverse=True)
        col_names = [f'x_lag{i}' for i in time_lags] + ['y']
        start_index = start_index + history_length
        if end_index is None:
            end_index = len(dataset) - target_step

        data_list = []

        for j in range(start_index, end_index):
            indices = range(j - 1, j - history_length - 1, -step_size)
            data = dataset[sorted(indices) + [j + target_step]]
            data_list.append(data)

        df_ts = pd.DataFrame(data=data_list, columns=col_names)
        return len(col_names) - 1, df_ts

    num_timesteps_data, df_train_windows = create_window(df_train_scaled,
                                                         start_index=0,
                                                         end_index=None,
                                                         history_length=history_length,
                                                         step_size=step_size,
                                                         target_step=target_step)
    tss = TimeSeriesLoader(df_train_windows)
    num_timesteps_val, df_val_windows = create_window(df_val_scaled,
                                                      start_index=0,
                                                      end_index=None,
                                                      history_length=history_length,
                                                      step_size=step_size,
                                                      target_step=target_step)

    tss_val = TimeSeriesLoader(df_val_windows)

    seed = 99
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()

    ts_inputs = tf.keras.Input(shape=(num_timesteps_data, 1))
    x = tf.keras.layers.Flatten()(ts_inputs)
    x = tf.keras.layers.Dense(units=128, activation='relu')(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(units=1, activation="linear")(x)

    model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mae'])

    val_X, val_y = tss_val.get_data()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=4,
                                                      restore_best_weights=True)
    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        save_freq='epoch',
        mode='min',
        save_best_only=True)

    X, y = tss.get_data()
    model.fit(x=X, y=y, epochs=num_epochs, batch_size=batch_size, validation_data=(val_X, val_y),
              callbacks=[early_stopping, checkpoint])

    raw_data = df_series.values[-history_length:]
    predictions = []
    for i in range(forecast_interval):
        if i % 100 == 0:
            print(f"{i}/{forecast_interval}")
        scaled_data = scaler.transform(raw_data.reshape(-1, 1)).reshape(1, -1, 1)
        next_data_scaled = model.predict(scaled_data)
        next_data = scaler.inverse_transform(next_data_scaled.reshape(-1, 1)).reshape(-1, )
        raw_data = np.append(raw_data, next_data)
        raw_data = raw_data[1:]
        predictions.append(next_data[0])

    prepared_df[feature_column] = predictions
    return prepared_df


#############################################


prediction_methods_map = {
    "Average forecast": avg_forecast,
    "Moving average forecast": moving_avg_forecast,
    "Simple exponential smoothing": partial(statsmodels_worker, partial(sm.tsa.SimpleExpSmoothing, )),
    "Holt linear": partial(statsmodels_worker, partial(sm.tsa.Holt, )),
    "Holt-Winter": partial(statsmodels_worker, partial(sm.tsa.ExponentialSmoothing, seasonal_periods=7,
                                                       trend='add', seasonal='add', freq='W')),
    "SARIMAX": partial(statsmodels_worker, partial(sm.tsa.SARIMAX, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7))),
    "Deep Neural Network": predict_dnn
}

if __name__ == "__main__":
    source_data = pd.read_csv("Train.csv")
    result = predict(source_data, "Deep Neural Network", 100, timedelta(hours=1), ["Count"], ["Datetime"], None)
    print()
