import pandas as pd

# Currently these are stupid stubs


def get_prediction_step_length():
    return 1


def get_time_column(df, timespan_keys):
    return [" ".join(tup)
            for tup in zip(*[df[k].to_list()
                             for k in timespan_keys])]


def squash_timespan_to_one_column(df: pd.DataFrame, timespan_columns):
    df["time_"] = get_time_column(df, timespan_columns)
    df.columns.drop(labels=timespan_columns)


def predict(df, method, prediction_length, feature_columns, timespan_columns):
    squash_timespan_to_one_column(df, timespan_columns)
    return pd.concat(
        [predict_one_feature(df, feature) for feature in feature_columns],
        axis=1
    )


def predict_one_feature(df, feature):
    prediction_df = pd.DataFrame()
    prediction_df[feature] = df[feature]
    prediction_df["time_"] = df["time_"]
    return prediction_df


