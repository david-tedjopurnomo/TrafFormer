from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data_dayhour(
        df, x_offsets_day, x_offsets_hour, y_offsets, add_time_in_day=True, add_day_in_week=True, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        # Convert to minutes
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]"))
        time_ind /= 300000000000
        time_ind = [int(x) for x in time_ind]
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        diw = df.index.dayofweek.tolist()
        diw = [int(x) for x in diw]
        diw = np.array(diw)
        diw = np.tile(diw, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(diw)
        

    data = np.concatenate(data_list, axis=-1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x_day, x_hour, y = [], [], []
    # t is the index of the last observation.
    min_t_day = abs(min(x_offsets_day))
    min_t_hour = abs(min(x_offsets_hour))
    delta = min_t_day - min_t_hour
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t_day, max_t):
        x_t_d = data[t + x_offsets_day, ...]
        x_t_h = data[t + x_offsets_hour, ...]
        y_t = data[t + y_offsets, ...]
        x_day.append(x_t_d)
        x_hour.append(x_t_h)
        y.append(y_t)
    x_day = np.stack(x_day, axis=0)
    x_hour = np.stack(x_hour, axis=0)
    y = np.stack(y, axis=0)
    
    return x_day, x_hour, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.traffic_df_filename)
    
    # Predict the next day with one day data and one hour data
    x_offsets_day = np.sort(np.arange(-264 ,1 ,24))
    x_offsets_hour= np.sort(np.arange(-11, 1, 1))
    y_offsets = np.sort(np.arange(1, 13, 1))
    #y_offsets = np.sort(np.arange(24, 289, 24))
    
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x_day, x_hour, y = generate_graph_seq2seq_io_data_dayhour(
        df,
        x_offsets_day=x_offsets_day,
        x_offsets_hour=x_offsets_hour,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )
    
    print("x shape: ", x_day.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x_day.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train_day, x_train_hour, y_train = (x_day[:num_train], 
                                          x_hour[:num_train], 
                                          y[:num_train]
    )
    
    # val
    x_val_day, x_val_hour, y_val = (
        x_day[num_train: num_train + num_val],
        x_hour[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test_day, x_test_hour, y_test = x_day[-num_test:], x_hour[-num_test:], y[-num_test:]
    
    x_train = np.concatenate((x_train_day, x_train_hour), axis=-1)
    x_val = np.concatenate((x_val_day, x_val_hour), axis=-1)
    x_test = np.concatenate((x_test_day, x_test_hour), axis=-1)

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets_day=x_offsets_day.reshape(list(x_offsets_day.shape) + [1]),
            x_offsets_hour=x_offsets_hour.reshape(list(x_offsets_hour.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_df_filename",
        type=str,
        default="data/metr-la.h5",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
