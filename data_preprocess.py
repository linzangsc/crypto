'''
tool for preprocess csv data
usage: 
python3.8 data_preprocess.py --raw_data_path /home/lzh/datasets/crypto_datasets/BTC_hourly/Binance_BTCUSDT_1hour_aft_preprocess2.csv
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arguments import preprocess_args

HOURS = set([i for i in range(24)])

def drop_duplicated(data_frame : pd.DataFrame, output_path):
    index = 10967
    less_than_index = data_frame.iloc[:index]
    tradecount_nan = less_than_index[less_than_index['tradecount'].isna()]
    new_data_frame = data_frame.drop(tradecount_nan.index)
    new_data_frame.to_csv(output_path, index=False)

def convert_date_format(data_frame : pd.DataFrame, output_path):
    index = 8318
    length = data_frame.shape[0]
    for i in range(index, length):
        date = data_frame.iloc[i]['date']
        date = date.split()
        if date[-1] == "12-PM":
            new_date = "12:00"
        elif date[-1] == "12-AM":
            new_date = "0:00"
        else:
            identification = date[-1].split('-')
            if identification[1] == 'PM':
                new_date = f"{int(identification[0]) + 12}:00"
            else:
                new_date = f"{int(identification[0])}:00"
        new_date = date[0] + ' ' + new_date
        # do not use chain indexing when modifying element in dataframe, use iloc or loc instead
        data_frame.iloc[i, 1] = new_date
    data_frame.to_csv(output_path, index=False)

def sort_descending(data_frame : pd.DataFrame, output_path):
    data_frame['date'] = pd.to_datetime(data_frame['date'])
    data_frame = data_frame.sort_values(by=['date'])
    data_frame.to_csv(output_path, index=False)

def check_missing(data_frame : pd.DataFrame, output_path):
    length = data_frame.shape[0]
    data_frame['date'] = pd.to_datetime(data_frame['date'])
    data_frame['day'] = data_frame['date'].dt.date
    data_frame['valid'] = 1
    data_frame['count'] = 1
    data_count = data_frame.groupby(['day'])['count'].sum()
    data_missing = data_count != 24
    for i in range(length):
        current_day = data_frame.iloc[i, 10]# 10th col is day
        if data_missing[current_day]:
            subset = data_frame[data_frame['day'] == current_day]
            existing_hours = set(subset['date'].dt.hour)
            missing_hours = HOURS.difference(existing_hours)
            for hour in missing_hours:
                new_row = pd.Series({
                    'unix': 100, 'date': str(current_day) + ' ' + f"{hour}:00:00", 'symbol': 'BTC/USDT',
                    'open': 0, 'high': 0, 'low': 0, 'close': 0, 'Volume BTC': 0, 'Volume USDT': 0,
                    'tradecount': 0, 'day': current_day, 'valid': 0, 'count':1
                })
                new_row['date'] = pd.to_datetime(new_row['date'])
                data_frame = data_frame.append(new_row, ignore_index=True)
            data_missing[current_day] = False
    print(data_frame.shape)
    sort_descending(data_frame, output_path)

def check_error(data_frame:pd.DataFrame, output_path):
    length = data_frame.shape[0]
    eps = 1e-8
    for i in range(length - 1):
        current_sample = data_frame.iloc[i]
        next_sample = data_frame.iloc[i + 1]
        if current_sample['valid']:
            if current_sample['open'] < eps or current_sample['high'] < eps or \
                current_sample['low'] < eps or current_sample['close'] < eps or \
                    current_sample['Volume BTC'] < eps or current_sample['Volume USDT'] < eps:
                data_frame.iloc[i, 11] = 0
                print(data_frame.iloc[[i]])
                continue
            elif next_sample['valid']:
                if current_sample['open'] > 2. * next_sample['open'] or \
                    current_sample['high'] > 2. * next_sample['high'] or \
                        current_sample['low'] > 2. * next_sample['low'] or \
                            current_sample['close'] > 2. * next_sample['close']:
                            data_frame.iloc[i, 11] = 0
                            print(data_frame.iloc[[i]])
                            continue
    drop_useless(data_frame, output_path)

def drop_useless(data_frame:pd.DataFrame, output_path):
    data_frame.drop(['unix', 'tradecount', 'count'], axis=1, inplace=True)
    data_frame.to_csv(output_path, index=False)

def draw_plot(data_frame:pd.DataFrame, data_path:str):
    figure_path = data_path.split('/')[:-1]
    figure_path = '/'.join(figure_path) + '/figure.jpg'
    print(figure_path)
    plt.figure(figsize=(25, 10))
    statistic_open = data_frame['open']
    statistic_close = data_frame['close']
    plt.plot(statistic_open, label='open', color='red')
    plt.plot(statistic_close, label='close', color='green')
    plt.legend()
    plt.savefig(figure_path)

if __name__ == "__main__":
    args = preprocess_args().parse()
    raw_data_path = args.raw_data_path
    output_path = args.output_path
    assert raw_data_path, "Please set data root of csv file"
    data_frame =  pd.read_csv(raw_data_path)
    data_frame['date'] = pd.to_datetime(data_frame['date'])
    # drop_duplicated(data_frame, output_path)
    # convert_date_format(data_frame, output_path)
    # sort_descending(data_frame, output_path)
    # check_missing(data_frame, output_path)
    # drop_useless(data_frame, output_path)
    # check_error(data_frame, output_path)
    draw_plot(data_frame, raw_data_path)


