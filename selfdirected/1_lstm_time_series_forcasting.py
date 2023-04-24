# %%
""" From https://www.youtube.com/watch?v=c0k-YLQGKjY&list=PLQY2H8rRoyvwLbzbnKJ59NkZvQAW9wLbx&index=2&t=0s

    This is a time series forcasting problem. We are trying to predict the next value in the sequence.
    We are using a LSTM (Long Short Term Memory) network to do this.

    We'll be predicting the temperature.
""" 
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
zip_path = tf.keras.utils.get_file(
    origin = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    fname = "jena_climate_2009_2016.csv.zip",
    extract = True
)
csv_path, _ = os.path.splitext(zip_path)

# Data format:
# Date Time,          p (mbar), T (degC), Tpot (K), Tdew (degC), rh (%), VPmax (mbar), VPact (mbar), VPdef (mbar), sh (g/kg), H2OC (mmol/mol), rho (g/m**3), wv (m/s), max. wv (m/s), wd (deg)
# 01.01.2009 00:10:00 996.52	-8.02	  265.4     -8.9         93.3    3.33          3.11          0.22          1.94       3.12             1307.75       1.03      1.75           152.3
# 01.01.2009 00:20:00 996.57	-8.41	  265.01    -9.28        93.4    3.23          3.02          0.21          1.89       3.03             1309.8        0.72      1.5            136.1
df = pd.read_csv(csv_path)
print(csv_path)
df.head()


# %%

# data is every 10 minutes. Let's drop samples so we just have 1 per hour
# - start on row 5, for hour 1:00 (first row is 0:10)
# - reindex with date-time.
df = df[5::6]
df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

# extract day of year and time of day into a new column
df['Day of Year'] = df.index.dayofyear
df['Hour of Day'] = df.index.hour

# %%
# Let's just use the temperature for now
temp_celcius = df['T (degC)']
temp_celcius.plot()

df['Day of Year'].plot()
df['Hour of Day'].plot()
plt.legend()

print('Columns: ', df.columns)

# TODO: continue tutorial at t=7:18.

# %%
# Each time, we're going to include one more item in the input, 
# then predict the next one.
#   - note: each item in x is actually a tensor of values at that time. If we're
#           just using temperature, it will be a 1-D tensor of length 1.
def get_data_from_df(df, x_cols, y_col, window_size=72, train_frac=0.8, val_frac=0.1, test_frac=0.1):

    assert train_frac + val_frac + test_frac - 1.0 < np.finfo(float).eps

    df = df[x_cols + [y_col]]
    df_as_np = df.to_numpy()
    df_x = df_as_np[:, :-1]
    df_y = df_as_np[:, -1]

    data = []

    for i in range(len(df_as_np) - window_size):
        x = df_x[i:i+window_size]
        y = df_y[i+window_size]
        data.append((x, y))

    # shuffle data
    np.random.shuffle(data)

    train_index = int(train_frac * len(data))
    val_index = int((train_frac + val_frac) * len(data))
    data_train, data_val, data_test = data[:train_index], data[train_index:val_index], data[val_index:]
    data_train = pd.DataFrame(data_train, columns=['x', 'y'])
    data_val = pd.DataFrame(data_val, columns=['x', 'y'])
    data_test = pd.DataFrame(data_test, columns=['x', 'y'])
    
    train_x = np.stack(data_train['x'].to_numpy())
    train_y = np.stack(data_train['y'].to_numpy())
    val_x = np.stack(data_val['x'].to_numpy())
    val_y = np.stack(data_val['y'].to_numpy())
    test_x = np.stack(data_test['x'].to_numpy())
    test_y = np.stack(data_test['y'].to_numpy())

    return train_x, train_y, val_x, val_y, test_x, test_y

# %%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# %%
window_size = 40
# x_cols = ['T (degC)']
x_cols = ['T (degC)', 'Day of Year', 'Hour of Day']
y_col = 'T (degC)'

# ensure that columns we're using are floats
df_used = df[x_cols + [y_col]].astype(float)


train_x, train_y, val_x, val_y, test_x, test_y = get_data_from_df(df_used, x_cols, y_col, window_size=window_size)

#print shapes of data
print('train_x.shape: ', train_x.shape)
print('train_y.shape: ', train_y.shape)
print('val_x.shape: ', val_x.shape)
print('val_y.shape: ', val_y.shape)
print('test_x.shape: ', test_x.shape)
print('test_y.shape: ', test_y.shape)

# %%
lstm_units = 64 # suggested to use multiples of 32. Could range from 32 to 512, 
# but multiple 128 cell layers is superior to a single 512 cell layer, due to the
# vanishing gradient problem.


# todo: figure out why this is crashing?
model = Sequential()


model.add(InputLayer((window_size, len(x_cols))))
model.add(LSTM(lstm_units))
model.add(Dense(8, 'relu'))
model.add(Dense(1, 'linear'))


model.summary()
# %%
