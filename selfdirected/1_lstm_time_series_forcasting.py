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

# %%
# Let's just use the temperature for now
temp_celcius = df['T (degC)']
temp_celcius.plot()

# TODO: continue tutorial at t=7:18.