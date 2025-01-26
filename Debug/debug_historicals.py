import numpy as np
import pandas as pd
import os
import pickle
import datetime
from finta import TA


def retrieve_local_pickle(filename):
    data = pd.DataFrame
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

list_debug_dir = os.listdir()
print(list_debug_dir)

hist: pd.DataFrame
data: np.ndarray
hist2: pd.DataFrame
for item in list_debug_dir:
    if "pickle" in item:
        hist = retrieve_local_pickle(item)
        print(hist)

        timestamp = []
        for begin in hist["Close"].index.tolist():
            # datetime_format = datetime.datetime.fromisoformat(str(begin)).strftime("%m/%d/%y")
            datetime_format = str(begin)
            timestamp.append(datetime_format)

        col = []
        for i in hist.columns:
            col.append(i[0])
        data = hist.to_numpy()
        data.transpose()
        print(data)
        print(type(data))
        hist2 = pd.DataFrame(data, columns=col, index=timestamp)
        # hist2["Prices"] = timestamp
        print(hist2)
        print(type(hist2))