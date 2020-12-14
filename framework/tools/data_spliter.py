import numpy as np
import pandas as pd

def at_least_one_label_in_test_set(df,train_percent, label_name, max_category):
    training_amount = int(df.shape[0] * train_percent)
    counting_map = {}
    rows_index = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        label_values = row[label_name]
        should_add = False
        for label in label_values:
            if not label in counting_map:
                counting_map[label] = 1
                should_add = True
        if should_add:
            rows_index.append(i)
        if len(counting_map) >= max_category:
            break

    test_set_temp = df.iloc[rows_index]
    df = df.drop(df.index[rows_index])
    training_amount = training_amount - test_set_temp.shape[0]

    df = df.reindex(np.random.permutation(df.index))
    

    training_set = df.iloc[:training_amount]
    test_set = df.iloc[training_set.shape[0]:]
    test_set = pd.concat([test_set_temp, test_set])
    return training_set, test_set
