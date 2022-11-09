import numpy as np
import pandas as pd
import numbers

def generalize(groups: list[pd.DataFrame], hierarchies: dict):
    output = []
    for group in groups:
        for attr in group:
            isnumeric = pd.api.types.is_numeric_dtype(group[attr])
            if isnumeric:
                result = "Average: " + np.average(group[attr]) + " Range: " + np.min(group[attr]) + " - " + np.max(group[attr])
            else:
                if attr in dict:
                    NotImplemented()
                else:
                    result = list(np.unique(group[attr]))
        output.append(result)
    return output