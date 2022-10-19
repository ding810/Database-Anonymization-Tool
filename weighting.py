import pandas as pd
import numpy as np

def weighting(table : pd.DataFrame, beta):
    numerical = table.select_dtypes(include='number')
    categorical = table.select_dtypes(exclude='number')
    numerical_avgs = []
    for attr in numerical:
        numerical_avgs.append(numerical[attr].mean())
    freqs = []
    for attr in categorical:
        dict = {}
        for record in categorical[attr]:
            if record in dict:
                dict[record] += 1
            else:
                dict[record] = 1
        freqs.append(dict)
    for ind, row in table.iterrows():
        print(row)
        # print(pd.DataFrame(row))
        # print(pd.Series(pd.DataFrame(row)))
        # print(row.select_dtypes(include='number'))
        # print(np.subtract(row.select_dtypes(include='number'), numerical_avgs))
        # print(pd.DataFrame(row).select_dtypes(include='number'))
        # num_score = np.sum(np.abs(np.subtract(pd.DataFrame(row).select_dtypes(include='number'), numerical_avgs)))
        # print(num_score)


def main():
    test1 = pd.DataFrame({'a': [1, 2] * 3,
                   'b': [True, False] * 3,
                   'c': [1.0, 2.0] * 3})
    weighting(test1, 0)

if __name__ == '__main__':
    main()