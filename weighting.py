import pandas as pd
import numpy as np

def weighting(table: pd.DataFrame, beta: int):
    num_records = table.shape[0]
    num_attr = table.shape[1]

    hasID = False
    if "ID" in table or "id" in table: # GENERALIZE LATER, find better way to see if records are IDed by attribute in table
        hasID = True
        if "ID" in table:
            ids = table["ID"]
            table = table.drop("ID", axis=1)
        else:
            ids = table["id"]
            table = table.drop("id", axis=1)

    if beta is None: 
        beta = int(0.05 * num_records)

    numerical = table.select_dtypes(include='number') # DataFrame with only numerical attributes
    num_numer_attr = numerical.shape[1]

    categorical = table.select_dtypes(exclude='number') # DataFrame with only categorical attributes
    num_categ_attr = categorical.shape[1]

    numerical_avgs = [] # array of averages of each numerical attribute
    for attr in numerical:
        numerical_avgs.append(numerical[attr].mean())
    numerical_avgs = pd.Series(data=numerical_avgs, index=numerical.columns) # converted into pd series

    freqs = [] # array of dictionaries, each corresponding to an attribute, each dict maps outcome to frequency of outcome
    for attr in categorical:
        dict = {}
        for record in categorical[attr]:
            if record in dict:
                dict[record] += 1
            else:
                dict[record] = 1
        freqs.append(dict)
    freqs = pd.Series(data=freqs, index=categorical.columns) # converted into pd series

    nscores = {} # dictionary mapping record to numerical score
    for ind1, record in numerical.iterrows():
        total = 0
        for ind2, value in record.items():
            total += np.abs(value - numerical_avgs[ind2])
        nscores[ind1] = total
    
    cscores = {} # dictionary mapping record to categorical score
    for ind1, record in categorical.iterrows():
        total = 0
        for ind2, value in record.items():
            total += freqs[ind2][value]
        cscores[ind1] = total / num_categ_attr

    nscores_sorted = sorted(nscores.items(), key = lambda x: x[1], reverse=True) # numerical scores sorted by score descending
    cscores_sorted = sorted(cscores.items(), key = lambda x: x[1]) # categorical scores sorted by score ascending
    num_rankings = {} # maps record number to numerical score ranking (starting from 1 to num_records)
    cat_rankings = {} # maps record number to categorical score ranking (starting from 1 to num_records)
    for ind in range(num_records):
        num_rankings[nscores_sorted[ind][0]] = ind+1
        cat_rankings[cscores_sorted[ind][0]] = ind+1
    arscores = [] # average rank scores and their corresponding record index
    weightscores = {} # maps records to their weight scores
    for recordind in range(num_records):
        score = np.average([num_rankings[recordind], cat_rankings[recordind]])
        arscores.append([score, recordind])
        weightscores[recordind] = num_records - score
    arscores.sort(key = lambda x: x[0])
    outliers = [] # the indexes of the records that are outliers
    for i in arscores[num_records-beta:]:
        outliers.append(i[1])
    
    weightscores_id = {} # maps record (by their true ID) to their weight scores
    if hasID:
        ind_to_id = ids.to_dict()
        for key in weightscores:
            weightscores_id[ind_to_id[key]] = weightscores[key]
        for ind, out in enumerate(outliers):
            outliers[ind] = ind_to_id[out]
    else:
        weightscores_id = weightscores

    return (weightscores_id, outliers)

def main():
    # test1 = pd.DataFrame({'a': [1, 2] * 3,
    #                'b': [True, False] * 3,
    #                'c': [1.0, 2.0] * 3})
    # print(weighting(test1, 0))
    test1 = pd.read_csv('testing.csv')
    print(weighting(test1,3))
    # test2 = pd.read_csv(filepath_or_buffer='adult.csv')
    # weightscores, outliers = weighting(test2, 3)
    # print(test2)
    # print(weightscores)
    # print(outliers)


if __name__ == '__main__':
    main()