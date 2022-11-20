import math
import numpy as np
import pandas as pd
from utils import *
from typing import Dict, Type, List, Tuple

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




def find_next_record(T, e,tree_dict, weight_dict):
    min_loss = math.inf
    min_record_ind = -1
    indices = list(T.index)
    for ind in indices:
        result = calculate_weighted_information_loss(pd.concat([e,T.loc[[ind]]]), tree_dict, weight_dict)
        if result < min_loss:
            min_loss = result
            min_record_ind = ind
    return min_record_ind

def find_next_centroid(T,T_copy, D, dic):
    max_dist = -1* math.inf
    max_record_ind = -1
    indices = list(T_copy.index)

        
    for ind in indices:
        distance = np.linalg.norm(D[dic[ind]])
        if distance > max_dist:
            max_dist = distance
            max_record_ind = ind
    return max_record_ind
        

def grouping_phase(T : pd.DataFrame, K : int, tree_dict : Dict[str,Type[Node]], weight_dict : Dict[int,int]) -> Tuple[List[pd.DataFrame], pd.DataFrame]: 
    D = np.empty([T.shape[0], math.ceil(T.shape[0]/K)])
    T_copy = T.copy(deep=True)
    row_to_ind_dic = {}

    E = []
    indices = list(T.index)
    for i, ind in enumerate(indices):
        row_to_ind_dic[ind] = i

    rand = np.random.randint(0,len(indices))
    rand_ind = indices[rand]
    print("rand_ind is: ", rand_ind)
    print()
    iteration = 1
    while T_copy.shape[0] >= K:
        print("starting iteration: ",iteration)
        print(D)
        print(T_copy)
        print("printing E: ")
        print(len(E))
        for cluster in E:
            print(cluster)
            print()
        print()
        e = None
        centroid_ind = None
        if not E:
            e = pd.DataFrame(T_copy.loc[[rand_ind]])
            T_copy = T_copy.drop([rand_ind])
            centroid_ind = rand_ind
        else:
            centroid_ind = find_next_centroid(T,T_copy,D,row_to_ind_dic)
            print("centroid_ind is: ", centroid_ind)
            print()
            e = pd.DataFrame(T_copy.loc[[centroid_ind]])
            T_copy = T_copy.drop(centroid_ind)
            
        while e.shape[0] < K:
            ind = find_next_record(T_copy,e,tree_dict, weight_dict)
            e = pd.concat([e,T_copy.loc[[ind]]])
            T_copy = T_copy.drop(ind)

        E.append(e)
        T_indices = list(T.index)
        for i,ind in enumerate(T_indices):
            D[i,len(E)-1] = dist(T.loc[centroid_ind], T.loc[ind], T,tree_dict)
        iteration += 1

    left_over = None
    if T_copy.shape[0] > 0:
        left_over = T_copy

    return E,left_over


def add_leftovers(clusters,outliers,leftovers, tree_dict, weight_dict):
    leftovers = list(leftovers)
    outliers = list(outliers)
    while leftovers:
        r = leftovers.pop(-1)
        min_ind = -1
        min_info_loss = math.inf
        for ind,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(np.append(cluster,r), tree_dict, weight_dict) - calculate_weighted_information_loss(cluster, tree_dict, weight_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind
        clusters[min_ind] = np.append(clusters[min_ind],r)
    
    while outliers:
        r = outliers.pop(-1)
        min_ind = -1
        min_info_loss = math.inf
        for ind,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(np.append(cluster,r), tree_dict) - calculate_weighted_information_loss(cluster, tree_dict, weight_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind
        clusters[min_ind] = np.append(clusters[min_ind],r)
    return clusters



print("fml")

test_data = pd.read_csv('testing.csv')
weight_dict, outliers = weighting(test_data, 3)
outlier_records = test_data.loc[test_data['id'].isin(outliers)]
print('Outlier records are: ')
print(outlier_records)
print()

# def better way to do this
outlier_inds = [np.where(test_data['id'] == outlier)[0][0] for outlier in outliers]
test_data = test_data.drop(test_data.index[outlier_inds])
print("remaining test_data is: ")
print(test_data)
print()

tree_dict = parse_heirarchies('heirarchy.txt')
ans,leftover = grouping_phase(test_data,3,tree_dict, weight_dict)
print("Printing ans")
for cluster in ans:
    print(cluster)
    print()

    



