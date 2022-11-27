import math
import numpy as np
import pandas as pd
from utils import *
from typing import Dict, Type, List, Tuple
from weighting import weighting
from generalization import generalize

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
        

def grouping_phase(T : pd.DataFrame, K : int, tree_dict : Dict[str,Type[Node]], weight_dict : Dict[int,int], L : int, sensitive_attribute : str) -> Tuple[List[pd.DataFrame], pd.DataFrame]: 
    D = np.empty([T.shape[0], math.ceil(T.shape[0]/K)])
    T_copy = T.copy(deep=True)
    row_to_ind_dic = {}

    E = []
    indices = list(T.index)
    for i, ind in enumerate(indices):
        row_to_ind_dic[ind] = i

    rand = np.random.randint(0,len(indices))
    rand_ind = indices[rand]
    iteration = 1
    while T_copy.shape[0] >= K:
        e = None
        centroid_ind = None
        seen = set()
        iter_copy  = T_copy.copy(deep=True)
        if not E:
            e = pd.DataFrame(T_copy.loc[[rand_ind]])
            seen.add(rand_ind)
            iter_copy = iter_copy.drop([rand_ind])
            T_copy = T_copy.drop([rand_ind])
            centroid_ind = rand_ind
            
        else:
            centroid_ind = find_next_centroid(T,T_copy,D,row_to_ind_dic)
            e = pd.DataFrame(T_copy.loc[[centroid_ind]])
            seen.add(centroid_ind)
            T_copy = T_copy.drop(centroid_ind)
            iter_copy = iter_copy.drop([centroid_ind])
            

            
        while e.shape[0] < K and iter_copy.shape[0] > 0:
            ind = find_next_record(iter_copy,e,tree_dict, weight_dict)
            if len(seen) >= L or ind not in seen:
                seen.add(ind)
                e = pd.concat([e,T_copy.loc[[ind]]])
                T_copy = T_copy.drop(ind)
                iter_copy = iter_copy.drop(ind)
            else:
                iter_copy = iter_copy.drop(ind)

        E.append(e)
        T_indices = list(T.index)
        for i,ind in enumerate(T_indices):
            D[i,len(E)-1] = dist(T.loc[centroid_ind], T.loc[ind], T,tree_dict)
        
    left_over = None
    if T_copy.shape[0] > 0:
        left_over = T_copy

    return E,left_over


def add_leftovers(clusters : List[pd.DataFrame], 
                  outliers : pd.DataFrame, 
                  leftovers : pd.DataFrame, 
                  tree_dict : Dict, 
                  weight_dict : Dict):
    for ind1, record in leftovers.iterrows():
        min_ind = -1
        min_info_loss = math.inf
        for ind2,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(pd.concat([cluster, record]), tree_dict, weight_dict) - calculate_weighted_information_loss(cluster, tree_dict, weight_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind2
        clusters[min_ind] = pd.concat(clusters[min_ind],record)
    
    for ind1, record in outliers.iterrows():
        min_ind = -1
        min_info_loss = math.inf
        for ind2,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(pd.concat([cluster, record]), tree_dict, weight_dict) - calculate_weighted_information_loss(cluster, tree_dict, weight_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind2
        clusters[min_ind] = pd.concat(clusters[min_ind],record)
    
    return clusters



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

tree_dict = parse_hierarchies('heirarchy.txt')
ans,leftover = grouping_phase(test_data,3,tree_dict, weight_dict, 3, "race")
print("Printing ans")
for cluster in ans:
    print(cluster)
    print()

    
