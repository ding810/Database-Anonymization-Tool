import pandas as pd
from filter_sens_attr import filter_sens_attr
from weighting import weighting
from k_anonymity import grouping_phase, add_leftovers
from utils import parse_hierarchies, BadParametersError
from generalization import generalize
from typing import Dict


sens_attr = "occupation"
original_table = pd.read_csv("adult.csv")
beta = 0
tree_dict = parse_hierarchies("hierarchy.txt")
record_arr = []
for record_num in [20,40,60,80]:
    k_arr = []
    for k in range(1,8):
        l_arr = []
        for l in range(1,k+1):
            table, sens_attr_column = filter_sens_attr(original_table, sens_attr)
            weightscores, outliers = weighting(table, beta)
            outlier_records = original_table.loc[outliers]
            no_outlier_original_table = original_table.drop(outliers)
            count = 0
            succeeded = False
            while count < 5 and not succeeded:
                try:
                    groups, leftovers = grouping_phase(no_outlier_original_table, k, tree_dict, weightscores, l, sens_attr)
                except BadParametersError:
                    count += 1
                else:
                    succeeded = True
            l_arr.append(succeeded)
        k_arr.apendd(l_arr)
    record_arr.append(k_arr)

print(record_arr)