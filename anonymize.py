import pandas as pd
from filter_sens_attr import filter_sens_attr
from weighting import weighting
from k_anonymity import grouping_phase, add_leftovers
from utils import parse_hierarchies, BadParametersError
from generalization import generalize
from l_diversity import l_diverse
from typing import Dict

def main():
    original_table : pd.DataFrame = pd.read_csv(input("Please enter a link/path to the table that you wish to be anonymized as a CSV: \n"), skipinitialspace=True).iloc[:20, :]
    beta : int = int(input("Please input the number of outliers in your dataset: \n"))
    sens_attr : str = input("Please input the name of the sensitive attribute in your dataset: \n")
    tree_dict : Dict = parse_hierarchies(input("Please enter the path to the text file containing the hierarchy of attributes: \n"))
    k : int = int(input("Please input the k value that you would like your anonymized table to have: \n"))
    l : int = int(input("Please input the l value that you would like your anonymized table to have: \n"))

    output_location : str = input("Please input the file path and file name where you would like your anonymized dataset to be stored: \n")
    got_path = False
    while not got_path:
        try:
            output_file = open(output_location, "x")
        except:
            output_location = input("This file already exists. \nPlease input the file path and file name where you would like your anonymized dataset to be stored: \n")
        else:
            got_path = True

    print("Got it! Anonymizing your dataset now. ")
    print("Filtering sensitive attribute...")
    table, sens_attr_column = filter_sens_attr(original_table, sens_attr)
    print("Weighting records...")
    weightscores, outliers = weighting(table, beta)
    outlier_records = original_table.loc[outliers]
    no_outlier_original_table = original_table.drop(outliers)
    print("Grouping records together...")
    got_groups = False
    while not got_groups:
        count = 0
        succeeded = False
        while count < 5 and not succeeded:
            try:
                groups, leftovers = grouping_phase(no_outlier_original_table, k, tree_dict, weightscores, l, sens_attr)
            except BadParametersError:
                count += 1
            else:
                succeeded = True
        if not succeeded:
            print("Unable to anonymize dataset with given k and l")
            k : int = int(input("Please input the k value that you would like your anonymized table to have: \n"))
            l : int = int(input("Please input the l value that you would like your anonymized table to have: \n"))
        else:
            got_groups = True
    
    print("Adding outiers and leftovers...")
    final_groups = add_leftovers(groups, outlier_records, leftovers, tree_dict, weightscores)
    print("\n\n")
    print(final_groups)
    print("\n\n")
    print("Generalizing groups...")
    generalized_groups = generalize(final_groups, tree_dict, sens_attr, sens_attr_column)

    print("Completed anonymization!")
    print("Storing at {} now...".format(output_location))
    generalized_groups.to_csv(output_file)
    print("Stored! Below is a preview of your anonynmized dataset. \n")
    print(generalized_groups.head(10))

if __name__ == '__main__':
    main()