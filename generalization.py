import numpy as np
import pandas as pd
import utils

def generalize(groups: list[pd.DataFrame], hierarchies: dict):
    output = []
    for group in groups:
        for attr in group:
            isnumeric = pd.api.types.is_numeric_dtype(group[attr])
            if isnumeric:
                result = "Average: " + str(round(np.average(group[attr]), 3)) + " Range: " + str(np.min(group[attr])) + " - " + str(np.max(group[attr]))
            else:
                if attr in hierarchies:
                    tree = hierarchies[attr]
                    lca = None
                    for ind in range(len(group[attr])):
                        if ind == 0:
                            lca = utils.find_parent_node(group[attr][ind], group[attr][ind+1], tree)
                        else:
                            lca = utils.find_parent_node(group[attr][ind], lca.value, tree)
                    result = str(lca.value)
                else:
                    result = str(list(np.unique(group[attr])))
            output.append(result)
    return output

test_group_1 = [[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'USA'],
 [ 2, 'Private',  9, 'Divorced', 'Handlers-cleaners', 'White', 'Male', 'USA'],
 [ 3, 'Self-emp', 13, 'Married', 'Exec-managerial', 'White', 'Male', 'USA'],
 [2, 'Private', 10, 'Married', 'Exec-managerial', 'Black', 'Male', 'USA'],
 [ 2, 'Private', 13, 'Married', 'Exec-managerial', 'White', 'Male', 'USA'],
 [ 3, 'Self-emp',  9, 'Married', 'Exec-managerial', 'White', 'Male', 'USA']]

test_group_2 = [[1, 'Private', 14, 'Never-married', 'Prof-specialty', 'White', 'Female', 'USA'],
 [2, 'State-gov', 13, 'Never-married', 'Adm-clerical', 'White', 'Male', 'USA'],
 [1, 'Private', 13, 'Married', 'Prof-specialty', 'Black', 'Female', 'Cuba'],
 [3, 'Private',  5, 'Any', 'Other-service', 'Black', 'Female', 'Jamaica'],
 [3, 'Private',  7, 'Married', 'Handlers-cleaners', 'Black', 'Male', 'USA']]

categories = ["age", "workclass", "education-num", "martial-status", "occupation", "race", "sex", "native-country"]
test_group_1 = pd.DataFrame(test_group_1, columns = categories)
test_group_2 = pd.DataFrame(test_group_2, columns = categories)
print(test_group_1)
print(test_group_2)
# output = generalize([test_group_1, test_group_2], {})
# for i in output:
#     print(i)
output = generalize([test_group_1, test_group_2], utils.parse_heirarchies('heirarchy.txt'))
for i in output:
    print(i)