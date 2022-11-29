import numpy as np
import pandas as pd
import utils

def generalize(groups: list[pd.DataFrame], hierarchies: dict, sens_attr: str, sens_attr_column: pd.Series) -> pd.DataFrame:
    output = []
    for group in groups:
        group_df = pd.DataFrame([])
        for attr in group:
            if attr != sens_attr:
                isnumeric = pd.api.types.is_numeric_dtype(group[attr])
                if isnumeric:
                    result = "~" + str(round(np.average(group[attr]), 3))
                else:
                    if attr in hierarchies:
                        tree = hierarchies[attr]
                        lca = None
                        for ind in range(len(group[attr])):
                            if ind == 0:
                                lca = utils.find_parent_node(group[attr][group.index[ind]], group[attr][group.index[ind+1]], tree)
                            else:
                                lca = utils.find_parent_node(group[attr][group.index[ind]], lca.value, tree)
                        result = str(lca.value)
                    else:
                        result = ""
                        for val in np.unique(group[attr]):
                            result += str(val)
                            result += ", "
                        result = result[:len(result)-2] # remove the ", " of the last value

                # add column to end of dataframe, with frequency = num rows
                col = [result for i in range(group.shape[0])]
                group_df.insert(group_df.shape[1], attr, col)
        group_df.insert(group_df.shape[1], sens_attr, sens_attr_column)
        output.append(group_df)
    return pd.concat(output, ignore_index=True)

def main():
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
    # output = generalize([test_group_1, test_group_2], utils.parse_hierarchies('hierarchy.txt'))
    # print(output)
    # for i in output:
    #     print(i)

if __name__ == '__main__':
    main()