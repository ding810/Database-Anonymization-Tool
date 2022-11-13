## input: list of dataframes
from collections import defaultdict
import math
import pandas as pd

def l_diverse(groups, sensitive_attribute):
    dic = defaultdict(int)
    sensitive_attribute_set = set()
    res = 0

    # compute sum
    for df in groups:
        # print(df)
        for ind, record in df.iterrows():
            sensitive_attribute_set.add(record.loc[sensitive_attribute])
            dic[record.loc[sensitive_attribute]] += 1
    # print(dic)
    # print(sensitive_attribute_set)

    for df in groups:
        temp = 0
        for sa in sensitive_attribute_set:
            val = 0
            for ind, record in df.iterrows():
                if record.loc[sensitive_attribute] == sa:
                    val += 1
            pqs = val/dic[sa]
            if pqs != 0:
                temp -= pqs*math.log(pqs)
        # print(temp)
        # both not 0, values stored in both, can take min now!
        if temp != 0 and res != 0:
            res = min(temp, res)
        else:
            # otherwise take the one that is not equal to 0
            res = max(temp, res)

    # print(res)
    # print(math.floor(math.e**res*100)/100)
    # print(math.e**res)
    return math.floor(math.e**res*100)/100

test = []
test_group_1 = [[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'HD'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'VI'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'C'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'C']]

test_group_2 = [[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'C'],
 [ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'HD'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'VI'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'VI']]

test_group_3 = [[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'C'],
 [ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'HD'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'VI'],
[ 2, 'Private', 14, 'Married', 'Exec-managerial', 'White', 'Female', 'C']]

categories = ["age", "workclass", "education-num", "martial-status", "occupation", "race", "sex", "sc"]
test_group_1 = pd.DataFrame(test_group_1, columns = categories)
test_group_2 = pd.DataFrame(test_group_2, columns = categories)
test_group_3 = pd.DataFrame(test_group_3, columns = categories)
test.append(test_group_1)
test.append(test_group_2)
test.append(test_group_3)
print(l_diverse(test, "sc"))

# https://personal.utdallas.edu/~mxk055100/courses/privacy08f_files/ldiversity.pdf