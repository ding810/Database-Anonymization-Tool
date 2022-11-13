## input: list of dataframes
from collections import defaultdict
import math
lst = []
sensitive_attribute = "sa"
dic = defaultdict(int)
sensitive_attribute_set = set()
res = 0

# compute sum
for df in lst:
    for row in df:
        sensitive_attribute_set.add(row[sensitive_attribute])
        dic[row[sensitive_attribute]] += 1

for df in lst:
    for sa in sensitive_attribute_set:
        val = 0
        for row in df:
            if dic[row[sensitive_attribute]] == sa:
                val += 1
        pqs = val/dic[row[sa]]
        res += pqs*math.log(pqs)

print(math.e**res)


