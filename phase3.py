from importlib.util import LazyLoader
import pandas as pd
import numpy as np

T = [] # data table of x records
E = set() # set of clusters
D = [] # distance matrix

# matrixDis = pd.DataFrame(np.zeros((len(T), len(E))), columns=feature_list)
matrixDis = np.zeros((len(T), len(E)))

for record in range(len(T)):
    for cluster in E:
        if D[record, cluster]:
            matrixDis[record, cluster] = D[record, cluster]
        else:
            dist = 0
            for ele in cluster:
                # categorical
                if isinstance(T[record], str):
                    # calculate categorical
                    break
                else:
                    dist += np.abs(T[record] - T[ele])/(max(T[ele] - min(T[ele])))

Next_centroid = np.argmax(np.max(matrixDis.to_numpy(), axis=0))