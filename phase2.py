import numpy as np
import math

def phase2(T, WT, K):
    T_copy = np.copy(T)
    num_records = T.shape[0]
    num_groups = math.ceil(num_records/K)
    D = np.empty((num_records, num_groups))
    current_ind = 0
    clusters = set()
    rand_record = np.random.randint(0,num_records)
    while (T_copy.shape[0] >= K):
        if (current_ind == 0):
            r = T[rand_record,:]
            T = np.delete(T,rand_record,0)
            new_cluster = set([r])
            while len(new_cluster) < K:
                record_ind = find_min(T,new_cluster)
                new_cluster.add(T[record_ind,:])
                #update dist
                T = np.delete(T,record_ind,0)

            clusters.add(new_cluster)

        else:
            next_centroid = find_next_centroid(T, clusters, D)
            new_cluster = set([next_centroid])
            while len(new_cluster) < K:
                record_ind = find_min(T,new_cluster)
                new_cluster.add(T[record_ind,:])
                #update dist
                T = np.delete(T,record_ind,0)

            clusters.add(new_cluster)
    return clusters
            
