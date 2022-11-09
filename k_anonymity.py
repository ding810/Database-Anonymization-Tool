import pandas as pd
import numpy as np


from math import ceil, sqrt
import numpy as np
import math

class Node:
    def __init__(self, val = None):
        self.value = val
        self.neighbors = []



def parse_heirarchies(path):
    graph_dict = {}
    with open(path,'r') as f:
        nextLine = f.readline()

        while nextLine:
            nextLine = nextLine.strip()
            if nextLine.split(':')[1] == 'continuous':
                f.readline()
                nextLine = f.readline()


            else:
                head = Node('Any')
                
                prev_stack = [head]
                name = nextLine.split(':')[0]
                graph_dict[name] = head
                f.readline()
                nextLine = f.readline()
                while nextLine and prev_stack:
                    nextLine = nextLine.strip()
                    num_closes = nextLine.count('}')
                    nextNode = Node(nextLine[1: len(nextLine) - num_closes])
                    prev_stack[-1].neighbors.append(nextNode)
                    prev_stack.append(nextNode)
                    for _ in range(num_closes):
                        prev_stack.pop(-1)
                    nextLine = f.readline().strip()



    return graph_dict






def numerical_dist(r1,r2,name,data):
    column_range = max(data[name]) - min(data[name])
    return abs(r1[name] - r2[name])/column_range





def find_num_leaf(node):
    if node is None: return 0
    if len(node.neighbors) == 0: return 1

    ans = 0
    for neigh in node.neighbors:
        ans += find_num_leaf(neigh)
    return ans

def find_path(val, node, path):
    if node is None: return None
    path.append(node)
    if node.value == val:
        return path
    for neigh in node.neighbors:
        ans = find_path(val,neigh,path)
        if ans is not None: return ans
    path.pop(-1)
    return None


def find_parent_node(a, b , tree):
    a_path = find_path(a,tree,[])
    b_path = find_path(b,tree,[])
    ans = None
    ind = 0
    while ind < len(a_path) and ind < len(b_path) and a_path[ind].value == b_path[ind].value:
        ans = a_path[ind]
        ind += 1
    return ans



def categorical_dist(r1,r2,name,tree):
    if r1[name] == r2[name]: return 0
    total_leaf_nodes = find_num_leaf(tree)
    parent = find_parent_node(r1[name],r2[name],tree)
    return find_num_leaf(parent)/total_leaf_nodes


def dist(r1,r2, T, tree_dict):
    ans = 0
    for name in r1.dtype.names:
        if name in tree_dict:
            ans += categorical_dist(r1,r2, name,tree_dict[name])
        else:
            ans += numerical_dist(r1,r2,name, T)
    return ans




def get_height(node, h):
    if not node.neighbors: return h
    return max([get_height(neigh,h+1) for neigh in node.neighbors])
    

def calc_categorical_information_loss(values, tree):
    elements = set()
    for ele in values:
        elements.add(ele)
    
    ele = elements.pop()
    lowest_common_ancestor = find_parent_node(ele,ele,tree)
    
    while elements:
            ele = elements.pop()
            lowest_common_ancestor = find_parent_node(lowest_common_ancestor.value,ele,tree)
    return get_height(lowest_common_ancestor,1)/ get_height(tree,1)

def calc_information_loss(equiv_class, tree_dict):
    
    ans = 0
    for name in equiv_class[0].dtype.names:
        if name in tree_dict:
            xd = calc_categorical_information_loss(equiv_class[name], tree_dict[name])
            ans += xd
        else:
            ans += (max(equiv_class[name]) - min(equiv_class[name])/len(equiv_class))
    ans *= len(equiv_class)
    # print("info loss for equiv class:")
    # print(equiv_class)
    # print("is: ", ans)
    # print()
    return ans

def get_weight_score(e):
    id_to_phase1_dict = {5:(1,10), 9:(2,9), 7:(3,8), 1:(4,7), 2:(5,6), 6:(5,6), 10:(6,5), 4:(7,4), 11:(7,4), 3:(8,3), 8:(8,3)}
    ans = 0
    for rec in e:
        weight = id_to_phase1_dict[rec['id']][1]
        ans += weight*weight
    return sqrt(ans)


def find_next_record(T, e,tree_dict):
    min_loss = math.inf
    min_record_ind = -1
    # print("starting finding next record")
    # print(T)
    # print(e)
    for ind in range(T.shape[0]):
        info_loss = calc_information_loss(np.append(e,T[ind]),tree_dict)
        weight_score = get_weight_score(np.append(e,T[ind]))
        if info_loss * weight_score < min_loss:
            min_loss = info_loss * weight_score
            min_record_ind = ind
    # print("next record is")
    # print(T[min_record_ind])
    return min_record_ind


def calculate_weighted_information_loss(cluster, tree_dict):
    info_loss = calc_information_loss(cluster,tree_dict)
    return get_weight_score(cluster) * info_loss


def find_next_centroid(T,T_copy, D):
    # print("finding next centroid")
    # print("T is")
    # print(T)
    # print("T_copy")
    # print(T_copy)
    # print("D is")
    # print(D)
    max_dist = math.inf
    max_record_ind = -1
    for ind in range(T_copy.shape[0]):
        ind = np.where(T['id'] == T_copy[ind]['id'])[0][0]
        distance = np.linalg.norm(D[ind])
        if distance > max_dist:
            max_dist = distance
            max_record_ind = ind
    # print("next centroid is")
    # print(T_copy[max_record_ind])
    return max_record_ind
        

def grouping_phase(T, WT, K, tree_dict):
    D = np.empty([T.shape[0], ceil(T.shape[0]/K)])
    T_copy = np.copy(T)
    E = []
    rand_ind = np.random.randint(0,T.shape[0])
    # print("starting grouping phase")
    # print("rand int is: ",rand_ind)
    # print()

    while T_copy.shape[0] >= K:
        # print("current E is: ")
        # print(E)
        # print("current D: ")
        # print(D)
        # print()
        if not E:
            e = np.array([T[rand_ind]])
            T_copy = np.delete(T_copy,rand_ind)
            while e.size < K:
                ind = find_next_record(T_copy,e,tree_dict)
                e = np.append(e,T_copy[ind])
                T_copy = np.delete(T_copy, ind)
            E.append(e)
            for i in range(D.shape[0]):
                D[i,len(E)-1] = dist(T[rand_ind],T[i],T,tree_dict)
            
        else:
            centroid_ind = find_next_centroid(T,T_copy,D)
            centroid = T_copy[centroid_ind]
            e = np.array([centroid])
            T_copy = np.delete(T_copy, centroid_ind)
            while e.size < K:
                ind = find_next_record(T_copy,e,tree_dict)
                e = np.append(e,T_copy[ind])
                T_copy = np.delete(T_copy, ind)
            E.append(e)
            for i in range(D.shape[0]):
                D[i,len(E)-1] = dist(centroid,T[i],T,tree_dict)
    left_over = None
    if T_copy.shape[0] > 0:
        left_over = T_copy

    return E,left_over


def final_fker(clusters,outliers,leftovers, tree_dict):
    leftovers = list(leftovers)
    outliers = list(outliers)
    while leftovers:
        r = leftovers.pop(-1)
        min_ind = -1
        min_info_loss = math.inf
        for ind,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(np.append(cluster,r), tree_dict) - calculate_weighted_information_loss(cluster, tree_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind
        clusters[min_ind] = np.append(clusters[min_ind],r)
    while outliers:
        r = outliers.pop(-1)
        min_ind = -1
        min_info_loss = math.inf
        for ind,cluster in enumerate(clusters):
            wil = calculate_weighted_information_loss(np.append(cluster,r), tree_dict) - calculate_weighted_information_loss(cluster, tree_dict)
            if wil < min_info_loss:
                min_info_loss, min_ind = wil, ind
        clusters[min_ind] = np.append(clusters[min_ind],r)
    return clusters

# workclass_tree = tree_dict['workclass']

# print(test_data[['id','workclass']])
# print(test_data[2])
# print(test_data[1])
# # print(numerical_dist(test_data[0],test_data[1],'Age',test_data))
# print(find_num_leaf(workclass_tree))
# # print(find_num_leaf(sex_tree))
# # print([node.value for node in find_path('State-gov',workclass_tree,[])])
# # print(find_parent_node('Government','Federal-gov',workclass_tree).value)
# print(categorical_dist(test_data[2],test_data[1], 'workclass', workclass_tree))
# print(parse_heirarchies('heirarchy.txt'))

# print(dist(test_data[0],test_data[1],test_data,tree_dict))


def weighting(table: pd.DataFrame, beta: int):
    hasID = False
    if "ID" in table: # GENERALIZE LATER, find better way to see if records are IDed by attribute in table
        hasID = True
        ids = table["ID"]
        table = table.drop("ID", axis=1)

    num_records = table.shape[0]
    num_attr = table.shape[1]

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


def main():
    # test1 = pd.DataFrame({'a': [1, 2] * 3,
    #                'b': [True, False] * 3,
    #                'c': [1.0, 2.0] * 3})
    # print(weighting(test1, 0))
    test2 = pd.read_csv('testing.csv')
    print("Example csv file: ")
    column_names = list(test2.columns)
    d_types = test2.dtypes
    # d_types = [d_type if (d_type == np.int64) else np.dtype('U30') for d_type in d_types ]
    np_dtypes = []
    for i in range(len(column_names)):
        np_dtypes.append((column_names[i],d_types[i]))

          
    test2 = test2.to_numpy()
    test2 = [list(row) for row in test2]
    print(test2)
    print(np_dtypes)  
    test2 = np.array(test2,np_dtypes)
    # newshit = np.array([],dtype=np_dtypes)
    # print(test2)
    print()
    print()
    weightscores, outliers = weighting(test2, 3)
    print("Weight scores and outliers: ")
    print(weightscores)
    print()
    print(outliers)
    print()
    print()
    
    # test_data = test2.to_numpy(dtype=np_dtypes)
    # print(test_data.dtype)
    print("fml")

    test_data = np.array([(1,2,'State-gov',13,'Male'),\
                            (2,3,'Self-emp',13,'Male'),\
                            (3,2,'Private',9,'Male'),\
                            (4,3,'Private',7,'Male'),\
                            (6,2,'Private',14,'Female'),\
                            (7,3,'Private',5,'Female'),\
                            (8,3,'Self-emp',9,'Male'),\
                            (9,1,'Private',14,'Female'),\
                            (10,2,'Private',13,'Male'),\
                            (11,2,'Private',10,'Male')], dtype=[('id','i4'),('age','i4'),('workclass','U20'),('education-num','i4'),('sex','U10')])

    outliers = np.array([(5,1,'Private',13,'Female')])
    tree_dict = parse_heirarchies('heirarchy.txt')
    # print("test_data is: ")
    # print(test_data)
    # print()

    # print("groupings are: ")
    # ans,leftover = grouping_phase(test_data,[],3,tree_dict)
    # for group in ans:
    #     print(group)
    #     print()
    # print(leftover)

    # print("after adjustment:")
    # ans = (final_fker(ans,leftover,[],tree_dict))
    # for group in ans:
    #     print(group)
    #     print()


if __name__ == '__main__':
    main()