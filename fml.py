from ast import parse
from math import ceil, sqrt
import numpy as np

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


def dist(r1,r2, data, tree_dict):
    ans = 0
    for name in r1.dtype.names:
        if name in tree_dict:
            ans += categorical_dist(r1,r2, name,tree_dict[name])
        else:
            ans += numerical_dist(r1,r2,name, data)
    return ans




def get_height(node, h):
    if not node.neighbors: return h
    return max([get_height(neigh,h+1) for neigh in node.neighbors])
    

def calc_categorical_information_loss(values, tree):
    # add case where only one value maybe

    lowest_common_ancestor = find_parent_node(values[0], values[1],tree)
    ind = 2
    while ind < len(values):
        lowest_common_ancestor = find_parent_node(lowest_common_ancestor,values[ind],tree)
        ind += 1
    return get_height(lowest_common_ancestor,1), get_height(tree,1)

def calc_information_loss(equiv_class, tree_dict):
    ans = 0
    for name in equiv_class[0].dtype.names:
        if name in tree_dict:
            ans += calc_categorical_information_loss(equiv_class[name], tree_dict[name])
        else:
            ans += (max(equiv_class[name]) - min(equiv_class[name])/len(equiv_class))
    ans *= len(equiv_class)
    return ans

def get_weight_score(e):
    id_to_phase1_dict = {5:(1,10), 9:(2,9), 7:(3,8), 1:(4,7), 2:(5,6), 6:(5,6), 10:(6,5), 4:(7,4), 11:(7,4), 3:(8,3), 8:(8,3)}
    ans = 0
    for rec in e:
        weight = id_to_phase1_dict[rec['id']][1]
        ans += weight*weight
    return sqrt(ans)


def find_next_record(T, e):
    min_loss = math.inf
    min_record_ind = -1
    for ind in range(T.shape[0]):
        info_loss = calc_information_loss(e.append(T[ind]))
        weight_score = get_weight_score(e.append(T[ind]))
        if info_loss * weight_score < min_loss:
            min_loss = info_loss * weight_score
            min_record_ind = ind
    return min_record_ind


def find_next_centroid(T,T_copy, E, D):
    max_dist = math.inf
    max_record_ind = -1
    for ind in range(T_copy.shape[0]):
        distances = T[T['id'] == T_copy[ind]['id']]
        distance = np.linalg.norm(distances)
        if distance > max_dist:
            max_dist = distance
            max_record_ind = ind
    return max_record_ind
        

def grouping_phase(T, WT, K):
    D = np.empty([T.shape[0], ceil(T.shape[0]/K)])
    T_copy = np.copy(T)
    E = []
    rand_ind = np.random.randint(0,T.shape[0])
    while T_copy.shape[0] >= K:
        if not E:
            e = np.array([T[rand_ind]])
            T_copy = np.delete(T_copy,rand_ind)
            while e.size < K:
                ind = find_next_record(T_copy,e)
                e = e.append(T_copy[ind])
                T_copy = np.delete(T_copy, ind)
            E.append(e)
            for i in range(D.shape[0]):
                D[i,len(E)-1] = dist(T[rand_ind],T[i])
            
        else:
            centroid_ind = find_next_centroid(T,T_copy,E,D)
            centroid = T_copy[centroid_ind]
            e = np.array([centroid])
            T_copy = np.delete(T_copy, centroid_ind)
            while e.size < K:
                ind = find_next_record(T_copy,e)
                e = e.append(T_copy[ind])
                T_copy = np.delete(T_copy, ind)
            E.append(e)
            for i in range(D.shape[0]):
                D[i,len(E)-1] = dist(centroid,T[i])
    if T_copy.shape[0] > 0:
        E.append(T_copy)

    return E


print("fml")

test_data = np.array([(1,2,'State-gov',13,'M'),\
                        (2,3,'Self-emp',13,'M'),\
                        (3,2,'Private',9,'M'),\
                        (4,3,'Private',7,'M'),\
                        (5,1,'Private',13,'F'),\
                        (6,2,'Private',14,'F'),\
                        (7,3,'Private',5,'F'),\
                        (8,3,'Self-emp',9,'M'),\
                        (9,1,'Private',14,'F'),\
                        (10,2,'Private',13,'M'),\
                        (11,2,'Private',10,'M')], dtype=[('id','i4'),('age','i4'),('workclass','U20'),('education-num','i4'),('sex','U1')])

tree_dict = parse_heirarchies('heirarchy.txt')




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

