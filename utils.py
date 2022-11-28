import math
import numpy as np
import pandas as pd
from typing import Dict, Type, List, Tuple

# helper functions for working with categorical data
class BadParametersError(Exception):
    pass

class Node:
    def __init__(self, val = None):
        self.value = val
        self.neighbors = []

    
def parse_hierarchies(path):
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


def find_num_leaf(node):
    if node is None: return 0
    if len(node.neighbors) == 0: return 1

    ans = 0
    for neigh in node.neighbors:
        ans += find_num_leaf(neigh)
    return ans


def find_path(val: str, node: Type[Node], path: List[Type[Node]]) -> List[Type[Node]]:
    if val == '?': return[node]
    if node is None: return None
    path.append(node)
    if node.value == val:
        return path
    for neigh in node.neighbors:
        ans = find_path(val,neigh,path)
        if ans is not None: return ans
    path.pop(-1)
    return None

def find_parent_node(a : str, b : str , tree : Type[Node]) -> Type[Node]:
    # print("xd3")
    # print(a)
    # print(b)
    # print([neigh.value for neigh in tree.neighbors])
    a_path = find_path(a,tree,[])
    b_path = find_path(b,tree,[])
    # print(a_path)
    # print(b_path)
    ans = None
    ind = 0
    while ind < len(a_path) and ind < len(b_path) and a_path[ind].value == b_path[ind].value:
        ans = a_path[ind]
        ind += 1
    return ans

def get_height(node, h):
    if not node.neighbors: return h
    return max([get_height(neigh,h+1) for neigh in node.neighbors])
    


# helper functions in calculating distances between two records
def dist(r1,r2, T, tree_dict):
    ans = 0
    for name in T.columns:
        if name in tree_dict:
            ans += categorical_dist(r1,r2, name,tree_dict[name])
        else:
            ans += numerical_dist(r1,r2,name, T)
    return ans

def categorical_dist(r1,r2,name,tree):
    # print("here xd")
    # print(r1)
    # print(r2)
    # print(name)
    if r1[name] == r2[name]: return 0
    total_leaf_nodes = find_num_leaf(tree)
    # print()
    # print()
    # print("xd1")
    # print(r1[name], "  ",r2[name])
    # print([neigh.value for neigh in tree.neighbors])
    parent = find_parent_node(r1[name],r2[name],tree)
    return find_num_leaf(parent)/total_leaf_nodes 

def numerical_dist(r1,r2,name,data):
    column_range = max(data[name]) - min(data[name])
    if column_range == 0: return 0

    return abs(r1[name] - r2[name])/column_range


# helper functions in calculating information loss in a dataset
def calculate_weighted_information_loss(cluster: pd.DataFrame, tree_dict, weight_dict: Dict[int,int]):
    info_loss = calc_information_loss(cluster,tree_dict)
    return get_weight_score(cluster, weight_dict) * info_loss

def get_weight_score(cluster : pd.DataFrame, weight_dict : Dict[int,int]):
    ans = 0
    indices = list(cluster.index)
    for ind in indices:
        # id = cluster.loc[ind]['id']
        weight = weight_dict[ind]
        ans += weight*weight
    return math.sqrt(ans)



def calc_information_loss(equiv_class: pd.DataFrame, tree_dict):
    
    ans = 0
    for name in equiv_class.columns:
        if name == 'id':
            continue
        if name in tree_dict:
            ans += calc_categorical_information_loss(equiv_class[name], tree_dict[name])
        else:
            ans += (max(equiv_class[name]) - min(equiv_class[name])/len(equiv_class))
    ans *= len(equiv_class)
    return ans


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

