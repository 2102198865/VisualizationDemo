from shapesimilarity import shape_similarity
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
from scipy.spatial import minkowski_distance
from scipy.spatial.distance import euclidean
import seaborn as sns

import math
 
# 这个方法是计算两点的距离公式
def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))
 
# 这个就是计算Frechet Distance距离的具体过程,是用递归方式计算
def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]
 
# 这个是给我们调用的方法
def frechet_distance(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)  # ca是a*b的矩阵(3*4),2,3




def frechet_distance2(exp_data,num_data):
    """
    cal fs by dynamic programming
    :param exp_data: array_like, (M,N) shape represents (data points, dimensions)
    :param num_data: array_like, (M,N) shape represents (data points, dimensions)
    # e.g. P = [[2,1] , [3,1], [4,2], [5,1]]
    # Q = [[2,0] , [3,0], [4,0]]
    :return:
    """
    P=exp_data
    Q=num_data
    p_length = len(P)
    q_length = len(Q)
    distance_matrix = np.ones((p_length, q_length)) * -1

    # fill the first value with the distance between
    # the first two points in P and Q
    distance_matrix[0, 0] = euclidean(P[0], Q[0])

    # load the first column and first row with distances (memorize)
    for i in range(1, p_length):
        distance_matrix[i, 0] = max(distance_matrix[i - 1, 0], euclidean(P[i], Q[0]))
    for j in range(1, q_length):
        distance_matrix[0, j] = max(distance_matrix[0, j - 1], euclidean(P[0], Q[j]))

    for i in range(1, p_length):
        for j in range(1, q_length):
            distance_matrix[i, j] = max(
                min(distance_matrix[i - 1, j], distance_matrix[i, j - 1], distance_matrix[i - 1, j - 1]),
                euclidean(P[i], Q[j]))
    # distance_matrix[p_length - 1, q_length - 1]
    sns.heatmap(distance_matrix, annot=True)
    return distance_matrix[p_length-1,q_length-1] # 最后一步即为弗雷彻距离

