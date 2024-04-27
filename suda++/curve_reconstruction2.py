import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
#from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
import math
import parameterization as pr
import time
import csv
from parameterize import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from shapesimilarity import shape_similarity
# == FUNCTIONS ========================================================================================================

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


# Takes points in [[x1, y1, z1], [x2, y2, z2]...] Numpy Array format
def thin_line(points, point_cloud_thickness=5, iterations=1, sample_points=0):
    itera = 0
    total_start_time =  time.perf_counter()
    while itera <iterations:
        if sample_points != 0:
            points = points[:sample_points]
        
        # Sort points into KDTree for nearest neighbors computation later ��������KDTree�У��Ա��Ժ������ٽ�����
        point_tree = spatial.cKDTree(points)

        # Empty array for transformed points ���洦����ĵ���
        new_points = []
        # Empty array for regression lines corresponding ^^ points ����ÿ����Ļع���
        regression_lines = []
        nn_time = 0
        rl_time = 0
        prj_time = 0
        for point in point_tree.data:
            # Get list of points within specified radius {point_cloud_thickness} ȡ���ض��뾶�ڵĵ�
            start_time = time.perf_counter()
            points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]
            nn_time += time.perf_counter()- start_time
            # plt.plot(points.T[0],points.T[1],'m*')
            # plt.plot(points_in_radius.T[0],points_in_radius.T[1],'r*')
            # plt.show()
            # Get mean of points within radius ȡ�ڰ뾶�ڵ����е�ľ�ֵ
            start_time = time.perf_counter()
            data_mean = points_in_radius.mean(axis=0)

            #����Ƥ��ѷ���ϵ��
            # transposed_points = points_in_radius.T
            # correlation_matrix = np.corrcoef(transposed_points)
            # pearson_correlation_coefficient = correlation_matrix[0, 1]
            # print("Pearson Correlation Coefficient:", pearson_correlation_coefficient)
            

            # Calulate 3D regression line/principal component in point form with 2 coordinates �����ľֲ�3D�ع���
            uu, dd, vv = np.linalg.svd(points_in_radius - data_mean) #ʹ������ֵ�ֽ�SVD��Ѱ�����ݷֲ�����Ҫά�ȣ���ԭʼ�ĸ�ά����ӳ�䵽��ά�ӿռ���ʵ�����ݽ�ά��������ÿ�����ȥ���ݾ�ֵ�������Ļ�������SVD����������������������ֵ������������
            #vv[0]����������еĵ�һ������������Ӧ��������ֵ��Ҳ�����ɷֵķ���np.grid[-1:1:2j]��ʾ ����һ���� -1 �� 1 �ľ��Ȳ��������������㡣[:, np.newaxis]���������ڽ��������һ���µ�ά�ȣ���һά����ת��Ϊ��ά����
            linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]   
            linepts += data_mean #���߶�ƽ�Ƶ�ȫ������ϵ�У��Եõ������Ļع��ߡ�
            regression_lines.append(list(linepts))
            rl_time += time.perf_counter() - start_time

            # Project original point onto 3D regression line ��pointͶӰ���ֲ��ع�����
            start_time = time.perf_counter()
            ap = point - linepts[0]
            ab = linepts[1] - linepts[0]
            point_moved = linepts[0] + np.dot(ap,ab) / np.dot(ab,ab) * ab
            prj_time += time.perf_counter()- start_time
            new_points.append(list(point_moved))
            
        points = new_points
        itera += 1
        # plt.plot(np.array(points).T[0],np.array(points).T[1],'bo')
        # plt.show()
        
    print("--- %s seconds to thin points ---" % (time.perf_counter() - total_start_time))
    print(f"Finding nearest neighbors for calculating regression lines: {nn_time}")
    print(f"Calculating regression lines: {rl_time}")
    print(f"Projecting original points on  regression lines: {prj_time}\n")
    return np.array(new_points), regression_lines


# Sorts points outputed from thin_points()
def sort_points2(points, index , regression_lines, sorted_point_distance):
    # plt.plot(points.T[0],points.T[1],'go')
    o_sorted_point_distance = sorted_point_distance
    sort_points_time = time.perf_counter()
    # Index of point to be sorted
    o_index = index
    # sorted points array for left and right of intial point to be sorted
    sort_points_left = [points[index]]
    sort_points_right = []

    # Regression line of previously sorted point
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points)

    cn = 0
    # Iterative add points sequentially to the sort_points_left array
    while 1:
        cn += 1
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line ����ع���������ȷ���������ķ�������һ���ع��ߵķ�����ͬ
        v = regression_lines[index][1] - regression_lines[index][0] #�ع�������
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0: #�ж����������ķ����Ƿ��෴������෴����ת��
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        #�ػع��߾���P*�Ҳ���Զ�ĵ㣬��δ����points_in_radius�С�
        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point �ҵ��ڻع����Ͼ���ԭʼ����sorted_point_distance����ĵ�distR_point
        distR_point = points[index] + ((v / np.linalg.norm(v)) * sorted_point_distance * 2 /3) #��
        
        
        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3} �ھ���distR_point�뾶Ϊsorted_point_distance / 3���ҵ㣬���ɵ㼯B
        points_in_radius = point_tree.data[point_tree.query_ball_point(points[index], sorted_point_distance )]
        distR_point_vector=distR_point-points[index]
        points_B1=[]
        

        
        for x in points_in_radius:
            x_vector=x-points[index]
            if np.dot(x_vector,distR_point_vector) >0:
                points_B1.append(x)

        # if len(points_B1) < 1:
        #     break
        #��ʱ�������޷��˳�ѭ�������������һ������
        if cn > 100 :
           # return sort_points(points,random.randint(1, len(points)-1), regression_lines, sorted_point_distance)
            break
        
        #Ϊ�˱�����Ϊ�м�ĳ�ξ����Դ��ֱ�ӶϿ�������sorted_point_distance�ľ��룬�����2���ж�
        if len(points_B1) < 1:
            if sorted_point_distance > 3 * o_sorted_point_distance:
                break
            sorted_point_distance = sorted_point_distance + 0.2
            continue
        else :
            sorted_point_distance = o_sorted_point_distance
       
        
        
        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0] #��ʼ��Ϊ��1����
        distR_point_vector = distR_point - points[index] #������Զ����������Ϊ��׼����
        nearest_point_vector = nearest_point - points[index]
        maxdis = 0
        #�Ƚϻ�׼������p*��points_in_radius�еĵ�������нǣ�ѡȡ�н���С�ĵ���Ϊ��Զ�㡣
        for x in points_B1: 
            x_vector = x - points[index]
            if  euclidean_distance(x,points[index]) > maxdis:
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
                maxdis = euclidean_distance(x, points[index])
        index = np.where(points == nearest_point)[0][0]

        # Add nearest point to 'sort_points_left' array
        #ax.plot(points[index][0],points[index][1],'bo')
        # plt.plot(nearest_point[0],nearest_point[1],'bo')
        sort_points_right.append(nearest_point)
        
    print(cn)
    
    sort_points_right.reverse()    
    sorted_point_distance = o_sorted_point_distance
    # Do it again but in the other direction of initial starting point 
    index = o_index
    cn = 0
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    while 1:
        cn = cn+1
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        # 
        # Now vector is substracted from the point to go in other direction
        # 
        distR_point = points[index] - ((v / np.linalg.norm(v)) * sorted_point_distance *2 /3)  #�ҵ��ع����Ͼ���ԭʼ�㸺sorted_point_distance�ĵ�distR_point
        
        # plt.plot(points[index][0],points[index][1],'ro')

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = point_tree.data[point_tree.query_ball_point(points[index], sorted_point_distance )]
        
        distR_point_vector=distR_point-points[index]
        
        points_B2=[]
        for x in points_in_radius:
            # plt.plot(x[0],x[1],'m*')
            x_vector=x-points[index]
            if np.dot(x_vector,distR_point_vector) >0:
                # plt.plot(x[0],x[1],'g*')
                points_B2.append(x)
        
        # if len(points_B2) < 1:
        #     break
        
        
        #��ʱ�������޷��˳�ѭ�������������һ������
        if cn > 100 :
            #return sort_points(points,random.randint(1, len(points))-1, regression_lines, sorted_point_distance)
            break
        
        #Ϊ�˱�����Ϊ�м�ĳ�ξ����Դ��ֱ�ӶϿ�������sorted_point_distance�ľ��룬�����2���ж�
        if len(points_B2) < 1:
            if sorted_point_distance > 3 * o_sorted_point_distance:
                break
            sorted_point_distance = sorted_point_distance + 0.2
            continue
        else :
            sorted_point_distance = o_sorted_point_distance
       
    

        # Neighbor of distR_point with smallest angle to regression line vector is selected as next point in order
        # 
        # CAN BE OPTIMIZED
        # 
        nearest_point = points_in_radius[0]
        distR_point_vector = distR_point - points[index] #ԭʼ�㵽distR_point������
        nearest_point_vector = nearest_point - points[index] #ԭʼ�㵽nearest_point������
        maxdis = 0
        
        for x in points_B2: 
            x_vector = x - points[index] #ԭʼ�㵽�ڵ㼯B�еĵ������
            #��x_vector��distR_point_vector�ļнǸ�С�������nearest_point
            if  euclidean_distance(x,points[index]) > maxdis:
                nearest_point_vector = nearest_point - points[index]
                nearest_point = x
                maxdis = euclidean_distance(x, points[index])
        index = np.where(points == nearest_point)[0][0]
        # plt.plot(nearest_point[0],nearest_point[1],'bo')
        # Add next point to 'sort_points_right' array
        sort_points_left.append(nearest_point)
        
    print(cn)
    sort_points_left.reverse()
    # Combine 'sort_points_right' and 'sort_points_left'
    sort_points_left.extend(sort_points_right[::-1])
    
    if sort_points_left[0][0] < sort_points_left[-1][0]:
        sort_points_left.reverse()
    # if(len(sort_points_left)<10):
    #     return sort_points(points,random.randint(1, len(points)-1), regression_lines, sorted_point_distance)
    print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
    return np.array(sort_points_left)
def normalize(points):
    transform = StandardScaler()
    columns_to_scale = points[:, [0, 1]]
    scaled_columns = StandardScaler().fit_transform(columns_to_scale)
    points[:, [0, 1]] = scaled_columns
    return points
def read_file(filename,r,l):
    df = pd.read_csv(filename)[r:l]
    x = df['r1'].values
    y = df['r2'].values
    angle = df['angle'].values
    points = np.vstack((x,y,angle)).T
    return points
def read_file2(filename, c, l):
    file = pd.read_csv(filename)
    file = np.array(file)
    circular = c
    lateral = l
    df = []
    for item in file:
        sh1 = item[8]
        sh2 = item[9]
        if sh1 == circular and sh2 == lateral:
            df.append(item)
    df= np.array(df)
    x = df.T[2]
    y = df.T[3]
    angle = df.T[1]
    points = np.vstack((x,y,angle)).T
    return points







# # points = read_file2("data/singlesj.csv",100,1)

# #points = read_file("data/sim_data3.csv",0,903)
# points = read_file("data/tested2.csv",0,500)
# # points = normalize(points)

# thinned_points, regression_lines=thin_line(points[:,:2])

# sorted_points = sort_points2(thinned_points, 130, regression_lines,1)



# # test(sorted_points)


# plt.plot(points.T[0],points.T[1],'m*')

# plt.plot(thinned_points.T[0], thinned_points.T[1], 'go')




# plt.plot(sorted_points.T[0], sorted_points.T[1], 'bo')
# plt.plot(sorted_points.T[0], sorted_points.T[1], '-b')
# #plt.title("circular = %d , lateral = %d ,angle range (%d %d)"%(circular,lateral,min(angle),max(angle)))

# # print("angle(",min(angle),",",max(angle),")")
# # print("r1(",min(y),",",max(y),")")
# # print("r2(",min(x),",",max(x),")")
# plt.show()
# # print("hh")

# # header = ['angle','r1','r2']
# # data = np.vstack((np.real(points.T[2]),np.real(points.T[0]),np.real(points.T[1]))).T

# # with open('D:\\suda\\curves-main\\data\\sim_data4.csv','w',encoding = 'iso-8859-15', newline = '') as f:
# #     writer = csv.writer(f)
# #     writer.writerow(header)
# #     writer.writerows(data)
