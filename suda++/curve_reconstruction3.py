import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
#from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
import math
import csv
import parameterization as pr
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from shapesimilarity import shape_similarity
from parameterize import *
from offeset_detection import *
from frechet_distance import *
from scipy.integrate import quad
# == FUNCTIONS ========================================================================================================

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


# Takes points in [[x1, y1, z1], [x2, y2, z2]...] Numpy Array format
def thin_line(points, point_cloud_thickness, iterations, sample_points=0):
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
        
    # print("--- %s seconds to thin points ---" % (time.perf_counter() - total_start_time))
    # print(f"Finding nearest neighbors for calculating regression lines: {nn_time}")
    # print(f"Calculating regression lines: {rl_time}")
    # print(f"Projecting original points on  regression lines: {prj_time}\n")
    return np.array(new_points), regression_lines

# Sorts points outputed from thin_points()
def sort_points(points, index ,  regression_lines, sorted_point_distance):
    #plt.plot(points.T[0],points.T[1],'go')
    o_sorted_point_distance = sorted_point_distance
    sort_points_time = time.perf_counter()
    # Index of point to be sorted
    #start_index = 0
    o_index = index
    # sorted points array for left and right of intial point to be sorted
    sort_points_left = []
    sort_points_right = []

    # Regression line of previously sorted point
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]

    # Sort points into KDTree for nearest neighbors computation later
    point_tree = spatial.cKDTree(points[:, :2])
    
    cnt = 0
    # Iterative add points sequentially to the sort_points_left array
    while 1:
        cnt +=1
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line ����ع���������ȷ���������ķ�������һ���ع��ߵķ�����ͬ
        v = regression_lines[index][1] - regression_lines[index][0] #�ع�������
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0: #�ж����������ķ����Ƿ��෴������෴����ת��
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        #�ػع��߾���P*�Ҳ���Զ�ĵ㣬��δ����points_in_radius�С�
        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point �ҵ��ڻع����Ͼ���ԭʼ����sorted_point_distance����ĵ�distR_point
        distR_point = points[index, :2] + ((v / np.linalg.norm(v)) * sorted_point_distance * 2 /3) #��
        
        indices_in_radius = point_tree.query_ball_point(points[index, :2],sorted_point_distance )
        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3} �ھ���distR_point�뾶Ϊsorted_point_distance / 3���ҵ㣬���ɵ㼯B
        points_in_radius = points[indices_in_radius]
        
        
        distR_point_vector=distR_point-points[index, :2]
        points_B1=[]
        for x in points_in_radius:
            x_vector=x[:2]-points[index, :2]
            if np.dot(x_vector,distR_point_vector) > 0:
                points_B1.append(x)
        
        # if len(points_B1) < 1:
        #     break
        if cnt >100:
            # return sort_points(points, random.randint(1, len(points)-1) ,  regression_lines, sorted_point_distance)
            break
        
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
        distR_point_vector = distR_point - points[index, :2] #������Զ����������Ϊ��׼����
        nearest_point_vector = nearest_point[:2] - points[index,:2]
        maxdis = 0
        #�Ƚϻ�׼������p*��points_in_radius�еĵ�������нǣ�ѡȡ�н���С�ĵ���Ϊ��Զ�㡣
        for x in points_B1: 
            x_vector = x[:2]- points[index,:2]
            
            
            if  euclidean_distance(x[:2],points[index,:2]) > maxdis:
                nearest_point_vector = nearest_point[:2] - points[index,:2]
                nearest_point = x
                maxdis = euclidean_distance(x[:2], points[index,:2])
                
                
            # if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
            #     nearest_point_vector = nearest_point[:2] - points[index,:2]
            #     nearest_point = x
            
            
        index = np.where(points == nearest_point)[0][0]#?????

        # Add nearest point to 'sort_points_left' array
        #plt.plot(points[index][0],points[index][1],'bo')
        sort_points_right.append(nearest_point)
    #plt.plot(np.array(sort_points_right).T[0],np.array(sort_points_right).T[1],'bo')
    #sort_points_right.reverse()    
    # Do it again but in the other direction of initial starting point 
    index =o_index
    cnt = 0
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    sorted_point_distance = o_sorted_point_distance
    while 1:
        cnt += 1
        # Calulate regression line vector; makes sure line vector is similar direction as previous regression line
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v ) / (np.linalg.norm(regression_line_prev) * np.linalg.norm(v))  < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        # Find point {distR_point} on regression line distance {sorted_point_distance} from original point 
        # 
        # Now vector is substracted from the point to go in other direction
        # 
        
        
        distR_point = points[index,:2] - ((v / np.linalg.norm(v)) * sorted_point_distance *2 /3)  #�ҵ��ع����Ͼ���ԭʼ�㸺sorted_point_distance�ĵ�distR_point
        
        indices_in_radius = point_tree.query_ball_point(points[index, :2],sorted_point_distance )

        # Search nearest neighbors of distR_point within radius {sorted_point_distance / 3}
        points_in_radius = points[indices_in_radius]
        
        distR_point_vector=distR_point[:2]-points[index,:2]
        
        points_B2=[]
        for x in points_in_radius:
            x_vector=x[:2]-points[index,:2]
            if np.dot(x_vector,distR_point_vector) >0:
                points_B2.append(x)
        
        # if len(points_B2) < 1:
        #     break
        if cnt >100:
            # return sort_points(points, random.randint(1, len(points)-1) ,  regression_lines, sorted_point_distance)
            break
    
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
        distR_point_vector = distR_point[:2] - points[index,:2] #ԭʼ�㵽distR_point������
        nearest_point_vector = nearest_point[:2] - points[index,:2] #ԭʼ�㵽nearest_point������
        maxdis = 0
        

            
        for x in points_B2: 
            x_vector = x[:2] - points[index,:2] #ԭʼ�㵽�ڵ㼯B�еĵ������
            
            #��x_vector��distR_point_vector�ļнǸ�С�������nearest_point
            if  euclidean_distance(x[:2],points[index,:2]) > maxdis:
                nearest_point_vector = nearest_point[:2] - points[index,:2]
                nearest_point = x
                maxdis = euclidean_distance(x[:2], points[index,:2])
            # if vg.angle(distR_point_vector, x_vector) < vg.angle(distR_point_vector, nearest_point_vector):
            #     nearest_point_vector = nearest_point[:2] - points[index,:2]
            #     nearest_point = x
            
            
            
        index = np.where(points == nearest_point)[0][0]
        # Add next point to 'sort_points_right' array
        sort_points_left.append(nearest_point)
        #plt.plot(nearest_point[0],nearest_point[1],'ro')
    sort_points_left.reverse()
    #plt.plot(np.array(sort_points_left).T[0],np.array(sort_points_left).T[1],'ro')
    # Combine 'sort_points_right' and 'sort_points_left'
    sort_points_left.extend(sort_points_right)
    
    if(len(sort_points_left)<5):
        return sort_points(points,random.randint(1, len(points)-1), regression_lines, sorted_point_distance)
    
    if sort_points_left[0][0] < sort_points_left[-1][0]:
        sort_points_left.reverse()
    
    
    # print("--- %s seconds to sort points ---" % (time.perf_counter() - sort_points_time))
    return np.array(sort_points_left)

def t_angle(t,angle,test_t):
    degree = 3
    coefficients_t = np.polyfit(t , angle , degree)
    fitted_angle = np.polyval(coefficients_t , test_t)
    # plt.plot(test_t, fitted_angle, 'bo')
    # plt.show()
    return fitted_angle

def read_file(filename,r,l):
    df = pd.read_csv(filename)[r:l]
    x = df['r1'].values
    y = df['r2'].values
    angle = df['angle'].values
    points = np.vstack((x,y,angle)).T
    return points

def get_sorted_points(points,point_cloud_thickness,interations,sorted_points_distance):
    thinned_points, regression_lines = thin_line(points[:,:2],point_cloud_thickness,interations)
    thinned_points = np.vstack((thinned_points.T[0],thinned_points.T[1],points.T[2])).T
    sorted_points = sort_points(thinned_points, 0,regression_lines,sorted_points_distance)
    return sorted_points
    
def get_sorted_points_t(sorted_points):
    t = centripetal(sorted_points[:,:2])
    return t

def get_parameterization(points):
    #coefficients_x,coefficients_y
    return parameterize(points)

def real_time_detection():
    window_size = 20
    offsets_number = 15
    points = read_file('data\singlesj_clean.csv',671,1321)
    sorted_points = get_sorted_points(points)
    cx , cy = get_parameterization(sorted_points)
    offsets =[]
    while(1):
        print("Please input the point:")
        r1,r2 = map(int,input("input r1,r2:").split())
        if r1 == -1 :
            break
        flag = detect(r1, r2, cx, cy, 10)
        
        if flag:
            print("The point is out of range.")
        else:
            print("The point is in range")
            
            
        offsets.append(flag)
        if(len(offsets)>20):
            offsets.pop(0)
        true_count = offsets.count(True)
        if true_count > offsets_number:
            print("The support is offset")
            break

def simulate_detection():
    window_size = 20
    offsets_number = 15
    points = read_file('data\singlesj_clean.csv',16586,16982)
    sorted_points = get_sorted_points(points)
    cx , cy = get_parameterization(sorted_points)
    offsets =[]
    # r1 = points.T[0]
    # r2 = points.T[1]
    
    points_detect = read_file('data\singlesj_clean.csv',19824,20084)
    detect_time = time.perf_counter()
    cnt = 0
    for point in points_detect:
        r1 = point[0]
        r2 = point[1]
        
        flag = detect(r1, r2, cx, cy, 12)
        
        if flag:
            print("The point (%d , %d) is out of range."%(r1,r2))
        else:
            print("The point (%d , %d) is in range"%(r1,r2))
            
        offsets.append(flag)
        
        if(len(offsets)>20):
            offsets.pop(0)
        cnt+= 1
        true_count = offsets.count(True)
        if true_count > offsets_number:
            print("--- %d points have been detected. The support is offset ---"%cnt)
            print("--- %s seconds to detect offset ---" % (time.perf_counter() - detect_time))
            break
    if cnt == len(points_detect):
        print("The support does not offset")    


def get_target_label():
    #source_points = read_file('data\singlesj_clean.csv',1980,2650)
    #source_points = read_file('data\sim_data.csv',0,503)
    source_points = read_file('data\sim_data3.csv',0,903)
    target_points = read_file2('data\singlesj.csv',0,2)
    #target_points.T[0] = np.real(target_points.T[0])
    
    #�ֶ������ǩ
    minn = min(target_points.T[2])
    maxx = max(target_points.T[2])
    indices = np.where((source_points[:,2]>= minn) & (source_points[:,2] <= maxx))
    source_points = source_points[indices]
    
    
    
    
    source_point_cloud_thickness = 5
    target_point_cloud_thickness = 15
    source_sorted_points_distance = 3
    target_sorted_points_distance = 8
    
    source_sorted_points = get_sorted_points(source_points, source_point_cloud_thickness,1, source_sorted_points_distance)
    target_sorted_points = get_sorted_points(target_points, target_point_cloud_thickness,1,  target_sorted_points_distance)
    # target_sorted_points = target_sorted_points[::-1]
    # source_sorted_points = normalize(source_sorted_points)
    # target_sorted_points = normalize(target_sorted_points)
    # target_points = normalize(target_points)
    
    
    

    
    
    print("source domain sorted points numbers:%d, source points distance:%d"%(len(source_sorted_points),source_sorted_points_distance))
    print("target domain sorted points numbers:%d, target points distance:%d"%(len(target_sorted_points),target_sorted_points_distance))

    plt.plot(source_points.T[0],source_points.T[1],'m*')
    plt.plot(target_points.T[0],target_points.T[1],'g*')
    plt.plot(source_sorted_points.T[0],source_sorted_points.T[1],'bo')
    plt.plot(target_sorted_points.T[0],target_sorted_points.T[1],'r')
    plt.show()
    
    
    print("source domain label range: (%d %d)"%(min(source_sorted_points.T[2]),max(source_sorted_points.T[2])))
    print("target domain label range: (%d %d)"%(min(target_sorted_points.T[2]),max(target_sorted_points.T[2])))
    
    
    
    
    source_sorted_points_t = centripetal(source_sorted_points[:,:2])
 
    
    #���Ŀ�������е�����λ��
    target_points_t = cal_t(target_points.T[0],target_points.T[1],target_sorted_points[:,:2])
    #��Դ���t-angleԤ��Ŀ����ı�ǩ
    target_angle = t_angle(source_sorted_points_t,source_sorted_points.T[2],target_points_t)
    
    # д���ļ�
    header = ['angle','r1','r2']
    data = np.vstack((np.real(target_angle),np.real(target_points.T[0]),np.real(target_points.T[1]))).T

    # with open('D:\\DeepDA\\DeepDA\\testdata\\singlesj_0_2_lable.csv','w',encoding = 'iso-8859-15', newline = '') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     writer.writerows(data)
        
    # data2 = np.vstack((target_points.T[2],target_points.T[0],target_points.T[1])).T
    # with open('D:\\DeepDA\\DeepDA\\testdata\\singlesj_350_-3.csv','w',encoding = 'iso-8859-15', newline = '') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     writer.writerows(data2)
    
    mae_mean = np.mean(np.abs(np.array(target_angle)-np.array(target_points.T[2])))
    print(mae_mean)

def get_target_label2():
    #source_points = read_file('data\singlesj_clean.csv',1980,2650)
    #source_points = read_file('data\sim_data.csv',0,503)
    source_points = read_file('data\sim_data3.csv',0,903)
    target_points = read_file('data\singlesj_clean.csv',12329,12738)
    #target_points.T[0] = np.real(target_points.T[0])
    source_points = normalize(source_points)
    target_points = normalize(target_points)
    
    
    source_point_cloud_thickness = 0.55
    target_point_cloud_thickness = 0.4
    source_sorted_points_distance = 0.25
    target_sorted_points_distance = 0.4
    
    source_sorted_points = get_sorted_points(source_points, source_point_cloud_thickness, source_sorted_points_distance)
    target_sorted_points = get_sorted_points(target_points, target_point_cloud_thickness, target_sorted_points_distance)
    
    
    # �ֶ������ǩ
    minn = min(target_points.T[2])
    maxx = max(target_points.T[2])
    indices = np.where((source_sorted_points[:,2]>= minn) & (source_sorted_points[:,2] <= maxx))
    source_sorted_points = source_sorted_points[indices]

    
    
    print("source domain sorted points numbers:%d, source points distance:%d"%(len(source_sorted_points),source_sorted_points_distance))
    print("target domain sorted points numbers:%d, target points distance:%d"%(len(target_sorted_points),target_sorted_points_distance))

    plt.plot(source_points.T[0],source_points.T[1],'m*')
    plt.plot(target_points.T[0],target_points.T[1],'g*')
    plt.plot(source_sorted_points.T[0],source_sorted_points.T[1],'bo')
    plt.plot(target_sorted_points.T[0],target_sorted_points.T[1],'ro')
    plt.show()
    
    
    print("source domain label range: (%d %d)"%(min(source_sorted_points.T[2]),max(source_sorted_points.T[2])))
    print("target domain label range: (%d %d)"%(min(target_sorted_points.T[2]),max(target_sorted_points.T[2])))
    
    
    
    
    source_sorted_points_t = centripetal(source_sorted_points[:,:2])
 
    
    #���Ŀ�������е�����λ��
    target_points_t = cal_t(target_points.T[0],target_points.T[1],target_sorted_points[:,:2])
    #��Դ���t-angleԤ��Ŀ����ı�ǩ
    target_angle = t_angle(source_sorted_points_t,source_sorted_points.T[2],target_points_t)
    
    # д���ļ�
    header = ['angle','r1','r2']
    data = np.vstack((np.real(target_angle),np.real(target_points.T[0]),np.real(target_points.T[1]))).T

    with open('D:\\DeepDA\\DeepDA\\testdata\\singlesj_10_-1_lable.csv','w',encoding = 'iso-8859-15', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        
    # data2 = np.vstack((target_points.T[2],target_points.T[0],target_points.T[1])).T
    # with open('D:\\DeepDA\\DeepDA\\testdata\\singlesj_350_-3.csv','w',encoding = 'iso-8859-15', newline = '') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     writer.writerows(data2)
    
    mae_mean = np.mean(np.abs(np.array(target_angle)-np.array(target_points.T[2])))
    print(mae_mean)


def max_min_normalize(curve):
    min_vals = np.min(curve, axis=0)
    max_vals = np.max(curve, axis=0)

    # ���������С��һ��
    normalized_curve = -1 + 2 * (curve - min_vals) / (max_vals - min_vals)

    return normalized_curve

def standardize(curve):
    # ����ÿ�������ľ�ֵ�ͱ�׼��
    mean_vals = np.mean(curve, axis=0)
    std_devs = np.std(curve, axis=0)

    # ���б�׼��
    standardized_curve = (curve - mean_vals) / std_devs

    return standardized_curve

def get_total_length(points):
    dists = []
    sum_dists = 0
    for id in range(1, len(points)):
        plt.plot(points[id][0],points[id][1],'bo')
        dists.append( np.sqrt(np.linalg.norm(points[id] - points[id - 1]).item()) )
        sum_dists += dists[-1]
    return sum_dists

def normalize(points):
    transform = StandardScaler()
    columns_to_scale = points[:, [0, 1]]
    scaled_columns = StandardScaler().fit_transform(columns_to_scale)
    points[:, [0, 1]] = scaled_columns
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
        
def interpolate(points):
    total_numbers = 80
    if len(points) < total_numbers:
        interpolate_numbers = total_numbers - len(points)
        cx, cy = parameterize(points)
        t_interpolated = np.linspace(0,1,interpolate_numbers)
        x_interpolated = np.polyval(cx,t_interpolated)
        y_interpolated = np.polyval(cy,t_interpolated)
        angle_interpolated = t_angle(centripetal(points[:,:2]),points.T[2],t_interpolated)
        points_interpolated = np.vstack((x_interpolated,y_interpolated,angle_interpolated)).T
        points = np.append(points, points_interpolated, axis=0)
        
        plt.plot(points.T[0],points.T[1],'bo')
        plt.show()
        
def cal_curve_length(points,dt=0.001):
    dt = dt
    t = centripetal(points[:,:2])
    x = points.T[0]
    y = points.T[1]
    
    area_list = []
    
    for i in range(1,len(t)):
        # ����ÿһ΢С���������߳��ȣ�dx = x_{i}-x{i-1}��������1��ʼ
        dl_i = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 ) 
        # ���������洢����
        area_list.append(dl_i)
    
    area = sum(area_list)
    print(area)
    
def align_label_range():
    circular = 0
    lateral = 0
    source_points = read_file('data\sim_data3.csv',0,903)
    # source_points = read_file2('data\singlesj.csv',15,0)
    target_points = read_file2('data\singlesj.csv',circular,lateral)
    
    source_points = normalize(source_points)
    target_points = normalize(target_points)
    
    source_point_cloud_thickness = 0.5
    target_point_cloud_thickness = 0.5

    source_sorted_points_distance = 0.2
    target_sorted_points_distance = 0.2
    
    
    
    source_sorted_points = get_sorted_points(source_points,source_point_cloud_thickness, 3,source_sorted_points_distance)
    target_sorted_points = get_sorted_points(target_points,target_point_cloud_thickness, 3,target_sorted_points_distance)
    # plt.plot(target_points.T[0],target_points.T[1],'m*')
    # for point in target_sorted_points:
    #     plt.plot(point[0],point[1],'bo')
    
    # ����sorted points�������
    # target_len = len(source_sorted_points)
    # if(len(target_sorted_points) < target_len - 10):
    #     while(len(target_sorted_points)<target_len - 10):
    #         target_sorted_points_distance -= 0.1
    #         target_sorted_points = get_sorted_points(target_points,target_point_cloud_thickness, target_sorted_points_distance)
    # elif (len(target_sorted_points) > target_len + 10):
    #     while(len(target_sorted_points) > target_len + 10):
    #         target_sorted_points_distance += 0.02
    #         target_sorted_points = get_sorted_points(target_points,target_point_cloud_thickness, target_sorted_points_distance)
    # print(len(source_sorted_points))
    # print(len(target_sorted_points))
   

    plt.plot(target_points.T[0],target_points.T[1],'m*')
    plt.plot(source_sorted_points.T[0],source_sorted_points.T[1],'bo')
    plt.plot(target_sorted_points.T[0],target_sorted_points.T[1],'go')
    plt.show()

    source_sorted_points_copy = source_sorted_points
    
    # len_target = get_total_length(target_sorted_points[:,:2])
    # len_source = get_total_length(source_sorted_points[:,:2])
    # len_target = get_total_length(target_sorted_points[:,:2])
    print("c= %d,l= %d"%(circular,lateral))
    print("source domain label range: (%d %d)"%(min(source_sorted_points.T[2]),max(source_sorted_points.T[2])))
    print("target domain label range: (%d %d), %d"%(min(target_sorted_points.T[2]),max(target_sorted_points.T[2]),max(target_sorted_points.T[2])-min(target_sorted_points.T[2])))
    cal_curve_length(source_sorted_points[:,:2])
    cal_curve_length(target_sorted_points[:,:2])
    result = frechet_distance(source_sorted_points[:,:2],target_sorted_points[:,:2])
    print(result)
    # print(curve_length_s)
    # print(curve_length_t)
    # print(len_source)
    # print(len_target)
    



    minn = min(target_points.T[2])
    maxx = max(target_points.T[2])
    indices = np.where((source_sorted_points[:,2]>= minn) & (source_sorted_points[:,2] <= maxx))
    source_sorted_points = source_sorted_points[indices]
    cal_curve_length(source_sorted_points[:,:2])
    
    
    plt.plot(source_sorted_points_copy.T[0],source_sorted_points_copy.T[1],'ro')
    plt.plot(source_sorted_points.T[0],source_sorted_points.T[1],'bo')
    plt.plot(target_sorted_points.T[0],target_sorted_points.T[1],'go')
    plt.show()



# get_target_label()
# align_label_range()


# points = read_file2('data/singlesj.csv',0,-4)
# sorted_points = get_sorted_points(points,20,1,10)
# plt.plot(sorted_points.T[0],sorted_points.T[1],'bo')
# for point in sorted_points:
#     plt.plot(point[0],point[1],'ro')


