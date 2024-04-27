from curve_reconstruction3 import *
from curve_reconstruction2 import *
import numpy as np
import socket
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
#from sklearn.neighbors import KDTree
from scipy.spatial import KDTree
import csv
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from shapesimilarity import shape_similarity
from parameterize import *
from offeset_detection import *
from scipy.integrate import quad
from read_data import *


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建Socket的客户端
client.connect(('10.27.194.49', 1113))  # 设置相对应的ip地址和端口号 我的IP:192.168.43.2 端口号：1111

def init_source():
    source_points = read_file('data\sim_data3.csv',0,903)
    # minn = 40
    # maxx = 162
    minn = 40
    maxx = 170
    indices = np.where((source_points[:,2]>= minn) & (source_points[:,2] <= maxx))
    source_points = source_points[indices]
    source_point_cloud_thickness = 5
    source_sorted_points_distance = 3
    source_sorted_points = get_sorted_points(source_points,source_point_cloud_thickness, 1,source_sorted_points_distance)
    if source_sorted_points[0][2] > source_sorted_points[-1][2]:
        source_sorted_points = source_sorted_points[::-1]
    plt.plot(source_sorted_points.T[0],source_sorted_points.T[1],'bo')
    plt.show()
    source_sorted_points_t = centripetal(source_sorted_points[:,:2])
    coefficients_tangle = np.polyfit(source_sorted_points_t[1:-1] , source_sorted_points.T[2][1:-1] , 3)
    return coefficients_tangle

def init_target(points):
    random.shuffle(points)
    fig, ax = plt.subplots()
    small_range = 2  # ����С��Χ�ƶ��Ĳ��������Ը�����Ҫ����
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        # ����С��Χ�ڵ����ƫ����
        delta_x = random.uniform(-small_range, small_range)
        delta_y = random.uniform(-small_range, small_range)
        # ��ƫ�������ӵ����������
        points[i] = (x + delta_x, y + delta_y)
    
    ax.plot(points.T[0],points.T[1],'m*')
    target_point_cloud_thickness = 5
    target_sorted_points_distance = 2
    interations = 1
    thinned_points, regression_lines = thin_line(points,target_point_cloud_thickness,interations)
    ax.plot(thinned_points.T[0],thinned_points.T[1],'go')
    # thinned_points = np.vstack((thinned_points.T[0],thinned_points.T[1],points.T[2])).T
    
    target_sorted_points = sort_points2(thinned_points, 130,regression_lines,target_sorted_points_distance)
    # target_sorted_points = get_sorted_points(np.array(points),target_point_cloud_thickness, 1, target_sorted_points_distance)
    
   
    ax.plot(target_sorted_points.T[0],target_sorted_points.T[1],'bo')
    plt.show()
    
    header = ['r1','r2']
    data = np.vstack((np.array(points).T[0],np.array(points).T[1])).T
    with open('D:\\suda\\curves-main\\data\\tested2.csv','w',encoding = 'iso-8859-15', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        
        
        
    x = target_sorted_points.T[0]
    y = target_sorted_points.T[1]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    t = centripetal(target_sorted_points[:,:2])
    coefficients_x = np.polyfit(t[1:-1] , x[1:-1] , 3)
    coefficients_y = np.polyfit(t[1:-1] , y[1:-1] , 3)
    
    return coefficients_x, coefficients_y

def init_target2(c,l):
    target_total_points = read_file2('data\\30_-2to350_0.csv',c,l)
    i = 0
    initialization_point_numbers = 150
    target_point_cloud_thickness = 10
    target_sorted_points_distance = 2
    target_points = []
    for i in range(0,initialization_point_numbers):
        target_points.append(target_total_points[i])
    
    target_sorted_points = get_sorted_points(np.array(target_points),target_point_cloud_thickness, 1, target_sorted_points_distance)
    # target_sorted_points = target_sorted_points[::-1]
    plt.plot(target_sorted_points.T[0],target_sorted_points.T[1],'bo')
    plt.show()
    
    x = target_sorted_points.T[0]
    y = target_sorted_points.T[1]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    t = centripetal(target_sorted_points[:,:2])
    coefficients_x = np.polyfit(t[1:-1] , x[1:-1] , 3)
    coefficients_y = np.polyfit(t[1:-1] , y[1:-1] , 3)
    return coefficients_x, coefficients_y
def run():
    coefficients_tangle =[-221.23094552,  300.23227743,   44.88889328,   40.18254011]
    
    is_offset = True
    last_angle = 0
    target_points = []
    
    window_size = 100
    allowed_offsets_number = 30
    allowed_distance = 10
    offsets = []
    cnt = 0
    
    
    while True:
        point = read_data()
        cnt += 1
        #Ŀ�����ʼ��
        if is_offset:
            start_time = time.perf_counter()
            print('--- getting data to intialize ---')
            # for i in range(0,300):
            #     target_points.append(read_data())
            while (time.perf_counter() - start_time) < 7:
                target_points.append(read_data())
            target_points = target_points[-300:]
            target_points = np.vstack((np.array(target_points).T[0],np.array(target_points).T[1])).T
            # plt.plot(target_points.T[0],target_points.T[1],'m*')
            # plt.show()
            coefficients_x, coefficients_y = init_target(target_points)
            is_offset = False
            print("--- %s seconds to initialize ---" % (time.perf_counter() - start_time))
            
            
            
            
            # plt.ion()
            # # Create a figure and a set of subplots.
            # figure, ax = plt.subplots()
            # # return AxesImage object for using.
            # lines, = ax.plot([], [])
            # ax.set_autoscaley_on(True)
            # # ax.set_aspect('equal')
            # ax.set_xlim(-1.5, 1.5)
            # ax.set_ylim(-1.5, 1.5)
            # ax.grid()
            cnt = 0
            
            continue
        
        
        t = cal_single_t2(coefficients_x, coefficients_y ,coefficients_tangle,last_angle,point[0],point[1])
        fitted_angle = np.polyval(coefficients_tangle , t)
        print('r1:%s,r2:%d,fitted angle:%d'%(point[0],point[1],fitted_angle))
        fitted_angle = int(fitted_angle)
        fitted_angle = str(fitted_angle)
        
        if cnt%2 ==0:
            client.send(fitted_angle.encode('utf-8'))
        
        #offset detect
        # is_offset_single = detect(point[0], point[1], coefficients_x, coefficients_y, allowed_distance)
        # offsets.append(is_offset_single)
        # if(len(offsets)>window_size):
        #     offsets.pop(0)
        # true_count = offsets.count(True)
        # if true_count > allowed_offsets_number:
        #     print("--- The support is offset. Reinitializing... ---\n\n\n")
        #     true_count = 0
        #     offsets.clear()
        #     target_points = []
        #     is_offset = True
        #     cnt = 0

       


        #visualize
        # if cnt%10 == 0:
        #     arm_points =np.array([[-1,1],[0,0]])
        #     vector = arm_points[0]-arm_points[1]
        #     theta = np.radians(fitted_angle)
        #     rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
        #                                 [-np.sin(theta), np.cos(theta)]])    
        #     rotated_vector = np.dot(rotation_matrix, vector)
        #     arm_points = list(arm_points)
        #     arm_points.append(rotated_vector)
            
        #     lines.set_xdata(np.array(arm_points).T[0])
        #     lines.set_ydata(np.array(arm_points).T[1])
        #     #Need both of these in order to rescale
        #     ax.relim()
        #     ax.autoscale_view()
        #     # draw and flush the figure .
        #     figure.canvas.draw()
        #     figure.canvas.flush_events()
            
    client.close()    
        
def run2():
    coefficients_tangle = init_source()
    start_time = time.perf_counter()
    print('--- getting data to intialize ---')
    
    coefficients_x, coefficients_y = init_target(30,-2)
    # target_total_points = read_file2('data\singlesj.csv',30,-2)
    target_total_points = read_file('data\\30_-2to350_0.csv',0,729)
    print("--- %s seconds to initialize ---" % (time.perf_counter() - start_time))
    
    target_fitted_angle = []
    target_real_angle = []
    window_size = 20
    allowed_offsets_number = 15
    allowed_distance = 25
    offsets = []
    
    i = 0
    last_angle = 0
    
    
    
    
    for i in range(0,len(target_total_points)):
        x = target_total_points[i][0]
        y = target_total_points[i][1]
        if i==0:
            t = cal_single_t(coefficients_x, coefficients_y ,x,y)
        else:
            t = cal_single_t2(coefficients_x, coefficients_y ,coefficients_tangle,last_angle,x,y)
        fitted_angle = np.polyval(coefficients_tangle , t)
        last_angle = fitted_angle
        target_fitted_angle.append(fitted_angle)
        target_real_angle.append(target_total_points[i][2])
        print("point %d :fitted angle %d , real angle %d "%(i,fitted_angle,target_total_points[i][2]))


        #offset detect
        is_offset = detect(x, y, coefficients_x, coefficients_y, allowed_distance)
        offsets.append(is_offset)
        if(len(offsets)>window_size):
            offsets.pop(0)
        true_count = offsets.count(True)
        if true_count > allowed_offsets_number:
            print("--- The support is offset. Reinitializing... ---")
            true_count = 0
            offsets.clear()
            coefficients_x, coefficients_y = init_target(350,0)
            
            

    mae_mean = np.mean(np.abs(np.array(target_fitted_angle)-np.array(target_real_angle)))
    print(mae_mean)
    
run()


# while True:
#         message = input("发送的信息：")  # 输入要发送的内容
#         client.send(message.encode('utf-8'))  # 发送
# client.close() #结束后关闭
 