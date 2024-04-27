import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import newton

# centripetal parameterization
def centripetal(points):
    dists = []
    sum_dists = 0
    for id in range(1, len(points)):
        dists.append( np.sqrt(np.linalg.norm(points[id] - points[id - 1]).item()) )
        sum_dists += dists[-1]
    t = [0]
    for d in dists:
        t.append(t[-1] + d / sum_dists)
    return t

def parameterize(points):
    x = points.T[0]
    y = points.T[1]
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    t = centripetal(points)
    
    degree = 3
    
    coefficients_x = np.polyfit(t , np.array(x) , degree)
    coefficients_y = np.polyfit(t , np.array(y) , degree)
    
    # fitted_x = np.polyval(coefficients_x , t)
    # fitted_y = np.polyval(coefficients_y , t)
    
    # print(coefficients_x)
    # print(coefficients_y)
    
    # plt.plot(fitted_x , fitted_y , 'm*')
    
    # plt.plot(x , y , 'bo')
    
    # plt.show()
    
    
    
    return coefficients_x ,coefficients_y

def cal_t(x , y , points):
    coefficients_x ,coefficients_y = parameterize(points)
    total_t = []
    for xi,yi in zip(x,y):
        cx = coefficients_x.copy()
        cx[-1] -= xi
        t_value = 0
        t_values = np.roots(cx)
        er=1000
        for ti in t_values:
            if 0 <= ti <=1:
                yt = np.polyval(coefficients_y , ti)
                if abs(yt - yi) <er:
                    er = abs(yt - yi)
                    t_value = ti
        total_t.append(t_value)
    return total_t
def cal_single_t(coefficients_x,coefficients_y,r1,r2):
    cx = coefficients_x.copy()
    cx[-1] -= r1
    t_value = 0
    t_values = np.roots(cx)    
    er=1000
    for ti in t_values:
        if -0.1 <= ti <=1.1:
            yt = np.polyval(coefficients_y , ti)
            if abs(yt - r2) <er:
                er = abs(yt - r2)
                t_value = ti
    
    return t_value


def cal_single_t2(coefficients_x,coefficients_y, coefficients_tangle ,last_angle,r1,r2):
    cx = coefficients_x.copy()
    cx[-1] -= r1
    t_value = 0
    t_values = np.roots(cx)
      
    t_values = [value for value in t_values if -1.1 <= value <= 1.1]
    t_values = np.array(np.real(t_values))  
    yt = np.polyval(coefficients_y, t_values)
    absolute_differences = np.abs(yt - r2)
    sorted_indices  = np.array(np.argsort(absolute_differences))
    sorted_t = t_values[sorted_indices]    
    fitted_angle = np.polyval(coefficients_tangle, sorted_t)
    if len(sorted_t)> 1 and abs(fitted_angle[0] - last_angle) and last_angle != 0 > 40:
        return sorted_t[1]
    else :
        return sorted_t[0]
    

def test(points):
    
    coefficients_x ,coefficients_y, t = parameterize(points)
    fitted_t = []
    # t_values=[]
    # for x in points.T[0]:
    #     t_value = inverse_function2(coefficients_x , x)
    #     t_values.append(t_value)
    start_time =  time.perf_counter()
    for point in points:
        x = point.T[0]
        y = point.T[1]
        cx = coefficients_x.copy()
        cx[-1] -= x
        t_values = np.roots(cx)
        t_values = np.real(t_values)
        er = 1000
        for ti in t_values:
            if 0 <= ti <= 1:
                yt = np.polyval(coefficients_y , ti)
                if abs(yt - y) < er:
                    er = abs(yt - y)
                    t_value = ti
        fitted_t.append(t_value)
        
    print("--- %s seconds to calculate t ---" % (time.perf_counter() - start_time))    
        
    # x_value = 742.4096930488425
    # y_value = 703.349943266794
    # coefficients_x[-1] -=x_value
    # t_values = np.roots(coefficients_x)
    # er = 1000
    # for ti in t_values:
    #     if ti > 1 or ti < 0:
    #         continue
    #     else:
    #         yt = np.polyval(coefficients_y , ti)
    #         if abs(yt - y_value) < er:
    #             er = abs(yt - y_value)
    #             t_value = ti
    
    test_mae_mean = np.mean(np.abs(np.array(t)-np.array(fitted_t)))
    
    print(t)
    # print(t_value)
    #print(t_values)
    print(fitted_t)
    
    print("The average error of the fitting value of t is:",test_mae_mean)
    
    
    # print("For x = " , x_value, "  y = ",y_value, "t is approximately" ,t_value)