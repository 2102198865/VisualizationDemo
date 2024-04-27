import numpy as np


def detect(x, y, coefficients_x, coefficients_y, allowed_distance):
    # x = point[0]
    # y = point[1]
    cx = coefficients_x.copy()
    cx[-1] -= x
    t_roots = np.roots(cx)
    t_roots = np.real(t_roots)
    result = True
    
    #两端的点容易被忽略。。带进去就解不出来， 是不是可以把曲线适当延长一下。
    
    fitted_y =[]
    for ti in t_roots:
        if -0.1<= ti <=1.1:
            fitted_y.append(np.polyval(coefficients_y, ti))
    
    if len(fitted_y) < 1:
        result = True
        return result
    
    for yi in fitted_y:
        if abs(yi-y) <= allowed_distance:
            return False
        
    return True



def real_time_detection():
    print("Please input the point:")
    r1,r2 = input("input r1,r2:")
    
    