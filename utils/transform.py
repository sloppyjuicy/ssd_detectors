
import numpy as np


def affine_transform(src_points, dest_points):
    """Calculates coefficients of affine transformation
    which maps src_points (xi,yi) to dest_points (ui,vi), (i=1,2,3)
    
    # Return
        c: (c00, c01, c02, c10, c11, c12)
    
    source: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/imgwarp.cpp
    """
    x, y = src_points[0::2], src_points[1::2]
    u, v = dest_points[0::2], dest_points[1::2]

    A = np.array([
        [x[0], y[0], 1, 0, 0, 0],
        [x[1], y[1], 1, 0, 0, 0],
        [x[2], y[2], 1, 0, 0, 0],
        [0, 0, 0, x[0], y[0], 1],
        [0, 0, 0, x[1], y[1], 1],
        [0, 0, 0, x[2], y[2], 1],
    ])

    b = np.array([u[0], u[1], u[2], v[0], v[1], v[2]])
    
    return np.linalg.solve(A,b)


def perspective_transform(src_points, dest_points):
    """Calculates coefficients of perspective transformation
    which maps src_points (xi,yi) to dest_points (ui,vi), (i=1,2,3,4)
    
    # Return
        c: (c00, c01, c02, c10, c11, c12, c20, c21)
    
    source: https://github.com/opencv/opencv/blob/master/modules/imgproc/src/imgwarp.cpp
    """
    x, y = src_points[0::2], src_points[1::2]
    u, v = dest_points[0::2], dest_points[1::2]

    A = np.array([
        [x[0], y[0], 1, 0, 0, 0, -x[0]*u[0], -y[0]*u[0]],
        [x[1], y[1], 1, 0, 0, 0, -x[1]*u[1], -y[1]*u[1]],
        [x[2], y[2], 1, 0, 0, 0, -x[2]*u[2], -y[2]*u[2]],
        [x[3], y[3], 1, 0, 0, 0, -x[3]*u[3], -y[3]*u[3]],
        [0, 0, 0, x[0], y[0], 1, -x[0]*v[0], -y[0]*v[0]],
        [0, 0, 0, x[1], y[1], 1, -x[1]*v[1], -y[1]*v[1]],
        [0, 0, 0, x[2], y[2], 1, -x[2]*v[2], -y[2]*v[2]],
        [0, 0, 0, x[3], y[3], 1, -x[3]*v[3], -y[3]*v[3]],
    ])

    b = np.array([u[0], u[1], u[2], u[3], v[0], v[1], v[2], v[3]])

    return np.linalg.solve(A,b)
