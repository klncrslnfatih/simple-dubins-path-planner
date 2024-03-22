"""

Dubins path planner | simple simulation with tkinter

author Atsushi Sakai(@Atsushi_twi) and fkilincarslan

"""
from ast import Delete
from cgitb import text
from email.mime import image
from importlib.resources import path
from logging.handlers import RotatingFileHandler
import sys
import pathlib
from tkinter.font import BOLD
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
from calendar import c
from tkinter import *
import time
from math import cos, sin, radians
from math import sin, cos, atan2, sqrt, acos, pi, hypot
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
        return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                     step_size=0.1, selected_types=None):
    """
    Plan dubins path

    Parameters
    ----------
    s_x : float
        x position of the start point [m]
    s_y : float
        y position of the start point [m]
    s_yaw : float
        yaw angle of the start point [rad]
    g_x : float
        x position of the goal point [m]
    g_y : float
        y position of the end point [m]
    g_yaw : float
        yaw angle of the end point [rad]
    curvature : float
        curvature for curve [1/m]
    step_size : float (optional)
        step size between two path points [m]. Default is 0.1
    selected_types : a list of string or None
        selected path planning types. If None, all types are used for
        path planning, and minimum path length result is returned.
        You can select used path plannings types by a string list.
        e.g.: ["RSL", "RSR"]

    Returns
    -------
    x_list: array
        x positions of the path
    y_list: array
        y positions of the path
    yaw_list: array
        yaw angles of the path
    modes: array
        mode list of the path
    lengths: array
        arrow_length list of the path segments.
    """
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size,
        planning_funcs)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths


def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL, }


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                      step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                      curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw


def main():
    print("Dubins path planner sample start!!")
    import matplotlib.pyplot as plt
    from utils.plot import plot_arrow

    start_x = 50.0  # [m] set to 0<x<1000
    start_y = 50.0  # [m] set to 0<y<500
    start_yaw = np.deg2rad(90)  # [rad]

    end_x = 800.0  # [m] set to 0<x<1000
    end_y = 400.0  # [m] set to 0<y<500
    end_yaw = np.deg2rad(-45.0)  # [rad]

    curvature = 0.01

    path_x, path_y, path_yaw, mode, lengths = plan_dubins_path(start_x,
                                                               start_y,
                                                               start_yaw,
                                                               end_x,
                                                               end_y,
                                                               end_yaw,
                                                               curvature)
    
    WIDTH = 1000
    HEIGHT = 500
    window = Tk()

    canvas = Canvas(window,width=WIDTH,height=HEIGHT)
    canvas.pack()
    window.title('Dubins Path Planner')

    background_photo = PhotoImage(file='background.png')
    background = canvas.create_image(0,0,image=background_photo,anchor=NW)

    photo_jet = PhotoImage(file='jet15.png')
    jet_image = canvas.create_image(end_x-30,end_y-30,image=photo_jet,anchor=NW)

    photo_boom=PhotoImage(file='boom50.png')
    
    photo_missile = PhotoImage(file='missile50.png')
    missile_image = canvas.create_image(start_x-30,start_y-30,image=photo_missile,anchor=NW)
    canvas.create_text(700, 15,fill='white', text='missile_x:  {}'.format(start_x))
    canvas.create_text(800, 15,fill='white', text='missile_y:  {}'.format(start_y))
    canvas.create_text(900, 15,fill='white', text='missile_yaw:  {}'.format(np.rad2deg(start_yaw)))
    canvas.create_text(700, 30,fill='white', text='jet_x:  {}'.format(end_x))
    canvas.create_text(800, 30,fill='white', text='jet_y:  {}'.format(end_y))
    canvas.create_text(900, 30,fill='white', text='jet_yaw:  {}'.format(np.rad2deg(end_yaw)))
    i=0
    
    while True:
         while i<len(path_x)-1:
            line = canvas.create_oval(path_x[i],path_y[i],path_x[i+1],path_y[i+1], width=1.5, fill="black")
            xVelocity = path_x[i+1]-path_x[i]
            yVelocity = path_y[i+1]-path_y[i]
            canvas.move(missile_image,xVelocity,yVelocity)
            missile_angle = np.rad2deg(path_yaw[i]) #degree
            window.update()
            time.sleep(0.05)
            i+=1
            
         canvas.create_image(end_x,end_y,image=photo_boom)
         window.update()
         time.sleep(1)
         break
    
    plt.plot(path_x, path_y, label="".join(mode))
    plot_arrow(start_x, start_y, start_yaw)
    plot_arrow(end_x, end_y, end_yaw)
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()
    

if __name__ == '__main__':
    main()
