from __future__ import absolute_import
from __future__ import print_function

import os
import queue
import sys
import optparse
import random
import traci
import numpy as np
import matplotlib.pyplot as plt
import time
from sumolib import checkBinary

# 检测是否已经添加环境变量
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


# sumo自带的，不知道有啥用
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

########## 碰撞参数 ##########################
l = 5
d = 2
ld = (l**2+d**2)**0.5
phi_ = np.arctan(d/l)   # 21.8°
arrow_length = 5
cs1 = 0.3                  # 调整衰减速率，越小危险区域区域越大
#############################################
m1 = 1000
lamta = 0.00001
tao = 0.8
aerfa1 = 0.01
aerfa2 = 0.001
kesai = 0.2

m2 = 1000
beita = 0.03
beita1 = 0.01
beita2 = 0.001

gama = 0.01

m3 = 1000

Ai = 0.1
y_l = -3.2
kexi = 0.1
delta = 0.5
sigema = 0.5
#############################################


# 碰撞时两车心距离
def get_oo_length(theta, phi):
    # 将输入的theta转换到0~pi/2，phi转换到0~pi。
    phi = phi % np.pi
    theta = theta % (2*np.pi)
    if theta < 0:   # 转换为正
        theta += 2*np.pi
    if theta > np.pi/2:
        if theta > np.pi*3/2:   # 第四象限
            theta = 2*np.pi - theta
            phi = -phi
        elif theta > np.pi:     # 第三象限
            theta = theta - np.pi
        elif theta > np.pi/2:   # 第二象限
            theta = np.pi - theta
            phi = -phi
    phi = phi % np.pi
    phi = (phi+2*np.pi) % np.pi
    if theta <= phi_:
        if phi <= 2*theta:
            return l/(2*np.cos(theta)) + (np.sin(phi)*(d-l*np.tan(theta)) + l)/(2*np.cos(phi-theta))
        if phi <= theta + phi_:
            return l/(2*np.cos(theta)) + np.sin(phi)*(d-l*np.tan(phi-theta))/(2*np.cos(theta)) + l/(2*np.cos(phi-theta))
        if phi <= 2*(theta + phi_):
            return l/(2*np.cos(theta)) + np.cos(phi)*(l-d/np.tan(phi-theta))/(2*np.cos(theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi/2:
            return l/(2*np.cos(theta)) + np.cos(phi)*(d+l*np.tan(theta))/(2*np.sin(phi-theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi + 2*theta - 2*phi_:
            return l/(2*np.cos(theta)) - np.cos(phi)*(d-l*np.tan(theta))/(2*np.sin(phi-theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi - theta:
            return l/(2*np.cos(theta)) - np.cos(phi)*(l+d/np.tan(phi-theta))/(2*np.cos(theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi:
            return l/(2*np.cos(theta)) + np.sin(phi)*(d-l/np.tan(phi-theta-np.pi/2))/(2*np.cos(theta)) - l/(2*np.cos(phi-theta))
    if theta <= np.pi/4:
        if phi <= theta - phi_:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta-phi))/(2*np.sin(theta)) + d/(2*np.sin(theta-phi))
        if phi <= 2*(theta-phi_):
            return d/(2*np.sin(theta)) + np.cos(phi)*(d-l*np.tan(theta-phi))/(2*np.sin(theta)) + l/(2*np.cos(theta-phi))
        if phi <= 2*theta:
            return d/(2*np.sin(theta)) + np.cos(phi)*(l - d/np.tan(theta))/(2*np.cos(phi-theta)) + l/(2*np.cos(phi-theta))
        if phi <= np.pi/2:
            return (l+ld*np.cos(phi-phi_))/(2*np.cos(theta))
        if phi <= np.pi:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta))/(2*np.sin(phi-theta)) + d/(2*np.sin(phi-theta))
    if theta <= np.pi/4 + phi_:
        if phi <= theta - phi_:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta-phi))/(2*np.sin(theta)) + d/(2*np.sin(theta-phi))
        if phi <= 2*(theta-phi_):
            return d/(2*np.sin(theta)) + np.cos(phi)*(d-l*np.tan(theta-phi))/(2*np.sin(theta)) + l/(2*np.cos(theta-phi))
        if phi <= np.pi/2:
            return d/(2*np.sin(theta)) + np.cos(phi)*(l-d/np.tan(theta))/(2*np.cos(phi-theta)) + l/(2*np.cos(phi-theta))
        if phi <= 2*theta:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(phi-theta))/(2*np.sin(theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta))/(2*np.sin(phi-theta)) + d/(2*np.sin(phi-theta))
    if theta <= np.pi/2:
        if phi <= theta - phi_:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta-phi))/(2*np.sin(theta)) + d/(2*np.sin(theta-phi))
        if phi <= np.pi/2:
            return d/(2*np.sin(theta)) + np.cos(phi)*(d-l*np.tan(theta-phi))/(2*np.sin(theta)) + l/(2*np.cos(theta-phi))
        if phi <= 2*theta:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(phi-theta))/(2*np.sin(theta)) + d/(2*np.sin(phi-theta))
        if phi <= np.pi:
            return d/(2*np.sin(theta)) + np.sin(phi)*(l-d/np.tan(theta))/(2*np.sin(phi-theta)) + d/(2*np.sin(phi-theta))
    return

# 计算P与E
def get_P_and_E(p1, p2, v1, v2, angle1, angle2, fai_v1, fai_v2, a1, a1_theta):
    theta = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) - fai_v1
    fai_v2 -= fai_v1
    P = get_oo_length(theta, fai_v2)/((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5
    Vx = v1*np.cos(angle1) - v2*np.cos(angle2)
    Vy = v1*np.sin(angle1) - v2*np.sin(angle2)
    V_delta = ((Vx**2)+(Vy**2))**0.5
    theta_V_delta = np.arctan2(Vy, Vx)
    E = P**cs1 * np.exp(0.002*v1 + 
                        0.002*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                        0.010*(a1)*abs(np.cos((theta-a1_theta)/2)))
    return P, E

# 场景A
def scene_A(fai_v1 = 0, fai_v2 = 0):
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    sumocfgfile = "E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_A\\2_1.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, "-c", sumocfgfile])  # 打开sumocfg文件

    time_list=[]
    P_list=[]
    ttci_list=[]
    v1_list=[]
    v2_list=[]
    a1_list = []
    a2_list = []

    E_list=[]
    E_1s_list=[]
    E_2s_list=[]
    E_3s_list=[]

    x1_list = []
    x2_list = []

    x1_3s_list = []
    x2_3s_list = []
    

    queue_p1 = queue.Queue(2)
    queue_p2 = queue.Queue(2)
    
    for step in range(0, 800):  # 仿真时间
        # time.sleep(0.1)
        traci.simulationStep()          # 逐步仿真
        t = traci.simulation.getTime()  # 仿真时间
        all_vehicle_id = traci.vehicle.getIDList()              # 获得所有车的id
        traci.gui.trackVehicle('View #0', all_vehicle_id[0])    # 在给定的视图上对给定的车辆进行视觉跟踪

        if len(all_vehicle_id) == 2:
            time_list.append(t)

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])   # 前车位置
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])   # 后车位置

            x1_list.append(p1[0])
            x2_list.append(p2[0])

            queue_p1.put(p1)
            queue_p2.put(p2)

            if queue_p1.full():
                old_p1 = queue_p1.get()
                fai_v1 = np.arctan2(p1[1]-old_p1[1], p1[0]-old_p1[0])
                # print('fai_v1=', fai_v1)

            if queue_p2.full():
                old_p2 = queue_p2.get()
                fai_v2 = np.arctan2(p2[1] - old_p2[1], p2[0] - old_p2[0])
                # print('fai_v2=', fai_v2)

            if p1[0] > 2000:
                traci.vehicle.setSpeed('1', 10)     # 行驶600米后，1车减速
            
            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            v1_list.append(v1)
            v2_list.append(v2)
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])
            a1_list.append(a1)
            a2_list.append(a2)
            # angle1 = traci.vehicle.getAngle(all_vehicle_id[0])
            # angle2 = traci.vehicle.getAngle(all_vehicle_id[1])
            angle1 = fai_v1
            angle2 = fai_v2

            pred_1 = []     # 1车坐标位置
            pred_2 = []     # 2车预测坐标
            for i in range(3):
                pred_1.append([p1[0], p1[1]])
                pred_2.append([p2[0], p2[1]])

            pred_1[0][0] += (v1 + 0.5 * a1)                 # 1车预测一秒横坐标
            pred_1[0][1] += (0.1 * random.uniform(-1, 1))   # 1车预测一秒纵坐标
            pred_1[1][0] += (v1 * 2 + 0.5 * a1 * 4)
            pred_1[1][1] += (0.1 * random.uniform(-1, 1))
            pred_1[2][0] += (v1 * 3 + 0.5 * a1 * 9)
            pred_1[2][1] += (0.1 * random.uniform(-0.5, 0.5))
            x1_3s_list.append(pred_1[2][0])


            pred_2[0][0] += (v2 + 0.5 * a2)                 # 2车预测一秒横坐标
            pred_2[0][1] += (0.1 * random.uniform(-1, 1))   # 2车预测一秒纵坐标
            pred_2[1][0] += (v2 * 2 + 0.5 * a2 * 4)
            pred_2[1][1] += (0.1 * random.uniform(-1, 1))
            pred_2[2][0] += (v2 * 3 + 0.5 * a2 * 9)
            pred_2[2][1] += (0.1 * random.uniform(-0.5, 0.5))
            x2_3s_list.append(pred_2[2][0])

            a1_theta = 0

            v1_1s = v1 + a1
            v1_2s = v1 + 2 * a1
            v1_3s = v1 + 3 * a1

            v2_1s = v2 + a2
            v2_2s = v2 + 2 * a2
            v2_3s = v2 + 3 * a2

            # 当前
            P, E = get_P_and_E(p1, p2, v1, v2, angle1, angle2, fai_v1, fai_v2, a1, a1_theta)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred_1[0], pred_2[0], v1_1s, v2_1s, angle1, angle2, fai_v1, fai_v2, a1, a1_theta)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred_1[1], pred_2[1], v1_2s, v2_2s, angle1, angle2, fai_v1, fai_v2, a1, a1_theta)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred_1[2], pred_2[2], v1_3s, v2_3s, angle1, angle2, fai_v1, fai_v2, a1, a1_theta)

            ttci=(v2-v1)/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5)

            P_list.append(P**cs1)
            ttci_list.append(ttci)
            E_list.append(E)
            E_1s_list.append(E_1s)
            E_2s_list.append(E_2s)
            E_3s_list.append(E_3s)


    assert len(time_list) == len(P_list)

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(1, 2, 2)
    ax4 = ax3.twinx()

    ax1.plot(time_list, v1_list, '-', c='tab:red', label='v1')
    ax1.plot(time_list, v2_list, '-', c='tab:blue', label='v2')
    ax2.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:green', label='Δx')
    # ax2.plot(time_list, a1_list, '-', c='tab:orange', label='a1')
    # ax2.plot(time_list, a2_list, '-', c='tab:cyan', label='a2')

    ax3.plot(time_list, P_list,  '-', c='tab:orange', label='P')
    ax3.plot(time_list, E_list, '-', c='tab:cyan', label='E')
    ax3.plot(time_list, E_3s_list, '--', c='tab:purple', label='E_3s', alpha=0.5)
    ax4.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:green', label='Δx')


    ax1.set_xlabel('t(s)')
    ax1.set_ylabel('v(m/s)')

    ax2.set_ylabel('x(m)')

    ax3.set_xlabel('t(s)')
    ax3.set_ylabel('E')

    ax4.set_ylabel('x(m)')

    # ax1.spines['left'].set_color(c='tab:blue')  # 注意ax1是left
    # ax2.spines['right'].set_color(c='tab:green')
    # ax3.spines['right'].set_color(c='tab:orange')
    # ax4.spines['right'].set_color(c='tab:purple')


    # ax2.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
    # ax3.tick_params(axis='y', color='tab:orange', labelcolor='tab:orange')
    # ax4.tick_params(axis='y', color='tab:purple', labelcolor='tab:purple')


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc='best')

    ax4.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc='center right')

    plt.tight_layout()
    # plt.grid()  # 网格
    plt.show()
    traci.close()
    return

# 场景B
def scene_B(fai = 0, fai_1 = 0):
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sumocfgfile = "E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_B\\4.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, "-c", sumocfgfile])  # 打开sumocfg文件

    time_list=[]
    F_list=[]
    ttci_list=[]
    v1_list=[]
    v2_list=[]
    # SM_list=[]

    # v3_list=[]
    Fv1_list=[]
    Fp_list=[]
    # Fv2_list=[]
    # Fl_list=[]
    dx_list=[]
    dy_list=[]
    d_list=[]

    queue_ego = queue.Queue(2)
    queue_p1 = queue.Queue(2)

    for step in range(0, 28):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        # print("simulation_time=", simulation_time)
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id
        # 获取所有车的position
        # all_vehicle_position = [traci.vehicle.getPosition(i) for i in all_vehicle_id]
        # all_vehicle_speed = [traci.vehicle.getSpeed(i) for i in all_vehicle_id]
        # all_vehicle_a = [traci.vehicle.getAcceleration(i) for i in all_vehicle_id]
        traci.gui.trackVehicle('View #0', all_vehicle_id[0])    # 在给定的视图上对给定的车辆进行视觉跟踪


        # if '1'in all_vehicle_id:
            # traci.vehicle.setLaneChangeMode("1", 0b001000000000)

        if len(all_vehicle_id) == 2:

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])
            # p3 = traci.vehicle.getPosition(all_vehicle_id[2])
            d=((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
            d_x = abs(p1[0] - p2[0])
            d_y = abs(p1[1] - p2[1])


            queue_ego.put(p2)
            queue_p1.put(p1)

            if queue_ego.full():
                old_p2 = queue_ego.get()
                fai = np.arctan2((p2[1]-old_p2[1]), (p2[0]-old_p2[0]))
                print('fai=', fai)

            if queue_p1.full():
                old_p1 = queue_p1.get()
                fai_1 = np.arctan2((p1[1]-old_p1[1]), (p1[0]-old_p1[0]))
                print('fai_1=', fai_1)

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            # v3 = traci.vehicle.getSpeed(all_vehicle_id[2])
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])

            pp = p1[0] - p2[0]
            if pp>10 and pp <25:
                traci.vehicle.changeLane('1', 0, 5)

            pred_ego = []
            pred = []
            for i in range(3):
                pred.append([p1[0], p1[1]])
                pred_ego.append([p2[0], p2[1]])

            if fai_1 == 0:
                pred[0][0] += (v1+0.5*a1)
                pred[0][1] += (0.1*random.uniform(-1, 1))

                pred[1][0] += (2*v1+0.5*a1*4)
                pred[1][1] += (0.1*random.uniform(-1, 1))

                pred[2][0] += (3 * v1 + 0.5 * a1 * 9)
                pred[2][1] += (0.1*random.uniform(-1, 1))
            else:
                pred[0][0] += (v1 + 0.5 * a1)
                pred[0][1] += (-0.64)

                pred[1][0] += (2 * v1 + 0.5 * a1 * 4)
                pred[1][1] += (-0.64*2)

                pred[2][0] += (3 * v1 + 0.5 * a1 * 9)
                pred[2][1] += (-0.64*3)
                pass

            pred_ego[0][0] += (v2+0.5*a2)
            pred_ego[0][1] += (0.1*random.uniform(-1, 1))
            pred_ego[1][0] += (v2*2 + 0.5 * a2*4)
            pred_ego[1][1] += (0.1 * random.uniform(-1, 1))
            pred_ego[2][0] += (v2*3 + 0.5 * a2*9)
            pred_ego[2][1] += (0.1 * random.uniform(-1, 1))

            v1_1s = v1 + a1
            v1_2s = v1 + 2 * a1
            v1_3s = v1 + 3 * a1

            v2_1s = v2 + a2
            v2_2s = v2 + 2 * a2
            v2_3s = v2 + 3 * a2

            # 当前
            P, E = get_P_and_E(p1, p2, v1, v2, fai_1, fai, fai_1, fai, a1, 0)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred[0], pred_ego[0], v1_1s, v2_1s, fai_1, fai, fai_1, fai, a1, 0)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred[1], pred_ego[1], v1_2s, v2_2s, fai_1, fai, fai_1, fai, a1, 0)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred[2], pred_ego[2], v1_3s, v2_3s, fai_1, fai, fai_1, fai, a1, 0)


            ttci = (v2 - v1) / (((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)

            time_list.append(t)
            Fv1_list.append(P)
            v1_list.append(v1)
            v2_list.append(v2)
            Fp_list.append(E_3s)
            F_list.append(E)
            ttci_list.append(ttci)
            d_list.append(d)
            dx_list.append(d_x)
            dy_list.append(d_y)

    # assert len(time_list) == len(F_list)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(1, 2, 2)
    ax4 = ax3.twinx()

    # ax4 = ax3.twinx()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax4 = ax1.twinx()

    # 将构造的ax右侧的spine向右偏移
    # ax3.spines['right'].set_position(('outward', 60))
    # ax4.spines['right'].set_position(('outward', 120))

    # img1_1, = ax1.plot(time_list, v1_list, '--', c='tab:blue')
    # img1_2, = ax1.plot(time_list, v2_list, '-.', c='tab:blue')
    # img2, = ax2.plot(time_list, F_list, c='tab:green')
    # img3, = ax3.plot(time_list, ttci_list, c='tab:orange')

    ax1.plot(time_list, v1_list, '--', c='tab:blue', label='v1')
    ax1.plot(time_list, v2_list, '--', c='tab:red', label='v2')
    # ax1.plot(time_list, d_list, '-.', c='tab:grey', label='distance')
    # ax1.plot(time_list, dx_list, '-.', c='tab:cyan', label='dx')
    # ax1.plot(time_list, dy_list, '-.', c='tab:purple', label='dy')
    ax2.plot(time_list, F_list, c='tab:orange', label='E')

    ax3.plot(time_list, dx_list, '-.', c='tab:grey', label='d_x')
    ax3.plot(time_list, dy_list, '-.', c='tab:green', label='d_y')

    ax4.plot(time_list, F_list, c='tab:orange', label='E')
    ax4.plot(time_list, Fv1_list, c='tab:cyan', label='P')
    ax4.plot(time_list, Fp_list, '--', c='tab:purple', label='E_p3')
    # ax3.plot(time_list, Fv2_list, c='tab:orange', label='DSI_v2')
    # ax3.plot(time_list, Fl_list, c='tab:grey', label='DSI_l')

    # ax3.plot(time_list, ttci_list, c='tab:orange', label='TTCI')
    # ax4.plot(time_list, SM_list, c='tab:purple', label='SM')

    # 获取对应折线图颜色给到spine ylabel yticks yticklabels
    axs = [ax1, ax2, ax3, ax4]
    # imgs = [img1_1, img1_2, img2, img3]

    ax1.set_xlabel('t(s)')
    ax3.set_xlabel('t(s)')

    ax1.set_ylabel('v(m/s)')
    ax2.set_ylabel('E')

    # ax3.set_xlabel('t(s)')
    ax3.set_ylabel('distance(m)')
    # ax3.set_ylabel('TTCI', c='tab:orange')
    ax4.set_ylabel('E')

    # ax1.spines['left'].set_color(c='tab:blue')  # 注意ax1是left
    # ax2.spines['right'].set_color(c='tab:green')
    # ax3.spines['right'].set_color(c='tab:orange')
    # ax4.spines['right'].set_color(c='tab:purple')


    # ax2.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
    # ax3.tick_params(axis='y', color='tab:orange', labelcolor='tab:orange')
    # ax4.tick_params(axis='y', color='tab:purple', labelcolor='tab:purple')


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc=(0.78,0.58))
    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc=(0.75,0.65))
    # ax3.legend()
    # ax1.legend(handles=handle1 + handle2 + handle3 ,
    #           labels=label1 + label2 + label3 ,loc=(0.75,0.72))

    plt.tight_layout()
    plt.show()


    traci.close()
    return

# 场景C
def scene_C(fai = 0):
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sumocfgfile = "E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_B\\3.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, "-c", sumocfgfile])  # 打开sumocfg文件

    time_list=[]
    F_list=[]
    P_list = []
    ttci_list=[]
    v1_list=[]
    v2_list=[]
    SM_list=[]

    v3_list=[]
    Fv1_list=[]
    Fp1_list=[]
    Fp2_list =[]
    Fv2_list=[]
    Fl_list=[]

    queue_ego = queue.Queue(2)

    for step in range(0, 80):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id
        if len(all_vehicle_id)==3:
            traci.gui.trackVehicle('View #0', all_vehicle_id[2])

        if '1' in all_vehicle_id:
            traci.vehicle.setLaneChangeMode("1", 0b001000000000)    # 换车时尊重他人的速度/制动间隙车道，调整速度以满足要求

        if len(all_vehicle_id) == 3:

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])
            p3 = traci.vehicle.getPosition(all_vehicle_id[2])

            queue_ego.put(p3)

            if queue_ego.full():
                old_ego = queue_ego.get()
                fai = np.arctan2((p3[1]-old_ego[1]), (p3[0]-old_ego[0]))
                print('fai=', fai)

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            v3 = traci.vehicle.getSpeed(all_vehicle_id[2])
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])
            a3 = traci.vehicle.getAcceleration(all_vehicle_id[2])

            pred_ego = []
            pred_1 = []
            pred_2 = []
            for i in range(3):
                pred_1.append([p1[0], p1[1]])
                pred_2.append([p2[0], p2[1]])
                pred_ego.append([p3[0], p3[1]])

            if fai == 0:
                if (v3 > 17 and step<30) or (v3>15 and(step>53 and step<57)):
                    pred_ego[0][0] += 0.65*v3
                    pred_ego[0][1] += (0.1 * random.uniform(-1, 1))
                else:
                    pred_ego[0][0] += (v3 + 0.5 * a3)
                    pred_ego[0][1] += (0.1 * random.uniform(-1, 1))

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4)
                pred_ego[1][1] += (0.1 * random.uniform(-1, 1))

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9)
                pred_ego[2][1] += (0.1 * random.uniform(-1, 1))
            elif fai > 0:
                pred_ego[0][0] += (v3 + 0.5 * a3)
                pred_ego[0][1] += (-0.64)

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4)
                pred_ego[1][1] += (-0.64 * 2)

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9)
                pred_ego[2][1] += (-0.64 * 3)
            else:
                pred_ego[0][0] += (v3 + 0.5 * a3)
                pred_ego[0][1] += (0.64)

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4)
                pred_ego[1][1] += (0.64 * 2)

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9)
                pred_ego[2][1] += (0.64 * 3)

            pred_1[0][0] += (v1 + 0.5 * a1)
            pred_1[0][1] += (0.1 * random.uniform(-1, 1))
            pred_1[1][0] += (v1 * 2 + 0.5 * a1 * 4)
            pred_1[1][1] += (0.1 * random.uniform(-1, 1))
            pred_1[2][0] += (v1 * 3 + 0.5 * a1 * 9)
            pred_1[2][1] += (0.1 * random.uniform(-1, 1))

            pred_2[0][0] += (v2 + 0.5 * a2)
            pred_2[0][1] += (0.1 * random.uniform(-1, 1))
            pred_2[1][0] += (v2 * 2 + 0.5 * a2 * 4)
            pred_2[1][1] += (0.1 * random.uniform(-1, 1))
            pred_2[2][0] += (v2 * 3 + 0.5 * a2 * 9)
            pred_2[2][1] += (0.1 * random.uniform(-1, 1))

            v1_1s = v1 + a1
            v1_2s = v1 + 2 * a1
            v1_3s = v1 + 3 * a1

            v2_1s = v2 + a2
            v2_2s = v2 + 2 * a2
            v2_3s = v2 + 3 * a2

            vego_1s = v3 + a3
            vego_2s = v3 + 2 * a3
            vego_3s = v3 + 3 * a3


            # 当前
            P1, E1 = get_P_and_E(p1, p3, v1, v3, 0, fai, 0, fai, a1, 0)
            P2, E2 = get_P_and_E(p2, p3, v2, v3, 0, fai, 0, fai, a2, 0)

            # 预测1s
            P1_1s, E1_1s = get_P_and_E(pred_1[0], pred_ego[0], v1_1s, vego_1s, 0, fai, 0, fai, a1, 0)
            P2_1s, E2_1s = get_P_and_E(pred_2[0], pred_ego[0], v2_1s, vego_1s, 0, fai, 0, fai, a2, 0)
            
            # 预测2s
            P1_2s, E1_2s = get_P_and_E(pred_1[1], pred_ego[1], v1_2s, vego_2s, 0, fai, 0, fai, a1, 0)
            P2_2s, E2_2s = get_P_and_E(pred_2[1], pred_ego[1], v2_2s, vego_2s, 0, fai, 0, fai, a2, 0)
            
            # 预测3s
            P1_3s, E1_3s = get_P_and_E(pred_1[2], pred_ego[2], v1_3s, vego_3s, 0, fai, 0, fai, a1, 0)
            P2_3s, E2_3s = get_P_and_E(pred_2[2], pred_ego[2], v2_3s, vego_3s, 0, fai, 0, fai, a2, 0)

            # ttci=(v2-v1)/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5)

            # temp = (0.15*v2/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5))+(v1+v2)*(v2-v1)/(1.5*9.8*(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5))
            # SM=1-temp

            time_list.append(t)
            Fv1_list.append(E1)
            Fv2_list.append(E2)
            Fp1_list.append(E1_3s)
            Fp2_list.append(E2_3s)
            F_list.append(max(E1,E2))
            P_list.append(max(P1,P2))
            # ttci_list.append(ttci)
            v1_list.append(v1)
            v2_list.append(v2)
            v3_list.append(v3)
            # SM_list.append(SM)

    Fv1_list_err = Fv2_list[1:]
    Fp_list_err = Fp2_list[:-1]

    len_rmse = len(Fv1_list_err)

    RMSE = 0.0
    bias = 0.0
    for i in range(len_rmse):
        RMSE += (Fv1_list_err[i] - Fp_list_err[i]) ** 2
        bias += (abs(Fv1_list_err[i] - Fp_list_err[i]) / Fv1_list_err[i])
    RMSE /= len_rmse
    RMSE = RMSE ** 0.5
    print("RMSE=", RMSE)
    bias /= len_rmse
    print("bias=", bias)


    assert len(time_list) == len(F_list)

    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(1, 2, 2)
    # ax4 = ax3.twinx()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax4 = ax1.twinx()

    # 将构造的ax右侧的spine向右偏移
    # ax3.spines['right'].set_position(('outward', 60))
    # ax4.spines['right'].set_position(('outward', 120))

    # img1_1, = ax1.plot(time_list, v1_list, '--', c='tab:blue')
    # img1_2, = ax1.plot(time_list, v2_list, '-.', c='tab:blue')
    # img2, = ax2.plot(time_list, F_list, c='tab:green')
    # img3, = ax3.plot(time_list, ttci_list, c='tab:orange')

    ax1.plot(time_list, v1_list, '--', c='tab:blue', label='v1')
    ax1.plot(time_list, v2_list, '-.', c='tab:orange', label='v2')
    ax1.plot(time_list, v3_list, ':', c='tab:red', label='v3')
    ax2.plot(time_list, F_list, c='tab:green', label='E')
    ax2.plot(time_list, P_list, linestyle=(0, (3, 1, 1, 1, 1, 1)), c='tab:cyan', label='P')

    ax3.plot(time_list, Fv1_list, c='tab:blue', label='E1')
    ax3.plot(time_list, Fv2_list, '-.', c='tab:orange', label='E2')
    ax3.plot(time_list, Fp1_list, linestyle=(0, (5, 1 )), c='tab:pink', label='E1_p3')
    ax3.plot(time_list, Fp2_list, '--', c='tab:cyan', label='E2_p3')
    # ax3.plot(time_list, F_list, '-', c='tab:green', label='max(E1,E2)')


    # 获取对应折线图颜色给到spine ylabel yticks yticklabels
    axs = [ax1, ax2]
    # imgs = [img1_1, img1_2, img2, img3]

    ax1.set_xlabel('t(s)')

    ax1.set_ylabel('v(m/s)')
    ax2.set_ylabel('DSI')

    ax3.set_xlabel('t(s)')
    ax3.set_ylabel('DSI')
    # ax3.set_ylabel('TTCI', c='tab:orange')
    # ax4.set_ylabel('SM', c='tab:purple')

    # ax1.spines['left'].set_color(c='tab:blue')  # 注意ax1是left
    # ax2.spines['right'].set_color(c='tab:green')
    # ax3.spines['right'].set_color(c='tab:orange')
    # ax4.spines['right'].set_color(c='tab:purple')


    # ax2.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
    # ax3.tick_params(axis='y', color='tab:orange', labelcolor='tab:orange')
    # ax4.tick_params(axis='y', color='tab:purple', labelcolor='tab:purple')


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    # handle3, label3 = ax3.get_legend_handles_labels()
    # handle4, label4 = ax4.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc=(0.78,0.65))
    ax3.legend()
    # ax1.legend(handles=handle1 + handle2 + handle3 ,
    #           labels=label1 + label2 + label3 ,loc=(0.75,0.72))

    plt.tight_layout()
    plt.show()


    traci.close()
    return

# 场景D
def scene_D():
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sumocfgfile = "E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_D\\5.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, "-c", sumocfgfile])  # 打开sumocfg文件

    time_list=[]
    F_list=[]
    ttci_list=[]
    v1_list=[]
    v2_list=[]
    # SM_list=[]

    # v3_list=[]
    Fv1_list=[]
    Fp_list=[]
    # Fv2_list=[]
    # Fl_list=[]
    dx_list=[]
    dy_list=[]
    d_list=[]

    vis1_list=[]
    vis2_list=[]


    time_t=0

    for step in range(0, 13):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        # print("simulation_time=", simulation_time)
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id

        if len(all_vehicle_id)==2:
            traci.gui.trackVehicle('View #0', all_vehicle_id[1])

            traci.vehicle.setLaneChangeMode("1", 0b000000000000)
            traci.vehicle.setLaneChangeMode("2", 0b000000000000)
            traci.vehicle.setSpeedMode("1", 00000)
            traci.vehicle.setSpeedMode("2", 00000)

        if len(all_vehicle_id) == 2:

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])
            # p3 = traci.vehicle.getPosition(all_vehicle_id[2])
            d=((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
            d_x = abs(p1[0] - p2[0])
            d_y = abs(p1[1] - p2[1])

            fai = 0.5*3.141592653
            fai_2 = 0

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            # v3 = traci.vehicle.getSpeed(all_vehicle_id[2])
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])



            pred_ego = []
            pred = []
            for i in range(3):
                pred_ego.append([p1[0], p1[1]])
                pred.append([p2[0], p2[1]])


            pred[0][0] += (v2+random.uniform(-1, 1))

            pred_ego[0][1] += (v1+random.uniform(-1, 1))


            v1_1s = v1 + a1
            v2_1s = v2 + a2

            v1_2s = v1 + 2*a1
            v2_2s = v2 + 2*a2

            v1_3s = v1 + 2*a1
            v2_3s = v2 + 2*a2


            # 当前
            P, E = get_P_and_E(p1, p2, v1, v2, np.pi/2, fai, np.pi/2, fai, a1, 0)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred[0], pred_ego[0], v1_1s, v2_1s, np.pi/2, fai, np.pi/2, fai, a1, 0)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred[1], pred_ego[1], v1_2s, v2_2s, np.pi/2, fai, np.pi/2, fai, a1, 0)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred[2], pred_ego[2], v1_3s, v2_3s, np.pi/2, fai, np.pi/2, fai, a1, 0)

            time_list.append(t)
            Fv1_list.append(P**cs1)
            v1_list.append(v1)
            v2_list.append(v2)
            Fp_list.append(E_3s)
            F_list.append(E)
            # ttci_list.append(ttci)
            d_list.append(d)
            dx_list.append(d_x)
            dy_list.append(d_y)

    # assert len(time_list) == len(F_list)
    file1_path = 'E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_D\\1.txt'
    with open(file1_path,'w') as f:
        s1=''
        for i in vis1_list:
            s1+=(str(i[0])+' '+str(i[1])+'\n')
        f.write(s1)
    f.close()

    file2_path = 'E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_D\\2.txt'
    with open(file2_path,'w') as f:
        s2=''
        for i in vis2_list:
            s2+=(str(i[0])+' '+str(i[1])+'\n')
        f.write(s2)
    f.close()

    Fv1_list_err = Fv1_list[1:]
    Fp_list_err = Fp_list[:-1]

    len_rmse=len(Fv1_list_err)

    RMSE = 0.0
    bias =0.0
    for i in range(len_rmse):
        RMSE += (Fv1_list_err[i]-Fp_list_err[i])**2
        bias += (abs(Fv1_list_err[i] - Fp_list_err[i]) / Fv1_list_err[i])
    RMSE /= len_rmse
    RMSE = RMSE**0.5
    print("RMSE=",RMSE)
    bias /= len_rmse
    print("bias=",bias)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(1, 2, 2)
    ax4 = ax3.twinx()

    # ax4 = ax3.twinx()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax4 = ax1.twinx()

    # 将构造的ax右侧的spine向右偏移
    # ax3.spines['right'].set_position(('outward', 60))
    # ax4.spines['right'].set_position(('outward', 120))

    # img1_1, = ax1.plot(time_list, v1_list, '--', c='tab:blue')
    # img1_2, = ax1.plot(time_list, v2_list, '-.', c='tab:blue')
    # img2, = ax2.plot(time_list, F_list, c='tab:green')
    # img3, = ax3.plot(time_list, ttci_list, c='tab:orange')

    # ax1.plot(time_list, v1_list, '--', c='k', label='v$_{1}$')
    # ax1.plot(time_list, v2_list, '-.', c='k', label='v$_{2}$')
    # # ax1.plot(time_list, d_list, '-.', c='tab:grey', label='distance')
    # # ax1.plot(time_list, dx_list, '-.', c='tab:cyan', label='dx')
    # # ax1.plot(time_list, dy_list, '-.', c='tab:purple', label='dy')
    # ax2.plot(time_list, F_list, c='k', label='DSI')
    #
    # ax3.plot(time_list, dx_list, '-.', c='k', label='d_x')
    # ax3.plot(time_list, dy_list, ':', c='k', label='d_y')
    #
    # ax4.plot(time_list, F_list, c='k', linestyle=(0,(3,1,1,1,1,1)),label='DSI')
    # ax4.plot(time_list, Fv1_list, c='k', label='DSI_c')
    # ax4.plot(time_list, Fp_list, '--', c='k', label='DSI_p')

    ax1.plot(time_list, v1_list, '--', c='tab:red', label='v$_{1}$')
    ax1.plot(time_list, v2_list, '-.', c='tab:blue', label='v$_{2}$')
    ax2.plot(time_list, F_list, c='tab:orange', label='E')
    ax3.plot(time_list, dx_list, '-.', c='tab:grey', label='d_x')
    ax3.plot(time_list, dy_list, ':', c='tab:green', label='d_y')
    ax4.plot(time_list, F_list, c='tab:orange', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='E')
    ax4.plot(time_list, Fv1_list, c='tab:cyan', label='P')
    ax4.plot(time_list, Fp_list, '--', c='tab:purple', label='E_p3')

    # 获取对应折线图颜色给到spine ylabel yticks yticklabels
    axs = [ax1, ax2, ax3, ax4]
    # imgs = [img1_1, img1_2, img2, img3]

    ax1.set_xlabel('t(s)')
    ax3.set_xlabel('t(s)')

    ax1.set_ylabel('v(m/s)')
    ax2.set_ylabel('E')

    # ax3.set_xlabel('t(s)')
    ax3.set_ylabel('distance(m)')
    # ax3.set_ylabel('TTCI', c='tab:orange')
    ax4.set_ylabel('E')

    # ax1.spines['left'].set_color(c='tab:blue')  # 注意ax1是left
    # ax2.spines['right'].set_color(c='tab:green')
    # ax3.spines['right'].set_color(c='tab:orange')
    # ax4.spines['right'].set_color(c='tab:purple')


    # ax2.tick_params(axis='y', color='tab:green', labelcolor='tab:green')
    # ax3.tick_params(axis='y', color='tab:orange', labelcolor='tab:orange')
    # ax4.tick_params(axis='y', color='tab:purple', labelcolor='tab:purple')


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc=(0.78,0.28))
    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc=(0.75,0.35),framealpha=1)
    # ax3.legend()
    # ax1.legend(handles=handle1 + handle2 + handle3 ,
    #           labels=label1 + label2 + label3 ,loc=(0.75,0.72))

    plt.tight_layout()
    plt.show()


    traci.close()
    return


if __name__ == "__main__":
    scene_D()

