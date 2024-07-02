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

# alpha = 0.005   # 他车速度, 0.003, 0.008
# beta = 0.006    # 速度差, 0.016, 0.016
# gamma = 0.025   # 他车加速度, 0.025, 0.025

alpha = 0.003   # 他车速度, 0.003, 0.008
beta = 0.016    # 速度差, 0.016, 0.016
gamma = 0.025   # 他车加速度, 0.025, 0.025
#############################################

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
    P = (get_oo_length(theta, fai_v2)/((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5)**cs1
    Vx = v1*np.cos(angle1) - v2*np.cos(angle2)
    Vy = v1*np.sin(angle1) - v2*np.sin(angle2)
    V_delta = ((Vx**2)+(Vy**2))**0.5
    theta_V_delta = np.arctan2(Vy, Vx)
    E = P * np.exp(alpha*v1 + 
                   beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                   gamma*abs(a1)*abs(np.cos((theta-a1_theta)/2)))
    return P, E


def get_P_and_E_and_ttc(p1, p2, v1, v2, angle1, angle2, fai_v1, fai_v2, a1, a1_theta):
    theta = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) - fai_v1
    fai_v2 -= fai_v1
    P = (get_oo_length(theta, fai_v2)/((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5)**cs1
    Vx = v1*np.cos(angle1) - v2*np.cos(angle2)
    Vy = v1*np.sin(angle1) - v2*np.sin(angle2)
    V_delta = ((Vx**2)+(Vy**2))**0.5
    theta_V_delta = np.arctan2(Vy, Vx)
    E = P * np.exp(alpha*v1 + 
                   beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                   gamma*abs(a1)*abs(np.cos((theta-a1_theta)/2)))
    ttci=V_delta*np.cos(theta_V_delta)/(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5)
    # ttci=V_delta/abs(p2[1]-p1[1])
    return P, E, ttci


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
    f_list = []
    E_1s_list=[]
    E_2s_list=[]
    E_3s_list=[]

    x1_list = []
    x2_list = []

    x1_3s_list = []
    x2_3s_list = []
    

    queue_p1 = queue.Queue(2)
    queue_p2 = queue.Queue(2)
    
    for step in range(0, 700):  # 仿真时间
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

            if p1[0] > 600:
                traci.vehicle.setSpeed('1', 10)     # 行驶600米后，1车减速
            
            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            v1_list.append(v1)
            v2_list.append(v2)
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0]) / 1
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1]) / 1
            if len(v1_list) < 3:
                a1 = traci.vehicle.getAcceleration(all_vehicle_id[0]) / 1
                a2 = traci.vehicle.getAcceleration(all_vehicle_id[1]) / 1
            else:
                a1 = (v1_list[-1] - v1_list[-3]) / 1
                a2 = (v2_list[-1] - v2_list[-3]) / 1
            a1_list.append(a1)
            a2_list.append(a2)

            # 预测属性
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

            if a1>=0:
                a1_theta = 0
            else:
                a1_theta = np.pi

            v1_1s = v1 + a1
            v1_2s = v1 + 2 * a1
            v1_3s = v1 + 3 * a1

            v2_1s = v2 + a2
            v2_2s = v2 + 2 * a2
            v2_3s = v2 + 3 * a2

            if step == 250:
                pass
            if step == 400:
                pass

            # 当前
            P, E = get_P_and_E(p1, p2, v1, v2, fai_v1, fai_v2, fai_v1, fai_v2, a1, a1_theta)


            f = E/P
            f_list.append(f)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred_1[0], pred_2[0], v1_1s, v2_1s, fai_v1, fai_v2, fai_v1, fai_v2, a1, a1_theta)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred_1[1], pred_2[1], v1_2s, v2_2s, fai_v1, fai_v2, fai_v1, fai_v2, a1, a1_theta)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred_1[2], pred_2[2], v1_3s, v2_3s, fai_v1, fai_v2, fai_v1, fai_v2, a1, a1_theta)

            # ttci=(v2-v1)/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5)

            ttt = 1
            if abs(pred_1[ttt][1] - pred_2[ttt][1]) < 2:
                if pred_1[ttt][0] > pred_2[ttt][0]:
                    ttci = (v2_2s - v1_2s) / abs(pred_1[ttt][0] - pred_2[ttt][0])
                else:
                    ttci = (v1_2s - v2_2s) / abs(pred_1[ttt][0] - pred_2[ttt][0])
            else:
                ttci = 0

            P_list.append(P)
            ttci_list.append(ttci)
            E_list.append(E)
            E_1s_list.append(E_1s)
            E_2s_list.append(E_2s)
            E_3s_list.append(E_3s)

    # np.save('./note/Risk_Assessment/sumo/scene_A/A_v2.npy',v2_list)
    assert len(time_list) == len(P_list)

    nnn = abs(np.array(E_2s_list[:-20]) - np.array(E_list[20:]))
    loss = sum(nnn/np.array(E_list[20:]))/len(nnn)
    print("loss", loss)

    fig = plt.figure(figsize=(9,7))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(2, 2, 3)
    ax6 = ax5.twinx()
    ax7 = fig.add_subplot(2, 2, 4)
    ax8 = ax7.twinx()
    # ax9 = fig.add_subplot(3, 2, 5)
    # ax10 = ax9.twinx()
    # ax11 = fig.add_subplot(3, 2, 6)
    # ax12 = ax11.twinx()

    # ax1.plot(time_list, v1_list, '-', c='tab:red', label='v$_{1}$')
    # ax1.plot(time_list, v2_list, '--', c='tab:blue', label='v$_{2}$')
    # ax2.plot(time_list, np.array(x1_list)-np.array(x2_list), '-.', c='tab:green', label='Δx')
    ax1.plot(time_list, v1_list, '-', c='black', label='v$_{1}$')
    ax1.plot(time_list, v2_list, '--', c='black', label='v$_{2}$')
    ax2.plot(time_list, np.array(x1_list)-np.array(x2_list), '-.', c='black', label='Δx')


    # ax3.plot(time_list, P_list,  '-', c='tab:blue', label='P$_{0}$')
    # ax3.plot(time_list, E_list, '--', c='tab:green', label='CRI$_{0}$')
    # ax3.plot(time_list, E_2s_list, '-.', c='tab:red', label='CRI$_{2}$', alpha=0.6)    # 
    # ax4.plot(time_list, np.array(x1_list)-np.array(x2_list), ':', c='tab:purple', label='Δx')   # linestyle=(0, (3, 1, 1, 1, 1, 1))
    ax3.plot(time_list, P_list,  '-', c='black', label='P$_{0}$')
    ax3.plot(time_list, E_list, '--', c='black', label='CRI$_{0}$')
    ax3.plot(time_list, E_2s_list, '-.', c='black', label='CRI$_{2}$', alpha=0.6)    # 
    ax4.plot(time_list, np.array(x1_list)-np.array(x2_list), ':', c='black', label='Δx')   # linestyle=(0, (3, 1, 1, 1, 1, 1))

    
    # ax5.plot(time_list, E_2s_list, '-', c='tab:blue', label='CRI$_{2}$')
    # ax6.plot(time_list, ttci_list, '--', c='tab:red', label='TTC$^{-1}$')
    ax5.plot(time_list, E_2s_list, '-', c='black', label='CRI$_{2}$')
    ax6.plot(time_list, ttci_list, '--', c='black', label='TTC$^{-1}$')

    # ax5.plot(time_list, np.array(E_list)/np.array(P_list)/(np.array(x1_list)-np.array(x2_list)), '--', c='tab:red', label='E/P/P', alpha=0.8)    # (速度+速度差+加速度)除等效距离
    # ax5.plot(time_list, np.array(E_list)/np.array(P_list)/(1/np.array(P_list)), '--', c='tab:red', label='E/P/P', alpha=0.8)    # (速度+速度差+加速度)除等效距离
    # ax5.plot(time_list, E_list, '--', c='tab:red', label='E')
    
    # ax7.plot(time_list, np.array(v2_list)-np.array(v1_list), '-', c='tab:blue', label='Δv')     # 动态属性, Δv, (-∞~+∞)
    # ax8.plot(time_list, f_list, '--', c='tab:red', label='E/P', alpha=0.8)                       # 动态属性, f, (0~∞)

    # ax7.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:blue', label='Δx')
    # ax8.plot(time_list, ttci_list, '-', c='tab:green', label='TTC$^{-1}$')      # (v2-v1)/Δx

    # ax9.plot(time_list, ttci_list, '-', c='tab:blue', label='TTC$^{-1}$') # (v2-v1)/Δx, 动态/静态
    # ax10.plot(time_list, E_list, '-', c='tab:red', label='E')  # f/
    
    fontsize = 12
    labelpad_y = 0
    ax1.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax1.set_ylabel('v(m/s)', fontsize=fontsize, labelpad=labelpad_y)
    ax2.set_ylabel('Δx(m)', fontsize=fontsize, labelpad=labelpad_y)

    ax3.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax3.set_ylabel('P$_{t}$ and CRI$_{t}$', fontsize=fontsize, labelpad=labelpad_y)
    ax4.set_ylabel('Δx(m)', fontsize=fontsize, labelpad=labelpad_y)

    ax5.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax5.set_ylabel('CRI$_{2}$', fontsize=fontsize, labelpad=labelpad_y)
    ax6.set_ylabel('TTC$^{-1}$', fontsize=fontsize, labelpad=labelpad_y)

    # ax7.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    # ax7.set_ylabel('P', fontsize=fontsize, labelpad=labelpad_y)
    # ax8.set_ylabel('TTC$^{-1}$', fontsize=fontsize, labelpad=labelpad_y)


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()
    handle5, label5 = ax5.get_legend_handles_labels()
    handle6, label6 = ax6.get_legend_handles_labels()
    handle7, label7 = ax7.get_legend_handles_labels()
    handle8, label8 = ax8.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc='best')

    ax4.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc='center right')
    
    ax5.legend(handles=handle5 + handle6,
              labels=label5 + label6, loc='best')

    ax8.legend(handles=handle7 + handle8,
               labels=label7 + label8, loc='best')

    plt.tight_layout()
    # plt.grid()  # 网格
    plt.show()
    traci.close()
    return


def scene_A_1(fai_v1 = 0, fai_v2 = 0):
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
    f_list = []

    x1_list = []
    x2_list = []

    theta_V_delta_list = []
    V_delta_list = []
    theta_list = []

    queue_p1 = queue.Queue(2)
    queue_p2 = queue.Queue(2)
    
    for step in range(0, 700):  # 仿真时间
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

            # queue_p1.put(p1)
            # queue_p2.put(p2)

            # if queue_p1.full():
            #     old_p1 = queue_p1.get()
            #     fai_v1 = np.arctan2(p1[1]-old_p1[1], p1[0]-old_p1[0])

            # if queue_p2.full():
            #     old_p2 = queue_p2.get()
            #     fai_v2 = np.arctan2(p2[1] - old_p2[1], p2[0] - old_p2[0])

            if p1[0] > 600:
                traci.vehicle.setSpeed('1', 10)     # 行驶600米后，1车减速
            
            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            v1_list.append(v1)
            v2_list.append(v2)
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])
            a1_list.append(a1)
            a2_list.append(a2)


            if a1>=0:
                a1_theta = 0
            else:
                a1_theta = np.pi

            if step == 250:
                pass
            if step == 400:
                pass
            
            # 当前
            theta = np.pi
            P = (get_oo_length(theta, fai_v2)/((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)**0.5)**cs1
            Vx = v1*np.cos(fai_v1) - v2*np.cos(fai_v2)
            Vy = v1*np.sin(fai_v1) - v2*np.sin(fai_v2)
            V_delta = ((Vx**2)+(Vy**2))**0.5
            theta_V_delta = np.arctan2(Vy, Vx)

            V_delta_list.append(V_delta)
            theta_V_delta_list.append(theta_V_delta)
            theta_list.append(theta)

            E = P * np.exp(alpha*v1 + 
                        beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                        gamma*abs(a1)*abs(np.cos((theta-a1_theta)/2)))

            
            f = E/P
            f_list.append(f)

            ttt = 1
            if abs(pred[ttt][1] - pred_ego[ttt][1]) < 2:
                if pred[ttt][0] > pred_ego[ttt][0]:
                    ttci = (v_ego_2s - v1_2s) / abs(pred[ttt][0] - pred_ego[ttt][0])
                else:
                    ttci = (v1_2s - v_ego_2s) / abs(pred[ttt][0] - pred_ego[ttt][0])
            else:
                ttci = 0

            # ttci = (v2-v1)/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5)
            # ttci = -V_delta*np.cos(theta_V_delta)/(((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**0.5)

            P_list.append(P)
            ttci_list.append(ttci)
            E_list.append(E)


    assert len(time_list) == len(P_list)

    fig = plt.figure(figsize=(15,8))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(2, 3, 2)
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(2, 3, 3)
    ax6 = ax5.twinx()
    ax7 = fig.add_subplot(2, 3, 4)
    ax8 = ax7.twinx()
    ax9 = fig.add_subplot(2, 3, 5)
    ax10 = ax9.twinx()
    ax11 = fig.add_subplot(2, 3, 6)
    ax12 = ax11.twinx()

    ax1.plot(time_list, v1_list, '-', c='tab:red', label='v$_{1}$')
    ax1.plot(time_list, v2_list, '-', c='tab:blue', label='v$_{2}$')
    ax2.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:green', label='Δx')

    ax3.plot(time_list, P_list,  '-', c='tab:blue', label='P')
    ax3.plot(time_list, E_list, '--', c='tab:red', label='E')
    ax4.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:green', label='Δx')

    
    ax5.plot(time_list, np.array(x1_list)-np.array(x2_list), '--', c='tab:blue', label='Δx')    # 静态属性, Δx, (20 ~ ∞)
    ax6.plot(time_list, 1/np.array(P_list), '--', c='tab:red', label='1/P', alpha=0.5)         # 静态属性, 1/P, (1 ~ ∞)
    
    ax7.plot(time_list, np.array(v2_list)-np.array(v1_list), '-', c='tab:blue', label='Δv')     # 动态属性, Δv, (-∞~+∞)
    ax8.plot(time_list, f_list, '--', c='tab:red', label='E/P', alpha=0.8)                       # 动态属性, f, (0~∞)

    ax9.plot(time_list, np.array(v2_list)-np.array(v1_list), '-', c='tab:blue', label='Δv', alpha=0.8)     # 动态属性, Δv, (-∞~+∞)
    ax10.plot(time_list, beta*np.array(V_delta_list)*abs(np.cos((np.array(theta_V_delta_list)-np.array(theta_list))/2)), '-', c='tab:red', label='f')  # f
    ax10.plot(time_list, np.exp(beta*np.array(V_delta_list)*abs(np.cos((np.array(theta_V_delta_list)-np.array(theta_list))/2))), '-', c='tab:grey', label='e^f')  # e^f

    ax11.plot(time_list, ttci_list, '-', c='tab:blue', label='TTC$^{-1}$') # Δv/Δx, TTC-1
    # ax12.plot(time_list, beta*np.array(V_delta_list)*abs(np.cos((np.array(theta_V_delta_list)-np.array(theta_list))/2))/(np.array(x1_list)-np.array(x2_list)), '-', c='tab:red', label='f/Δx')  # f/Δx
    ax12.plot(time_list, 5*np.log(np.array(f_list))/(np.array(x1_list)-np.array(x2_list)), '--', c='tab:red', label='5f/Δx')  # f/Δx
    # ax12.plot(time_list, np.exp(beta*np.array(V_delta_list)*abs(np.cos((np.array(theta_V_delta_list)-np.array(theta_list))/2)))/(np.array(x1_list)-np.array(x2_list)), '-', c='tab:grey', label='e$^{f}$/Δx')  # e^f/Δx
    ax12.plot(time_list, np.array(f_list)/(np.array(x1_list)-np.array(x2_list)), '-.', c='tab:grey', label='e$^{f}$/Δx')  # e^f/Δx
    ax12.plot(time_list, np.array(E_list)/10, ':', c='tab:pink', label='E/10')   # e^f/(1/P)   linestyle=(0, (3, 1, 1, 1, 1, 1))

    
    
    ax1.set_xlabel('t(s)')
    ax1.set_ylabel('v(m/s)')
    ax2.set_ylabel('Distance(m)')

    ax3.set_xlabel('t(s)')
    ax3.set_ylabel('P & E')
    ax4.set_ylabel('Distance(m)')

    ax5.set_xlabel('t(s)')
    ax5.set_ylabel('E/P')
    ax6.set_ylabel('TTC$^{-1}$')

    ax7.set_xlabel('t(s)')
    ax7.set_ylabel('P')
    ax8.set_ylabel('TTC$^{-1}$')

    ax11.set_xlabel('t(s)')
    ax11.set_ylabel('TTC$^{-1}$')
    ax12.set_ylabel('E')


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()
    handle5, label5 = ax5.get_legend_handles_labels()
    handle6, label6 = ax6.get_legend_handles_labels()
    handle7, label7 = ax7.get_legend_handles_labels()
    handle8, label8 = ax8.get_legend_handles_labels()
    handle9, label9 = ax9.get_legend_handles_labels()
    handle10, label10 = ax10.get_legend_handles_labels()
    handle11, label11 = ax11.get_legend_handles_labels()
    handle12, label12 = ax12.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc='best')

    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc='center right')
    
    ax5.legend(handles=handle5 + handle6,
              labels=label5 + label6, loc='best')

    ax7.legend(handles=handle7 + handle8,
               labels=label7 + label8, loc='best')
    
    ax9.legend(handles=handle9 + handle10,
               labels=label9 + label10, loc='best')
    
    ax11.legend(handles=handle11 + handle12,
               labels=label11 + label12, loc='center right')
    

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

    for step in range(0, 280):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id
        traci.gui.trackVehicle('View #0', all_vehicle_id[0])    # 在给定的视图上对给定的车辆进行视觉跟踪


        # if '1'in all_vehicle_id:
            # traci.vehicle.setLaneChangeMode("1", 0b001000000000)

        if len(all_vehicle_id) == 2:

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])
            
            d_x = abs(p1[0] - p2[0])
            d_y = abs(p1[1] - p2[1])
            d=(d_x**2+d_y**2)**0.5

            
            queue_ego.put(p2)

            if queue_ego.full():
                old_p2 = queue_ego.get()
                fai = np.arctan2((p2[1]-old_p2[1]), (p2[0]-old_p2[0]))
                # print('fai=', fai)

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])
            if a1>0:
                a1_theta = 0
            else:
                a1_theta = np.pi

            # pp = p1[0] - p2[0]
            # if pp>10 and pp <25:
            #     traci.vehicle.changeLane('1', 0, 5)

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
            P, E = get_P_and_E(p1, p2, v1, v2, 0, fai, 0, fai, a1, a1_theta)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred[0], pred_ego[0], v1_1s, v2_1s, 0, fai, 0, fai, a1, a1_theta)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred[1], pred_ego[1], v1_2s, v2_2s, 0, fai, 0, fai, a1, a1_theta)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred[2], pred_ego[2], v1_3s, v2_3s, 0, fai, 0, fai, a1, a1_theta)


            ttci = (v2 - v1) / (((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5)

            time_list.append(t)
            Fv1_list.append(P)
            v1_list.append(v1)
            v2_list.append(v2)
            Fp_list.append(E_2s)
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

    ax1.plot(time_list, v1_list, '-', c='tab:blue', label='v$_{1}$')
    ax1.plot(time_list, v2_list, '--', c='tab:red', label='v$_{2}$')
    ax2.plot(time_list, d_list, '-.', c='tab:grey', label='D')
    # ax2.plot(time_list, dx_list, '-.', c='tab:cyan', label='dx')
    # ax2.plot(time_list, dy_list, '-.', c='tab:purple', label='dy')
    # ax2.plot(time_list, F_list, c='tab:orange', label='E')

    
    ax3.plot(time_list, F_list, '-', c='tab:orange', label='E')
    ax3.plot(time_list, Fv1_list, '--', c='tab:cyan', label='P')
    ax3.plot(time_list, Fp_list, '-.', c='tab:purple', label='E$^{\'}$')
    ax4.plot(time_list, dx_list, ':', c='tab:grey', label='Δx')
    ax4.plot(time_list, dy_list, linestyle=(0, (3, 1, 1, 1, 1, 1)), c='tab:green', label='Δy')
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
    ax2.set_ylabel('Distance(m)')

    # ax3.set_xlabel('t(s)')
    ax3.set_ylabel('E & P')
    ax4.set_ylabel('Distance(m)')
    # ax3.set_ylabel('TTCI', c='tab:orange')
    

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
    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc='center right')    # (0.75,0.65)
    # ax3.legend()
    # ax1.legend(handles=handle1 + handle2 + handle3 ,
    #           labels=label1 + label2 + label3 ,loc=(0.75,0.72))

    plt.tight_layout()
    plt.show()


    traci.close()
    return


# 场景B, 区分了一下车
def scene_B_1(fai_ego = 0, fai_1 = 0):
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
    v_ego_list=[]
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

    for step in range(0, 280):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id
        # traci.gui.trackVehicle('View #0', all_vehicle_id[0])    # 在给定的视图上对给定的车辆进行视觉跟踪


        # if '1'in all_vehicle_id:
            # traci.vehicle.setLaneChangeMode("1", 0b001000000000)

        if len(all_vehicle_id) == 2:
            traci.gui.trackVehicle('View #0', all_vehicle_id[1])

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])   # 1
            p_ego = traci.vehicle.getPosition(all_vehicle_id[1])   # ego
            
            d_x = abs(p_ego[0] - p1[0])
            d_y = abs(p_ego[1] - p1[1])
            d=(d_x**2+d_y**2)**0.5

            queue_p1.put(p1)        # 他车
            queue_ego.put(p_ego)    # 自车

            if queue_p1.full():
                old_p1 = queue_p1.get()
                fai_1 = np.arctan2((p1[1]-old_p1[1]), (p1[0]-old_p1[0]))
                # print('fai=', fai)
            if queue_ego.full():
                old_p_ego = queue_ego.get()
                fai_ego = np.arctan2((p_ego[1]-old_p_ego[1]), (p_ego[0]-old_p_ego[0]))
                # print('fai_ego=', fai_ego)

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])  # 1
            v_ego = traci.vehicle.getSpeed(all_vehicle_id[1])  # ego
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])/1   # 1
            a_ego = traci.vehicle.getAcceleration(all_vehicle_id[1])/1   # ego
            
            if a1>0:
                a1_theta = 0
            else:
                a1_theta = np.pi

            # pp = p1[0] - p2[0]
            # if pp>10 and pp <25:
            #     traci.vehicle.changeLane('1', 0, 5)

            pred_ego = []
            pred = []
            for i in range(3):
                pred_ego.append([p_ego[0], p_ego[1]])
                pred.append([p1[0], p1[1]])

            kkk = 0.8
            if fai_ego == 0:
                pred_ego[0][0] += (v_ego+0.5*a_ego)
                pred_ego[0][1] += (0.1*random.uniform(-1, 1))

                pred_ego[1][0] += (2*v_ego+0.5*a_ego*4)
                pred_ego[1][1] += (0.1*random.uniform(-1, 1))

                pred_ego[2][0] += (3 * v_ego + 0.5 * a_ego * 9)
                pred_ego[2][1] += (0.1*random.uniform(-1, 1))
            else:
                pred_ego[0][0] += (v_ego + 0.5 * a_ego) * kkk
                pred_ego[0][1] += (-0.64)

                pred_ego[1][0] += (2 * v_ego + 0.5 * a_ego * 4) * kkk
                pred_ego[1][1] += (-0.64*2)

                pred_ego[2][0] += (3 * v_ego + 0.5 * a_ego * 9) * kkk
                pred_ego[2][1] += (-0.64*3)
                pass

            pred[0][0] += (v1+0.5*a1)
            pred[0][1] += (0.1*random.uniform(-1, 1))
            pred[1][0] += (v1*2 + 0.5 * a1*4)
            pred[1][1] += (0.1 * random.uniform(-1, 1))
            pred[2][0] += (v1*3 + 0.5 * a1*9)
            pred[2][1] += (0.1 * random.uniform(-1, 1))

            v_ego_1s = v_ego + a_ego
            v_ego_2s = v_ego + 2 * a_ego
            v_ego_3s = v_ego + 3 * a_ego

            v1_1s = v1 + a1
            v1_2s = v1 + 2 * a1
            v1_3s = v1 + 3 * a1

            # 当前
            P, E = get_P_and_E(p1, p_ego, v1, v_ego, fai_1, 0, fai_1, 0, a1, a1_theta)    # 一车是自车，二车是他车

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred[0], pred_ego[0], v1_1s, v_ego_1s, fai_1, 0, fai_1, 0, a1, a1_theta)
            
            # 预测2s
            P_2s, E_2s = get_P_and_E(pred[1], pred_ego[1], v1_2s, v_ego_2s, fai_1, 0, fai_1, 0, a1, a1_theta)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred[2], pred_ego[2], v1_3s, v_ego_3s, fai_1, 0, fai_1, 0, a1, a1_theta)

            # if abs(p1[1] - p_ego[1]) < 2:
            #     ttci = (v_ego - v1) / abs(p1[0] - p_ego[0])
            # else:
            #     ttci = 0

            # if abs(p1[1] - p3[1]) < 2:
            #     if p1[0] > p3[0]:
            #         ttci_1 = (v3 - v1) / abs(p1[0] - p3[0])
            #     else:
            #         ttci_1 = (v1 - v3) / abs(p1[0] - p3[0])
            # else:
            #     ttci_1 = 0

            ttt = 1
            if abs(pred[ttt][1] - pred_ego[ttt][1]) < 2:
                if pred[ttt][0] > pred_ego[ttt][0]:
                    ttci = (v_ego_2s - v1_2s) / abs(pred[ttt][0] - pred_ego[ttt][0])
                else:
                    ttci = (v1_2s - v_ego_2s) / abs(pred[ttt][0] - pred_ego[ttt][0])
            else:
                ttci = 0
            # ttci = (v_ego - v1) / (((p_ego[0] - p1[0]) ** 2 + (p_ego[1] - p1[1]) ** 2) ** 0.5)
            if E_2s >= 1:
                print(E_2s)
                print(t) 
            time_list.append(t)
            Fv1_list.append(P_2s)
            v1_list.append(v1)
            v_ego_list.append(v_ego)
            Fp_list.append(E_2s)
            F_list.append(E)
            ttci_list.append(ttci)
            d_list.append(d)
            dx_list.append(d_x)
            dy_list.append(d_y)

    # assert len(time_list) == len(F_list)
    nnn = abs(np.array(Fp_list[:-20]) - np.array(F_list[20:]))
    loss = sum(nnn/np.array(F_list[20:]))/len(nnn)
    print("loss", loss)
    fig = plt.figure(figsize=(9, 7))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(2, 2, 3)
    ax6 = ax5.twinx()

    # ax4 = ax3.twinx()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax3 = ax1.twinx()
    # ax4 = ax1.twinx()

    # ax1.plot(time_list, v_ego_list, '-', c='tab:red', label='v$_{1}$')
    # ax1.plot(time_list, v1_list, '--', c='tab:blue', label='v$_{2}$')
    # ax2.plot(time_list, d_list, '-.', c='tab:grey', label='D')
    ax1.plot(time_list, v_ego_list, '-', c='black', label='v$_{1}$')
    ax1.plot(time_list, v1_list, '--', c='black', label='v$_{2}$')
    ax2.plot(time_list, d_list, '-.', c='black', label='D')

    
    # ax3.plot(time_list, F_list, '-', c='tab:orange', label='CRI$_{0}$')
    # ax3.plot(time_list, Fv1_list, '--', c='tab:cyan', label='P$_{2}$')
    # ax3.plot(time_list, Fp_list, '-.', c='tab:purple', label='CRI$_{2}$')
    # ax4.plot(time_list, dx_list, ':', c='tab:grey', label='Δx')
    # ax4.plot(time_list, dy_list, linestyle=(0, (5, 1 )), c='tab:green', label='Δy')
    ax3.plot(time_list, F_list, '-', c='black', label='CRI$_{0}$')
    # ax3.plot(time_list, Fv1_list, '--', c='black', label='P$_{2}$')
    ax3.plot(time_list, Fp_list, '-.', c='black', label='CRI$_{2}$')
    ax4.plot(time_list, dx_list, ':', c='black', label='Δx')
    ax4.plot(time_list, dy_list, linestyle=(0, (5, 1 )), c='black', label='Δy')

    ax5.plot(time_list, Fp_list, '-', c='black', label='CRI$_{2}$')
    ax6.plot(time_list, ttci_list, '--', c='black', label='TTC$^{-1}$')

    # 获取对应折线图颜色给到spine ylabel yticks yticklabels
    axs = [ax1, ax2, ax3, ax4]
    # imgs = [img1_1, img1_2, img2, img3]

    fontsize=12
    ax1.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax3.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax5.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)

    ax1.set_ylabel('v(m/s)', fontsize=fontsize, labelpad=0)
    ax2.set_ylabel('Distance(m)', fontsize=fontsize, labelpad=0)

    # ax3.set_xlabel('t(s)')
    ax3.set_ylabel('P$_{t}$ and CRI$_{t}$', fontsize=fontsize, labelpad=0)
    ax4.set_ylabel('Distance(m)', fontsize=fontsize, labelpad=0)
    # ax3.set_ylabel('TTCI', c='tab:orange')
    ax5.set_ylabel('CRI$_{2}$', fontsize=fontsize, labelpad=0)
    ax6.set_ylabel('TTC$^{-1}$', fontsize=fontsize, labelpad=5)

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
    handle5, label5 = ax5.get_legend_handles_labels()
    handle6, label6 = ax6.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc='best')
    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc=(0.77,0.25))    # (0.75,0.65)
    ax5.legend(handles=handle5 + handle6,
               labels=label5 + label6, loc=(0.03,0.65))    # (0.75,0.65)
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

    sumocfgfile = "E:\\Code\\github\\Daily\\note\\Risk_Assessment\\sumo\\scene_C\\3.sumocfg"  # sumocfg文件的位置
    traci.start([sumoBinary, "-c", sumocfgfile])  # 打开sumocfg文件

    time_list=[]
    F_list=[]
    P1_list = []
    P2_list = []
    ttci_list_1 = []
    ttci_list_2 = []
    ttci_list = []
    v1_list=[]
    v2_list=[]
    SM_list=[]

    v3_list=[]
    Fv1_list=[]
    Fp1_list=[]
    Fp2_list =[]
    Fv2_list=[]
    Fl_list=[]

    a1_list = []
    a2_list = []
    a3_list = []

    y3_list = []
    delta_x1_list = []
    delta_x2_list = []

    queue_ego = queue.Queue(2)

    for step in range(0, 800):  # 仿真时间, 800
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id

        if '1' in all_vehicle_id:
            traci.vehicle.setLaneChangeMode("1", 0b001000000000)    # 换车时尊重他人的速度/制动间隙车道，调整速度以满足要求

        if len(all_vehicle_id) == 3:
            # traci.gui.trackVehicle('View #0', all_vehicle_id[2])
            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])
            p3 = traci.vehicle.getPosition(all_vehicle_id[2])
            delta_x1_list.append(abs(p1[0]-p3[0]))
            delta_x2_list.append(abs(p2[0]-p3[0]))
            y3_list.append(p3[1])

            queue_ego.put(p3)

            if queue_ego.full():
                old_ego = queue_ego.get()
                fai = np.arctan2((p3[1]-old_ego[1]), (p3[0]-old_ego[0]))    # fai是基于仿真软件返回的车辆状态实时计算而来的
                # print('fai=', fai)

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            v3 = traci.vehicle.getSpeed(all_vehicle_id[2])
            # a1 = traci.vehicle.getAcceleration(all_vehicle_id[0]) / 2
            # a2 = traci.vehicle.getAcceleration(all_vehicle_id[1]) / 2
            # a3 = traci.vehicle.getAcceleration(all_vehicle_id[2]) / 2
            if len(v1_list) < 3:
                a1 = traci.vehicle.getAcceleration(all_vehicle_id[0]) / 1
                a2 = traci.vehicle.getAcceleration(all_vehicle_id[1]) / 1
                a3 = traci.vehicle.getAcceleration(all_vehicle_id[2]) / 1
            else:
                a1 = (v1_list[-1] - v1_list[-3]) / 1
                a2 = (v2_list[-1] - v2_list[-3]) / 1
                a3 = (v3_list[-1] - v3_list[-3]) / 1
            a1_list.append(a1)
            a2_list.append(a2)
            a3_list.append(a3)

            if a1 > 0:
                a1_theta = 0
            else:
                a1_theta = np.pi
            if a2 > 0:
                a2_theta = 0
            else:
                a2_theta = np.pi

            pred_ego = []
            pred_1 = []
            pred_2 = []
            for i in range(3):
                pred_1.append([p1[0], p1[1]])
                pred_2.append([p2[0], p2[1]])
                pred_ego.append([p3[0], p3[1]])


            kkk = 0.8
            if fai == 0:
                pred_ego[0][0] += (v3 + 0.5 * a3)
                pred_ego[0][1] += (0.1 * random.uniform(-1, 1))

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4)
                pred_ego[1][1] += (0.1 * random.uniform(-1, 1))

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9)
                pred_ego[2][1] += (0.1 * random.uniform(-1, 1))
            elif fai > 0:
                pred_ego[0][0] += (v3 + 0.5 * a3) * kkk
                pred_ego[0][1] += (0.64)

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4) * kkk
                pred_ego[1][1] += (0.64 * 2)

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9) * kkk
                pred_ego[2][1] += (0.64 * 3)
            else:
                pred_ego[0][0] += (v3 + 0.5 * a3) * kkk
                pred_ego[0][1] += (-0.64)

                pred_ego[1][0] += (2 * v3 + 0.5 * a3 * 4) * kkk
                pred_ego[1][1] += (-0.64 * 2)

                pred_ego[2][0] += (3 * v3 + 0.5 * a3 * 9) * kkk
                pred_ego[2][1] += (-0.64 * 3)

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
            P1, E1 = get_P_and_E(p1, p3, v1, v3, 0, fai, 0, fai, a1, a1_theta)
            P2, E2 = get_P_and_E(p2, p3, v2, v3, 0, fai, 0, fai, a2, a2_theta)
            

            # 预测1s
            P1_1s, E1_1s = get_P_and_E(pred_1[0], pred_ego[0], v1_1s, vego_1s, 0, fai, 0, fai, a1, a1_theta)
            P2_1s, E2_1s = get_P_and_E(pred_2[0], pred_ego[0], v2_1s, vego_1s, 0, fai, 0, fai, a2, a2_theta)
            
            # 预测2s
            P1_2s, E1_2s = get_P_and_E(pred_1[1], pred_ego[1], v1_2s, vego_2s, 0, fai, 0, fai, a1, a1_theta)
            P2_2s, E2_2s = get_P_and_E(pred_2[1], pred_ego[1], v2_2s, vego_2s, 0, fai, 0, fai, a2, a2_theta)
            
            # 预测3s
            P1_3s, E1_3s = get_P_and_E(pred_1[2], pred_ego[2], v1_3s, vego_3s, 0, fai, 0, fai, a1, a1_theta)
            P2_3s, E2_3s = get_P_and_E(pred_2[2], pred_ego[2], v2_3s, vego_3s, 0, fai, 0, fai, a2, a2_theta)

            E = max(E1_2s, E2_2s)

            # ttci_1=(v3-v1)/(((p3[0]-p1[0])**2+(p3[1]-p1[1])**2)**0.5)
            # ttci_2=(v3-v2)/(((p3[0]-p2[0])**2+(p3[1]-p2[1])**2)**0.5)

            # if abs(p1[1] - p3[1]) < 2:
            #     if p1[0] > p3[0]:
            #         ttci_1 = (v3 - v1) / abs(p1[0] - p3[0])
            #     else:
            #         ttci_1 = (v1 - v3) / abs(p1[0] - p3[0])
            # else:
            #     ttci_1 = 0
            # if abs(p2[1] - p3[1]) < 2:
            #     if p2[0] > p3[0]:
            #         ttci_2 = (v3 - v2) / abs(p2[0] - p3[0])
            #     else:
            #         ttci_2 = (v2 - v3) / abs(p2[0] - p3[0])
            # else:
            #     ttci_2 = 0

            ttt = 1
            if abs(pred_1[ttt][1] - pred_ego[ttt][1]) < 2:
                if pred_1[ttt][0] > pred_ego[ttt][0]:
                    ttci_1 = (vego_2s - v1_2s) / abs(pred_1[ttt][0] - pred_ego[ttt][0])
                else:
                    ttci_1 = (v1_2s - vego_2s) / abs(pred_1[ttt][0] - pred_ego[ttt][0])
            else:
                ttci_1 = 0
            if abs(pred_2[ttt][1] - pred_ego[ttt][1]) < 2:
                if pred_2[ttt][0] > pred_ego[ttt][0]:
                    ttci_2 = (vego_2s - v2_2s) / abs(pred_2[ttt][0] - pred_ego[ttt][0])
                else:
                    ttci_2 = (v2_2s - vego_2s) / abs(pred_2[ttt][0] - pred_ego[ttt][0])
            else:
                ttci_2 = 0

            ttci = max(ttci_1, ttci_2)
            # temp = (0.15*v2/(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5))+(v1+v2)*(v2-v1)/(1.5*9.8*(((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)**0.5))
            # SM=1-temp
            if E1_2s >= 1:
                print(t)
                print("E1_2s="+str(E1_2s))
            if E2_2s >= 1:
                print(t)
                print("E2_2s="+str(E2_2s))
            time_list.append(t)
            Fv1_list.append(E1)
            Fv2_list.append(E2)
            F_list.append(E)
            
            P1_list.append(P1_2s)
            P2_list.append(P2_2s)

            Fp1_list.append(E1_2s)
            Fp2_list.append(E2_2s)

            ttci_list_1.append(ttci_1)
            ttci_list_2.append(ttci_2)
            ttci_list.append(ttci)

            v1_list.append(v1)
            v2_list.append(v2)
            v3_list.append(v3)
            # SM_list.append(SM)


    nnn1 = abs(np.array(Fp1_list[:-20]) - np.array(Fv1_list[20:]))
    loss1 = sum(nnn1/np.array(Fv1_list[20:]))/len(nnn1)
    print("loss1", loss1)
    nnn2 = abs(np.array(Fp2_list[:-20]) - np.array(Fv2_list[20:]))
    loss2 = sum(nnn2/np.array(Fv2_list[20:]))/len(nnn2)
    print("loss2", loss2)

    # Fv1_list_err = Fv2_list[1:]
    # Fp_list_err = Fp2_list[:-1]

    # len_rmse = len(Fv1_list_err)

    # RMSE = 0.0
    # bias = 0.0
    # for i in range(len_rmse):
    #     RMSE += (Fv1_list_err[i] - Fp_list_err[i]) ** 2
    #     bias += (abs(Fv1_list_err[i] - Fp_list_err[i]) / Fv1_list_err[i])
    # RMSE /= len_rmse
    # RMSE = RMSE ** 0.5
    # print("RMSE=", RMSE)
    # bias /= len_rmse
    # print("bias=", bias)


    assert len(time_list) == len(P1_list)

    fig = plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(2, 2, 2)
    # ax4 = ax3.twinx()
    ax5 = fig.add_subplot(2, 2, 3)
    ax6 = ax5.twinx()
    ax7 = fig.add_subplot(2, 2, 4)
    ax8 = ax7.twinx()

    ax1.plot(time_list, v1_list, '-', c='tab:blue', label='v$_{A}$')
    ax1.plot(time_list, v2_list, '--', c='tab:orange', label='v$_{B}$')
    ax1.plot(time_list, v3_list, '-.', c='tab:red', label='v$_{ego}$')
    ax2.plot(time_list, delta_x1_list, ':', c='tab:pink', label='D$_{A}$')
    ax2.plot(time_list, delta_x2_list, linestyle=(0, (5, 1 )), c='tab:green', label='D$_{B}$')  # linestyle=(0, (5, 1 ))
    ax2.plot(time_list, np.array(y3_list)*10, linestyle=(0, (3, 1, 1, 1, 1, 1)), c='tab:cyan', label='10·y$_{ego}$')   # linestyle=(0, (3, 1, 1, 1, 1, 1))

    # ax3.plot(time_list, Fv1_list, linestyle=(0, (5, 1 )), c='tab:blue', label='CRI$_{A,0}$')
    # ax3.plot(time_list, Fv2_list, linestyle=(0, (3, 1, 1, 1, 1, 1)), c='tab:red', label='CRI$_{A,0}$')
    ax3.plot(time_list, P1_list, '-', c='tab:blue', label='P$_{A,2}$')
    ax3.plot(time_list, P2_list, '--', c='tab:red', label='P$_{B,2}$')
    ax3.plot(time_list, Fp1_list, '-.', c='tab:pink', label='CRI$_{A,2}$', alpha=0.8)    # $^{\'}$
    ax3.plot(time_list, Fp2_list, ':', c='tab:cyan', label='CRI$_{B,2}$', alpha=0.8)
    # ax4.plot(time_list, np.array(y3_list), linestyle=(0, (5, 1 )), c='tab:cyan', label='10·y$_{ego}$')   # linestyle=(0, (3, 1, 1, 1, 1, 1))
    # ax4.plot(time_list, delta_x1_list, linestyle=(0, (5, 1 )), c='tab:orange', label='D$_{1}$')
    # ax4.plot(time_list, delta_x2_list, linestyle=(0, (3, 1, 1, 1, 1, 1)), c='tab:green', label='D$_{2}$')  # linestyle=(0, (5, 1 ))

    # ax5.plot(time_list, Fv1_list, c='tab:blue', label='CRI$_{A,0}$')
    # ax5.plot(time_list, Fv2_list, '--', c='tab:red', label='CRI$_{A,0}$')
    # ax6.plot(time_list, ttci_list_1, '-.', c='tab:orange', label='TTC1$^{-1}$')
    # ax6.plot(time_list, ttci_list_2, ':', c='tab:green', label='TTC2$^{-1}$')

    ax5.plot(time_list, Fp1_list, '-', c='tab:red', label='CRI$_{A,2}$')
    ax6.plot(time_list, ttci_list_1, '--', c='tab:blue', label='TTC$_{A}$$^{-1}$')    # , alpha=0.6
    ax7.plot(time_list, Fp2_list, '-', c='tab:red', label='CRI$_{B,2}$')
    ax8.plot(time_list, ttci_list_2, '--', c='tab:blue', label='TTC$_{B}$$^{-1}$')
    

    # 获取对应折线图颜色给到spine ylabel yticks yticklabels
    axs = [ax1, ax2]
    # imgs = [img1_1, img1_2, img2, img3]

    ax1.set_xlabel('t(s)')

    ax1.set_ylabel('v(m/s)')
    ax2.set_ylabel('Distance(m)')

    ax3.set_xlabel('t(s)')
    ax3.set_ylabel('P$_{t}$ and CRI$_{t}$')
    # ax3.set_ylabel('TTCI', c='tab:orange')
    # ax4.set_ylabel('y$_{ego}$')
    # ax4.set_ylim(-5, 15)

    ax5.set_xlabel('t(s)')
    ax5.set_ylabel('CRI$_{2}$')
    ax6.set_ylabel('TTC$^{-1}$')

    ax7.set_xlabel('t(s)')
    ax7.set_ylabel('CRI$_{2}$')
    ax8.set_ylabel('TTC$^{-1}$')

    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    # handle3, label3 = ax3.get_legend_handles_labels()
    # handle4, label4 = ax4.get_legend_handles_labels()
    handle5, label5 = ax5.get_legend_handles_labels()
    handle6, label6 = ax6.get_legend_handles_labels()
    handle7, label7 = ax7.get_legend_handles_labels()
    handle8, label8 = ax8.get_legend_handles_labels()

    # ax1.legend(handles=handle1 + handle2,
    #           labels=label1 + label2, loc=(0.75,0.05))
    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc=(1.13,-0.08))
    # ax3.legend(handles=handle3 + handle4,
    #           labels=label3 + label4, loc='best')
    ax3.legend()
    ax5.legend(handles=handle5 + handle6,
              labels=label5 + label6, loc='best')
    ax7.legend(handles=handle7 + handle8,
              labels=label7 + label8, loc='best')
    
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
    P_list=[]
    E_list=[]
    P1_list=[]
    E1_list=[]
    P2_list=[]
    E2_list=[]
    P3_list=[]
    E3_list=[]
    # Fv2_list=[]
    # Fl_list=[]
    dx_list=[]
    dy_list=[]
    d_list=[]
    d1_list = []
    d2_list = []
    d3_list = []

    P_warm = []
    E_warm = []
    vis1_list=[]
    vis2_list=[]


    time_t=0

    for step in range(0, 120):  # 仿真时间
        traci.simulationStep()  # 一步一步（一帧一帧）进行仿真
        # time.sleep(0.1)
        t = traci.simulation.getTime()  # 获得仿真时间
        # print("simulation_time=", simulation_time)
        all_vehicle_id = traci.vehicle.getIDList()  # 获得所有车的id

        if len(all_vehicle_id) == 2:
            traci.gui.trackVehicle('View #0', all_vehicle_id[1])    # 2车为自车
            # 如果要故意碰撞需要设置setLaneChangeMode和setSpeedMode
            traci.vehicle.setLaneChangeMode("1", 0b000000000000)    # 关闭换道模型
            traci.vehicle.setLaneChangeMode("2", 0b000000000000)
            traci.vehicle.setSpeedMode("1", 00000)  # 关闭跟驰模型
            traci.vehicle.setSpeedMode("2", 00000)

            p1 = traci.vehicle.getPosition(all_vehicle_id[0])
            p2 = traci.vehicle.getPosition(all_vehicle_id[1])

            d=((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
            d_x = abs(p1[0] - p2[0])
            d_y = abs(p1[1] - p2[1])

            fai = 0.5*np.pi     # 自车横摆角
            fai_2 = 0

            v1 = traci.vehicle.getSpeed(all_vehicle_id[0])
            v2 = traci.vehicle.getSpeed(all_vehicle_id[1])
            a1 = traci.vehicle.getAcceleration(all_vehicle_id[0])
            a2 = traci.vehicle.getAcceleration(all_vehicle_id[1])

            if a1>0:
                a1_theta = 0
            else:
                a1_theta = np.pi


            pred_ego = []
            pred = []
            for i in range(3):
                pred_ego.append([p1[0], p1[1]])
                pred.append([p2[0], p2[1]])


            # pred[0][0] += (v1*1 + 1*a1/2 + random.uniform(-1, 1))
            # pred[1][0] += (v1*2 + 4*a1/2 + random.uniform(-1, 1))
            # pred[2][0] += (v1*3 + 9*a1/2 + random.uniform(-1, 1))

            # pred_ego[0][1] += (v2*1 + 1*a2/2 + random.uniform(-1, 1))
            # pred_ego[1][1] += (v2*2 + 4*a2/2 + random.uniform(-1, 1))
            # pred_ego[2][1] += (v2*3 + 9*a2/2 + random.uniform(-1, 1))

            pred[0][0] += (v1*1 + random.uniform(-1, 1))
            pred[1][0] += (v1*2 + random.uniform(-1, 1))
            pred[2][0] += (v1*3 + random.uniform(-1, 1))

            pred_ego[0][1] += (v2*1 + random.uniform(-1, 1))
            pred_ego[1][1] += (v2*2 + random.uniform(-1, 1))
            pred_ego[2][1] += (v2*3 + random.uniform(-1, 1))

            d1 = ((pred[0][0]-pred_ego[0][0])**2+(pred[0][1]-pred_ego[0][1])**2)**0.5
            d2 = ((pred[1][0]-pred_ego[1][0])**2+(pred[1][1]-pred_ego[1][1])**2)**0.5
            d3 = ((pred[2][0]-pred_ego[2][0])**2+(pred[2][1]-pred_ego[2][1])**2)**0.5
            d1_list.append(d1)
            d2_list.append(d2)
            d3_list.append(d3)

            v1_1s = v1 + a1
            v2_1s = v2 + a2

            v1_2s = v1 + 2*a1
            v2_2s = v2 + 2*a2

            v1_3s = v1 + 3*a1
            v2_3s = v2 + 3*a2

            # 当前
            P, E = get_P_and_E(p1, p2, v1, v2, 0, fai, 0, fai, a1, a1_theta)

            # 预测1s
            P_1s, E_1s = get_P_and_E(pred[0], pred_ego[0], v1_1s, v2_1s, 0, fai, 0, fai, a1, a1_theta)

            # 预测2s
            P_2s, E_2s = get_P_and_E(pred[1], pred_ego[1], v1_2s, v2_2s, 0, fai, 0, fai, a1, a1_theta)
            
            # 预测3s
            P_3s, E_3s = get_P_and_E(pred[2], pred_ego[2], v1_3s, v2_3s, 0, fai, 0, fai, a1, a1_theta)

            ttci=v2/abs(p2[1]-p1[1])
            ttci_2s = v2_2s/abs(pred_ego[1][1] - pred[1][1])
            ttci_3s = v2_3s/abs(pred_ego[2][1] - pred[2][1])

            if P_2s >= 1:
                P_warm.append(t)
                print("P warm " + format(t, '.1f'))
            if E_2s >= 1:
                E_warm.append(t)
                print("E warm " + format(t, '.1f'))

            time_list.append(t)
            v1_list.append(v1)
            v2_list.append(v2)

            P_list.append(P)
            E_list.append(E)
            P1_list.append(P_1s)
            E1_list.append(E_1s)
            P2_list.append(P_2s)
            E2_list.append(E_2s)
            P3_list.append(P_3s)
            E3_list.append(E_3s)
            

            ttci_list.append(ttci_2s)
            d_list.append(d)
            dx_list.append(d_x)
            dy_list.append(d_y)

    nnn = abs(np.array(d2_list[:-20]) - np.array(d_list[20:]))
    loss = sum(nnn/np.array(d_list[20:]))/len(nnn)
    print("loss", loss)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = ax1.twinx()
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = ax3.twinx()
    ax5 = fig.add_subplot(2, 2, 3)
    ax6 = ax5.twinx()

    ax1.plot(time_list, v1_list, '-', c='tab:red', label='v$_{1}$')
    ax1.plot(time_list, v2_list, '--', c='tab:blue', label='v$_{2}$')
    ax2.plot(time_list, d_list, '-.', c='tab:orange', label='D$_{0}$')
    ax2.plot(time_list, d2_list, ':', c='tab:grey', label='D$_{2}$')
    # ax2.plot(time_list, dy_list, ':', c='tab:grey', label='d_y')

    # ax3.plot(time_list, P_list, '-', c='tab:cyan', label='P')
    # ax3.plot(time_list, E_list, linestyle=(0, (5, 1 )), c='tab:cyan', label='CRI$_{0}$')
    ax3.plot(time_list, P2_list, '-', c='tab:red', label='P$_{2}$') # linestyle=(0, (3, 1, 1, 1, 1, 1)), 
    ax3.plot(time_list, E2_list, '--', c='tab:blue', label='CRI$_{2}$')
    ax4.plot(time_list, d_list, '-.', c='tab:orange', label='D$_{0}$')  # linestyle=(0, (5, 1 ))
    ax4.plot(time_list, d2_list, ':', c='tab:grey', label='D$_{2}$')    # linestyle=(0, (3, 1, 1, 1, 1, 1))

    # ax4.plot(time_list, dy_list, ':', c='tab:green', label='d_y')
    ax5.plot(time_list, E2_list, '-', c='tab:red', label='CRI$_{2}$')
    # ax5.plot(time_list, E2_list, '-', c='tab:red', label='E$^{\'}$')
    ax6.plot(time_list, ttci_list, '--', c='tab:blue', label='TTC$^{-1}$')


    fontsize = 12
    ax1.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax3.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)
    ax5.set_xlabel('t(s)', fontsize=fontsize, labelpad=0)

    ax1.set_ylabel('v(m/s)', fontsize=fontsize, labelpad=0)
    ax2.set_ylabel('distance(m)', fontsize=fontsize, labelpad=0)

    # ax3.set_xlabel('t(s)')
    ax3.set_ylabel('CRI$_{t}$ and P$_{t}$', fontsize=fontsize, labelpad=0)
    # ax3.set_ylabel('TTCI', c='tab:orange')
    ax4.set_ylabel('distance(m)', fontsize=fontsize, labelpad=0)

    ax5.set_ylabel('CRI$_{2}$', fontsize=fontsize, labelpad=0)
    ax6.set_ylabel('TTC$^{-1}$', fontsize=fontsize, labelpad=3)


    handle1, label1 = ax1.get_legend_handles_labels()
    handle2, label2 = ax2.get_legend_handles_labels()
    handle3, label3 = ax3.get_legend_handles_labels()
    handle4, label4 = ax4.get_legend_handles_labels()
    handle5, label5 = ax5.get_legend_handles_labels()
    handle6, label6 = ax6.get_legend_handles_labels()

    ax1.legend(handles=handle1 + handle2,
              labels=label1 + label2, loc='center right')
    ax3.legend(handles=handle3 + handle4,
               labels=label3 + label4, loc='center left')  # ,framealpha=1
    ax5.legend(handles=handle5 + handle6,
               labels=label5 + label6, loc='center left')  # ,framealpha=1
    # ax3.legend()
    # ax1.legend(handles=handle1 + handle2 + handle3 ,
    #           labels=label1 + label2 + label3 ,loc=(0.75,0.72))

    plt.tight_layout()
    plt.show()

    traci.close()
    return


if __name__ == "__main__":
    scene_C()

