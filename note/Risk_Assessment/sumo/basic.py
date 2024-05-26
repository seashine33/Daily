import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


########## 碰撞参数 ##########################
l = 5
d = 2
ld = (l**2+d**2)**0.5
phi_ = np.arctan(d/l)   # 21.8°
arrow_length = 2.5    # 5
alpha = 0.005
beta = 0.006
gamma = 0.025
########## 风险场参数 ##########################

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

# 作风险来源车
def draw_center_car():
    fine = 500
    x = np.linspace(-l/2, l/2, fine)
    y = np.linspace(-d/2, d/2, fine)
    line_color = 'k'
    plt.plot(x, np.ones(fine)*d/2, color = line_color)
    plt.plot(x, np.ones(fine)*(-d/2), color = line_color)
    plt.plot(np.ones(fine)*(l/2), y, color = line_color)
    plt.plot(np.ones(fine)*(-l/2), y, color = line_color)
    plt.plot(0,0,'o',color = line_color)
    xx = np.array([-l/2, l/2])
    y1 = np.array([-d/2, -d/2])
    y2 = np.array([d/2, d/2])
    # plt.fill_between(xx, y1, y2, where=(y2>y1), color='yellow', alpha=1)
    plt.annotate('', xy=(arrow_length,0),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='r'))
    # plt.annotate('', xy=(0,arrow_length),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='r'))
    return

# 作自车
def draw_self_car(x, y, phi, linestyle='-'):
    fine = 500
    X = np.linspace(-l/2, l/2, fine)
    Y = np.ones(fine)*d/2
    xx = np.cos(phi) * X - np.sin(phi) * Y + x      # xx == X
    yy = np.sin(phi) * X + np.cos(phi) * Y + y     # yy == Y
    plt.plot(xx, yy, color = 'b', linestyle=linestyle)
    Y = np.ones(fine)*(-d/2)
    xx = np.cos(phi) * X - np.sin(phi) * Y + x      # xx == X
    yy = np.sin(phi) * X + np.cos(phi) * Y + y     # yy == Y
    plt.plot(xx, yy, color = 'b', linestyle=linestyle)
    X = np.ones(fine)*l/2
    Y = np.linspace(-d/2, d/2, fine)
    xx = np.cos(phi) * X - np.sin(phi) * Y + x      # xx == X
    yy = np.sin(phi) * X + np.cos(phi) * Y + y     # yy == Y
    plt.plot(xx, yy, color = 'b', linestyle=linestyle)
    X = np.ones(fine)*(-l/2)
    xx = np.cos(phi) * X - np.sin(phi) * Y + x      # xx == X
    yy = np.sin(phi) * X + np.cos(phi) * Y + y     # yy == Y
    plt.plot(xx, yy, color = 'b', linestyle=linestyle)

    X = np.linspace(0, x, fine)
    Y = np.linspace(0, y, fine)
    plt.plot(X, Y, color = 'black')
    plt.plot(x,y,'o',color = 'b')
    plt.annotate('', xy=(x+arrow_length*np.cos(phi),y+arrow_length*np.sin(phi)),xytext=(x,y), arrowprops=dict(arrowstyle="->", color='b'))
    # E.append(f(x, y, 20, 0))
    # print(E[-1])
    return

# 作自车碰撞下位置_xy
def draw_danger_car_xy(x, y, phi):
    theta = np.arctan2(y, x)
    length = get_oo_length(theta, phi)
    x = length * np.cos(theta)
    y = length * np.sin(theta)
    draw_self_car(x, y, phi, linestyle='--')
    return

# 作自车碰撞下位置_theta
def draw_danger_car_theta(theta, phi):
    length = get_oo_length(theta, phi)
    x = length * np.cos(theta)
    y = length * np.sin(theta)
    draw_self_car(x, y, phi, linestyle='--')
    return

# 作出自车在任一点(x,y)与车头方向phi下，对应的碰撞位置
def demo2():
    x = 1.71
    y = 1
    phi = np.pi/180*75     # np.pi/8
    draw_center_car()
    # draw_self_car(x, y, phi)
    draw_danger_car_xy(x, y, phi)
    plt.gca().set_aspect(1)
    plt.show()
    return

# 临界碰撞展示：theta不变，phi变
def demo3():
    theta = 30 * np.pi/180 # phi_ + np.pi/4
    phi = np.linspace(0, 2*np.pi, 100)   # np.pi/2, np.pi + 2*theta - 2*phi_, np.pi - theta, np.pi
    for p in phi:
        draw_center_car()
        draw_danger_car_theta(theta, p)
        plt.xlim(-8, 8)
        plt.ylim(-7, 7)
        plt.gca().set_aspect(1)
        plt.pause(0.001)
        # if p == phi[99]:
        #     plt.pause(1)
        plt.ioff()
        plt.clf()
    return

# 临界碰撞展示：theta变，phi不变
def demo4():
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.pi/8  # 自车方向角   # np.pi/2, np.pi + 2*theta - 2*phi_, np.pi - theta, np.pi
    for t in theta:
        draw_center_car()
        draw_danger_car_theta(t, phi)
        plt.xlim(-8, 8)
        plt.ylim(-7, 7)
        plt.gca().set_aspect(1)
        plt.pause(0.002)
        # if p == phi[99]:
        #     plt.pause(1)
        plt.ioff()
        plt.clf()
    return

# 碰撞概率展示
def demo5():
    x = np.linspace(-20, 20, 400)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵, [200, 400]
    theta = np.arctan2(Y, X)
    length_ = np.zeros((theta.shape[0], theta.shape[1]))
    cs1 = 0.28                  # 调整衰减速率，越小危险区域区域越大

    # 静态因素
    phi_self =       0              # 自车横摆角

    # 动态因素
    v_center =       20              # 他车速度大小
    v_center_theta = 0              # 他车速度方向
    v_self =         20             # 自车速度大小
    v_self_theta =   0              # 自车速度方向
    a_center =       0              # 他车加速度大小
    a_center_theta = 0              # 他车加速度方向


    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            length_[i][j] = get_oo_length(theta[i][j], phi_self)
    x = length_ * np.cos(theta)     # 各点对应临界碰撞坐标
    y = length_ * np.sin(theta)
    P = ((x**2 + y**2)**0.5/(X**2 + Y**2)**0.5)**cs1    # 碰撞概率：临界碰撞距离/当前两车距离
    Vx = v_center*np.cos(v_center_theta) - v_self*np.cos(v_self_theta)
    Vy = v_center*np.sin(v_center_theta) - v_self*np.sin(v_self_theta)
    V_delta = ((Vx**2)+(Vy**2))**0.5
    theta_V_delta = np.arctan2(Vy, Vx)
    # Z = P * np.exp(alpha*v_center + 
    #                     beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
    #                     gamma*(a_center)*abs(np.cos((theta-a_center_theta)/2)))
    Z = np.exp(alpha*v_center + 
                beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                gamma*(a_center)*abs(np.cos((theta-a_center_theta)/2))) - 1
    # for i in range(len(Z)):
    #     for j in range(len(Z[i])):
    #         if Z[i][j]>1.5:
    #             Z[i][j]=1.5

    ax = plt.subplot(1, 1, 1)
    line = [0.60, 0.70, 0.80, 0.90, 1.0, 1.5]
    # line = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
    # line = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
    plt.contourf(X, Y, Z, 6, cmap=plt.get_cmap('Blues'))
    c = plt.contour(X, Y, Z, 6, colors='black')
    plt.clabel(c, inline=True, fontsize=10)
    # draw_center_car()   # 自车车身
    demo6(phi_self)     # 碰撞环
    title = '(' + str(1) + ')' + "Va=" + str(v_center) + ",Vb=" + str(v_self)
    title += ",φ="+ str(int(phi_self/np.pi*180)) + "°,Aa=" + str(a_center) + ",θAa=" + str(int(a_center_theta/np.pi*180)) + '°'
    ax.set_title(title, family='monospace', fontsize=10)     # Times New Roman
    ax.set_xlabel('X/m', fontsize=8, labelpad=0)
    ax.set_ylabel('Y/m', fontsize=8, labelpad=-10)
    plt.annotate('', xy=(arrow_length*np.cos(theta_V_delta),arrow_length*np.sin(theta_V_delta)),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='red'))
    plt.xlim(-20, 20)
    plt.ylim(-10, 10)
    plt.gca().set_aspect(1)
    # plt.grid()  # 网格
    # plt.tight_layout()
    plt.show()
    return

# 碰撞概率集中展示
def demo5_1():
    x = np.linspace(-20, 20, 400)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵, [200, 400]
    theta = np.arctan2(Y, X)
    length_ = np.zeros((theta.shape[0], theta.shape[1]))
    cs1 = 0.28                  # 调整衰减速率，越小危险区域区域越大

    alpha = 0.005
    beta = 0.006
    gamma = 0.025
    # # 静态因素
    # phi_self =       [0,  0,  0,  0,     0,       0, np.pi/2, np.pi]              # 自车方向角，车头朝向
    # # 动态因素
    # v_center =       [0, 20,  0, 20,    20,      20,      20,    20]              # 速度越大，需要的跟车距离越大
    # v_center_theta = [0,  0,  0,  0,     0,       0,       0,     0]
    # v_self =         [0, 20, 20, 20,    20,      20,      20,    20]              # 自车速度
    # v_self_theta =   [0,  0,  0,  0,     0,       0, np.pi/2, np.pi]
    # a_center =       [0,  0,  0,  5,     5,       5,       0,     0]              # 相对加速度，只有正值
    # a_center_theta = [0,  0,  0,  0, np.pi, np.pi/2,       0,     0]              # 相对加速度方向

    # 静态因素
    phi_self =       [ 0,  0,     0,       0]              # 自车方向角，车头朝向
    # 动态因素
    v_center =       [20, 20,    20,      20]              # 速度越大，需要的跟车距离越大
    v_center_theta = [ 0,  0,     0, np.pi/4]
    v_self =         [20, 20,    20,      20]              # 自车速度
    v_self_theta =   [ 0,  0,     0,       0]
    a_center =       [ 0,  5,    10,       0]              # 相对加速度，只有正值
    a_center_theta = [ 0,  0, np.pi,       0]              # 相对加速度方向
    
    for k in range(0, len(phi_self)):    # len(phi_self)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                length_[i][j] = get_oo_length(theta[i][j], phi_self[k]) # 临界碰撞距离
        x = length_ * np.cos(theta)     # 临界碰撞坐标
        y = length_ * np.sin(theta)
        P = ((x**2 + y**2)**0.5/(X**2 + Y**2)**0.5)**cs1    # 碰撞概率：临界碰撞距离/当前两车距离
        Vx = v_center[k]*np.cos(v_center_theta[k]) - v_self[k]*np.cos(v_self_theta[k])
        Vy = v_center[k]*np.sin(v_center_theta[k]) - v_self[k]*np.sin(v_self_theta[k])
        V_delta = ((Vx**2)+(Vy**2))**0.5
        theta_V_delta = np.arctan2(Vy, Vx)
        Z = P * np.exp(alpha*v_center[k] + 
                       beta*V_delta*abs(np.cos((theta_V_delta-theta)/2)) + 
                       gamma*(a_center[k])*abs(np.cos((theta-a_center_theta[k])/2)))
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                if Z[i][j]>1.2:
                    Z[i][j]=1.2

        ax = plt.subplot(2, 2, k+1)
        
        line = [0.60, 0.70, 0.80, 0.90, 1.0, 1.2]
        # line = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
        # line = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
        plt.contourf(X, Y, Z, line, cmap=plt.get_cmap('Blues'), extend='both') # magma_r, Blues, rainbow
        cbar = plt.colorbar(ax=ax, fraction=0.025)
        cbar.set_ticks([0.6, 0.7, 0.8, 0.9, 1.0, 1.2])
        cbar.set_ticklabels(['0.6', '0.7', '0.8', '0.9', '1.0', 'Collision'])
        c = plt.contour(X, Y, Z, line, colors='black')
        plt.clabel(c, inline=True, fontsize=10)

        
        demo6(phi_self[k])     # 碰撞环
        draw_center_car()   # 自车车身
        title = '(' + str(k+1) + ')' + "v$_{A}$=" + str(v_center[k]) + "m/s,θ$_{A}$="+ str(int(v_center_theta[k]/np.pi*180)) + "°,a$_{A}$="
        if a_center_theta[k] == np.pi:
            a_center[k] = -a_center[k]
        title +=  str(a_center[k]) + "m/s$^{2}$"
        # ax.set_title(title, family='monospace', fontsize=12, y=-0.25)     # Times New Roman
        ax.set_xlabel('X/m', fontsize=8, labelpad=0)
        ax.set_ylabel('Y/m', fontsize=8, labelpad=-10)
        
        plt.xlim(-20, 20)
        plt.ylim(-10, 10)
        plt.gca().set_aspect(1)

        
        # plt.grid()  # 网格
    # plt.tight_layout()
    plt.show()
    return

# 碰撞环
def demo6(phi_self = np.pi*2/3):
    theta = np.linspace(0, 2*np.pi, 500)
    x_col = []
    y_col = []

    for t in theta:
        l_col = get_oo_length(t, phi_self)
        x_col.append(l_col*np.cos(t))
        y_col.append(l_col*np.sin(t))

    plt.fill(x_col, y_col, color = 'darkblue')   # red, midnightblue, navy, darkblue
    plt.plot(x_col, y_col, color = 'k')
    # plt.annotate('', xy=(arrow_length*np.cos(phi_self),arrow_length*np.sin(phi_self)),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='pink'))
    # plt.show()
    return

# 概率环
def demo7():
    x = np.linspace(-20, 20, 800)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵, [200, 400]

    fai = 0 # 风险来源车转向角
    xx = np.cos(fai) * X - np.sin(fai) * Y      # xx == X
    yy = np.sin(fai) * X + np.cos(fai) * Y     # yy == Y

    theta = np.arctan2(yy, xx)
    phi_self = np.pi*0/6  # 自车方向角np.pi*7/8

    length_ = np.zeros((theta.shape[0], theta.shape[1]))
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            length_[i][j] = get_oo_length(theta[i][j], phi_self)

    x = length_ * np.cos(theta)
    y = length_ * np.sin(theta)
    
    # zz = f(X, Y)/f(x, y)
    zz = (x**2 + y**2)**0.10/(X**2 + Y**2)**0.10
    # v_self = 20
    # zz = f(X, Y)/f(x, y)*np.exp(0.004*v_self*np.abs(np.sin((phi_self-theta)/2))-0.03)
    for i in range(0,len(zz)):
        for j in range(0,len(zz[i])):
            if zz[i][j]>1.5:
                zz[i][j]=1.5
    line = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
    # line = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5]
    plt.contourf(xx, yy, zz, line, cmap=plt.get_cmap('Blues'))   # 8表示画8条等高线，
    c = plt.contour(xx, yy, zz, line, colors='black')
    plt.clabel(c, inline=True, fontsize=10)

    # 自车
    car_x = np.linspace(-2.5, 2.5, 500)
    car_y = np.ones(500)*1
    plt.plot(car_x, car_y, color='red', linewidth=2)
    car_y = -car_y
    plt.plot(car_x, car_y, color='red', linewidth=2)
    car_x = np.ones(200)*2.5
    car_y = np.linspace(-1, 1, 200)
    plt.plot(car_x, car_y, color='red', linewidth=2)
    car_x = -car_x
    plt.plot(car_x, car_y, color='red', linewidth=2)

    theta_self = np.linspace(0, 2*np.pi, 100)
    x_col = []
    y_col = []
    for t in theta_self:
        l_col = get_oo_length(t, phi_self)
        x_col.append(l_col*np.cos(t))
        y_col.append(l_col*np.sin(t))
    plt.plot(x_col, y_col, color = 'pink')

    plt.annotate('', xy=(arrow_length*np.cos(phi_self),arrow_length*np.sin(phi_self)),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='black'))
    plt.annotate('', xy=(arrow_length,0),xytext=(0,0), arrowprops=dict(arrowstyle="->", color='red'))
    plt.xlim(-20, 20)
    plt.ylim(-10, 10)
    plt.gca().set_aspect(1)
    plt.grid()  # 网格
    plt.show()
    return

# get_oo_length各区间的可视化
def demo8():
    # theta = 10*np.pi/180      # 1
    # phi_self = [0, theta, 2*theta, theta + phi_, 2*(theta + phi_),np.pi/2, np.pi + 2*theta - 2*phi_, np.pi-theta, np.pi]
    # theta = 30*np.pi/180      # 2
    # phi_self = [0, theta - phi_, 2*(theta-phi_), 2*theta, np.pi/2, np.pi]
    # theta = 50*np.pi/180      # 3
    # phi_self = [0, theta - phi_, 2*(theta-phi_), np.pi/2, 2*theta, 150*np.pi/180]
    theta = 80*np.pi/180      # 4
    phi_self = [0, theta - phi_, np.pi/2, 120*np.pi/180, 2*theta, 175*np.pi/180]
    for i, p in enumerate(phi_self):
        ax = plt.subplot(2, 3, i+1)
        draw_center_car()
        draw_danger_car_theta(theta, p)
        # plt.xlim(-3, 8)       # 1
        # plt.ylim(-2.5, 4)
        # plt.xlim(-3, 8)       # 2
        # plt.ylim(-2.5, 5.5)
        # plt.xlim(-3, 5.5)         # 3
        # plt.ylim(-2, 6.5)
        plt.xlim(-3, 4)         # 4
        plt.ylim(-2, 6.5)
        title = '(' + str(i+1) + ')' + 'θ=' + format(theta/np.pi*180, '.1f') + '°,φ=' + format(phi_self[i]/np.pi*180, '.1f') + '°'
        ax.set_title(title, family='monospace', fontsize=12, y=-0.25)
        ax.set_xlabel('X/m', fontsize=8, labelpad=0)
        ax.set_ylabel('Y/m', fontsize=8, labelpad=-10)
        plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.show()
    return

# 碰撞概率P
def demo9():
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵, [200, 400]
    theta = np.arctan2(Y, X)
    length_ = np.zeros((theta.shape[0], theta.shape[1]))
    cs1 = 0.28                  # 调整衰减速率，越小危险区域区域越大
    phi_self = [0, np.pi/4, np.pi/2, np.pi*2/3]
    lamb =     [1.2,   1.0,     0.8,       0.5]
    # phi_self = [0, np.pi/4]
    # lamb =     [1.2,   1.0]
    for k in range(0, len(phi_self)):    # len(phi_self)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                length_[i][j] = get_oo_length(theta[i][j], phi_self[k])
        x = length_ * np.cos(theta)     # 各点对应临界碰撞坐标
        y = length_ * np.sin(theta)
        Z = ((x**2 + y**2)**0.5/(X**2 + Y**2)**0.5)**lamb[k]    # 碰撞概率：临界碰撞距离/当前两车距离
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                if Z[i][j]>1.5:
                    Z[i][j]=1.5
        ax = plt.subplot(2, 2, k+1)
        line = [0.40, 0.60, 0.80, 1.0, 1.5]
        plt.contourf(X, Y, Z, line, cmap=plt.get_cmap('Blues'))
        c = plt.contour(X, Y, Z, line, colors='black')
        plt.clabel(c, inline=True, fontsize=10)
        title = "(" + str(k+1) + ")" + "φ="+ str(int(phi_self[k]/np.pi*180)) + "°,λ=" + format(lamb[k], '.1f')
        ax.set_title(title, family='monospace', fontsize=10)     # Times New Roman
        draw_center_car()   # 自车车身
        demo6(phi_self[k])     # 碰撞环
        ax.set_xlabel('X/m', fontsize=8, labelpad=0)
        ax.set_ylabel('Y/m', fontsize=8, labelpad=-10)
        plt.gca().set_aspect(1)
    plt.show()
    return

# 碰撞概率P
def demo9_1():
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵, [200, 400]
    theta = np.arctan2(Y, X)
    length_ = np.zeros((theta.shape[0], theta.shape[1]))
    cs1 = 0.08                  # 0.28, 调整衰减速率，越小危险区域区域越大
    # phi_self = [0, np.pi/4, np.pi/2, np.pi*2/3]
    # lamb =     [1.2,   1.0,     0.8,       0.5]
    phi_self = [0, np.pi/4]
    lamb =     [0.6,   1.2]
    for k in range(0, len(phi_self)):    # len(phi_self)
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                length_[i][j] = get_oo_length(theta[i][j], phi_self[k])
        x = length_ * np.cos(theta)     # 各点对应临界碰撞坐标
        y = length_ * np.sin(theta)
        Z = ((x**2 + y**2)**0.5/(X**2 + Y**2)**0.5)**lamb[k]    # 碰撞概率：临界碰撞距离/当前两车距离
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                if Z[i][j]>1.2:
                    Z[i][j]=1.2
        ax = plt.subplot(1, 2, k+1)
        line = [0.20, 0.40, 0.60, 0.80, 1.0]
        # line = [0.60, 0.70, 0.80, 0.90, 1.0]
        plt.contourf(X, Y, Z, line, cmap=plt.get_cmap('Blues'), extend='both') # magma_r, Blues, rainbow, Greys

        cbar = plt.colorbar(ax=ax, fraction=0.05)
        cbar.set_ticks([0.20, 0.40, 0.60, 0.80, 1.0])
        cbar.set_ticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        # cbar.set_ticks([0.60, 0.70, 0.80, 0.90, 1.0])
        # cbar.set_ticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'])

        c = plt.contour(X, Y, Z, line, colors='black')
        plt.clabel(c, inline=True, fontsize=10)

        draw_center_car()   # 自车车身
        # demo6(phi_self[k])     # 碰撞环

        title = "(" + str(k+1) + ")" + "φ="+ str(int(phi_self[k]/np.pi*180)) + "°,λ=" + format(lamb[k], '.1f')
        ax.set_title(title, family='monospace', fontsize=10, y=-0.20)     # Times New Roman
        ax.set_xlabel('X/m', fontsize=8, labelpad=0)
        ax.set_ylabel('Y/m', fontsize=8, labelpad=-10)
        plt.gca().set_aspect(1)
    plt.show()
    return

if __name__ == "__main__":
    demo5_1()
    # P = (E[0]/E[1])
    # print(P)