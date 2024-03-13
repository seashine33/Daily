import matplotlib.pyplot as plt

def demo():
    x = [0,1,2,3,4]
    y = [0,1,2,3,4]
    # plt.plot(
    #     x,y,
    #     ".",color='red',
    #     alpha=1, linewidth=20,
    # )
    plt.scatter(
        x, y,
        s=100, c='blue',
        alpha=1, marker="*",
    )
    plt.xlim(-20, 20)
    plt.ylim(-10, 10)
    plt.gca().set_aspect(1) # xy等轴距
    plt.grid()  # 网格
    plt.show()

if __name__ == "__main__":
    demo()