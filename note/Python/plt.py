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
    plt.show()

if __name__ == "__main__":
    demo()