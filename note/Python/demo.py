import numpy as np
import matplotlib.pyplot as plt

def d2024_5_9():
    R0 = np.arange(0.1, 10, 0.1)
    Rc = (3*1.4/2/0.567*(101167+2*0.567/R0-14551)*R0**(3*1.4))**(1/(3*1.4-1))
    # Rc = (1.5*1.4/0.567*(101300+2*0.567/R0-133)*R0**4.2)**(1/3.2)
    plt.plot(R0, Rc)
    plt.show()


if __name__ == '__main__':
    d2024_5_9()