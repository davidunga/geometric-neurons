import matplotlib.pyplot as plt
from motorneural.motor import KinData
import numpy as np


def draw_kinematics(kin: KinData, color_by="crv2"):

    c = np.abs(kin[color_by]) ** .5
    si = np.argsort(c)
    c = c[si]
    x, y = kin['X'][si].T

    plt.figure()
    plt.scatter(x, y, c=c, s=20, alpha=.5, edgecolor='none', cmap='jet')
    plt.colorbar()
    plt.show()
