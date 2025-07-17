import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
def create_dataset(n):
    x=np.empty((0,2))
    y=np.empty(0)
    for i in range(n):
        example=np.random.random(2)
        if example[0]>example[1]:
            y=np.append(y,1)
        else:
            y=np.append(y,-1)
        x=np.append(x, np.array([example]), axis=0)
    return x, y

x_train, y_train=create_dataset(20)
#p_no_hinge=Perceptron(x_train=x_train, y_train=y_train)
#p_no_hinge.train()
plt.figure()
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap='tab10', s=20)  # punti colorati
plt.plot([0,1], [0,1], 'k-', label='y=x')                            # retta nera
plt.legend(['y=x'])
#p_hinge=Perceptron(x_train=x_train, y_train=y_train)
#p_hinge.train(hinge=True)
x_test, y_test=create_dataset(1000)
plt.figure()
plt.scatter(x_test[:,0], x_test[:,1], c=y_test, cmap='tab10', s=20)  # punti colorati
plt.plot([0,1], [0,1], 'k-', label='y=x')                            # retta nera
plt.legend(['y=x'])
plt.show()
#plt.plot(xpoints, ypoints, 'o')
#plt.show()