import numpy as np
import matplotlib.pyplot as plt

# Read data
Training_loss = np.loadtxt('../logs/TrainLoss.txt', skiprows=1)
Validation_loss = np.loadtxt('../logs/ValidationLoss.txt', skiprows=1)

# Compute moving average
xT = Training_loss[:,0]
yT = Training_loss[:,1]

xV = Validation_loss[:,0]
yV = Validation_loss[:,1]


# plot original and  dataset
plt.semilogy(xT, yT, 'r-', alpha=0.3, label='Training Loss')
plt.semilogy(xV, yV, 'r-', label='Validation Loss')
plt.legend()
plt.show()
