## Code to plot loss data from model training
## Could be included in modely.py, but wrote it later, so did it separately to avoid running 
## the entire model for documentation graph
##
## Uses output from model.py for data. An example below
# 38572/38572 [==============================] - 50s 1ms/step - loss: 0.0249 - val_loss: 0.0223
# Epoch 2/5
# 3572/38572 [==============================] - 48s 1ms/step - loss: 0.0176 - val_loss: 0.0217
# Epoch 3/5
# 38572/38572 [==============================] - 49s 1ms/step - loss: 0.0153 - val_loss: 0.0178
# Epoch 4/5
# 38572/38572 [==============================] - 48s 1ms/step - loss: 0.0137 - val_loss: 0.0188
# Epoch 5/5
# 38572/38572 [==============================] - 48s 1ms/step - loss: 0.0132 - val_loss: 0.0182


# import the libraries

import matplotlib
#%matplotlib inline
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data from training output for model.py run
loss_train = [0.0249, 0.0176, 0.0153, 0.0137, 0.0132]
loss_valid = [0.0223, 0.0217, 0.0178, 0.0188, 0.0182]
epochs = [1, 2, 3, 4, 5]

plt.plot(epochs, loss_train, label='Training')
plt.plot(epochs, loss_valid, label='Validation')
plt.ylabel('Loss')
plt.xlabel('EPOCHS')
plt.title('Loss Progression During Training')
plt.legend()
plt.savefig('plot_loss.png', format='png')
#plt.show()