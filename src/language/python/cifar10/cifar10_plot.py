# read result and plot them
import matplotlib.pyplot as plt

train_loss = '/home/zhuyangyang/project/test/src/language/python/training_loss.txt'
train_accuracy = '/home/zhuyangyang/project/test/src/language/python/training_accuracy.txt'

# Read training loss data
epochs_loss = []
loss_values = []
# with open('training_loss.txt', 'r') as f_loss:
with open(train_loss, 'r') as f_loss:
    for line in f_loss:
        epoch, loss = line.split()
        epochs_loss.append(int(epoch))
        loss_values.append(float(loss))

# Read training accuracy data
epochs_accuracy = []
accuracy_values = []
with open(train_accuracy, 'r') as f_accuracy:
    for line in f_accuracy:
        epoch, accuracy, _ = line.split()
        epochs_accuracy.append(int(epoch))
        accuracy_values.append(float(accuracy))

# Plot training loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_loss, loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_accuracy, accuracy_values, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy over Epochs')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
