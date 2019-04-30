import matplotlib.pyplot as plt
learning_rate = 0.001
n_epoch = 100
dataset = [[0.5,0,0,0],
[0,0.5,1,0],
[0.5,1,0,1],
[1,1,0,1],
[0,0.5,0.5,0],
[0.5,0.5,0,1],
[0,0,0,0],
[0,1,0.5,0],
[0.5,0,0.5,0],
[0.5,0,1,0],
[1,0.5,0,1],
[0.5,0.5,0.5,1],
[0,0,1,0],
[0.5,1,0.5,1],
[1,0.5,0.5,1],
[1,1,0.5,1],
[0,1,1,0],
[0.5,0.5,1,0],
[0.5,1,1,1],
[1,0,1,0],
[1,0.5,1,1],
[1,1,1,1],
[0,0.5,0,0],
[0,1,0,0],
[1,0,0.5,1],
[0,0,0.5,0]]
def predict(row, weights):
activation = weights[0]
for i in range(len(row)-1):
activation += weights[i + 1] * row[i]
return 1.0 if activation >= 0.0 else 0.0
# Train using SGD
def train_weights(train, learning_rate, n_epoch):
weights = [0.0 for i in range(len(train[0]))]
error_list=[]
for epoch in range(n_epoch):
sum_error = 0.0
for row in train:
prediction = predict(row, weights)
perceptron_from_scratch.py
error = row[-1] - prediction
sum_error += error**2
weights[0] = weights[0] + learning_rate * error
for i in range(len(row)-1):
weights[i + 1] = weights[i + 1] + learning_rate *
error * row[i]
print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, learning_rate,
sum_error/len(train)))
print('weights = ', weights)
error_list.append(1-sum_error/len(train))
return [weights, error_list]
[weights, error_list] = train_weights(dataset, learning_rate, n_epoch)
# Plot the training accuracy
plt.plot(error_list)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig('perceptron_accuracy.pdf')
plt.show()