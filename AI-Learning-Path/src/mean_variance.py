import numpy as np
data = [1.1, 0.8, 2.3, 4.4]

mean = np.mean(data)
var = np.var(data)
print(mean)
print(var)

output = [float((i - mean) / var**0.5) for i in data]
print(output)

data = [-0.74, -0.95, 0.1, 1.59]

print(np.mean(output))
print(np.var(output))
