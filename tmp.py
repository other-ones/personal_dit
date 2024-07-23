import numpy as np
import matplotlib.pyplot as plt

# Generate Gumbel noise
num_samples = 10
gumbel_noise = np.random.gumbel(loc=0, scale=1, size=num_samples)
print(gumbel_noise)