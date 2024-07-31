import numpy as np
import matplotlib.pyplot as plt

# Generating detailed analysis data (example values)
# Simulated data for Precision@k for different models
precision_at_k = {
    'User-Based': [0.72, 0.75, 0.78, 0.80, 0.82],
    'Item-Based': [0.74, 0.76, 0.79, 0.81, 0.83],
    'Content-Based': [0.68, 0.70, 0.73, 0.75, 0.77],
    'Hybrid': [0.76, 0.78, 0.80, 0.82, 0.85]
}

k_values = [1, 2, 3, 4, 5]

# Plotting Precision@k for different models
plt.figure(figsize=(12, 8))
for model, precisions in precision_at_k.items():
    plt.plot(k_values, precisions, marker='o', label=model)

plt.xlabel('k')
plt.ylabel('Precision@k')
plt.title('Precision@k for Different Recommendation Models')
plt.legend()
plt.grid(True)
plt.xticks(k_values)

# Display the plot
plt.show()
