import pandas as pd
import matplotlib.pyplot as plt

loss_data = pd.read_csv("data\metrics\losses_centralized.csv")
accuracy_data = pd.read_csv("data\metrics\metrics_centralized.csv")

plt.figure(figsize=(10, 6))

# Vẽ accuracy theo round
plt.plot(accuracy_data['round'], accuracy_data['accuracy'], label='Accuracy', marker='o')

# Vẽ loss theo round
plt.plot(loss_data['round'], loss_data['loss'], label='Loss', marker='x')

plt.title('Loss and Accuracy over Rounds')
plt.xlabel('Round')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.show()
