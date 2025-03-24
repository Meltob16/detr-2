import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json

file_path = "models\out\sentinel2_20_epochs\log.txt"

try:
    with open(file_path, "r") as f:
        logs = json.load(f)
except json.JSONDecodeError:
    logs = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                print(f"Invalid JSON in line: {line.strip()}")

# At this point, 'logs' is a list of dictionaries containing the log entries
print("Logs loaded successfully. Number of entries:", len(logs))

# Initialize a dictionary to group metrics by epoch
metrics_by_epoch = defaultdict(lambda: {'train_loss': [], 'train_loss_bbox': [], 'train_class_error': []})

# Populate the dictionary with metrics from each log entry
for log in logs:
    epoch = log['epoch']
    metrics_by_epoch[epoch]['train_loss'].append(log['train_loss'])
    metrics_by_epoch[epoch]['train_loss_bbox'].append(log['train_loss_bbox'])
    metrics_by_epoch[epoch]['train_class_error'].append(log['train_class_error'])

# Compute average metrics for each epoch
epochs = sorted(metrics_by_epoch.keys())
avg_train_loss = [np.mean(metrics_by_epoch[epoch]['train_loss']) for epoch in epochs]
avg_train_loss_points = [np.mean(metrics_by_epoch[epoch]['train_loss_bbox']) for epoch in epochs]
avg_train_class_error = [np.mean(metrics_by_epoch[epoch]['train_class_error']) for epoch in epochs]

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot train_loss and train_loss_points on the left y-axis
ax1.plot(epochs, avg_train_loss, label='Train Loss', color='blue')
ax1.plot(epochs, avg_train_loss_points, label='Train Loss Bbox', color='green')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a secondary y-axis for train_class_error
ax2 = ax1.twinx()
ax2.plot(epochs, avg_train_class_error, label='Train Class Error', color='red')
ax2.set_ylabel('Class Error (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

# Adjust layout and add title
fig.tight_layout()
plt.title('Average Training Metrics over Epochs')
plt.show()