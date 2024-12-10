import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Đọc dữ liệu từ file CSV
data = pd.read_csv("data\metrics\detection_metrics.csv")

# Tính trung bình các độ đo
average_accuracy = data['accuracy'].mean()
average_recall = data['recall'].mean()
average_precision = data['precision'].mean()
average_f1_score = data['f1_score'].mean()

print(f"Average Accuracy: {average_accuracy}")
print(f"Average Recall: {average_recall}")
print(f"Average Precision: {average_precision}")
print(f"Average F1 Score: {average_f1_score}")
averages = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score'],
    'Average': [average_accuracy, average_recall, average_precision, average_f1_score]
})

averages.to_csv("data\plots\Average_metrics.csv", index=False)

# Lấy confusion matrix từ file CSV
# Chuyển đổi chuỗi thành mảng numpy
def parse_confusion_matrix(matrix_str):
    # Loại bỏ dấu ngoặc vuông và chuyển đổi thành danh sách các số nguyên
    matrix_str = matrix_str.strip('[]')
    matrix_list = [int(x) for x in matrix_str.split()]
    # Chuyển đổi danh sách thành mảng numpy
    return np.array(matrix_list).reshape(2, 2)

confusion_matrices = data['confusion_matrix'].apply(parse_confusion_matrix)

# Tính trung bình confusion matrix
average_confusion_matrix = np.mean(confusion_matrices.tolist(), axis=0)
print("Average Confusion Matrix:", average_confusion_matrix)
# Vẽ confusion matrix
rounded_confusion_matrix = np.round(average_confusion_matrix).astype(int)
print("Rounded Confusion Matrix:", rounded_confusion_matrix)
fig, ax = plt.subplots()
cax = ax.matshow(rounded_confusion_matrix, cmap=plt.cm.Blues)
plt.title('Average Confusion Matrix')
fig.colorbar(cax)

# Thêm nhãn cho các ô
for (i, j), val in np.ndenumerate(rounded_confusion_matrix):
    ax.text(j, i, f'{int(val)}', ha='center', va='center')

# Thêm nhãn cho các trục
ax.set_xticklabels(['', 'Malicious', 'Benign'])
ax.set_yticklabels(['', 'Malicious', 'Benign'])

plt.xlabel('Predicted')
plt.ylabel('True')
# Lưu kết quả ra file
plt.savefig("data\plots\confusion_matrix_detection_plot.png")
plt.show()