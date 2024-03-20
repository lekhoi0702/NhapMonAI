# Import các thư viện cần thiết
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dữ liệu Iris
iris = load_iris()
X = iris.data
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Khởi tạo mô hình Naive Bayes
model = GaussianNB()

# Huấn luyện mô hình với dữ liệu huấn luyện
model.fit(X_train, y_train)

# Dự đoán nhãn của dữ liệu kiểm tra
y_pred = model.predict(X_test)


# In ra các mẫu test cùng với kết quả dự đoán
print("Mẫu test\t\tDự đoán")
print("--------------------------------------")
for i in range(len(X_test)):
    print(f"{X_test[i]}\t{y_pred[i]}")

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print("Độ chính xác của mô hình: {:.2f}%".format(accuracy * 100))
