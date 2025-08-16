# 🌸 Flask KNN Iris Classifier
## 📌 Giới thiệu

Đây là một ứng dụng web đơn giản sử dụng Flask và scikit-learn để phân loại loài hoa Iris.
Ứng dụng cho phép người dùng nhập 4 thông số của hoa:

🌿 Sepal length (cm)

🌿 Sepal width (cm)

🌸 Petal length (cm)

🌸 Petal width (cm)

Sau khi nhập, hệ thống sẽ dự đoán loài hoa và hiển thị xác suất theo từng lớp.

## 🛠️ Công nghệ và công cụ sử dụng
Thành phần	Công nghệ sử dụng
Ngôn ngữ lập trình	Python 3
Web framework	Flask
Machine Learning	scikit-learn (Pipeline, StandardScaler, KNN)
Cơ sở dữ liệu	sklearn.datasets.load_iris()
Frontend	HTML, CSS, Bootstrap 5
Template engine	Jinja2 (tích hợp trong Flask)
## 🧠 Logic & thuật toán

Dữ liệu: Lấy từ sklearn.datasets.load_iris().

Xử lý dữ liệu: Chia train/test theo tỷ lệ 80/20.

Huấn luyện mô hình:

StandardScaler() để chuẩn hóa dữ liệu.

KNeighborsClassifier(n_neighbors=5) để phân loại.

Dự đoán: Khi người dùng nhập form → Model dự đoán → Trả về tên loài hoa và xác suất từng lớp.

## 🎨 Giao diện
🔹 Form nhập dữ liệu

Người dùng nhập 4 thông số hoa Iris và bấm Dự đoán.

🔹 Kết quả dự đoán

Hiển thị tên loài hoa dự đoán.

Hiển thị xác suất từng lớp bằng progress bar (Bootstrap 5).

## 📊 Ví dụ kết quả

Dự đoán: 🌼 Setosa

Xác suất:

Setosa: 98.5%

Versicolor: 1.2%

Virginica: 0.3%

🚀 Khởi chạy ứng dụng
 ```bash
## 1. Cài đặt thư viện
pip install flask scikit-learn numpy

## 2. Chạy ứng dụng Flask
python app.py

## 3. Truy cập trình duyệt
http://127.0.0.1:5000
```
## 💡 Ghi chú

Có thể thay đổi tham số n_neighbors trong KNN để quan sát sự khác biệt.

Nếu muốn truy cập từ thiết bị khác trong cùng mạng LAN, thay đổi:

app.run(host="0.0.0.0", port=5000)
