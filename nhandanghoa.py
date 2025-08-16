# app.py
from flask import Flask, request, render_template_string
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

app = Flask(__name__)

# ---- Train model (KNN) once at startup ----
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
model.fit(X_train, y_train)

# Quick test accuracy printed to console
acc = model.score(X_test, y_test)
print(f"KNN test accuracy: {acc:.3f}")

# ---- HTML (Bootstrap 5) ----
TEMPLATE = """
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Iris Classifier – KNN (Bootstrap 5)</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      body{background:#f7f7fb}
      .card{border:none; box-shadow:0 8px 30px rgba(0,0,0,.05)}
      .progress{height: 14px}
      .footer{font-size:.9rem; color:#666}
    </style>
  </head>
  <body>
    <div class="container py-5">
      <div class="row justify-content-center">
        <div class="col-lg-8">
          <div class="card p-4 p-md-5 rounded-4">
            <div class="d-flex justify-content-between align-items-center mb-4">
              <h1 class="h3 m-0">Nhận diện loài hoa Iris (KNN)</h1>
              <span class="badge text-bg-primary">Độ chính xác test: {{ accuracy }}</span>
            </div>
            <form method="post" action="/" class="row g-3">
              <div class="col-sm-6">
                <label class="form-label">Sepal length (cm)</label>
                <input type="number" step="0.01" class="form-control" name="sepal_length" placeholder="Nhập chiều dài đài hoa (cm)" required>
              </div>
              <div class="col-sm-6">
                <label class="form-label">Sepal width (cm)</label>
                <input type="number" step="0.01" class="form-control" name="sepal_width" placeholder="Nhập chiều rộng đài hoa (cm)" required>
              </div>
              <div class="col-sm-6">
                <label class="form-label">Petal length (cm)</label>
                <input type="number" step="0.01" class="form-control" name="petal_length" placeholder="Nhập chiều dài cánh hoa (cm)" required>
              </div>
              <div class="col-sm-6">
                <label class="form-label">Petal width (cm)</label>
                <input type="number" step="0.01" class="form-control" name="petal_width" placeholder="Nhập chiều rộng cánh hoa (cm)" required>
              </div>
              <div class="col-12 d-grid d-sm-flex gap-2 mt-2">
                <button type="submit" class="btn btn-primary px-4">Dự đoán</button>
                <a href="/" class="btn btn-outline-secondary">Nhập lại</a>
              </div>
            </form>

            {% if result %}
            <hr class="my-4">
            <h2 class="h5 mb-3">Kết quả</h2>
            <div class="alert alert-success" role="alert">
              Dự đoán: <strong>{{ result.label }}</strong>
            </div>
            <p class="mb-2">Xác suất theo từng lớp:</p>
            <ul class="list-unstyled">
              {% for name, prob in result.probs %}
              <li class="mb-2">
                <div class="d-flex justify-content-between"><span>{{ name }}</span><span>{{ '{:.1f}%'.format(prob*100) }}</span></div>
                <div class="progress">
                  <div class="progress-bar" role="progressbar" style="width: {{ prob*100 }}%" aria-valuenow="{{ prob*100 }}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
              </li>
              {% endfor %}
            </ul>
            {% endif %}

            <div class="footer mt-4">
              <p class="mb-1">Mô hình: <code>StandardScaler → KNeighborsClassifier(k=5)</code></p>
              <p class="mb-0">Dữ liệu: <code>sklearn.datasets.load_iris()</code> – 4 biến: sepal_length, sepal_width, petal_length, petal_width.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
"""

# ---- Routes ----
@app.route("/", methods=["GET", "POST"]) 
def index():
    result = None

    if request.method == "POST":
        try:
            sl = float(request.form.get("sepal_length"))
            sw = float(request.form.get("sepal_width"))
            pl = float(request.form.get("petal_length"))
            pw = float(request.form.get("petal_width"))
            sample = np.array([[sl, sw, pl, pw]])
            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0]

            result = {
                "label": str(class_names[pred]).title(),
                "probs": list(zip([n.title() for n in class_names], map(float, proba)))
            }
        except Exception as e:
            result = {"label": f"Lỗi: {e}", "probs": []}

    return render_template_string(
        TEMPLATE,
        result=result,
        accuracy=f"{acc*100:.1f}%"
    )


if __name__ == "__main__":
    # Set host="0.0.0.0" nếu muốn truy cập từ máy khác trong LAN
    app.run(debug=True)