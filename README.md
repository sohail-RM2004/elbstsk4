<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">

</head>
<body>

<h1>Breast Cancer Classification — Logistic Regression</h1>
<p>Binary classification pipeline that predicts malignant (1) vs benign (0) tumors using the Breast Cancer Wisconsin dataset.</p>

<h2>Overview</h2>
<p>This project trains and evaluates a Logistic Regression classifier to detect malignant tumors. The pipeline includes data cleaning, encoding, scaling, model training, evaluation, and threshold tuning.</p>

<h2>Dataset</h2>
<ul>
  <li><strong>Source:</strong> Breast Cancer Wisconsin (CSV version)</li>
  <li><strong>Main columns used:</strong> <code>diagnosis</code>, <code>radius_mean</code>, <code>texture_mean</code>, <code>perimeter_mean</code>, <code>area_mean</code>, …, <code>fractal_dimension_worst</code></li>
  <li><strong>Dropped columns:</strong> <code>id</code>, <code>Unnamed: 32</code></li>
  <li><strong>Target encoding:</strong> <code>'M' → 1 (malignant), 'B' → 0 (benign)</code></li>
</ul>

<h2>Preprocessing Steps</h2>
<ul>
  <li>Drop non-informative columns: <code>id</code>, <code>Unnamed: 32</code>.</li>
  <li>Map target: <code>diagnosis = {'M': 1, 'B': 0}</code>.</li>
  <li>Split data: 80% train / 20% test with <code>stratify=y</code> to preserve class balance.</li>
  <li>Standardize numeric features using <code>StandardScaler</code> (fit on train only, transform train & test).</li>
</ul>

<h2>Modeling</h2>
<p>Trained a <code>LogisticRegression(max_iter=1000)</code> model on the standardized training set.</p>

<h3>Key Code</h3>
<pre>
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

# predict & eval
y_pred = model.predict(X_test_s)
y_prob = model.predict_proba(X_test_s)[:,1]
</pre>

<h2>Evaluation</h2>
<ul>
  <li><strong>ROC-AUC:</strong> 0.996031746031746</li>
  <li><strong>Precision:</strong> 0.97 </li>
  <li><strong>Recall:</strong> 0.93 </li>
  <li><strong>Confusion Matrix:</strong> [[71  1]
 [ 1 41]]</li>
</ul>

<h3>Metric meanings</h3>
<ul>
  <li><strong>Confusion matrix</strong> shows true/false positives & negatives — useful to see types of errors.</li>
  <li><strong>Precision</strong> = TP / (TP + FP) — few false alarms if high.</li>
  <li><strong>Recall</strong> = TP / (TP + FN) — few missed malignant cases if high.</li>
  <li><strong>ROC-AUC</strong> measures discriminative ability across thresholds (1.0 perfect, 0.5 random).</li>
</ul>

<h2>Threshold Tuning & Sigmoid</h2>
<p>Logistic Regression outputs a probability: <code>p = sigmoid(z) = 1 / (1 + e<sup>-z</sup>)</code>, where <code>z = w·x + b</code>. The default classifier threshold is 0.5 (predict malignant if p ≥ 0.5).</p>
<ul>
  <li>Lowering the threshold (e.g., 0.3) increases recall but reduces precision.</li>
  <li>For medical screening, high recall is often prioritized to avoid missed malignancies.</li>
</ul>

<pre>
# Example: tune threshold
y_prob = model.predict_proba(X_test_s)[:,1]
y_pred_custom = (y_prob >= 0.3).astype(int)
</pre>

<h2>Usage</h2>
<ol>
  <li>Place the dataset CSV in the project folder (e.g., <code>breast_cancer_data.csv</code>).</li>
  <li>Run the analysis script: <code>python breast_cancer_lr.py</code>.</li>
  <li>View the printed metrics in your console or save results to a file.</li>
</ol>


<p><em>Prepared by: Sohail — Breast Cancer Classification project</em></p>

</body>
</html>
