# 🎓 Graduate Admission Prediction using Artificial Neural Network

This project focuses on predicting graduate admissions based on various academic and personal parameters using an **Artificial Neural Network (ANN)**. The dataset used is the **Graduate Admission 2** dataset available on Kaggle.

---

## 📊 About the Dataset

The [Graduate Admission 2](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) dataset contains information about students' academic records and test scores, which are commonly used by graduate schools to assess applicants.

The target variable is the **Chance of Admit**, a continuous value between 0 and 1 indicating the likelihood of a student's admission into a graduate program.

---

## 🔍 Dataset Features

| Feature               | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `GRE Score`            | Graduate Record Examination score (out of 340)               |
| `TOEFL Score`          | TOEFL score (out of 120)                                     |
| `University Rating`    | University Rating (1 to 5)                                   |
| `SOP`                  | Strength of Statement of Purpose (1 to 5)                    |
| `LOR`                  | Strength of Letter of Recommendation (1 to 5)                |
| `CGPA`                 | Undergraduate GPA (on a scale of 10)                         |
| `Research`             | Research Experience (0 = No, 1 = Yes)                        |
| `Chance of Admit`      | Target variable (probability of admission, 0.0 to 1.0)       |

---

## 📊 Exploratory Data Analysis (EDA)

Before training the model, I performed thorough EDA to understand the relationships between features and the target variable.  
This involved plotting various charts such as:

- Histograms for feature distributions  
- Correlation heatmap  
- Scatter plots between CGPA, GRE, TOEFL vs. Chance of Admit  
- Histplots to visualize the effect of Research and SOP/LOR on admissions  

These visual insights helped in understanding the data trends and informed feature selection for the ANN model.

---

## 🧠 Model: Artificial Neural Network

An ANN model was built using **TensorFlow** and **Keras** to predict the `Chance of Admit` based on the provided features. The ANN helps capture non-linear relationships in the data more effectively than traditional linear regression.

### ⚙️ Model Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(7,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
```

- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  


---

## 🧪 Evaluation

The model was evaluated using the following metrics:

- **Mean Squared Error (MSE)**
- **R² Score (Coefficient of Determination)**

---

## 📈 Results

- **Train MAE:** ~0.04  
- **Test MAE:** ~0.05  
- **R² Score:** **0.77** (77%)

✅ This means the model explains **77% of the variance** in the graduate admission outcomes — a strong performance for a regression task with limited features.

---

## 🧰 Tools & Libraries Used

- Python 
- Pandas & NumPy  
- Seaborn & Matplotlib  
- Scikit-learn  
- TensorFlow & Keras  

---

## 📌 Conclusion

The ANN model successfully predicts the probability of graduate admission based on academic and personal features. With proper feature scaling, EDA, and a well-structured neural network, the model achieves robust performance on the test data.

---
