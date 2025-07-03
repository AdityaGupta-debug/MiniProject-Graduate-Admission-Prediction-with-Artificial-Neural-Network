# 🎓 Graduate Admission Prediction using Artificial Neural Network

This project focuses on predicting graduate admissions based on various academic and personal parameters using an **Artificial Neural Network (ANN)**. The dataset used is the **Graduate Admission 2** dataset available on Kaggle.

---

## 📊 About the Dataset

The [Graduate Admission 2](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) dataset contains information about students' academic records and test scores, which are commonly used by graduate schools to assess applicants.

The target variable is the **Chance of Admit**, a continuous value between 0 and 1 indicating the likelihood of a student's admission into a graduate program.

---

## 🔍 Dataset Features

| Feature               | Description                                         |
|----------------------|-----------------------------------------------------|
| `GRE Score`          | Graduate Record Examination score (out of 340)      |
| `TOEFL Score`        | Test of English as a Foreign Language score (out of 120) |
| `University Rating`  | Rating of the university (1 to 5)                   |
| `SOP`                | Statement of Purpose strength (1 to 5)              |
| `LOR`                | Letter of Recommendation strength (1 to 5)          |
| `CGPA`               | Undergraduate GPA (on a scale of 10)               |
| `Research`           | Research experience (0 = No, 1 = Yes)              |
| `Chance of Admit`    | Target variable (0.0 to 1.0)                        |

---

## 📊 Exploratory Data Analysis (EDA)

Before training the model, I performed thorough EDA to understand the relationships between features and the target variable.  
This involved plotting various charts such as:

- Histograms for feature distributions  
- Correlation heatmap  
- Scatter plots between CGPA, GRE, TOEFL vs. Chance of Admit  
- Boxplots to visualize effect of research and SOP/LOR on admissions  

These visual insights helped in understanding the data trends and informed feature selection for the ANN model.

---

## 🧠 Model: Artificial Neural Network

An ANN model was built using **TensorFlow** and **Keras** to predict the `Chance of Admit` based on the provided features. The ANN helps capture non-linear relationships in the data more effectively than traditional linear regression.

### ⚙️ Model Architecture

- **Input Layer:** 7 input neurons (1 for each feature)
- **Hidden Layers:** 
  - Dense(64, activation='relu')
  - Dense(32, activation='relu')
- **Output Layer:** 
  - Dense(1, activation='linear') — for regression output
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Metrics:** Mean Absolute Error (MAE)

### 🧪 Evaluation

The model was evaluated using metrics like **MSE**, **MAE**, and **R² Score**. It performed well on the test set, accurately predicting admission chances for most students.

---

## 📈 Results

- **Train MAE:** ~0.04  
- **Test MAE:** ~0.05  
- **R² Score:** **0.77** (77%)

✅ This means the model explains **77% of the variance** in the graduate admission outcomes — a strong performance for a regression task with limited features.

---


---

## 💻 Requirements

Add these to `requirements.txt`:

