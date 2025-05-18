# 🧠 Mental Health Detection using Machine Learning

This project explores how machine learning algorithms can detect mental health issues based on social media and survey data.

## 📄 Description
The goal of this project is to assess different ML models such as:
- Support Vector Machines (SVM)
- Logistic Regression
- Random Forest
- Naive Bayes

It uses a labeled dataset of 27,000+ samples with text and binary classification (0 = no issue, 1 = mental health issue).

## 📁 Project Structure
```
mental-health-detection-ml/
├── data/                  # Contains the dataset
├── notebook/              # Jupyter notebook for ML modeling
├── docs/                  # PDF report
├── requirements.txt       # Python dependencies
├── README.md              # Project overview
└── .gitignore             # Files to exclude from Git
```

## 📊 Dataset
The dataset includes:
- `text`: social media post or comment
- `label`: 0 (no issue) or 1 (mental health issue)

## 📈 Results
SVM and Logistic Regression achieved the highest accuracies:
- SVM: 92%
- Logistic Regression: 91%

## 📚 References
Based on the research paper by Abdullah Al Hadi and Mohammad Saiful Arefin from AIU.

## 🚀 How to Run
1. Clone the repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `Final.ipynb` in Jupyter and run cells.
