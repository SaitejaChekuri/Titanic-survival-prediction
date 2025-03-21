# **Titanic Analysis with TensorFlow Decision Forests 🚢**

## **Overview**
This project applies **Gradient Boosted Trees (GBT)** from **TensorFlow Decision Forests (TF-DF)** to analyze the famous **Titanic dataset**. The goal is to predict passenger survival based on key features and evaluate feature importance, tree structure, and model performance.

## **Why TensorFlow Decision Forests?**
Decision forests, including Gradient Boosted Trees, are powerful for tabular data analysis. TF-DF provides:
- **Automated feature selection**  
- **High interpretability** (unlike deep learning models)  
- **Robust handling of missing and categorical data**  
- **Faster training with fewer hyperparameters**  

## **Project Goals**
- Build and train a **Gradient Boosted Trees model** using **TF-DF**  
- Analyze **feature importance** to understand which variables impact survival  
- Evaluate the **model structure** (tree depth, number of nodes)  
- Measure the **validation loss** to assess model performance  

---

## **Dataset**
The **Titanic dataset** contains passenger details such as:
- **Survival** (`0` = No, `1` = Yes)  
- **Passenger Class (Pclass)** (1st, 2nd, or 3rd)  
- **Sex** (Male/Female)  
- **Age**  
- **SibSp** (Number of siblings/spouses aboard)  
- **Parch** (Number of parents/children aboard)  
- **Fare** (Ticket price)  
- **Embarked** (Port of Embarkation: C, Q, S)  

---

## **Feature Importance**
Feature importance measures how much each variable contributes to predictions.  
### **Key Insights:**
1. **Sex** – The most important factor (Females had a higher survival rate).  
2. **Pclass** – Higher-class passengers had better survival chances.  
3. **Age & Fare** – Moderate importance, influencing survival likelihood.  
4. **SibSp & Parch** – Less significant but still affecting survival.  
5. **Name, Ticket, and Embarked** – Least important features.

---

## **Model Details**
- **Algorithm:** Gradient Boosted Trees (TF-DF)  
- **Number of Trees:** 36  
- **Total Nodes:** 1836  
- **Average Nodes per Tree:** 51  
- **Max Depth:** 5  
- **Loss Function:** Binomial Log Likelihood  
- **Validation Loss:** 1.04625 (Lower is better)  

---

## **Installation & Setup**
### **1. Clone the Repository**
```bash
git clone https://github.com/SaitejaChekuri/Titanic-TFDF-Analysis.git
cd Titanic-TFDF-Analysis
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter Notebook**
```bash
jupyter notebook
```
Then, open **Titanic_analysis.ipynb** and execute all cells.

---

## **Libraries Used**
- **TensorFlow Decision Forests** (`tensorflow_decision_forests`)  
- **Pandas** (`pandas`)  
- **NumPy** (`numpy`)  
- **Matplotlib & Seaborn** (for data visualization)  
- **Scikit-learn** (`sklearn`) (for data preprocessing)

---

## **Results & Findings**
- **Gradient Boosted Trees performed well on the dataset, highlighting key survival factors.**  
- **Females, younger passengers, and those in higher classes had a higher survival rate.**  
- **Decision forests provided an interpretable approach to survival prediction.**  

---

## **Future Improvements**
- **Hyperparameter tuning** to optimize model performance.  
- **Incorporating additional features** like family size and cabin location.  
- **Experimenting with other ML models** (Random Forest, XGBoost).  

---

## **Author**
👤 **Saiteja Chekuri**  
📌 [GitHub](https://github.com/SaitejaChekuri) | [LinkedIn](https://linkedin.com/in/saiteja-chekuri-b38465220)
