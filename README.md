# 🚔 Houston Crime Type Classification (2010–2024)

**University of Houston–Clear Lake | Jan 2025 – Apr 2025**

An end-to-end machine learning pipeline designed to predict crime categories (NIBRS Description) using 14 years of Houston Police Department data (~3.6 million records). The project simulates real-world deployment by using **2024 as a true future holdout year**.

---

## 📌 Project Objective

The goal of this project is to predict the **type of crime (NIBRS Description)** based on time and location-based features.

### Target Variable
- `NIBRS Description` (Crime Category)

### Features Used
- Beat (Geographic patrol area)
- Premise (Location type: home, street, business, etc.)
- Month
- Weekday
- Hour Group (6-hour time buckets)

Instead of using a random train/test split, this project uses:

- **Training Data:** 2010–2023  
- **Testing Data (Holdout):** 2024  

This approach better reflects real-world prediction scenarios.

---

## 📊 Dataset Overview

- Source: Houston Police Department public crime data
- Time Range: January 2010 – 2024
- Original Size: ~3.6 million records
- Multi-label crime entries expanded into single-label rows
- Final modeling dataset balanced across top 5 crime types

---

## 🧹 Data Engineering Pipeline

### 1️⃣ Multi-Source Integration
Merged three differently formatted data sources:
- Monthly Excel files (2010–2018)
- Mid-2018 supplemental dataset
- 2019–2024 NIBRS dataset

Standardized:
- Column names
- Offense count fields
- Date formats
- Removed irrelevant and duplicate columns

---

### 2️⃣ Multi-Label Expansion
Some crime records contained multiple categories (e.g., `"Theft, Assault"`).  
These were split into separate rows to ensure proper supervised classification.

---

### 3️⃣ Feature Engineering

From the timestamp:
- Extracted Year
- Extracted Month
- Extracted Weekday
- Created `Hour_Group` (4 six-hour buckets)

Final feature set:
Beat
Premise
Month
Weekday
Hour_Group


---

## ⚖️ Handling Class Imbalance

Crime data is naturally imbalanced. To ensure fair training:

- Selected top 5 most frequent crime types (based on 2024)
- Used stratified sampling: **20,000 samples per class**
- Created balanced datasets:

| Split | Years | Rows |
|--------|--------|--------|
| Train | 2010–2023 | 100,000 |
| Test  | 2024 | 100,000 |

---

## 🧠 Models Implemented

All models were built using **scikit-learn pipelines** with:

- Imputation
- One-hot encoding
- 3-fold stratified cross-validation

### Models Compared
1. **Neural Network (MLPClassifier)**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**

---

## 📈 Results

| Model | CV Accuracy | Test Accuracy (2024) | Macro AUC |
|--------|-------------|---------------------|------------|
| **Neural Network (MLP)** | 0.3052 | **0.2937** | **0.5895** |
| Random Forest | 0.2279 | 0.2911 | 0.5835 |
| KNN | 0.1848 | 0.2414 | 0.5412 |

### 🏆 Final Model Selected: Neural Network

Selected because:
- Highest test accuracy
- Highest macro AUC
- Better balance across crime categories
- Stronger nonlinear pattern detection

---

## 📊 Evaluation Methods

- 3-fold Stratified Cross Validation
- Confusion Matrices
- One-vs-Rest ROC Curves
- Macro AUC (class-balanced performance metric)

Macro AUC was prioritized to ensure fairness across multiple crime categories.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

---

## 🔬 Key Learnings

- Real-world datasets require extensive cleaning and schema alignment.
- Multi-label expansion is crucial for proper supervised learning.
- Stratified sampling improves fairness across classes.
- Temporal holdout validation better reflects production deployment.
- Neural networks capture complex time-location interactions effectively.

---

## 🚀 Future Improvements

- Incorporate geospatial clustering (latitude/longitude)
- Experiment with XGBoost or LightGBM
- Add socioeconomic or weather data features
- Deploy as a crime prediction API
- Explore deep embedding approaches

---

## 👨‍💻 Author

**Govardhan Reddy Narala**  
M.S. Data Science  
University of Houston–Clear Lake  




