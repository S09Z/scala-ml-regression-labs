# 🏡 House Price Prediction using Scala & Apache Spark

## 📌 Project Overview

This project implements a **House Price Prediction Model** using **Scala** and **Apache Spark MLlib**.\
It uses **Linear Regression** to predict house prices based on **Size (sqft)** and **Bedrooms**.

---

## 📂 Project Structure

```
house-price-prediction/
│—— build.sbt                                # SBT configuration file
│—— data/
│   ├—— housing.csv                          # Dataset containing house prices
│—— src/
│   ├—— main/
│   │   ├—— scala/
│   │   │   ├—— HousePricePrediction.scala   # Main Scala file for ML Model
│—— README.md                                # Project documentation
```

---

## ⚙️ Installation & Setup

### **1⃣ Install Dependencies**

Ensure you have **Apache Spark** and **Scala SBT** installed.

#### **For Mac/Linux (Homebrew)**

```sh
brew install apache-spark sbt
```

#### **For Ubuntu**

```sh
sudo apt update && sudo apt install -y default-jdk scala sbt
```

### **2⃣ Clone This Repository**

```sh
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### **3⃣ Run SBT Setup**

```sh
sbt clean compile
```

---

## 🚀 How to Run

### **1⃣ Train and Predict House Prices**

```sh
sbt run
```

### **2⃣ Expected Output**

```
🚀 Spark Session Started!
+-----------+--------+--------+
|Size (sqft)|Bedrooms|Price ($)|
+-----------+--------+--------+
|       1500|       3|  300000|
|       1800|       4|  350000|
|       1200|       2|  200000|
+-----------+--------+--------+

Root Mean Squared Error (RMSE): 24000.5
```

---

## 📊 Model Details

- **Algorithm:** Linear Regression
- **Library:** Apache Spark MLlib
- **Features Used:** `Size (sqft)`, `Bedrooms`
- **Metric Used:** Root Mean Squared Error (RMSE)

---

## 🎯 Future Improvements

- ✅ Add **Feature Engineering** (Normalization, Polynomial Features)
- ✅ Try **Random Forest Regression**
- ✅ Deploy as an **API using Play Framework**

---

## 📚 License

This project is licensed under the **MIT License**.