# Predicting Extreme Weather in Madrid  
### Principal Component Analysis & Regression Forecasting

This project builds a next-day temperature forecasting model for **Madrid**, using meteorological data from the **Copernicus CERRA** dataset.  
It combines exploratory analysis, PCA, Principal Component Regression (PCR), linear regression, bootstrap inference, and a decile-based error study to assess how well models predict extreme temperatures.

Developed for **FEM11149 – Introduction to Data Science**  
*Erasmus School of Economics, Erasmus University Rotterdam*

---

## Repository Structure

- data/
     - a2_data_group_22.csv # Raw dataset
- report/
     - Assignment2_IDS.Rmd # Full analysis (source + narrative)
     - Assignment2_IDS.pdf # Final knitted report
- README.md


---

## Overview

The goal is to forecast **tomorrow’s maximum temperature** from **today’s meteorological conditions**.  
The analysis explores:

- Which variables are most predictive  
- Whether PCA improves model stability  
- PCR versus standard linear regression  
- Performance across cold, mid, and hot temperature deciles  
- How well models predict temperature extremes  

---

## Methods

### **Data Preparation**
- Renamed and standardized variables  
- Created a supervised learning structure:  
  **X = today’s weather**, **y = tomorrow’s maximum temperature**

### **Correlation Analysis**
- Strong relationship between today’s and tomorrow’s max temperature  
- High multicollinearity → motivates PCA  

### **Principal Component Analysis (PCA)**
- Chose number of PCs via:
  - cumulative variance  
  - Kaiser (> 1)  
  - scree plot  
- Interpreted PC1–PC2 with a biplot  
- Bootstrap CI for PC1 variance ratio  

### **Predictive Models**
- Principal Component Regression (PCR)  
- Multiple Linear Regression (baseline)  
- Sensitivity analysis: 3, 4, and 5 components  
- Evaluation with RMSE, MAE, R²  

### **Decile MSE Analysis**
- Split test data into 10 temperature deciles  
- Compared performance for:  
  - ❄️ coldest days  
  - 🌤️ mid-range days  
  - 🔥 hottest days  

---

## Key Findings

- Today’s maximum temperature is an **excellent predictor** (r ≈ 0.96).  
- First **4 PCs explain ~87%** of variance.  
- **Linear regression outperformed all PCR models**, with lower RMSE, lower MAE, and higher R².  
- PCR performance is sensitive to component choice.  
- All models struggle more with **extreme** temperatures.  
- PCR(3) performs best on the coldest days.  
- **Linear regression performs best on the hottest days**, which matter most for extreme-weather prediction.

---

## Reproducing the Analysis

### **Install packages**

```r
install.packages(c(
  "pacman", "dplyr", "tibble", "knitr", "readr",
  "tidyr", "ggplot2", "boot", "pls"
))
```
---

## Skills Demonstrated

- Data cleaning
- Exploratory analysis
- PCA, loadings, eigenvalues
- Bootstrap inference
- PCR & linear regression modeling
- Chronological train/test splitting
- Sensitivity analysis
- Decile-based diagnostics
- Reproducible RMarkdown reporting

---
## Authors

- Daphne
- Riya
- Vera
- Zsófi

**MSc Data Science & Marketing Analytics, Erasmus School of Economics**
