Predicting Extreme Weather in Madrid
Principal Component Analysis & Regression Forecasting (IDS Assignment 2)

This project models tomorrow’s maximum temperature in Madrid using historical meteorological data from the Copernicus CERRA dataset.
It combines exploratory analysis, PCA, Principal Component Regression (PCR), linear regression, bootstrap inference, and decile-level performance evaluation to understand and forecast temperature extremes.

This project was developed as part of the course FEM11149 – Introduction to Data Science (ESE, Erasmus University Rotterdam).

Repository Structure
.
├── data/
│   └── a2_data_group_22.csv            # Dataset (raw)
│
├── report/
│   ├── Assignment2_IDS.Rmd            # Full analysis (source code + narrative)
│   └── Assignment2_IDS.pdf            # Knitted report (final output)
│
└── README.md                          # Project documentation

🔍 Project Overview

The goal of this project is to forecast next-day maximum temperature using today’s weather conditions, exploring:

Which meteorological variables are most predictive

Whether PCA improves forecasting performance

How PCR compares to a standard linear regression

How prediction accuracy changes across cold vs hot temperature deciles

Whether extreme weather events are harder to predict

The full analysis is available in the RMarkdown report.

🧪 Methods & Techniques
1. Data Setup

Cleaned and renamed variables for clarity

Constructed a supervised learning dataset where
X = today’s weather,
y = tomorrow’s maximum temperature

2. Correlation Analysis

Temperature variables show the strongest predictive power

Strong multicollinearity across meteorological variables → justification for PCA

3. Principal Component Analysis (PCA)

Calculated principal components on training data

Chose optimal number of components using:

Cumulative variance explained

Kaiser criterion (eigenvalues > 1)

Scree plot

Interpreted PC1 and PC2 using a biplot

Bootstrap confidence intervals for PC1 variance ratio

4. Predictive Models

Principal Component Regression (PCR)

Multiple Linear Regression (baseline)

Sensitivity analysis: 3, 4, and 5 components

Evaluation metrics:

RMSE

MAE

R²

5. Decile MSE Analysis

Divided test observations into 10 temperature deciles

Compared model performance on:

coldest days

mid-range days

hottest days

📊 Key Findings

Today’s maximum temperature is the strongest predictor of tomorrow’s (r ≈ 0.96).

The first 4 principal components explain ~87% of variance.

Linear Regression outperformed all PCR models, with:

lower RMSE & MAE

higher R²

PCR’s performance is sensitive to the number of components chosen.

All models struggle with rare extreme temperatures (very hot or very cold).

The decile MSE plot shows:

PCR(3) performs best on the coldest days

Linear Regression performs best on the hottest days (most relevant for extreme weather forecasting)

▶️ How to Reproduce the Analysis
Requirements

Install required packages:

install.packages(c(
  "pacman", "dplyr", "tibble", "knitr", "readr", 
  "tidyr", "ggplot2", "boot", "pls"
))


Or load using pacman:

pacman::p_load(dplyr, tibble, knitr, readr, tidyr,
               ggplot2, boot, pls)

Steps

Clone this repository

Open Assignment2_IDS.Rmd in RStudio

Make sure the dataset is located at:

data/a2_data_group_22.csv


Click Knit to generate the full report

🧠 Skills Demonstrated

Data cleaning & preprocessing

Exploratory analysis & visualization

PCA, eigenvalues, loadings, biplots

Bootstrap inference

Regression modeling (PCR & LM)

Handling time-ordered data

Model selection & sensitivity analysis

Decile-based model diagnostics

Reproducible reporting in RMarkdown

👩‍💻 Authors

Daphne

Riya

Vera

Zsófi

(Erasmus School of Economics, Data Science & Marketing Analytics MSc)
