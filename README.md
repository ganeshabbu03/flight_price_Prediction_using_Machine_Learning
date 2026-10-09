# Flight Ticket Price Prediction & Analytics Web App

An end-to-end interactive web application built with **Streamlit** and **Scikit-Learn (Linear Regression)** based on the flight booking dataset described in `problem statement.docx` and raw datasets (`business.csv` and `economy.csv`).

---

## 🌟 Key Features

1. **Comprehensive 5-Stage EDA Process**:
   - **Stage 1 (Data Quality Audit)**: Total record count, schema types, missing value audit, and duplicate detection.
   - **Stage 2 (Target & Outlier Analysis)**: Ticket price distributions across cabins and log-price transformations.
   - **Stage 3 (Categorical Breakdown)**: Fare comparisons across Airline companies, Cabin Classes, Stops, and Departure time windows.
   - **Stage 4 (Numerical Correlations & Metro Route Matrix)**: Pearson correlation matrix and Metro City-to-City route pricing heatmaps.
   - **Stage 5 (Booking Dynamics)**: Booking lead time trajectory showing surge pricing patterns.

2. **Step-by-Step Linear Regression Studio**:
   - **Flow of Execution & Mathematical Foundations**:
     - Interactive flowchart mapping all 7 execution stages: `Data Ingestion` ➔ `Dummy Encoding` ➔ `Train/Test Partition` ➔ `Loss Minimization` ➔ `Weight Estimation` ➔ `Evaluation Metrics` ➔ `Residual Diagnostics`.
     - **Mathematical Formulations**:
       - Multiple Linear Regression Hypothesis: $\mathbf{\hat{Y}} = \mathbf{X} \boldsymbol{\beta}$
       - OLS Objective Loss Function: $J(\boldsymbol{\beta}) = \frac{1}{2n} \|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2$
       - Normal Equation Analytical Closed-Form: $\boldsymbol{\hat{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$
       - Regularized Losses (Ridge $L_2$ & Lasso $L_1$)
       - Evaluation formulas ($R^2$, Adjusted $R^2$, MAE, MSE, RMSE).
   - **Step 1 - Prepare Data (Flow: Ingestion & Feature Transformation)**: Review uploaded/sampled rows, target definitions, and one-hot encoding summary.
   - **Step 2 - Train/Test Split (Flow: Partitioning & Validation Guard)**: Configurable test share ratio with random seed reproducibility.
   - **Step 3 - Train & Inspect Coefficients (Flow: Weight Optimization)**:
     - Feature coefficients table, directional impact (Positive/Negative), model intercept, and CSV export.
     - **Linear Regression Model Calculation**:
       - Full fitted regression equation ($\hat{Y} = \beta_0 + \beta_1 x_1 + \dots$).
       - Step-by-step dot product vector calculation table showing feature values, beta coefficients, individual dollar contributions ($x_j \cdot \beta_j$), final predicted price ($\hat{y}$), ground truth price ($y$), and residual error ($e$).
   - **Step 4 - Test-Set Evaluation (Flow: Generalization Scoring)**:
     - $R^2$, Adjusted $R^2$, MAE, MSE, and RMSE summary metrics.
     - **Detailed Step-by-Step Mathematical Calculations**:
       - Displays empirical test set quantities: Test samples ($n$), Features ($p$), Mean fare ($\bar{y}$), $\text{SSR}$, $\text{SST}$, and $\text{SAE}$.
       - Shows step-by-step plugged-in formulas with real dataset numbers for $R^2$, Adjusted $R^2$, MAE, MSE, and RMSE.
   - **Step 5 - Predictions & Residuals (Flow: Residual Error Inspection)**: Test sample preview displaying Actual Price, Predicted Price, and Residual Errors with downloadable CSV.

3. **Live Price Predictor**:
   - Real-time fare estimator with statistical Confidence Interval ($\pm \text{MAE}$).
   - Configurable route, airline, cabin class, stops, duration, and booking lead days.

4. **Model Tester & Statistical Diagnostics**:
   - Actual vs. Predicted scatter plot with ideal 1:1 reference line.
   - Residuals vs. Predicted homoscedasticity plot.
   - Residual error distribution histogram & box plot for normality verification.
   - 5-Fold cross-validation consistency and error tolerance bands ($\pm ₹5,000$ and $\pm ₹10,000$).

5. **Live Price Predictor**:
   - Real-time fare estimator with **Confidence Interval (± MAE)** bounds.
   - Customizable airline, origin/destination cities, stops, class, flight duration, and booking lead time.

6. **Exploratory Data Analysis (EDA)**:
   - Interactive Plotly charts: Average fare by airline, Business vs Economy box plots, Booking lead time curve, Stops comparison, and City-to-City Route heatmap.

7. **Raw & Processed Data View**:
   - Filterable table and one-click processed CSV download.

---

## 🚀 How to Run the App

Open PowerShell or Command Prompt in the project folder and run:

```powershell
streamlit run app.py
```

Or using the full Python executable path:
```powershell
python -m streamlit run app.py --server.port 8502
```

The app will open automatically in your browser at:
`http://localhost:8502`
