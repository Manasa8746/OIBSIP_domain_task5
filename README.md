# OIBSIP_domain_task5
Task 5: SALES PREDICTION USING PYTHON
📊 Project Title: Sales Prediction Using Python
🎯 Objective:
   The objective of this project is to predict the sales of a product based on the amount spent on different advertisement channels such as TV, Radio, and Newspaper.
   This helps businesses make data-driven decisions about advertising strategies to maximize revenue.
⚙️ Steps Performed:
   Import Libraries:
      Imported essential Python libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn for data analysis, visualization, and modeling.
   Load Dataset:
      Loaded the Advertising.csv dataset using pandas to explore advertising and sales data.
   Data Exploration:
      Displayed dataset info, summary statistics, and checked for missing values.
      Used heatmap and pair plots to analyze feature correlations with sales.
   Feature Selection:
      Selected TV, Radio, and Newspaper as input features and Sales as the target variable.
   Data Splitting:
      Split the dataset into 80% training and 20% testing sets using train_test_split().
   Model Training:
      Trained a Linear Regression model to understand the relationship between advertisement spending and sales.
   Prediction & Evaluation:
      Made predictions using the test dataset.
   Evaluated model performance using metrics:
     R² Score
     Mean Absolute Error (MAE)
     Mean Squared Error (MSE)
  Visualization:
     Created a scatter plot to compare Actual vs Predicted Sales for visual performance verification.
  Custom Prediction:
     Predicted sales for new advertising inputs (e.g., TV=150, Radio=20, Newspaper=15).
🧰 Tools Used:
Tool / Library	Purpose
Python	Programming Language
Pandas, NumPy	Data Handling and Analysis
Matplotlib, Seaborn	Data Visualization
Scikit-learn	Machine Learning Model & Metrics
Jupyter Notebook / VS Code	Code Execution Environment

📈 Output / Results:
  R² Score: ~0.90 → indicates 90% accuracy in predicting sales.
  Mean Absolute Error: Low value → minimal prediction error.
  Observation:
   TV and Radio advertisement spending have the most impact on sales.
   Newspaper advertising shows lesser influence.
   The model can accurately predict future sales based on new advertising budgets.
   Generated a scatter plot showing strong alignment between actual and predicted val
