# Heather Moore Data Science Portfolio

  Hello! My name is Heather Moore and this is my professional portfolio. I graduated with Bachelors degree in economics with a minor in data science at Iowa State University. Currently I am finishing my Masters (MS) in Data Science. During my academic career I gained proficiency in the programming languages Python, R, and SQL. 

  Currently, I hold a civilian service position, Operations Research Analyst,for the Air Force. At the Studies and Analysis Squadron, my team and I gather data across the Air Education and Training Command to conduct analyze as well as create products for our customers. Our job is to create accurate yet useful analytics to aid non-technical customers in automating daily tasks  and making data-driven decision. This role helped me improve my coding in all of my known languages as well help me gain experience in creating dashboards through Power BI and Envision by Palantir. 

  With my studies and work experience, I honed my ability to work with complex data and developed a keen eye for identifying patterns and trends. I also gained experience in data management and statistical analysis, which I believe will be valuable assets moving forward in my career.

## [Resume](https://github.com/hmoore200/Portfolio/blob/885d61afa387a1c47df1acc93fa2e9811d22722e/Resume_2025.pdf)


## Projects: 
## [Natural Language Processing - Airline Reviews](https://github.com/hmoore200/Portfolio/blob/main/Milestone%202%20-Natural%20Language%20Processing.ipynb)

### Overview:
This project leverages Natural Language Processing to classify IMDB movie reviews as positive or negative. The workflow includes cleaning and preprocessing text, feature extraction with TF-IDF, and training several classification models.

### Key Features:

Preprocessing: Removal of punctuation, stopwords, stemming, and tokenization.

Feature Extraction: TF-IDF vectorization to represent text numerically.

Models Used: Naive Bayes and Logistic Regression.

Visualization: Word clouds and confusion matrix for interpretability.

### Tools & Libraries:
pandas, scikit-learn, nltk, wordcloud, matplotlib

### Results:

Best Accuracy: ~88% using Logistic Regression with TF-IDF features.

Confusion Matrix: Demonstrated balanced performance on both positive and negative classes.

Insights: Frequent positive review words included “great”, “amazing”, while negative reviews featured “bad”, “worst” prominently.

## [Boston Housing Market Analysis](https://github.com/hmoore200/Portfolio/blob/main/Term%20Project%201-Boston%20Housing%20Market.ipynb) 

### Overview:
This regression project explores the Boston Housing dataset to build predictive models for median home prices. It involves exploratory analysis, multivariate regression, and model comparison.

### Key Features:

Data Analysis: Correlation matrix, pairplots, and feature importance visualization.

Modeling: Linear Regression, Decision Tree, Random Forest.

Evaluation Metrics: MAE, MSE, RMSE, R².

Tools & Libraries:
pandas, matplotlib, seaborn, scikit-learn

### Results:

Best Model: Random Forest Regressor with an R² score of ~0.87 on test data.

Top Features: Number of rooms (RM), % lower status population (LSTAT), and pupil-teacher ratio (PTRATIO).

Insight: More rooms and lower LSTAT values correlate with higher housing prices.


## [Walmart Time Series Analysis using ARIMA](https://github.com/hmoore200/Portfolio/blob/main/Milestone%203_%20Walmart%20Time%20Series.ipynb)

This project focuses on analyzing and forecasting weekly sales data from Walmart using time series modeling techniques, specifically the ARIMA (AutoRegressive Integrated Moving Average) model. The dataset includes various economic indicators such as fuel price, temperature, CPI, and unemployment, along with holiday flags for specific weeks.

### Key Components:

Data Cleaning & Exploration: Handled missing values, converted date columns, and visualized trends such as total weekly sales and store-level performance.

Exploratory Data Analysis (EDA): Generated visual insights through time plots, heatmaps, boxplots (holiday vs. non-holiday sales), and store comparison bar charts.

Time Series Modeling: Applied ARIMA models to both the full dataset and individual stores (e.g., Store #4). Forecasted future sales and evaluated model accuracy using Root Mean Squared Error (RMSE).

### Insights & Recommendations:

Sales consistently spike during holiday seasons.

High-performing stores should be studied to replicate success.

Store-specific modeling yielded better forecasting accuracy than overall modeling.

Tools & Libraries Used:

Python (Pandas, Matplotlib, Seaborn, Statsmodels)

Jupyter Notebook


## [Student Dropout Rate Analysis](https://github.com/hmoore200/Portfolio/blob/main/Student%20Dropout%20Rates.ipynb)
### Overview:
This classification project analyzes student demographic and academic data to predict the likelihood of dropouts. It aims to help educators intervene early with at-risk students.

### Key Features:

Data Cleaning: Handling missing values and encoding categorical variables.

Visualization: Heatmaps, boxplots, and distribution plots.

Models Used: Logistic Regression and Random Forest Classifier.

Evaluation Metrics: Accuracy, precision, recall, and F1-score.

Tools & Libraries:
pandas, matplotlib, seaborn, scikit-learn

### Results:

Best Model: Random Forest Classifier with accuracy around 91%.

Important Predictors: GPA, number of absences, parental education level.

Key Insight: Low GPA and high absenteeism were the strongest indicators of dropout risk.

## [Employee Attrition Prediction with Logistic Regression](https://github.com/hmoore200/Portfolio/blob/main/Employee%20Attrition.ipynb)

### Goal:
To identify the key factors that influence employee attrition and build a logistic regression model to predict whether an employee is likely to leave the organization.

### Description:
This project explores employee attrition using a labeled dataset. It focuses on data exploration, logistic regression modeling, and evaluation to understand the relationship between various job-related features and attrition outcomes.

  #### 1. Data Preparation & Exploration:
  Loaded and inspected the employee attrition dataset.
  
  Checked for null values and confirmed data cleanliness.
  
  Conducted exploratory analysis to understand:
  
  The distribution of attrition (Yes/No).
  
  Key categorical variables (e.g., Department, Job Role, Marital Status).
  
  Numeric relationships with attrition (e.g., Monthly Income, Age, Overtime).
  
  Visualized patterns using bar plots and value counts to identify potential predictors.
  
 #### 2. Model Implementation:
  Converted categorical variables into dummy variables using one-hot encoding.
  
  Split the dataset into training (75%) and testing (25%) sets.
  
  Trained a Logistic Regression model on the preprocessed data.
  
  Evaluated the model using accuracy score and confusion matrix:
  
  Model Accuracy: 87.67% on the test set.
  
  Correctly identified most "No Attrition" cases but slightly underpredicted "Yes" cases.
  
  Visualized prediction distribution and confusion matrix to assess misclassification areas.

### Skills:
Data preprocessing (encoding, feature selection).

Exploratory data analysis (categorical vs. target relationships).

Supervised learning (logistic regression modeling).

Classification metrics (accuracy, confusion matrix).

Visualization (bar charts, value counts, prediction histograms).

### Technology:
Python, pandas, matplotlib, seaborn, sklearn

### Results:
  Logistic Regression Performance:
  
  Achieved 87.67% accuracy on the test data.
  
  Model was better at predicting "No Attrition" than "Yes Attrition."
  
  Important Factors Identified:
  
  Overtime, Monthly Income, and Job Role showed strong correlations with attrition
  
  Employees with overtime were more likely to leave.

### Key Insight:

Logistic regression effectively modeled the probability of attrition, but improvements could be made with techniques like SMOTE (for imbalance) or more complex classifiers (e.g., Random Forests).



## [Machine Learning Classifiers and Clustering](https://github.com/hmoore200/Portfolio/blob/main/Classifiers_Code.Rmd)

### Goal: 
To evaluate the performance of k-Nearest Neighbors (k-NN) and k-means clustering algorithms on different datasets, comparing their accuracy and determining optimal cluster counts.

### Description: 
The project involved analyzing two classification datasets (binary and trinary). 

### Data Preparation & Exploration:

Loaded and visualized binary and trinary datasets using scatter plots to assess linear separability.

Computed Euclidean distances between feature vectors to understand data distribution.

### Model Implementation:

Applied k-NN classifiers (k=3,5,10,15,20,25) to both datasets, splitting data into train/test sets (75/25).

Evaluated accuracy, noting k-NN outperformed logistic regression (from a prior project) due to non-linear data patterns.

Clustering Analysis:

Used k-means (k=2 to 12) on a clustering dataset, visualizing results for each *k*.

Identified the "elbow point" (k=8) to determine optimal cluster count based on average distance metrics.

### Skills:
Data visualization (ggplot2, scatter plots).

Distance metrics (Euclidean).

Supervised learning (k-NN hyperparameter tuning, accuracy evaluation).

Unsupervised learning (k-means clustering, elbow method).

Statistical analysis (model comparison).

Technology: R, ggplot2, class, e1071, stats libraries.

### Results:
k-NN Performance:

Binary dataset achieved ~97% accuracy across most *k* values.

Trinary dataset accuracy declined slightly (80% to 76%) as *k* increased.

### Clustering:

k-means revealed diminishing returns beyond k=8 clusters.

Key Insight:

k-NN was better suited than linear classifiers (e.g., logistic regression) due to non-linear data patterns.


## [Natural Language Processing(NLP) - Text Classification with Logistic Regression](https://github.com/hmoore200/Portfolio/blob/main/NLP.ipynb)

### Goal:
To build and evaluate a binary text classification model using traditional Natural Language Processing (NLP) techniques and logistic regression, focusing on accuracy and effectiveness of preprocessing steps.

### Description:
This project applied core NLP and machine learning principles to classify text data into binary categories (e.g., positive or negative sentiment). Key steps included:

  1. Text Preprocessing & Feature Engineering:
  Loaded a labeled text dataset containing raw sentences and corresponding sentiment labels.

Cleaned text by:

  - Lowercasing all characters.

  - Removing stopwords using NLTK.

  - Tokenizing and reconstructing text.

  - Converted processed text into numerical features using CountVectorizer (Bag-of-Words model), creating sparse feature vectors.

  2. Model Training & Evaluation:
  Split the dataset into 75% training and 25% testing sets.
  
  Trained a Logistic Regression classifier using the count-based features.
  
  Evaluated model performance using accuracy

### Skills:
- NLP preprocessing (stopwords, tokenization, normalization).

- Feature extraction (CountVectorizer).

- Model training (Logistic Regression).

- Supervised learning evaluation (train/test split, accuracy score).

- Data pipeline design (clean → vectorize → model → evaluate).

### Technology:
Python, pandas, sklearn, nltk

### Results:
Logistic Regression Performance:

  - Achieved 96.25% accuracy on unseen data.

  - Demonstrated strong generalization with minimal overfitting.

  - Preprocessing Impact:

  - Stopword removal and normalization significantly improved model performance.

### Key Insight:
Simpler models like logistic regression, when paired with clean and well-engineered text features, can produce high accuracy in classification tasks—even without deep learning or embeddings

## [Kia & Hyundai Theft Awareness Analysis](https://github.com/hmoore200/Portfolio/blob/main/Week5_6_Thefts.Rmd)

This project uses real-world data to raise public awareness about the increasing rate of Kia and Hyundai vehicle thefts across the U.S., with a special focus on Milwaukee. Designed for a general audience—especially local residents and car owners—the project translates complex data into clear, impactful visualizations that encourage proactive safety measures.

### Project Goals:

Inform Milwaukee residents and car owners (particularly Kia/Hyundai drivers) about the rising trend of vehicle thefts.

Empower the public to take preventative action through accessible, engaging, and accurate data storytelling.

Use infographic-style visuals suitable for platforms like local news, social media, or community bulletins.

### Visualizations:

Stacked Area Chart: Shows rise in thefts over time by state.

Faceted Bar Charts: Compare seasonal theft patterns by vehicle type and state.

Donut & Pie Charts: Highlight top cities for thefts and monthly theft distributions.

Stacked Bar Chart: Ranks top agencies by total reported thefts (2019–2022).

Line & Column Charts: Emphasize trends in Milwaukee-specific data.

### Data Cleaning:

Merged and reshaped datasets from multiple sources.

Filled missing values, recalculated percentages, renamed columns, and formatted dates.

Applied thoughtful data preparation to ensure clean, accurate, and meaningful outputs.

### Design Approach:

Gestalt principles (alignment, proximity, similarity) guide visual layout.

Accessible fonts, intuitive color coding, and thoughtful label placement ensure readability.

Visuals optimized for clarity and impact—no prior data literacy required.

### Tools & Technologies Used:

R (Tidyverse, ggplot2, dplyr, tidyr)

CSV and Google Sheets data sources

Infographic-style data storytelling

## [Netflix Viewership Analysis](https://github.com/hmoore200/Portfolio/blob/main/DSC_640_Wk%201_2.ipynb)
This project explores and visualizes Netflix viewership trends using global and country-specific datasets. The goal is to identify the most popular titles, examine the impact of language and format (TV vs Film), and analyze performance over time.

### Key Objectives:
Identify the top-performing English and Non-English titles on Netflix.

Compare weekly viewership trends for TV shows and movies.

Analyze box office-style metrics like views in the first 91 days.

Explore geographic trends in viewership across different countries.

### Tools Used:
Python (Pandas, Matplotlib, Seaborn)

Jupyter Notebook

Datasets from Netflix Top 10

### Key Visuals & Insights:
Bar plots of the top 20 most-viewed titles globally.

Boxplots comparing performance between TV and Film formats.

Geographic comparisons of popular content by country.

Insights on how language affects viewership reach.

### Notable Findings:
English-language titles dominate globally but several Non-English titles have strong regional performance.

Sales/viewership often spikes during holidays or notable release months.

The first 91 days of release are critical for content success.

## [Are Taller NBA Players More Successful?](https://github.com/hmoore200/Portfolio/blob/main/TallNBAPlayers.ipynb )

### Goal: 
This project aims to analyze the performance trends of tall NBA players (6'7" and above) from 2000 onward. By examining key metrics like scoring efficiency, rebounds, and career achievements, the analysis seeks to uncover patterns in how height impacts player success in the modern NBA. The findings provide data-driven insights for player evaluation and team strategy.

### Description: 
  The project analyzed a dataset of NBA players from 2000 onward who are taller than 6'7". The dataset included player statistics such as effective field goal percentage (eFG%), points per game (PTS), rebounds per game (TRB), All-NBA team appearances, and Player Efficiency Rating (PER). The project involved loading and filtering the data, cleaning and preprocessing it, performing exploratory data analysis (EDA), and visualizing performance distributions across key metrics.


### Skills: 
Data cleaning Data preprocessing, Exploratory data analysis (EDA), Statistical analysis, Data visualization

### Technology: 
 Python, NumPy, Matplotlib, Jupyter Notebook


### Results:
The analysis revealed that:

Effective field goal percentage (eFG%) for tall players averages 48.16%, with most falling between 44.5% and 53.7%.

Points per game (PTS) show a right-skewed distribution, with most players averaging 2–8 PPG.

Rebounding (TRB) clusters around 1.9–4.8 RPG for the majority of players.

All-NBA team selections are rare, with an average of just 0.17 appearances per player.

The findings provide insights into the performance trends of taller NBA players in the modern era.


