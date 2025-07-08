# Heather Moore Data Science Portfolio

  Hello! My name is Heather Moore and this is my professional portfolio. I graduated with Bachelors degree in economics with a minor in data science at Iowa State University. Currently I am finishing my Masters (MS) in Data Science. During my academic career I gained proficiency in the programming languages Python, R, and SQL. 

  Currently, I hold a civilian service position, Operations Research Analyst,for the Air Force. At the Studies and Analysis Squadron, my team and I gather data across the Air Education and Training Command to conduct analyze as well as create products for our customers. Our job is to create accurate yet useful analytics to aid non-technical customers in automating daily tasks  and making data-driven decision. This role helped me improve my coding in all of my known languages as well help me gain experience in creating dashboards through Power BI and Envision by Palantir. 

  With my studies and work experience, I honed my ability to work with complex data and developed a keen eye for identifying patterns and trends. I also gained experience in data management and statistical analysis, which I believe will be valuable assets moving forward in my career.

 [Resume](https://github.com/hmoore200/Portfolio/blob/885d61afa387a1c47df1acc93fa2e9811d22722e/Resume_2025.pdf)


## Projects: 

## [Are Taller NBA Players More Successful?](https://github.com/hmoore200/Portfolio/blob/main/TallNBAPlayers.ipynb )

Goal: This project aims to analyze the performance trends of tall NBA players (6'7" and above) from 2000 onward. By examining key metrics like scoring efficiency, rebounds, and career achievements, the analysis seeks to uncover patterns in how height impacts player success in the modern NBA. The findings provide data-driven insights for player evaluation and team strategy.

Description: The project analyzed a dataset of NBA players from 2000 onward who are taller than 6'7". The dataset included player statistics such as effective field goal percentage (eFG%), points per game (PTS), rebounds per game (TRB), All-NBA team appearances, and Player Efficiency Rating (PER). The project involved loading and filtering the data, cleaning and preprocessing it, performing exploratory data analysis (EDA), and visualizing performance distributions across key metrics.


Skills: Data cleaning, Data preprocessing, Exploratory data analysis (EDA), Statistical analysis, Data visualization


Technology: Python, NumPy, Matplotlib, Jupyter Notebook


Results:
The analysis revealed that:

Effective field goal percentage (eFG%) for tall players averages 48.16%, with most falling between 44.5% and 53.7%.

Points per game (PTS) show a right-skewed distribution, with most players averaging 2–8 PPG.

Rebounding (TRB) clusters around 1.9–4.8 RPG for the majority of players.

All-NBA team selections are rare, with an average of just 0.17 appearances per player.

The findings provide insights into the performance trends of taller NBA players in the modern era.


## [Machine Learning Classifiers and Clustering](https://github.com/hmoore200/Portfolio/blob/main/Classifiers_Code.Rmd)

Goal: To evaluate the performance of k-Nearest Neighbors (k-NN) and k-means clustering algorithms on different datasets, comparing their accuracy and determining optimal cluster counts.

Description: The project involved analyzing two classification datasets (binary and trinary). 

Data Preparation & Exploration:

Loaded and visualized binary and trinary datasets using scatter plots to assess linear separability.

Computed Euclidean distances between feature vectors to understand data distribution.

Model Implementation:

Applied k-NN classifiers (k=3,5,10,15,20,25) to both datasets, splitting data into train/test sets (75/25).

Evaluated accuracy, noting k-NN outperformed logistic regression (from a prior project) due to non-linear data patterns.

Clustering Analysis:

Used k-means (k=2 to 12) on a clustering dataset, visualizing results for each *k*.

Identified the "elbow point" (k=8) to determine optimal cluster count based on average distance metrics.

Skills:
Data visualization (ggplot2, scatter plots).

Distance metrics (Euclidean).

Supervised learning (k-NN hyperparameter tuning, accuracy evaluation).

Unsupervised learning (k-means clustering, elbow method).

Statistical analysis (model comparison).

Technology: R, ggplot2, class, e1071, stats libraries.

Results:
k-NN Performance:

Binary dataset achieved ~97% accuracy across most *k* values.

Trinary dataset accuracy declined slightly (80% to 76%) as *k* increased.

Clustering:

k-means revealed diminishing returns beyond k=8 clusters.

Key Insight:

k-NN was better suited than linear classifiers (e.g., logistic regression) due to non-linear data patterns.

