# Truths in the Shadow of Lies

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/01_FakeNews.jpg?raw=true">
</div>

## Project Definition

### Project Overview
Fake news detection is a crucial tool in today's information society, where the spread of misinformation has increased [[1]](#ref1). In an era where social media and online platforms serve as primary news sources, distinguishing between truthful and false information is more important than ever. Fake news can distort public awareness, undermine trust in institutions and even interfere with democratic decision-making.

The impact of fake news is far-reaching: it can influence elections, deepen social divisions and even lead to violent conflict. In light of these challenges, the ability to effectively recognize and combat fake news is becoming an indispensable part of media literacy. Automated fake news detection technologies, such as machine learning and natural language processing, play a key role by making it possible to analyze large amounts of data and quickly identify potential disinformation.

The aim is to use the WELFake dataset [[2]](#ref2) to make a prediction about whether an article is real or fake information. The WELFake dataset was compiled for the development of algorithms for detecting fake news. This dataset contains 72k news articles with 35k real and 37k fake news. 

### Problem Statement
The problem to be solved is to predict the truthfulness of the available information. Machine learning and natural language processing algorithms are used for this. The goal is to achieve the highest possible accuracy. To increase the accuracy of the model, additional input parameters are developed using feature engineering. Suitable input variables (features) are generated from the raw data (articles). Sometimes important information is not directly visible in the raw data. Feature engineering can be used to extract this information. The generated dataset is then classified using various algorithms. An attempt is then made to further increase the accuracy using parameter optimization.

### Metrics
To evaluate the classification results, common classification metrics from machine learning [[3]](#ref3) are used. These are summarized using a classification report provided by the scikit-learn framework [[4]](#ref4).
- Accuracy: This is the proportion of correctly classified messages (both true and false) out of the total number of messages. A high accuracy means that your model is working well overall.
- Precision: This indicates how many of the messages classified as "true" are actually true. High precision means that your model does not often classify "lies" as "true".
- Recall: This shows how good your model is at finding all the "true" messages. High recall means that your model missed very few "true" messages.
- F1 score: This is a combined metric of precision and recall that provides a balanced measure of the model's performance. A high F1 score means your model performs well in both identifying true news and avoiding false alarms.

As mentioned at the beginning, the focus is on the highest possible accuracy. The other metrics are also taken into account in the evaluations in order to obtain additional information on whether the algorithm performs well or poorly in terms of false positives and/or false negatives.

## Analysis

### Data Exploration and Visualisation
The WELFake dataset contains 72k news articles with 35k real and 37k fake news. There are four columns: serial number (starting at 0); title (headline of the text message); text (content of the message); and label (0 = real and 1 = fake). During the data exploration, it was determined that approximately 0.8 percent of the data in the 'title' column and 0.05% of the data in the 'text' column were missing. This concerns 597 data records which were excluded from further analysis. Figure 1 shows the distribution of the raw data in classes 0 (REAL) and 1 (FAKE). It is easy to see that it is roughly balanced.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/03_DistributionArticles.png?raw=true" style="width: 80%; height: auto;">
</div>
<div align="center">
	<i>Figure 1 – Distribution of the raw data in classes 0 (REAL) and 1 (FAKE)</i>
</div>
<br>

Additionally, a search was carried out for duplicates. 8416 duplicates were found and removed. Deduplication removes redundancies that can lead to bias results. Figure 2 shows the distribution of the cleaned data in classes 0 (REAL) and 1 (FAKE). Most of the duplicate records were found in the fake news class (8776 duplicates). After cleaning, the distribution is somewhat more uneven than before but still good enough for classification.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/03_DistributionArticlesCleaned.png?raw=true" style="width: 80%; height: auto;">
</div>
<div align="center">
	<i>Figure 2 – Distribution of the cleaned data in classes 0 (REAL) and 1 (FAKE)</i>
</div>
<br>




## Methodolgy

### Data Preprocessing

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/02_FeatureEngineering.png?raw=true">
</div>
<div align="center">
	<i>Figure 2 – Histogram of the survey results in response to the question about the level of education required</i>
</div>
<br>

### Implementation

### Refinement

## Results

### Model Evaluation and Validation

Feature importance is a crucial concept in machine learning that is used to understand which features (or characteristics) of a dataset contribute most to predicting a target value. Analyzing feature importance helps interpret models, improve decision making, and optimize performance. This information can help identify irrelevant or redundant features, leading to better model performance and shorter training times. The output of feature importance is often presented in the form of numbers that represent the relative importance of the features. Higher values ​​indicate that the corresponding feature is more important for the prediction.

Figure 6 shows the top 15 features and their importance. It can be seen that two of the engineered features are included in the top 15. This shows the importance of feature engineering to achieve higher model accuracy.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/06_FeatureImportance.png?raw=true">
</div>
<div align="center">
	<i>Figure 6 – Top 15 feature importance</i>
</div>
<br>


### Justification


## Conclusion

### Reflection

### Improvement

## References
1. <a name="ref1">[Reuters Institute](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2024/dnr-executive-summary)</a>
2. <a name="ref2">[WELFake dataset for fake news detection in text data](https://zenodo.org/records/4561253)</a>
3. <a name="ref3">[Classification Metrics | scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)</a>
4. <a name="ref4">[Classification Report Metrics | scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)</a>


This blog post is part of my [Udacity Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Nanodegree program. If you are interested in the evaluation (Jupyter Notebook) you can find it on [Github](https://github.com/TechDataMan/FakeNews).


