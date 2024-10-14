# How AI and Machine Learning Uncover Fake News

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

The WELFake dataset [[2]](#ref2) contains 72k news articles with 35k real and 37k fake news. There are four columns: serial number (starting at 0); title (headline of the text message); text (content of the message); and label (0 = real and 1 = fake). During the data exploration, it was determined that approximately 0.8 percent of the data in the 'title' column and 0.05% of the data in the 'text' column were missing. This concerns 597 data records which were excluded from further analysis. Figure 1 shows the distribution of the raw data in classes 0 (REAL) and 1 (FAKE). It is easy to see that it is roughly balanced.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/03_DistributionArticles.png?raw=true" style="width: 80%; height: auto;">
</div>
<div align="center">
	<i>Figure 1 – Distribution of the raw data in classes 0 (REAL) and 1 (FAKE)</i>
</div>
<br>

Additionally, a search was carried out for duplicates. 8416 duplicates were found and removed. Deduplication removes redundancies that can lead to bias results. Figure 2 shows the distribution of the cleaned data in classes 0 (REAL) and 1 (FAKE). Most of the duplicate records were found in the fake news class (8776 duplicates). After cleaning, the distribution is somewhat more uneven than before but still good enough for classification. Highly imbalanced classes can lead to issues with the quality of the classification algorithm. This is because the model tends to favor the majority class, which can result in high accuracy, but simultaneously significantly impairs performance on the minority class.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/03_DistributionArticlesCleaned.png?raw=true" style="width: 80%; height: auto;">
</div>
<div align="center">
	<i>Figure 2 – Distribution of the cleaned data in classes 0 (REAL) and 1 (FAKE)</i>
</div>
<br>

## Methodology

In the previous section, we examined the data, removed obvious errors (NaNs), and eliminated duplicates. Natural language processing largely relies on breaking down sentences into individual words to count their frequencies. To enrich the dataset with additional features, various characteristics were extracted from the texts through feature engineering and stored as additional columns in the input matrix. The following features were generated: 
- Number of sentences
- Number of words
- Number of verbs
- Number of nouns
- Number of adjectives and
- Number of punctuation marks

All distributions presented in Figure 3 regarding word types (nouns, verbs, adjectives) and sentence lengths show similar patterns. However, it is apparent that there are significant differences between real and fake articles. Particularly in the area of short sentences/few words, there is a strong overrepresentation in the real articles. Furthermore, the maxima between real and fake articles do not overlap. However, punctuation marks show a completely different distribution. It appears that fake articles use significantly more punctuation marks than real articles.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/02_FeatureEngineering.png?raw=true" style="width: 90%; height: auto;">
</div>
<div align="center">
	<i>Figure 3 – Histograms of the engineered features</i>
</div>
<br>

The heatmap in Figure 4 is showing the correlations between the different engineered features and is an effective tool for visualizing the relationships within a dataset. Most of the engineered features are highly correlated with each other, while only a single feature shows no significant correlation (Number of punctuation marks). The following interpretations can be derived from this: First, the high correlation of most features suggests that they may represent similar information or patterns in the data. This may indicate redundancy, which means that some features may be redundant and could affect model performance.

On the other hand, the isolated feature that shows no correlation could represent a unique or independent variable that may provide important information not captured by other features. The analysis should therefore focus on understanding the importance of this feature in the context of the overall model and evaluating whether it may play an important role in certain predictions.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/03_CorrelationFeatureEngineering.png?raw=true" style="width: 50%; height: auto;">
</div>
<div align="center">
	<i>Figure 4 – Correlation matrix of the engineered features</i>
</div>
<br>

For the reasons presented, not all correlated features were used for training the models. Only the number of sentences and the number of punctuation marks were retained. The remaining features were discarded. The extraction of features from the respective articles is a computationally intensive and extremely time-consuming process. Therefore, during the implementation, this step was initially tested only on selected articles and subsequently executed once. The data generated from this extraction was then written to a database to facilitate easy loading of the data from there.

A key component of Natural Language Processing (NLP) is the conversion of text into a numerical form that can subsequently be processed by machine learning models. Two important steps are involved in this process:
- First, a document-feature matrix is created. In this matrix, the rows correspond to the articles and the columns to the recognized words. Each cell in the matrix contains the count of occurrences of a specific word in a document, allowing for straightforward counting of word frequencies.
- Second, the matrix is further processed. Using the TF-IDF transformer, the raw frequencies are converted into TF-IDF values. TF-IDF stands for Term Frequency-Inverse Document Frequency and is a measure of how important a word is in a document relative to a corpus. It reduces the weight of common words that appear in many documents while giving greater importance to rarer words that are specific to a particular document.

With the available computing power, it was not possible to use the entirety of the articles. For this reason, the number of articles was reduced to 2000. The articles were chosen randomly - however, the statistical distribution between real and fake articles was maintained. When examining the input matrix for the algorithms, it becomes evident that there is a very large number of individual features. The matrix has approximately 45k columns after preprocessing, while the number of articles used has already been reduced to 2000. The reasons for this are manifold. Linguistic Diversity: Differences in dialects, slang, and regional expressions can affect the consistency of the data. Noise in the Data: Texts often contain noise, such as typos, irrelevant information or unstructured data.

The choice of machine learning algorithms is crucial for the success of a model and depends on various factors [[5]](#ref5). Firstly, the type of data and the specific problem to be solved are paramount. For classification tasks, such as distinguishing between real and fake news, algorithms like 
- Naive Bayes and
- Support Vector Machines (SVM)
  
are often suitable. Naive Bayes offers advantages due to its simplicity and speed, while SVMs provide robust performance in high-dimensional spaces. Another important aspect is the interpretability of the models. Algorithms like
- Logistic Regression
  
are easy to understand and explain, which is beneficial when communicating results to stakeholders. Ensemble methods like
- AdaBoost

can also perform well, especially with more complex datasets, as they combine multiple models and thereby reduce overfitting. Scalability and computational resources are also considerations. Some models require more computational power and time for training. Ultimately, the selection of algorithms should also be based on experimental results, making it advisable to try multiple models and compare their performance through cross-validation to identify the best model for the specific application. All four algorithms will be implemented and compared against each other. The results are shown in the next chapter.

## Results

The results of the four tested classification models - Naive Bayes, AdaBoost, Logistic Regression, and Support Vector Machine (SVM) - provide interesting insights into the performance of each model in classifying the two categories: real articles (Label 0) and fake articles (Label 1). The results are shown in Figure 5.

The Naive Bayes model achieved an average accuracy (avg / total) of 0.84 for precision, recall, and F1-score. The results show that the model has a precision of 0.82 and a recall of 0.87 for real articles, while it achieves a precision of 0.85 and a recall of 0.80 for fake articles. These values suggest that the model is capable of correctly identifying a substantial number of relevant instances, although it shows some weaknesses in identifying fake articles, as the recall is lower in this case. Overall, the performance of Naive Bayes is solid but not outstanding.

AdaBoost shows the best overall performance among the tested models, with an average precision, recall, and F1-score of 0.92. Notably, it has a high precision of 0.94 for real articles and also a high recall of 0.94 for fake articles. This means the model is able to make many correct positive predictions while also correctly classifying the majority of actual positive instances. AdaBoost thus appears to be a robust choice, especially when a balance between the two classes is required.

Logistic Regression shows similar results to the SVM, with an average F1-score of 0.89. For real articles, the precision is 0.92, while the recall is 0.86. For fake articles, the precision is 0.86 and the recall is 0.92. These values indicate that the model is relatively balanced, but it is somewhat weaker in identifying real articles. Logistic Regression provides a good foundation for classification but could be improved by applying more complex models like AdaBoost.

The Support Vector Machine (SVM) achieved identical results to Logistic Regression, suggesting that both models used similar approaches to separate the classes. Here, too, the precision for real articles is 0.92, with a recall of 0.86, while fake articles have a precision of 0.86 and a recall of 0.93. The SVM is known for its efficiency in high-dimensional spaces and also offers strong performance, but without the advantage of AdaBoost.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/05_ConfusionMatrix.png?raw=true" style="width: 85%; height: auto;">
</div>
<br>
<div align="center">
	<i>Figure 5 – The results of the four classifiers used in comparison</i>
</div>
<br>

Overall, the results indicate that AdaBoost is the most powerful model in this analysis, followed by Naive Bayes, Logistic Regression, and SVM. While all models achieve respectable performance, it is crucial to consider the chosen model in the context of specific use cases and requirements, particularly regarding the prioritization of precision or recall depending on business or analytical goals. For instance, AdaBoost could be favored if minimizing misclassifications in both classes is critical.

The results of the model parameter optimization reveal interesting advancements in classification accuracy and overall model performance. Two approaches were analyzed: the VotingClassifier, which combines the results of AdaBoost, Logistic Regression, and SVM, and the targeted optimization of the best individual algorithm, AdaBoost, through GridSearch.

The VotingClassifier achieves an average precision, recall, and F1-score of 0.90. This indicates that the combination of the three models leads to robust performance. For the classification of real articles (Label 0), the VotingClassifier reaches a precision of 0.93 and a recall of 0.87. These values suggest that the model is capable of accurately identifying many real articles but is somewhat weaker in detecting fake articles (Label 1), with a precision of 0.87 and a recall of 0.93.

The fact that both precision and recall for both classes are relatively high results in an F1-score of 0.90, indicating balanced performance. The VotingClassifier clearly benefits from the diversity of the underlying models, which capture different aspects of the data. This illustrates the strength of ensemble methods, which often mitigate the weaknesses of individual models and enhance overall performance.

The targeted optimization of AdaBoost through GridSearch yields an average precision, recall, and F1-score of 0.91. Compared to the original results of AdaBoost, which had an average F1-score of 0.92, the improvements here are more subtle. For real articles, the optimized AdaBoost model shows a precision of 0.95, which is a positive development, while the recall drops to 0.88. This indicates that the model is making fewer false positive predictions but is also identifying fewer real articles correctly.

For fake articles, the precision remains at 0.88, indicating a slight deterioration, while the recall increases to 0.95. This shows that the optimized model has become better at identifying fake articles, albeit at the cost of precision.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/06_Optimization.png?raw=true" style="width: 85%; height: auto;">
</div>
<br>
<div align="center">
	<i>Figure 6 – The results of the voting classifier and optimized AdaBoost algorithm</i>
</div>
<br>

In summary, both the VotingClassifier and the optimized AdaBoost model achieve good results in classification, though neither shows drastic improvements compared to the original results of AdaBoost. The VotingClassifier provides robust performance, while the targeted optimization of AdaBoost has improved specific aspects of classification, albeit with some trade-offs.







Feature importance is a crucial concept in machine learning that is used to understand which features (or characteristics) of a dataset contribute most to predicting a target value. Analyzing feature importance helps interpret models, improve decision making, and optimize performance. This information can help identify irrelevant or redundant features, leading to better model performance and shorter training times. The output of feature importance is often presented in the form of numbers that represent the relative importance of the features. Higher values ​​indicate that the corresponding feature is more important for the prediction.

Figure 7 shows the top 15 features and their importance. It can be seen that two of the engineered features are included in the top 15. This shows the importance of feature engineering to achieve higher model accuracy.

<div align="center">
	<img src="https://github.com/techdataman/techdataman.github.io/blob/main/_posts/_img/02_article/07_FeatureImportance.png?raw=true" style="width: 60%; height: auto;">
</div>
<div align="center">
	<i>Figure 7 – Top 15 feature importance</i>
</div>
<br>


## Conclusion

### Reflection

### Improvement

<!--
Feature Engineering im Kontext von Natural Language Processing (NLP) bringt mehrere Schwierigkeiten und Herausforderungen mit sich:

1. **Textkomplexität**: Sprache ist vielschichtig und kontextabhängig. Die gleiche Phrase kann unterschiedliche Bedeutungen haben, abhängig von ihrem Kontext. Dies erschwert die Identifizierung relevanter Merkmale.
2. **Sprachvielfalt**: Unterschiede in Dialekten, Slang und regionalen Ausdrücken können die Konsistenz der Daten beeinträchtigen und die Extraktion von Merkmalen erschweren.
3. **Rauschen in den Daten**: Texte enthalten oft Rauschen, wie Tippfehler, irrelevante Informationen oder unstrukturierte Daten, die die Feature-Extraktion und die Modellleistung beeinträchtigen können.
4. **Herausforderung bei der Auswahl von Merkmalen**: Es kann schwierig sein zu entscheiden, welche Merkmale tatsächlich relevant sind. Eine Überanpassung kann auftreten, wenn zu viele irrelevante Merkmale in das Modell einfließen.
5. **Skalierung und Verarbeitung**: Die Verarbeitung großer Textmengen erfordert erhebliche Rechenressourcen. Die Extraktion von Merkmalen kann zeitaufwendig sein, insbesondere bei sehr großen Datensätzen.
6. **Vektorisierung**: Der Prozess, Text in numerische Formate zu konvertieren (z. B. durch Bag-of-Words oder TF-IDF), kann Informationen verlieren und die Bedeutung des Textes verzerren.
7. **Feature-Korrelation**: Merkmale können stark miteinander korreliert sein, was die Interpretation der Ergebnisse erschwert und zu Problemen bei der Modellierung führen kann.
8. **Feature-Dimension**: Eine hohe Dimensionalität kann zu Überanpassung führen und die Rechenleistung erhöhen. Es ist oft notwendig, Techniken zur Dimensionsreduktion anzuwenden.
9. **Semantische Bedeutung**: Die Herausforderung, semantische Beziehungen zwischen Wörtern und Phrasen zu erfassen, ist entscheidend, aber komplex. Die Verwendung von Word Embeddings oder anderen Methoden zur Erfassung von Bedeutungen kann notwendig sein.
10. **Evaluation**: Die Bewertung der Wirksamkeit der extrahierten Merkmale ist entscheidend, erfordert jedoch oft spezifische Metriken und Techniken, um die Qualität der Merkmale zu bestimmen.

Diese Herausforderungen erfordern sorgfältige Planung, Experimentieren und Anpassungen während des Feature Engineering-Prozesses, um die besten Ergebnisse in NLP-Anwendungen zu erzielen.
-->

## References
[1] <a name="ref1">[Reuters Institute](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2024/dnr-executive-summary)</a><br>
[2] <a name="ref2">[WELFake dataset for fake news detection in text data](https://zenodo.org/records/4561253)</a><br>
[3] <a name="ref3">[Classification Metrics | scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)</a><br>
[4] <a name="ref4">[Classification Report Metrics | scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)</a><br>
[5] <a name="ref5">[Methods and Algorithms for Text Classification](https://www.elastic.co/de/what-is/text-classification)</a><br>


<br><br>This blog post is part of my [Udacity Data Scientist](https://www.udacity.com/course/data-scientist-nanodegree--nd025) Nanodegree program. If you are interested in the evaluation (Jupyter Notebook) you can find it on [Github](https://github.com/TechDataMan/FakeNews). Thank you for reading, and stay critical.


