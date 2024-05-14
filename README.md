# Hate-Speech-Detection
## Project Scope
Develop a system that utilizes Machine Learning (ML) and Natural Language Processing (NLP) techniques to automatically detect hate speech in text data. <br> 

Data Source is available in Kaggle, named [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset/data) and it is [Licensed](https://creativecommons.org/publicdomain/zero/1.0/). The most important part of this data collection is the comments section. By studying these comments, we can identify and categorize them as Hate speech, Offensive language, or Neutral language. This will be helpful for further analysis.

## Data Collection and Preprocessing 

The data contains the following information, Unnamed: 0, count,	hate_speech, offensive_language,	neither,	class,	tweet, as column names. Our analysis will only consider the "class" and "tweet" columns from the dataset. To ensure data quality, we'll first check for any missing values (null values) in these columns before taking a sample of the data. <br> 

1. <code> data.isnull() </code> Returns table of Boolean values. It returns False if values are not null.
2. <code> data.isnull().sum() </code> Returns information about how many null values are present in the dataset.

### Data Cleaning 
In this phase, we use Natural Language ToolKit ( NLTK ) for data processing. 

Cleaning is done using, [Stopwords](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/) & [Stemmer](https://saturncloud.io/glossary/stemming/#:~:text=Stemming%20is%20a%20text%20preprocessing,classification%2C%20and%20other%20NLP%20tasks). tools in NLTK. 
1. <code> stop_words = set(stopwords.words('english')) </code>.
2. <code> stemmer = nltk.stem.SnowballStemmer('english') </code>.

<i> This paraphrase clarifies that the initial steps mentioned help get the data ready for the cleaning process, which will be done using the Natural Language Toolkit (NLTK). </i>

## Model Training and Evaluation 
For Model building, we use [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  The basic idea behind decision trees is to recursively partition the data space into smaller and smaller subsets, until each subset contains only one type of data point. This is done by selecting a feature and a threshold value at each node of the tree. The data points are then split into two groups, based on whether their value for the selected feature is less than or greater than the threshold.


<pre>from sklearn.tree import DecisionTreeClassifier   
dt = DecisionTreeClassifier() 
dt.fit(x_train, y_train)
</pre>

<i> These lines of code are essential for constructing and fine-tuning a machine learning model. </i> 

## Model Evaluation 
### Confusion Matrix </code> 

The confusion matrix function is used to check the accuracy of the model. <br>
<br>
<code>from sklearn.metrics import confusion_matrix. <br>
cr = confusion_matrix(y_test, y_pred) </code>

Output: 
<p>
  array( [ [ 126,   50,  289],
       [  37, 1177,  165],
       [ 192,  212, 5931 ] ] )
</p>

## Visualization

[Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/) are library used in visualizing the output in the form of [HeatMap](https://seaborn.pydata.org/generated/seaborn.heatmap.html).

## Limitations 

Hate Speech Detection projects, while valuable tools have some inherent limitations. Here are a few challenges to consider:

* **Data Bias:**  Training data for these projects can be biased. If the data primarily reflects certain demographics or types of hate speech, the model might struggle with others. This can lead to unfair targeting of certain groups.

* **Context-Dependence:**  Hate speech often relies on context, like sarcasm or satire. Text alone might not capture the intent, leading to misclassifications. 

* **Evolving Language:**  Hate speech creators adapt their language over time. New slang or euphemisms might confuse the model, requiring constant updates.

* **Freedom of Speech Issues:**  Defining what exactly constitutes hate speech can be tricky. Overly aggressive filtering might infringe on legitimate free speech.

* **Nuances of Language:**  Understanding sarcasm, humor, and cultural references is difficult for machines. A model might flag harmless jokes as offensive.


## Conclusion

Accuracy Level of Model: 88.44% 

This project successfully explored the potential of Machine Learning (ML) and Natural Language Processing (NLP) techniques for automatic hate speech detection in text data. We achieved the following:

<b> Developed a Machine Learning Model: </b> A trained and optimized ML model capable of classifying text data as containing hate speech or not.<br> 
<br>
<b> Evaluated Model Performance:</b> The model achieved 88.4% in detecting hate speech text data. The evaluation report provides a detailed analysis of the model's strengths and weaknesses.
