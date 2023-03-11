import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

from jupyterthemes import jtplot
jtplot.style(theme='monokai',context='notebook',ticks=True, grid=False)

#
reviews_df = pd.read_csv('amazon_reviews.csv')
#print(reviews_df)

#
#reviews_df.info()
#print(reviews_df.describe())

#
from pylab import *
sns.countplot(x = reviews_df['rating'])
#show()

#
reviews_df['length'] = reviews_df['verified_reviews'].apply(len)
#print(reviews_df)

#
reviews_df['length'].plot(bins=100, kind='hist')
#show()

#plotting feedback
#sns.countplot(x=reviews_df['feedback'])
#show()

#separarting positive and negative reviews
positive = reviews_df[reviews_df['feedback'] == 1]
#print(positive)

negative = reviews_df[reviews_df['feedback'] == 0]
#print(negative)

#converting all positive reviews to list
sentences = positive['verified_reviews'].tolist()
#print(len(sentences))

#join all reviews into one large string
sentences_as_one_string = " ".join(sentences)
#print(sentences_as_one_string)

#showing wordcloud for positive reviews
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
#show()

#showing wordcloud for negative reviews
sentences = negative['verified_reviews'].tolist()
sentences_as_one_string = " ".join(sentences)

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))
#show()

#function to remove punctuation and stopwords
def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean
    #print(Test_punc_removed_join)
    #return Test_punc_removed_join

#
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)

#show the original review
print(reviews_df['verified_reviews'][5])

#show the cleaned up version
print(reviews_df_clean[5])

#
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

#print(vectorizer.get_feature_names_out())

#
#print(reviews_countvectorizer.toarray())

#
reviews_countvectorizer.shape

#
reviews = pd.DataFrame(reviews_countvectorizer.toarray())

#
X = reviews

#
y = reviews_df['feedback']
#print(y)

#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#By Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

#
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
#sns.heatmap(cm, annot=True)
#plt.show()


print(classification_report(y_test, y_predict_test))

#By logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred,y_test)
#sns.heatmap(cm, annot=True)
#plt.show()



#By gradient boosting classfier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred,y_test)
sns.heatmap(cm, annot=True)
plt.show()









