import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

lemmatizer = WordNetLemmatizer()

# Variable for the path of the CSV dataset
resturaunt_reviews = pd.read_csv('/Users/mahmoudibrahim/PycharmProjects/20025336_MahmoudElsayed_NLP_Project/Restaurant_Reviews.csv')

# Here is the preprocessing method that has lowercase, and symbol removal, stop words, remove extra spaces
def preprocessing(txt):
    txt = txt.lower()
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = re.sub(r"\s+[a-zA-Z]\s+", ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    wrds = txt.split()
    stop_words = set(stopwords.words('english'))
    wrds = [lemmatizer.lemmatize(wrd) for wrd in wrds if wrd not in stop_words]
    txt = ' '.join(wrds)
    return txt

# Here the dataset is inserted into a set to seperate the preprocessed from the raw data
resturaunt_reviews['preprocessed_reviews'] = resturaunt_reviews['Review'].apply(preprocessing)

# using the NTKL stopword list, and using count vectorizer with stop word removal
vectorization = CountVectorizer(stop_words='english')
x = vectorization.fit_transform(resturaunt_reviews['preprocessed_reviews'])

# Bag of Words
BoW_reviews = pd.DataFrame(x.toarray(), columns=vectorization.get_feature_names_out())
print(BoW_reviews)


y = resturaunt_reviews['Liked']

# splitting the training and testing data to 90:10 with a random state of 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

# variable with the model Naive Bayes being used and fitting the train into the model
model = MultinomialNB()
model.fit(x_train, y_train)

# predictions of the test data
y_pred = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

print("Test Accuracy: %", accuracy_score(y_test, y_pred) * 100)

# classifing unseen reviews
def classify_unseen_reviews(unseen):
    unseen_review_preprocessed = preprocessing(unseen)
    unseen_review_vectorization = vectorization.transform([unseen_review_preprocessed])
    return model.predict(unseen_review_vectorization)

# testing a handwritten review and seeing the results
unseen_review = "this food is horrible"
print('')
print("The review: ", unseen_review)
print("This review is predicted to be: ", "Postive" if classify_unseen_reviews(unseen_review) == 1 else "Negative")

# creating a confusion matrix
cnfsn_mtrx = confusion_matrix(y_test, y_pred)

# displaying a confusion matrix
plt.figure(figsize = (8, 6))
sns.heatmap(cnfsn_mtrx, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = ['Negative', 'Postive'], yticklabels = ['Negative', 'Postive'])
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.title('Confusion Matrix')
plt.show()


