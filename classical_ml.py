import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

data = pd.read_csv("data.csv", encoding='utf-8')

x_train, x_test, y_train, y_test = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=40)

# learn vocab and apply mapping
# TfidfVectorizer(ngram_range=(1, 2))
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# model = KNeighborsClassifier(n_neighbors=1)
# classifier = LinearSVC(max_iter=10000, class_weight='balanced')
classifier = LogisticRegression(max_iter=1000)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("classification report:", classification_report(y_test, y_pred))

while True:
      user_input = input("\nEnter text: ")
      
      if user_input.lower().strip() in ['quit', 'exit', 'q']:
          break
      
      print("Prediction:", classifier.predict(vectorizer.transform([user_input])))
      