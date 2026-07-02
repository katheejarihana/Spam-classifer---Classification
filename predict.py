import pickle

model = pickle.load(open("model/spam_model.pkl","rb"))

vectorizer = pickle.load(open("model/vectorizer.pkl","rb"))

text = input("Enter message: ")

text = vectorizer.transform([text])

prediction = model.predict(text)

if prediction[0]==1:

    print("Spam")

else:

    print("Not Spam")