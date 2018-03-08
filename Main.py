from classifier.Model import Model

model = Model()
# model.create_word_embedding()
model.load_word_embeddings()

model.create_model()


str1 = "wo gibt es einen parkplatz an der hochschule"
str2 = "alta sag mir ma eure adresse"



print(str1)
print("Prediction str1: ", model.predict(str1))
print(str2)
print("Prediction str2: ", model.predict(str2))