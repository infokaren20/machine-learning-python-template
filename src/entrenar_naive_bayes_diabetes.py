# Entrenamiento de modelo Naive Bayes Multinomial para predicción de diabetes
# Utiliza el dataset data/raw/diabetes.csv y guarda el modelo entrenado en models/naive_bayes_diabetes.pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Cargar el dataset
df = pd.read_csv('data/raw/diabetes.csv')

# Separar variables predictoras y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# MultinomialNB requiere variables no negativas (ya lo son en este dataset)

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluar modelo
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc:.2f}')

# Guardar modelo entrenado
with open('models/naive_bayes_diabetes.pkl', 'wb') as f:
    pickle.dump(model, f)

print('Modelo guardado en models/naive_bayes_diabetes.pkl')