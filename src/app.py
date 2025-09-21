from utils import db_connect
engine = db_connect()

from flask import Flask, render_template_string, request
import pickle
import numpy as np

# Cargar el modelo entrenado
with open('models/naive_bayes_diabetes.pkl', 'rb') as f:
	model = pickle.load(f)

app = Flask(__name__)

# HTML simple para el formulario
form_html = '''
<!doctype html>
<html lang="es">
	<head>
		<meta charset="utf-8">
		<title>Predicción de Diabetes</title>
		<style>
			body { font-family: Arial; margin: 40px; }
			.container { max-width: 500px; margin: auto; }
			input[type=number] { width: 100%; padding: 8px; margin: 5px 0; }
			input[type=submit] { padding: 10px 20px; }
		</style>
	</head>
	<body>
		<div class="container">
			<h2>Predicción de Diabetes</h2>
			<form method="post">
				<label>Embarazos:</label><input type="number" name="Pregnancies" required><br>
				<label>Glucosa:</label><input type="number" name="Glucose" required><br>
				<label>Presión sanguínea:</label><input type="number" name="BloodPressure" required><br>
				<label>Pliegue cutáneo:</label><input type="number" name="SkinThickness" required><br>
				<label>Insulina:</label><input type="number" name="Insulin" required><br>
				<label>IMC:</label><input type="number" step="any" name="BMI" required><br>
				<label>Función Pedigree:</label><input type="number" step="any" name="DiabetesPedigreeFunction" required><br>
				<label>Edad:</label><input type="number" name="Age" required><br>
				<input type="submit" value="Predecir">
			</form>
			{% if pred is not none %}
				<h3>Resultado: {{ pred }}</h3>
			{% endif %}
		</div>
	</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def predict():
		pred = None
		if request.method == 'POST':
				# Obtener datos del formulario
				features = [
						float(request.form['Pregnancies']),
						float(request.form['Glucose']),
						float(request.form['BloodPressure']),
						float(request.form['SkinThickness']),
						float(request.form['Insulin']),
						float(request.form['BMI']),
						float(request.form['DiabetesPedigreeFunction']),
						float(request.form['Age'])
				]
				# Predecir
				result = model.predict([features])[0]
				pred = 'Posible diabetes' if result == 1 else 'No diabetes'
		return render_template_string(form_html, pred=pred)

if __name__ == '__main__':
		app.run(debug=True)

