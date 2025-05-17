import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELOS_DIR = BASE_DIR
STATIC_DIR = os.path.join(BASE_DIR, "static")

class SimpleFCM:
    def __init__(self, adjacency_matrix, threshold_function=None):
        self.W = np.array(adjacency_matrix)
        self.n = self.W.shape[0]
        self.threshold = threshold_function if threshold_function else self.sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def infer(self, initial_state, steps=10):
        state = np.array(initial_state, dtype=float)
        for _ in range(steps):
            state = self.threshold(np.dot(state, self.W))
        return state

# Cargar modelos y scaler
scaler = pickle.load(open(os.path.join(MODELOS_DIR, "escalador.pkl"), "rb"))
model_log = pickle.load(open(os.path.join(MODELOS_DIR, "Regresion_logistica.pkl"), "rb"))
model_mlp = pickle.load(open(os.path.join(MODELOS_DIR, "red_neuronal.pkl"), "rb"))
model_svm = pickle.load(open(os.path.join(MODELOS_DIR, "svm.pkl"), "rb"))
fcm_data = pickle.load(open(os.path.join(MODELOS_DIR, "fcm_model.pkl"), "rb"))
modelo_fcm = SimpleFCM(adjacency_matrix=fcm_data["adjacency_matrix"])

features = [f'dato{i}' for i in range(1, 31)]

def cargar_nombres_y_rangos():
    nombres_campos = [
        "Edad",
        "IMC",
        "Edad gestacional al parto (semanas)",
        "Embarazo",
        "Paridad",
        "Síntomas iniciales (SII)",
        "Edad gestacional al inicio de los SII (semanas)",
        "Intervalo desde el inicio de los SII hasta el parto (días)",
        "Edad gestacional al inicio de la hipertensión (semanas)",
        "Intervalo desde el inicio de la hipertensión hasta el parto (días)",
        "Edad gestacional al inicio del edema (semanas)",
        "Intervalo desde el inicio del edema hasta el parto (días)",
        "Edad gestacional al inicio de la proteinuria (semanas)",
        "Intervalo desde el inicio de la proteinuria hasta el parto (días)",
        "Tratamiento expectante",
        "Terapia antihipertensiva antes de la hospitalización",
        "Antecedentes",
        "Presión arterial sistólica máxima",
        "Presión arterial diastólica máxima",
        "Motivo del parto",
        "Vía de parto",
        "Valor máximo de BNP",
        "Valores máximos de creatinina",
        "Valor máximo de ácido úrico",
        "Valor máximo de proteinuria",
        "Valor máximo de proteínas totales",
        "Valor máximo de albúmina",
        "Valor máximo de ALT",
        "Valor máximo de AST",
        "Valor máximo de plaquetas"
    ]

    df = pd.read_csv(os.path.join(BASE_DIR, "dataset_FGR_corregido.csv"))
    rangos_campos = []
    for col in features:
        min_val = df[col].min()
        max_val = df[col].max()
        rangos_campos.append({'min': min_val, 'max': max_val})

    return nombres_campos, rangos_campos

def predecir_con_fcm(entrada):
    estados = modelo_fcm.infer(initial_state=entrada.flatten(), steps=10)
    return 1 if np.mean(estados) > 0.5 else 0

app = Flask(__name__)

@app.route('/')
def index():
    nombres_campos, rangos_campos = cargar_nombres_y_rangos()
    return render_template("index.html", nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

@app.route('/predecir', methods=['POST'])
def predecir():
    nombres_campos, rangos_campos = cargar_nombres_y_rangos()
    try:
        entrada = []
        selects_valores = {
            5: [0, 1, 2],       # dato6: Síntomas iniciales (SII)
            16: [0, 1, 2],      # dato17: Antecedentes
            19: list(range(6)), # dato20: Motivo del parto (0-5)
            20: [0, 1]          # dato21: Vía de parto
        }
        
        valores_form = {}

        for i in range(30):
            campo = f'dato{i+1}'
            val_str = request.form.get(campo, None)
            if val_str is None:
                raise ValueError(f"Falta el campo {campo}")

            valores_form[campo] = val_str

            if i in selects_valores:
                val = int(val_str)
                if val not in selects_valores[i]:
                    raise ValueError(f"Valor inválido para {campo}: {val}")
                entrada.append(val)
            else:
                val = float(val_str)
                min_val = rangos_campos[i]['min']
                max_val = rangos_campos[i]['max']
                if not (min_val <= val <= max_val):
                    raise ValueError(f"El valor de {campo} debe estar entre {min_val} y {max_val}")
                entrada.append(val)

        entrada_np = np.array(entrada).reshape(1, -1)
        modelo_sel = request.form.get("modelo")
        valores_form["modelo"] = modelo_sel

        entrada_esc = scaler.transform(entrada_np)

        if modelo_sel == "fcm":
            pred = predecir_con_fcm(entrada_esc)
        elif modelo_sel == "logistica":
            pred = model_log.predict(entrada_esc)[0]
        elif modelo_sel == "mlp":
            pred = model_mlp.predict(entrada_esc)[0]
        elif modelo_sel == "svm":
            pred = model_svm.predict(entrada_esc)[0]
        else:
            return render_template("index.html",
                                   error_individual="Modelo no válido",
                                   nombres_campos=nombres_campos,
                                   rangos_campos=rangos_campos,
                                   valores_form=valores_form)

        return render_template("index.html",
                               resultado_individual={'prediccion': pred, 'modelo': modelo_sel},
                               nombres_campos=nombres_campos,
                               rangos_campos=rangos_campos,
                               valores_form=valores_form)

    except Exception as e:
        return render_template("index.html",
                               error_individual=str(e),
                               nombres_campos=nombres_campos,
                               rangos_campos=rangos_campos,
                               valores_form=valores_form if 'valores_form' in locals() else None)

@app.route('/lote', methods=['POST'])
def lote():
    nombres_campos, rangos_campos = cargar_nombres_y_rangos()
    try:
        archivo = request.files['archivo']
        modelo_sel = request.form['modelo']

        if not archivo.filename.endswith(('.csv', '.xlsx', '.xls', '.xml')):
            return render_template("index.html", error_lote="Solo se permiten archivos CSV, Excel (.xlsx, .xls) o XML",
                                   nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

        if archivo.filename.endswith('.csv'):
            df = pd.read_csv(archivo)
        elif archivo.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(archivo)
        else:
            try:
                df = pd.read_xml(archivo)
            except Exception as e:
                return render_template("index.html", error_lote=f"Error leyendo archivo XML: {e}",
                                       nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

        if not all(col in df.columns for col in features) or 'C31' not in df.columns:
            return render_template("index.html", error_lote="El archivo debe contener columnas dato1..dato30 y C31",
                                   nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

        X = df[features]
        y = df['C31']
        X_esc = scaler.transform(X)

        if modelo_sel == "fcm":
            preds = [predecir_con_fcm(x.reshape(1, -1)) for x in X_esc]
        elif modelo_sel == "logistica":
            preds = model_log.predict(X_esc)
        elif modelo_sel == "mlp":
            preds = model_mlp.predict(X_esc)
        elif modelo_sel == "svm":
            preds = model_svm.predict(X_esc)
        else:
            return render_template("index.html", error_lote="Modelo no válido",
                                   nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)

        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Matriz de Confusión - {modelo_sel.upper()}")
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        path_img = os.path.join(STATIC_DIR, 'confusion_matrix.png')
        plt.savefig(path_img)
        plt.close()

        report = classification_report(y, preds, output_dict=False)

        return render_template("index.html", resultados_lote=report, exactitud_lote=acc, modelo_usado_lote=modelo_sel,
                               nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

    except Exception as e:
        return render_template("index.html", error_lote=str(e),
                               nombres_campos=nombres_campos, rangos_campos=rangos_campos, valores_form=None)

if __name__ == "__main__":
    app.run(debug=True)
