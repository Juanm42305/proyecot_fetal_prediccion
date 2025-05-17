import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Cargar dataset corregido
df = pd.read_excel("FGR_dataset_corregido.xlsx")

# Variables y etiqueta
features = [f'dato{i}' for i in range(1,31)]
X = df[features]
y = df['C31']

# Dividir train-test 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Regresión Logística
model_log = LogisticRegression(max_iter=500, random_state=42)
model_log.fit(X_train_scaled, y_train)
pred_log = model_log.predict(X_test_scaled)
acc_log = accuracy_score(y_test, pred_log)
print(f"Regresión Logística - Exactitud: {acc_log*100:.2f}%")

# 2. Red Neuronal (MLP)
model_mlp = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=1000, random_state=42)
model_mlp.fit(X_train_scaled, y_train)
pred_mlp = model_mlp.predict(X_test_scaled)
acc_mlp = accuracy_score(y_test, pred_mlp)
print(f"Red Neuronal - Exactitud: {acc_mlp*100:.2f}%")

# 3. SVM
model_svm = SVC(kernel='rbf', probability=True, random_state=42)
model_svm.fit(X_train_scaled, y_train)
pred_svm = model_svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, pred_svm)
print(f"SVM - Exactitud: {acc_svm*100:.2f}%")

# 4. Mapa Cognitivo Difuso (FCM) con 30 nodos
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

# Generar matriz de pesos aleatoria simétrica para FCM
np.random.seed(42)
W = np.random.uniform(-1, 1, (30, 30))
W = (W + W.T) / 2  # Simetría

fcm = SimpleFCM(adjacency_matrix=W)

def predecir_fcm(X):
    preds = []
    for x in X:
        estados = fcm.infer(x, steps=10)
        pred = 1 if np.mean(estados) > 0.5 else 0
        preds.append(pred)
    return np.array(preds)

X_test_scaled = scaler.transform(X_test)
pred_fcm = predecir_fcm(X_test_scaled)
acc_fcm = accuracy_score(y_test, pred_fcm)
print(f"FCM - Exactitud: {acc_fcm*100:.2f}%")

# Guardar modelos
pickle.dump(scaler, open("escalador.pkl", "wb"))
pickle.dump(model_log, open("Regresion_logistica.pkl", "wb"))
pickle.dump(model_mlp, open("red_neuronal.pkl", "wb"))
pickle.dump(model_svm, open("svm.pkl", "wb"))
pickle.dump({"adjacency_matrix": W}, open("fcm_model.pkl", "wb"))

# Guardar dataset por si quieres usarlo en la app
df.to_csv("dataset_FGR_corregido.csv", index=False)
