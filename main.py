import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# ==============================
# 1 - Carregar dataset
# ==============================

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 2 - Criar modelo
# ==============================

def criar_modelo():

    model = keras.Sequential([

        layers.Dense(10, activation='relu', input_shape=(4,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(3, activation='softmax')

    ])

    model.compile(

        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']

    )

    return model


# ==============================
# 3 - Loop de treinamentos
# ==============================

epocas = [10, 25, 50, 75, 100]
acuracias = []

for e in epocas:

    print("\n==============================")
    print("Treinando com", e, "épocas")
    print("==============================")

    model = criar_modelo()

    model.fit(X_train, y_train, epochs=e, verbose=0)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    acc = accuracy_score(y_test, y_pred)

    acuracias.append(acc)

    print("Acurácia:", acc)


# ==============================
# 4 - Plot épocas vs acurácia
# ==============================

plt.figure(figsize=(8,5))

plt.plot(epocas, acuracias, marker='o')

plt.xlabel("Número de Épocas")
plt.ylabel("Acurácia")
plt.title("Acurácia vs Épocas")

plt.grid(True)

plt.show()


# ==============================
# 5 - Estrutura da rede na CLI
# ==============================

print("\n==============================")
print("Estrutura da Rede Neural")
print("==============================")

model = criar_modelo()

for i, layer in enumerate(model.layers):

    print("\nCamada", i+1)

    print("Tipo:", type(layer).__name__)

    if hasattr(layer, "units"):
        print("Neurônios:", layer.units)


# ==============================
# 6 - Criar grafo dos perceptrons
# ==============================

print("\nGerando grafo da rede neural...")

# estrutura: entrada + camadas
estrutura = [4, 10, 8, 3]

G = nx.DiGraph()

pos = {}

# criar nós organizados por camada
for camada, n_neuronios in enumerate(estrutura):

    for neuronio in range(n_neuronios):

        nome = f"L{camada}N{neuronio}"

        G.add_node(nome)

        pos[nome] = (camada, -neuronio)


# criar conexões entre camadas
for camada in range(len(estrutura)-1):

    for n1 in range(estrutura[camada]):

        for n2 in range(estrutura[camada+1]):

            G.add_edge(
                f"L{camada}N{n1}",
                f"L{camada+1}N{n2}"
            )


# ==============================
# 7 - Plot do grafo
# ==============================

plt.figure(figsize=(12,6))

nx.draw(

    G,
    pos,
    with_labels=False,
    node_size=500,
    arrows=True

)

plt.title("Grafo da Rede Neural Organizada por Camadas")

plt.show()
