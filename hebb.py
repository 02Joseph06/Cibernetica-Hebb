import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation



data_path = "mobs/"   
classes = ["aldeano", "aldeano_zombi", "pillager"]


def extract_features(path):
    try:
        img = Image.open(path).convert("L")     # escala de grises
        img = img.resize((8, 8))                # reduce a 8x8
        arr = np.array(img).astype(np.float32)
        arr = (arr / 127.5) - 1.0               
        return arr.flatten()                    
    except Exception as e:
        print(f"Error procesando {path}: {e}")
        return None

# DATASET

X, Y, file_list = [], [], []

print("Imágenes cargadas:")
for i, c in enumerate(classes):
    folder = os.path.join(data_path, c)
    if not os.path.exists(folder):
        print(f"Carpeta no encontrada: {folder}")
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, file)
            feats = extract_features(path)
            if feats is not None:
                X.append(feats)
                Y.append(i)
                file_list.append(path)
                print(path)

X = np.array(X)
Y = np.array(Y)

if len(X) == 0:
    raise ValueError("No se encontraron imágenes de mobs.")

n_features = X.shape[1]   
n_hidden = 8              
n_classes = len(classes)

# PESOS INICIALES
W1 = np.random.randn(n_hidden, n_features) * 0.1
W2 = np.random.randn(n_classes, n_hidden) * 0.1



fig, ax = plt.subplots(figsize=(10, 7))
plt.title("Red Hebbiana Multicapa – Clasificación de Mobs Minecraft", fontsize=14)

input_pos = [(1, i) for i in range(n_features)]
hidden_pos = [(2.5, i) for i in range(n_hidden)]
output_pos = [(4, i) for i in range(n_classes)]

input_nodes = [plt.Circle(pos, 0.08, color='skyblue', ec='black') for pos in input_pos]
hidden_nodes = [plt.Circle(pos, 0.1, color='lightgreen', ec='black') for pos in hidden_pos]
output_nodes = [plt.Circle(pos, 0.12, color='salmon', ec='black') for pos in output_pos]

for node in input_nodes + hidden_nodes + output_nodes:
    ax.add_patch(node)

for j, c in enumerate(classes):
    ax.text(4.4, output_pos[j][1], c.replace("_", " ").capitalize(), va='center', fontsize=12, color='darkred')

connections_1 = []
for i, inp in enumerate(input_pos):
    for j, hid in enumerate(hidden_pos):
        line, = ax.plot([inp[0], hid[0]], [inp[1], hid[1]], 'gray', alpha=0.25)
        connections_1.append(line)

connections_2 = []
for i, hid in enumerate(hidden_pos):
    for j, out in enumerate(output_pos):
        line, = ax.plot([hid[0], out[0]], [hid[1], out[1]], 'gray', alpha=0.25)
        connections_2.append(line)

ax.set_xlim(0, 5)
ax.set_ylim(-1, max(n_features, n_hidden, n_classes))
ax.axis('off')


def activation(x):
    return np.tanh(x)


def update(frame):
    global W1, W2
    ax.set_title(f"Aprendizaje Hebbiano – Imagen {frame+1}/{len(X)}", fontsize=14)

    x = X[frame]
    target = np.zeros(n_classes)
    target[Y[frame]] = 1

    h = activation(W1 @ x)
    y = activation(W2 @ h)

    W2 += np.outer(target, h) * 0.05
    W1 += np.outer(h, x) * 0.02

    idx = 0
    for i in range(n_features):
        for j in range(n_hidden):
            w = W1[j, i]
            connections_1[idx].set_linewidth(1 + abs(w) * 2)
            connections_1[idx].set_color('red' if w > 0 else 'blue')
            idx += 1

    idx = 0
    for i in range(n_hidden):
        for j in range(n_classes):
            w = W2[j, i]
            connections_2[idx].set_linewidth(1 + abs(w) * 2)
            connections_2[idx].set_color('orange' if w > 0 else 'purple')
            idx += 1

    return connections_1 + connections_2

ani = animation.FuncAnimation(fig, update, frames=len(X), interval=1000, blit=False, repeat=False)
plt.show()



def predict_mob(path):
    feats = extract_features(path)
    if feats is None:
        return "Error procesando la imagen."
    h = activation(W1 @ feats)
    activ = W2 @ h
    idx = np.argmax(activ)
    return classes[idx]

print("\n--- CLASIFICACIÓN DE IMAGEN ---")
test_file = input("Ingresa la ruta de una imagen de un mob:\n> ").strip()

if os.path.exists(test_file):
    pred = predict_mob(test_file)
    print(f"\nLa imagen '{os.path.basename(test_file)}' es clasificada como: {pred.upper()}")
else:
    print("Archivo no encontrado.")
