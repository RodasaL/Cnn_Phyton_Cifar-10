# -*- coding: utf-8 -*-
"""
"""
# Imports
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from random import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Gray Wolf Optimization Parameters
POPULATION_SIZE = 3
N_ITERATIONS = 1
allowed_dense_units = [32, 64, 128, 256, 512]

# Funções de Carregamento e Pré-processamento
def load_cifar_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        images = batch['data']
        labels = batch['labels']
        images = images.reshape(-1, 3, 32, 32)
        images = np.transpose(images, (0, 2, 3, 1))
        return images, labels

def load_cifar10(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar_batch(batch_file)
        x_train.append(images)
        y_train += labels
    x_train = np.concatenate(x_train)
    y_train = np.array(y_train)
    test_file = os.path.join(data_dir, 'test_batch')
    x_test, y_test = load_cifar_batch(test_file)
    return x_train, y_train, x_test, y_test

def filter_classes(x, y, classes):
    x = np.array(x)
    y = np.array(y).astype(int)
    mask = np.isin(y, classes)
    return x[mask], y[mask]

def normalize_data(x): # [0,1]
    return x.astype('float32') / 255.0

# Define the Convolutional Neural Network model
# Define the Convolutional Neural Network model
def create_model(learning_rate, dense_units, dropout_rate):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        layers.Conv2D(64,(3,3), activation='relu', padding = 'same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
            
        layers.Flatten(),
        layers.Dropout(0.2),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        
        layers.Dense(dense_units, activation='relu'),   # Mantém camada existente
        layers.Dropout(0.5),
        layers.BatchNormalization(),          
        
        layers.Dense(6, activation='softmax')  # 6 classes para animais
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, epochs=2, batch_size=64):
    print("A gerar + dados rodando imagens etc")
    # Configuração do gerador de dados de treino com aumento de dados
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    

    train_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    
    # Treina o modelo com 'fit', passando 'epochs' e 'validation_data'
    return model.fit(
        train_data_generator,
        epochs=epochs,
        validation_data=(x_test_filtered, y_test_filtered)
    )


# Initialize wolves
def initialize_wolves():
    wolves = []
    for _ in range(POPULATION_SIZE):
        learning_rate = random() * 0.01
        dense_units = np.random.choice(allowed_dense_units)
        dropout_rate = random() * 0.5
        wolves.append((learning_rate, dense_units, dropout_rate))
    return wolves

# Update the positions of wolves with constraints
def update_position(wolf, alpha, beta, delta, a):
    new_position = []
    for i in range(len(wolf)):
        r1, r2 = random(), random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha[i] - wolf[i])
        X1 = alpha[i] - A1 * D_alpha
        
        r1, r2 = random(), random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta[i] - wolf[i])
        X2 = beta[i] - A2 * D_beta
        
        r1, r2 = random(), random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta[i] - wolf[i])
        X3 = delta[i] - A3 * D_delta
        
        # Calculate the new position
        param_value = (X1 + X2 + X3) / 3
        
        # Constrain the values within acceptable ranges
        if i == 0:  # learning_rate
            param_value = np.clip(param_value, 1e-4, 1e-2)
        elif i == 1:  # dense_units
           param_value = np.random.choice(allowed_dense_units)
        elif i == 2:  # dropout_rate
            param_value = np.clip(param_value, 0, 0.5)
        
        new_position.append(param_value)
    return tuple(new_position)


# Gray Wolf Optimization
def gray_wolf_optimization(num_wolves=POPULATION_SIZE, num_iterations=N_ITERATIONS):
    print("Iniciando o Gray Wolf Optimization...")  
    positions = initialize_wolves()  # Inicializar os lobos
    alpha, beta, delta = None, None, None

    for iteration in range(num_iterations):
        print(f"Iteração {iteration + 1}/{num_iterations}")
        fitness_scores = []
        a = 2 * (1 - (iteration / num_iterations) ** 2)  # Atualização do fator de convergência 'a'

        for wolf, pos in enumerate(positions):
            print(f"  Lobo {wolf + 1}/{num_wolves} com posição: LR={pos[0]:.5f}, Units={int(pos[1])}, Dropout={pos[2]:.2f}")
            K.clear_session()
            model = create_model(pos[0], int(pos[1]), pos[2])
            history = model.fit(x_train_filtered, y_train_filtered, epochs=1, batch_size=128, validation_data=(x_test_filtered, y_test_filtered), verbose=0 )
            val_loss = history.history['val_loss'][-1]
            fitness_scores.append((wolf, val_loss))
            print(f"  -> Val Loss para o lobo {wolf + 1}: {val_loss:.4f}")

        # Classificar pelo fitness e definir alpha, beta, delta
        fitness_scores.sort(key=lambda x: x[1])
        alpha, beta, delta = [positions[fitness_scores[i][0]] for i in range(3)]
        print(f"  Alpha (melhor): {alpha}, Beta: {beta}, Delta: {delta}")

        # Atualização da posição dos lobos
        for i in range(num_wolves):
            positions[i] = update_position(positions[i], alpha, beta, delta, a)
            print(f"  Nova posição do lobo {i + 1}: {positions[i]}")

    print("Otimização concluída.")
    return alpha


# Funções para avaliação
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def evaluate_model_metrics(y_true, y_pred, y_prob, num_classes=6):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')

    specificity = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[:, i].sum() + cm[i, :].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(tn / (tn + fp))
    specificity = np.mean(specificity)
    
    f1 = f1_score(y_true, y_pred, average='macro')
    
    auc_per_class = {}
    for i in range(num_classes):
        class_true = (y_true == i).astype(int)
        class_prob = y_prob[:, i]
        try:
            auc_per_class[i] = roc_auc_score(class_true, class_prob)
        except ValueError:
            auc_per_class[i] = None
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Sensibilidade (Recall): {sensitivity:.4f}')
    print(f'Precisao: {precision:.4f}')
    print(f'Especificidade: {specificity:.4f}')
    print(f'F1-Score: {f1:.4f}')
    for cls, auc in auc_per_class.items():
        print(f'AUC para classe {cls}: {auc if auc is not None else "Indisponível"}')
    
    return accuracy, sensitivity, specificity, f1, auc_per_class



# Caminho para os arquivos do CIFAR-10
data_dir = r'D:\OneDriveIsec\OneDrive - ISEC\IC\files'
animal_classes = [2, 3, 4, 5, 6, 7]

# Load CIFAR-10
x_train, y_train, x_test, y_test = load_cifar10(data_dir)

# Filtrar os dados
x_train_filtered, y_train_filtered = filter_classes(x_train, y_train, animal_classes)
x_test_filtered, y_test_filtered = filter_classes(x_test, y_test, animal_classes)

# Ajustando os labels entre 0 e 5
y_train_filtered = np.array([animal_classes.index(label) for label in y_train_filtered])
y_test_filtered = np.array([animal_classes.index(label) for label in y_test_filtered])

# Normalizar os dados filtrados
x_train_filtered = normalize_data(x_train_filtered)
x_test_filtered = normalize_data(x_test_filtered)

# Executar o GWO para encontrar os melhores parâmetros
best_params = gray_wolf_optimization()
learning_rate, dense_units, dropout_rate = best_params
print("Melhores parâmetros encontrados pelo GWO:", best_params)

# Criar e treinar o modelo otimizado
print("Treinando modelo com os melhores parâmetros encontrados pelo GWO...")
model = create_model(learning_rate, int(dense_units), dropout_rate)
#model = create_model(0.025024874568846402, 39, 0.013647792135632266)

history = train_model(model, x_train_filtered, y_train_filtered, epochs=10)

# Avaliar o modelo otimizado
evaluate_model(model, x_test_filtered, y_test_filtered)

# Previsões e matriz de confusão
y_prob = model.predict(x_test_filtered)
y_pred = np.argmax(y_prob, axis=1)
plot_confusion_matrix(y_test_filtered, y_pred)

# Avaliar as métricas adicionais
evaluate_model_metrics(y_test_filtered, y_pred, y_prob, num_classes=6)

# Plot da accuracy e loss
plt.figure(figsize=(12, 4))

# Plot da accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy de Treino')
plt.plot(history.history['val_accuracy'], label='Accuracy de Validação')
plt.title('Accuracy do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

# Plot da loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss de Treino')
plt.plot(history.history['val_loss'], label='Loss de Validação')
plt.title('Loss do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.show()
