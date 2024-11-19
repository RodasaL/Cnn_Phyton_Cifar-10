# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from random import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parâmetros do PSO
POPULATION_SIZE = 2  # Tamanho da população
N_ITERATIONS = 2  # Número de iterações
W = 0.5  # Fator de inércia
C1 = 1.5  # Constante de aceleração para melhor pessoal
C2 = 1.5  # Constante de aceleração para melhor global
allowed_dense_units = [32, 64, 128, 256, 512]  # Valores permitidos para unidades densas

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

def normalize_data(x):
    return x.astype('float32') / 255.0

# Define o modelo CNN
def create_model(learning_rate, dense_units, dropout_rate):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
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
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(6, activation='softmax')  # 6 classes para animais
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=4, batch_size=128):
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    train_data_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    return model.fit(
        train_data_generator,
        epochs=epochs,
        validation_data=(x_test_filtered, y_test_filtered),
        verbose=1
    )

# Plotar gráficos de loss e accuracy
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Inicializa as partículas (hiperparâmetros) para PSO
def initialize_particles():
    particles = []
    velocities = []
    for _ in range(POPULATION_SIZE):
        learning_rate = random() * 0.01
        dense_units = np.random.choice(allowed_dense_units)
        dropout_rate = random() * 0.5
        velocity = [random() * 0.1, random() * 10, random() * 0.1]
        particles.append((learning_rate, dense_units, dropout_rate))
        velocities.append(velocity)
    return particles, velocities

# Atualiza a posição das partículas com restrições
def update_particle_position(particle, velocity):
    new_position = []
    for i in range(len(particle)):
        param_value = particle[i] + velocity[i]
        if i == 0:  # learning_rate
            param_value = np.clip(param_value, 1e-4, 1e-2)
        elif i == 1:  # dense_units
            param_value = np.random.choice(allowed_dense_units)
        elif i == 2:  # dropout_rate
            param_value = np.clip(param_value, 0, 0.5)
        new_position.append(param_value)
    return tuple(new_position)

# PSO para Otimização de Hiperparâmetros
def particle_swarm_optimization(num_particles=POPULATION_SIZE, num_iterations=N_ITERATIONS):
    print("Iniciando o Particle Swarm Optimization...")
    particles, velocities = initialize_particles()
    p_best_positions = particles[:]
    p_best_scores = [float('inf')] * num_particles
    g_best_position = None
    g_best_score = float('inf')

    for iteration in range(num_iterations):
        print(f"Iteração {iteration + 1}/{num_iterations}")
        for i, particle in enumerate(particles):
            K.clear_session()
            model = create_model(particle[0], int(particle[1]), particle[2])
            history = model.fit(x_train_filtered, y_train_filtered, epochs=2, batch_size=128,
                                validation_data=(x_test_filtered, y_test_filtered), verbose=1)
            val_loss = history.history['val_loss'][-1]
            print(f"  -> Val Loss para a partícula {i + 1}: {val_loss:.4f}")
            if val_loss < p_best_scores[i]:
                p_best_scores[i] = val_loss
                p_best_positions[i] = particle
            if val_loss < g_best_score:
                g_best_score = val_loss
                g_best_position = particle
        for i in range(num_particles):
            new_velocity = [
                W * velocities[i][j] + C1 * random() * (p_best_positions[i][j] - particles[i][j]) + C2 * random() * (
                            g_best_position[j] - particles[i][j])
                for j in range(len(particles[i]))
            ]
            velocities[i] = new_velocity
            particles[i] = update_particle_position(particles[i], velocities[i])
            print(f"  Nova posição da partícula {i + 1}: {particles[i]}")

    print("Otimização concluída.")
    return g_best_position

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
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    return accuracy, recall, precision, f1, auc

# Carregar dados e normalizar
data_dir = r'D:\OneDriveIsec\OneDrive - ISEC\IC\files'
x_train, y_train, x_test, y_test = load_cifar10(data_dir)
animal_classes = [2, 3, 4, 5, 6, 7]
x_train_filtered, y_train_filtered = filter_classes(x_train, y_train, animal_classes)
x_test_filtered, y_test_filtered = filter_classes(x_test, y_test, animal_classes)
y_train_filtered = np.array([animal_classes.index(label) for label in y_train_filtered])
y_test_filtered = np.array([animal_classes.index(label) for label in y_test_filtered])
x_train_filtered = normalize_data(x_train_filtered)
x_test_filtered = normalize_data(x_test_filtered)

# Otimização de hiperparâmetros
best_hyperparameters = particle_swarm_optimization()
print(f"Melhores hiperparâmetros: {best_hyperparameters}")

# Treinamento e avaliação com os melhores hiperparâmetros
model = create_model(*best_hyperparameters)
history = train_model(model, x_train_filtered, y_train_filtered)

# Exibir gráficos de loss e accuracy
plot_training_history(history)

# Avaliação do modelo
evaluate_model(model, x_test_filtered, y_test_filtered)

# Previsões e matriz de confusão
y_pred = np.argmax(model.predict(x_test_filtered), axis=1)
y_prob = model.predict(x_test_filtered)
plot_confusion_matrix(y_test_filtered, y_pred)

# Avaliação das métricas
accuracy, recall, precision, f1, auc = evaluate_model_metrics(y_test_filtered, y_pred, y_prob)
print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1-score: {f1}")
print(f"AUC: {auc}")
