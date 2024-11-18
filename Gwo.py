import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import models, layers, Input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping


# Funções de carregamento e pré-processamento (mesmas que antes)
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


def load_and_preprocess_data():
    data_dir = r'D:\OneDriveIsec\OneDrive - ISEC\IC\files'
    animal_classes = [2, 3, 4, 5, 6, 7]
    x_train, y_train, x_test, y_test = load_cifar10(data_dir)
    x_train_filtered, y_train_filtered = filter_classes(x_train, y_train, animal_classes)
    x_test_filtered, y_test_filtered = filter_classes(x_test, y_test, animal_classes)
    y_train_filtered = np.array([animal_classes.index(label) for label in y_train_filtered])
    y_test_filtered = np.array([animal_classes.index(label) for label in y_test_filtered])
    x_train_filtered = normalize_data(x_train_filtered)
    x_test_filtered = normalize_data(x_test_filtered)
    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered


# Função para inicializar os lobos (GWO)
def initialize_wolves(search_space, num_wolves):
    dimensions = len(search_space)
    wolves = np.zeros((num_wolves, dimensions))
    for i in range(num_wolves):
        wolves[i] = np.random.uniform(search_space[:, 0], search_space[:, 1])
    return wolves


# Função para calcular métricas de desempenho, incluindo AUC por classe
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
    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    # Calcule AUC por classe
    auc_per_class = []
    for i in range(num_classes):
        auc_class = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        auc_per_class.append(auc_class)

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "auc_per_class": auc_per_class
    }

    return metrics


# Função de fitness que treina o modelo CNN e retorna a perda de validação e métricas
def fitness_function(params, epochs=6):
    # Ajuste os parâmetros com base nos valores fornecidos pelo GWO
    num_filters_1 = max(int(params[0]), 32)  # Número mínimo de 32 filtros na primeira camada
    num_filters_2 = max(int(params[1]), 64)  # Número mínimo de 64 filtros na segunda camada
    num_neurons = max(int(params[2]), 64)  # Número mínimo de 64 neurônios na camada densa
    learning_rate = max(params[3], 1e-5)  # Taxa de aprendizado mínima de 1e-5

    # Crie o modelo com os parâmetros ajustados
    model = models.Sequential([
        Input(shape=(32, 32, 3)),
        layers.Conv2D(num_filters_1, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(num_filters_2, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(num_neurons, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_neurons * 2, activation='relu'),  # Nova camada densa com metade dos neurônios
        layers.Dropout(0.35),  # Novo Dropout
        layers.Dense(6, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # Adicione EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=1,
                        validation_data=(x_test, y_test), callbacks=[early_stopping])

    val_loss = history.history['val_loss'][-1]
    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = evaluate_model_metrics(y_test, y_pred, y_prob)

    return val_loss, metrics, model, history


# Algoritmo GWO otimizado
def gwo_algorithm(search_space, num_wolves, max_iterations):
    alpha_wolf = None
    beta_wolf = None
    gamma_wolf = None

    wolves = initialize_wolves(search_space, num_wolves)
    fitness_values = []

    # Calcule o fitness inicial para todos os lobos
    print("Executa cnn para obter o fitness inicial\n")
    metrics_list = []  # Lista para armazenar métricas intermediárias
    for wolf in wolves:
        fitness, metrics, _, _ = fitness_function(wolf)
        fitness_values.append(fitness)
        metrics_list.append(metrics)

    for iteration in range(max_iterations):
        a = 2 - (iteration / max_iterations) * 2
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # Atualize alpha, beta, e gamma baseado nos valores de fitness
        sorted_indices = np.argsort(fitness_values)
        alpha_wolf = wolves[sorted_indices[0]]
        beta_wolf = wolves[sorted_indices[1]] if len(wolves) > 1 else alpha_wolf
        gamma_wolf = wolves[sorted_indices[2]] if len(wolves) > 2 else beta_wolf

        for i in range(num_wolves):
            for j in range(len(search_space)):
                r1 = np.random.random()
                r2 = np.random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_wolf[j] - wolves[i, j])
                X1 = alpha_wolf[j] - A1 * D_alpha

                r1 = np.random.random()
                r2 = np.random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_wolf[j] - wolves[i, j])
                X2 = beta_wolf[j] - A2 * D_beta

                r1 = np.random.random()
                r2 = np.random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_gamma = abs(C3 * gamma_wolf[j] - wolves[i, j])
                X3 = gamma_wolf[j] - A3 * D_gamma

                wolves[i, j] = (X1 + X2 + X3) / 3

            # Calcule o novo fitness para o lobo atualizado
            fitness, metrics, _, _ = fitness_function(wolves[i])
            fitness_values[i] = fitness
            metrics_list[i] = metrics
            print(
                f"\nLobo {i + 1}: num_filters_1={int(wolves[i][0])}, num_filters_2={int(wolves[i][1])}, num_neurons={int(wolves[i][2])}, learning_rate={wolves[i][3]:.5f}, Val Loss: {fitness:.4f}\n")
            print(f"Metrics: {metrics}\n")
            print("--------------------------------------------\n")

    # Pegue o melhor lobo (alpha_wolf) e suas métricas
    best_index = np.argmin(fitness_values)
    best_params = wolves[best_index]
    best_metrics = metrics_list[best_index]

    # Treine o modelo novamente com os melhores parâmetros e salve o modelo
    print("\nTreinando o modelo final com os melhores hiperparâmetros encontrados...")
    final_val_loss, final_metrics, final_model, history = fitness_function(best_params, epochs=10)
    final_model.save("best_model.h5")  # Salve o melhor modelo

    # Mostrar gráficos de perda e precisão
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

    print("\nMelhores Métricas da Melhor Solução Encontrada:")
    for metric, value in best_metrics.items():
        if metric == "auc_per_class":
            print(f"{metric.capitalize()}: {[f'{auc:.4f}' for auc in value]}")
        else:
            print(f"{metric.capitalize()}: {value:.4f}")

    print("\nMétricas Finais do Modelo:")
    for metric, value in final_metrics.items():
        if metric == "auc_per_class":
            print(f"{metric.capitalize()}: {[f'{auc:.4f}' for auc in value]}")
        else:
            print(f"{metric.capitalize()}: {value:.4f}")

    return best_params


# Exemplo de uso
search_space = np.array([
    [32, 256],  # num_filters_1
    [64, 512],  # num_filters_2
    [128, 512],  # num_neurons
    [0.0001, 0.005]  # learning_rate
])

num_wolves = 4
max_iterations = 5

optimal_solution = gwo_algorithm(search_space, num_wolves, max_iterations)
print("Solução Ótima:", optimal_solution)
