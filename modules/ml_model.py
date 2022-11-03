from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

classes = {0: 'rock', 1: 'paper', 2: 'scissors'}

def train(show_results=False) -> svm.SVC:

    mnist_dataframe = pd.read_csv(
    "./datasets/mnist_train_small.csv",
    sep=";",
    header=None)

    mnist_dataframe = mnist_dataframe.head(20000)

    target, features = parse_labels_and_features(mnist_dataframe)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    clf = svm.SVC(probability=True)
    clf.fit(X_train, y_train)
    dump(clf, './models/mnist_trained_model.joblib')

    y_predict = clf.predict(X_test)

    if show_results:

        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, y_predict)}\n"
        )

        disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predict)
        disp.figure_.suptitle("Confusion Matrix")

        plt.show()

    return clf


def load_model() -> svm.SVC:
    return load('./models/mnist_trained_model.joblib')


def parse_labels_and_features(dataset):

    labels = dataset[0]

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:,1:784]
    # Scale the data to [0, 1] by dividing out the max value, 255.
    features = features / 255

    return labels, features


def predict_photo(clf, image):

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:

        x_max, x_min, y_max, y_min = 0, 10000, 0, 10000

        # Recorre cada punto de cada mano y lo mete en la imagen de salida
        for handLms in results.multi_hand_landmarks:

            prediction = 0 # Cambiar
            for id, lm in enumerate(handLms.landmark):

                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                x_max = max(x_max, cx)
                x_min = min(x_min, cx)
                y_max = max(y_max, cy)
                y_min = min(y_min, cy)
            
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)

        # Draw a rectangle around the hand giving 10 pixels of margin and a thickness of 5
        cv2.rectangle(image, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 5)
        # Draw another rectangle below the first one
        cv2.rectangle(image, (x_min - 10, y_max + 10), (x_max + 10, y_max + 50), (0, 255, 0), cv2.FILLED)
        # Write the text "No hotdog: 83%" in the rectangle in arial font, color black fitted to the rectangle and a thickness of 2.
        cv2.putText(image, classes[prediction] + ": 83%", (x_min - 10, y_max + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return image
    

def predict_video(clf, file):

    cap = cv2.VideoCapture("./inputs/" + file)
    name = file.split(".")[0]
    # Archivo de salida. Parametros: archivo, codec, FPS, resolucion. Trata de poner la misma resolucion que la del video de entrada, si no da bateo
    out = cv2.VideoWriter("./inputs/results/" + name + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280, 720)) 

    while True:

        success, image = cap.read() # Lee un frame como imagen
        if success == False: break # Si no quedan imagenes rompe el ciclo
        out.write(predict_photo(clf, image))

    return out
        
