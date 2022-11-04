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

classes = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}

def train(show_results=False) -> svm.SVC:

    slr_dataframe = pd.read_csv(
    "./datasets/slr_rps.csv",
    sep=",")

    # Shuffle the dataset
    slr_dataframe = slr_dataframe.sample(frac=1).reset_index(drop=True)
    print(slr_dataframe)

    target, features = parse_labels_and_features(slr_dataframe)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    clf = svm.SVC(probability=True)
    clf.fit(X_train.values, y_train)
    dump(clf, './models/slr_rps_trained.joblib')

    y_predict = clf.predict(X_test.values)

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
    return load('./models/slr_rps_trained.joblib')


def parse_labels_and_features(dataset):

    # Variable 'labels' equals to the last column of the dataset
    labels = dataset.iloc[:, -1]

    # variable 'features' equals to all the columns except the last one
    features = dataset.iloc[:, 0:-1]

    # Substract every feature with index congruent to 1 mod 3 by the first feature of the same hand
    # for i in range(0, len(features.columns), 3):
    #     features.iloc[:, i] = features.iloc[:, i] - features.iloc[:, 0]
    #     features.iloc[:, i + 1] = features.iloc[:, i + 1] - features.iloc[:, 1]
    #     features.iloc[:, i + 2] = features.iloc[:, i + 2] - features.iloc[:, 2]

    return labels, features


def predict_photo(clf, image, predict = True, last = None):

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:

        x_max, x_min, y_max, y_min = 0, 10000, 0, 10000

        # Recorre cada punto de cada mano y lo mete en la imagen de salida
        for handLms in results.multi_hand_landmarks:

            marks = []

            for id, lm in enumerate(handLms.landmark):
                
                marks.append(lm.x)
                marks.append(lm.y)
                marks.append(lm.z)

                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                x_max = max(x_max, cx)
                x_min = min(x_min, cx)
                y_max = max(y_max, cy)
                y_min = min(y_min, cy)
            
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            
            prediction, proba = -1, 0
            if predict == True or last[0] == -1:
                # Create a dataframe with the marks and the first row with elements from 0 to the number of columns
                df = pd.DataFrame([marks])
                proba = clf.predict_proba(df.values)
                # Variable 'prediction' equals to the class with the highest probability
                prediction = np.argmax(proba)
                print(prediction, proba[0][prediction])
            else:
                prediction = last[0]
                proba = last[1]

        # Draw a rectangle around the hand giving 10 pixels of margin and a thickness of 3%
        cv2.rectangle(image, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), int((x_max - x_min) * 0.03))

        text = classes[prediction] + ": " + str(round(proba[0][prediction] * 100, 2)) + "%"
        # Draw a rectangle below the hand with the predicted class
        cv2.rectangle(image, (x_min - 10, y_max + 10), (x_max + 10, y_max + 100), (0, 255, 0), cv2.FILLED)
        cv2.putText(image, text, (x_min, y_max + 80), cv2.FONT_HERSHEY_PLAIN, int((x_max - x_min) * 0.008), (0, 0, 0), int((x_max - x_min) * 0.01))

        return (image, prediction, proba)
    return (image, -1, 0)
    

def predict_video(clf, file):

    cap = cv2.VideoCapture("./inputs/" + file)
    name = file.split(".")[0]
    # Archivo de salida. Parametros: archivo, codec, FPS, resolucion. Trata de poner la misma resolucion que la del video de entrada, si no da bateo
    out = cv2.VideoWriter("./inputs/results/" + name + ".avi", cv2.VideoWriter_fourcc(*'XVID'), 30.0, (1280, 720)) 

    framecount = 0
    last = (-1, 0)
    while True:
        
        framecount += 1
        predict = framecount % 6 == 0
        success, image = cap.read() # Lee un frame como imagen
        if success == False: break # Si no quedan imagenes rompe el ciclo
        result, prediction, proba = predict_photo(clf, image, predict, last)
        last = (prediction, proba)
        out.write(result)

    return out
        
