import modules.ml_model as ml_model
import sys
import time
import os
import cv2

if __name__ == '__main__':
    
    clf = None

    if len(sys.argv) == 1:
        print('Please pass an argument')

    elif sys.argv[1] == 'train':

        print("Training model...")
        # Start a timer
        start = time.time()
        clf = ml_model.train(show_results=True)
        print(f"Training completed in {time.time() - start} seconds")

    elif sys.argv[1] == 'predict':
        
        # clf = ml_model.load_model()

        # Process each png or jpg in the folder 'inputs'
        for file in os.listdir('inputs'):
            if file.endswith('.png') or file.endswith('.jpg'):
                print(f"Predicting {file}...")
                image = ml_model.predict_photo(clf, cv2.imread("./inputs/" + file))
                # Save the image to 'results' folder
                cv2.imwrite("./inputs/results/" + file, image)

        # Process each mp4 in the folder 'inputs'
        for file in os.listdir('inputs'):
            if file.endswith('.mp4'):
                print(f"Predicting {file}...")
                video = ml_model.predict_video(clf, file)
                video.release()

    else:
        print('Invalid argument')

    