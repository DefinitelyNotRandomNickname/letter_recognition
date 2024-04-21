import argparse
import cv2
import os

import numpy as np
import tensorflow as tf

from typing import Dict


EPOCHS = 10
IMG_WIDTH = 32
IMG_HEIGHT = 32
NUM_CATEGORIES = 0


class LetterRecognitor:

    def __init__(self, test_data: str = "test", train_data: str = "train", load: bool = False):
        self.model = None
        
        if load and os.path.exists("model"):
            self.model = tf.keras.models.load_model("model")
        else:
            self.training_data = {}
            self.testing_data = {}
            
            self.load_data(test_data, train_data)
            self.create_model()
            self.train_model()
            self.test_model()


    def load_data(self, test_data: str = "test", train_data: str = "train"): 
        global NUM_CATEGORIES
        
        for label in os.listdir(train_data):
            print(f"Loading {label}...")
            data = []
            directory = os.path.join(train_data, label)
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                image = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                data.append(image)
            self.training_data[ord(label) - 65] = data
            NUM_CATEGORIES += 1
        
        for label in os.listdir(test_data):
            data = []
            directory = os.path.join(test_data, label)
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                image = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                data.append(image)
            self.testing_data[ord(label) - 65] = data


    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
        ])
        
        self.model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


    def train_model(self):
        train_images = []
        train_labels = []
        for label, data in self.training_data.items():
            train_images.extend(data)
            train_labels.extend([label] * len(data))

        self.model.fit(np.array(train_images), np.array(train_labels), epochs=EPOCHS)
        
        
    def test_model(self):
        test_images = []
        test_labels = []
        for label, data in self.testing_data.items():
            test_images.extend(data)
            test_labels.extend([label] * len(data))
        
        self.model.evaluate(np.array(test_images), np.array(test_labels), verbose=2)
        
        
    def predict_image(self, image_path, info: bool = False) -> Dict[str, float]:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image, verbose=0)
        if info:
            predicted_labels = [chr(i + 65) for i in range(len(prediction[0]))]
            percentage_probabilities = [prob * 100 for prob in prediction[0]]

            for label, percentage in zip(predicted_labels, percentage_probabilities):
                print(f"Label: {label}, Probability: {percentage:.2f}%")

        return (chr(prediction.argmax() + 65), max(prediction[0]) * 100)
    
    
    def save(self, path: str = "model") -> None:
        self.model.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Input image')
    parser.add_argument('--load', action='store_true', help='Load existing model from directory model')
    parser.add_argument('--info', action='store_true', help='Print all percantages')
    args = parser.parse_args()
    
    recognizer = LetterRecognitor(load=args.load)
    recognizer.save()
    if not args.input:
        print("Usage: python main.py -i image.png")
        return 1

    predicted_label, percentage = recognizer.predict_image(args.input, info=args.info)
    print(f"Predicted label: {predicted_label} for {percentage:.2f}%")
    

if __name__ == "__main__":
    main()