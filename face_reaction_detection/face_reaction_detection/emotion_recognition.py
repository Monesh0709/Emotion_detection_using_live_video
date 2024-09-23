import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class EmotionRecognition:
    def __init__(self, mode):
        self.mode = mode
        self.model = None
        self.train_dir = 'data/train'
        self.val_dir = 'data/test'
        self.num_train = 28709
        self.num_val = 7178
        self.batch_size = 64
        self.num_epoch = 50
        self.emotion_dict = {
            0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
            4: "Neutral", 5: "Sad", 6: "Surprised"
        }

    def plot_model_history(self, model_history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
        axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) / 10))
        axs[0].legend(['train', 'val'], loc='best')
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))
        axs[1].legend(['train', 'val'], loc='best')
        fig.savefig('plot.png')
        plt.show()

    def prepare_data(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(48, 48),
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode='categorical')

        self.validation_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=(48, 48),
            batch_size=self.batch_size,
            color_mode="grayscale",
            class_mode='categorical')

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

    def train_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model_info = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.num_train // self.batch_size,
            epochs=self.num_epoch,
            validation_data=self.validation_generator,
            validation_steps=self.num_val // self.batch_size
        )
        self.plot_model_history(model_info)
        self.model.save_weights('model.weights.h5')

    def display_mode(self):
        self.model.load_weights('model.weights.h5')
        cv2.ocl.setUseOpenCL(False)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = self.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                cv2.putText(frame, self.emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                print(self.emotion_dict[maxindex])
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    emotion_recognition = EmotionRecognition(mode)
    emotion_recognition.prepare_data()
    emotion_recognition.build_model()
    if mode == "train":
        emotion_recognition.train_model()
    elif mode == "display":
        emotion_recognition.display_mode()
    else:
        print("Invalid mode. Use 'train' or 'display'.")

if __name__ == "__main__":
    main()
