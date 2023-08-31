from tkinter import filedialog, messagebox
from tkinter import Tk, Label, Button, Frame, Listbox, Scrollbar, END, SINGLE
from PIL import Image, ImageTk
import os
import cv2
import threading
import time
from tensorflow.keras.models import load_model
import numpy as np

# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ImageBrowser:
    def __init__(self, root):
        self.root = root
        root.title("Image Browser")

        frame = Frame(root)
        frame.pack()

        self.delete_button = Button(frame, text="Remove image", command=self.delete_image)
        self.delete_button.grid(row=0, column=0)

        self.start_button = Button(frame, text="Start", command=self.start_process, state="disabled")
        self.start_button.grid(row=0, column=1)

        self.next_button = Button(frame, text="Next image", command=self.next_image, state="disabled")
        self.next_button.grid(row=0, column=2)

        scrollbar = Scrollbar(frame)
        scrollbar.grid(row=1, column=3, sticky='ns')

        self.listbox = Listbox(frame, yscrollcommand=scrollbar.set, selectmode=SINGLE)
        self.listbox.grid(row=1, column=0, columnspan=3)
        self.listbox.bind('<<ListboxSelect>>', self.show_selected_image)

        scrollbar.config(command=self.listbox.yview)

        self.image_label = Label(root)
        self.image_label.pack()

        self.image_paths = []
        self.current_image_index = -1

        self.camera_started = threading.Event()  # Initialize camera_started event
        self.camera_stop = threading.Event()  # Initialize camera_stop event
        self.camera_thread = None  # Initialize camera thread

        self.model = load_model('emotion_model_FER+_kod3_Epoch50.h5')  # Load the trained model

        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # Replace this with your actual labels order
        self.sentiments = []
        self.photo_sentiments = []  # Store the average sentiment for each photo
        self.best_photo_index = None  # The index of the photo with the highest average sentiment

        self.load_images()  # Load images on program start

    def load_images(self):
        directory = 'images'
        new_image_paths = [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(('.png', '.jpg', '.jpeg'))]
        if new_image_paths:
            for path in new_image_paths:
                self.image_paths.append(path)
                self.listbox.insert(END, path.split('/')[-1])
            self.start_button['state'] = 'normal'  # Enable the Start button when images are loaded

    def delete_image(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "No image selected to delete.")
            return
        selected_index = selected_indices[0]
        self.listbox.delete(selected_index)
        del self.image_paths[selected_index]
        if self.current_image_index == selected_index:
            self.current_image_index = -1
            self.image_label.config(image='')

    def start_process(self):
        self.listbox.selection_clear(0, END)
        if self.image_paths:
            self.camera_started = threading.Event()
            self.camera_stop = threading.Event()

            self.camera_thread = threading.Thread(target=self.start_camera)
            self.camera_thread.start()

            self.root.after(500, self.wait_for_camera_start)

    def next_image(self):
        if self.image_paths:
            self.current_image_index += 1
            if self.current_image_index >= len(self.image_paths):
                self.current_image_index = -1
                self.image_label.config(image='')
                self.camera_stop.set()
                messagebox.showinfo("Info", "All images viewed.")
                self.start_button['state'] = 'normal'
                self.next_button['state'] = 'disabled'
                self.show_best_photo()
            else:
                self.display_image(self.current_image_index)

    def display_image(self, image_index):
        image = Image.open(self.image_paths[image_index])
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def show_selected_image(self, evt):
        selected_index = self.listbox.curselection()[0]
        self.display_image(selected_index)

    def analyze_sentiment(self, frame):
        # Detect the face in the frame
        faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minSize=(30, 30))
        # If a face is detected
        if len(faces) > 0:
            x, y, w, h = faces[0]  # Assume the first detected face is the relevant one
            # Crop the face from the frame
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Convert the face to grayscale
            face = cv2.resize(face, (48, 48))  # Resize the face to match the model input size
            face = face / 255  # Normalize pixel values
            face = np.expand_dims(face, axis=0)  # Add a batch dimension
            face = np.expand_dims(face, axis=-1)  # Add a color channel dimension
            # Predict the sentiment of the face
            sentiment_probs = self.model.predict(face)[0]
            sentiment_index = np.argmax(sentiment_probs)  # Get the index of the maximum sentiment probability
            emotion = self.emotions[sentiment_index]
            probability = sentiment_probs[sentiment_index]
            return emotion, probability, frame  # Return the emotion label, probability, and the frame
        return None, None, frame


    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.camera_started.set()

        while not self.camera_stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                sentiment, probability, frame = self.analyze_sentiment(frame)
                if sentiment is not None:
                    print(f"Emotion: {sentiment}, Probability: {probability:.2f}")
                    self.sentiments.append((self.current_image_index, sentiment, probability))
        self.cap.release()

    def wait_for_camera_start(self):
        if self.camera_started.is_set():
            self.next_image()
            self.start_button['state'] = 'disabled'
            self.next_button['state'] = 'normal'
        else:
            self.root.after(500, self.wait_for_camera_start)

    def show_best_photo(self):
        best_average_sentiment = -1
        best_photo_index = -1

        for photo_index, sentiment, probability in self.sentiments:
            if photo_index != -1:
                if len(self.photo_sentiments) < photo_index + 1:
                    self.photo_sentiments.append([])
                self.photo_sentiments[photo_index].append((sentiment, probability))

        if self.photo_sentiments:
            for photo_index, photo_sentiments in enumerate(self.photo_sentiments):
                average_sentiment = np.mean([sentiment[1] for sentiment in photo_sentiments])
                if average_sentiment > best_average_sentiment:
                    best_average_sentiment = average_sentiment
                    best_photo_index = photo_index

            if best_photo_index != -1:
                best_photo_path = self.image_paths[best_photo_index]
                best_sentiments = self.photo_sentiments[best_photo_index]

                messagebox.showinfo("Info", f"The best photo is: {best_photo_path}\nAverage Sentiment: {best_average_sentiment}")
                self.display_image(best_photo_index)


if __name__ == "__main__":
    root = Tk()
    image_browser = ImageBrowser(root)
    root.mainloop()
