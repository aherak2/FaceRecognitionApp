import torch
import cv2
from ultralytics import YOLO
from tkinter import Tk, filedialog, Button, Label, Canvas, Frame
from PIL import Image, ImageTk
import threading
import queue


identity_model = YOLO('best.pt')


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def upload_new_image(upload_window, canvas, label_widget):

    canvas.delete("all")


    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG Image", "*.jpg"), ("JPEG Image", "*.jpeg"), ("PNG Image", "*.png"), ("All Images", "*.jpg *.jpeg *.png")])

    if not file_path:
        return


    frame = cv2.imread(file_path)
    if frame is None:
        print("Error loading image!")
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_frame)
    img = img.resize((300, 300), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)


    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk  
    
    label = "Unknown"  
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        predictions = identity_model.predict(face_img)
        top_index = predictions[0].probs.top1

        person_name = predictions[0].names[top_index]
        confidence = predictions[0].probs.top1conf.item()

        print(f"Detected name: {person_name} with confidence: {confidence:.2f}")

        label = f"Detected: {person_name}" 
        

    label_widget.config(text=label)

def switch_to_live_camera(upload_window):

    upload_window.destroy()

    live_camera_mode()
def close_camera(window, cap):
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

def switch_to_image_upload(window):
    window.quit()
    image_upload_mode()

def live_camera_mode():

    live_window = Tk()
    live_window.title("Live Camera Mode")


    button_frame = Frame(live_window)
    button_frame.pack(pady=10)


    canvas = Canvas(live_window, width=640, height=480)
    canvas.pack(pady=10)


    close_button = Button(button_frame, text="Close Camera", command=lambda: close_camera(live_window, cap), width=20, height=2)
    close_button.pack(side="top", pady=10)


    switch_button = Button(button_frame, text="Switch to Image Upload", command=lambda: switch_to_image_upload(live_window), width=20, height=2)
    switch_button.pack(side="top", pady=10)


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                predictions = identity_model.predict(face_img)
                top_index = predictions[0].probs.top1

                person_name = predictions[0].names[top_index]
                confidence = predictions[0].probs.top1conf.item()

                print(f"Detected name: {person_name} with confidence: {confidence:.2f}")

                label = f"{person_name}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)


        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk 
        
        live_window.update_idletasks()
        live_window.update()

    cap.release()
    cv2.destroyAllWindows()
    live_window.mainloop()


def image_upload_mode():

    upload_window = Tk()
    upload_window.title("Image Upload Mode")

    canvas = Canvas(upload_window, width=300, height=300)
    canvas.pack(pady=10)

    label_widget = Label(upload_window, text="Unknown", font=("Arial", 14))
    label_widget.pack(pady=10)


    upload_new_image(upload_window, canvas, label_widget)

    Button(upload_window, text="Upload New Image", command=lambda: upload_new_image(upload_window, canvas, label_widget), width=20, height=2).pack(pady=10)
    Button(upload_window, text="Switch to Live Camera", command=lambda: switch_to_live_camera(upload_window), width=20, height=2).pack(pady=10)
    Button(upload_window, text="Close", command=upload_window.destroy, width=20, height=2).pack(pady=10)

    upload_window.mainloop()

def main_menu():
    def open_live_camera():
        root.destroy()
        live_camera_mode()

    def open_image_upload():
        root.destroy()
        image_upload_mode()

    root = Tk()
    root.title("Mode Selection")
    root.geometry("300x150")

    Button(root, text="Live Camera", command=open_live_camera, width=20, height=2).pack(pady=10)
    Button(root, text="Image Upload", command=open_image_upload, width=20, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_menu()
