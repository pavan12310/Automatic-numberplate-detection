import os
import cv2
import easyocr
import sqlite3
from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime

app = Flask(__name__)

db = sqlite3.connect('database.db', check_same_thread=False)
cursor = db.cursor()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS ocr_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path TEXT NOT NULL,
        extracted_text TEXT,
        date TEXT,
        time TEXT
    )
''')
db.commit()



frameWidth = 1000
frameHeight = 480
cascade_path = "haarcascade_plate_number.xml"


if not os.path.exists(cascade_path):
    raise ValueError(f"Cascade file '{cascade_path}' not found.")
plateCascade = cv2.CascadeClassifier(cascade_path)

if plateCascade.empty():
    raise ValueError("Error loading cascade file. Check if the XML file is correct.")

minArea = 500
count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return 'No image uploaded'
    
    image = request.files['image']
    if image.filename == '':
        return 'No selected file'
    
    image_path = os.path.join('IMAGES', image.filename)
    image.save(image_path)
    
    img = cv2.imread(image_path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    if len(numberPlates) == 0:
        return 'Error: No number plate detected in the uploaded image.'

    detected_text = ""

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]
            
            
            imgRoi_gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
            imgRoi_gray = cv2.bilateralFilter(imgRoi_gray, 11, 17, 17)  # Reduce noise
            imgRoi_thresh = cv2.adaptiveThreshold(imgRoi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(imgRoi_thresh)

            detected_text = "Detected Text:\n" + "\n".join([text for (_, text, _) in result])

            
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H:%M:%S")

            
            cursor.execute(
                "INSERT INTO ocr_results (image_path, extracted_text, date, time) VALUES (?, ?, ?, ?)",
                (image_path, detected_text, current_date, current_time)
            )
            db.commit()

            count += 1
            
            return detected_text
    
    return 'No number plate detected'

@app.route('/capture', methods=['POST'])
def capture():
    global count

    
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)

    while True:
        success, img = cap.read()
        if not success:
            cap.release()
            return 'Error: Failed to capture image from the webcam.'

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        numberPlates = plateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        for (x, y, w, h) in numberPlates:
            area = w * h
            if area > minArea:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                imgRoi = img[y:y + h, x:x + w]
                cv2.imshow("Number Plate", imgRoi)

        cv2.imshow("Result", img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            if 'imgRoi' in locals():
                cv2.imwrite(f"IMAGES/{str(count)}.jpg", imgRoi)
                cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("Result", img)
                cv2.waitKey(500)

                
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H:%M:%S")

                
                cursor.execute(
                    "INSERT INTO ocr_results (image_path, extracted_text, date, time) VALUES (?, ?, ?, ?)",
                    (f"IMAGES/{str(count)}.jpg", "", current_date, current_time)
                )
                db.commit()

                count += 1
                cap.release()
                cv2.destroyAllWindows()
                return 'Image saved successfully.'
            else:
                cap.release()
                cv2.destroyAllWindows()
                return 'No Number Plate detected to save.'
        elif key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return 'Capture cancelled.'

@app.route('/process_image', methods=['POST'])
def process_image():
    image_path = request.form['image_path']
    
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Failed to load image"
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(imgGray)
    
    detected_text = "Detected Text:\n" + "\n".join([text for (_, text, _) in result])
    
    
    print(f"Image Path: {image_path}")

    try:
        
        cursor.execute(
            "INSERT INTO ocr_results (image_path, extracted_text, date, time) VALUES (?, ?, ?, ?)",
            (image_path, detected_text, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"))
        )
        db.commit()  
        print("Data inserted successfully.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return "Error inserting data into the database."
    
    return detected_text



if __name__ == '__main__':
    app.run(debug=True)
