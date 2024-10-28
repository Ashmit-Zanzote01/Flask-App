# app.py  
from flask import Flask, render_template, request, jsonify, Response, url_for
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from tensorflow import keras

app = Flask(__name__)

# Path to store captured images temporarily
TEMP_IMAGE_DIR = 'D:\\Python\\Health_AI\\static\\temp_images'
if not os.path.exists(TEMP_IMAGE_DIR):
    os.makedirs(TEMP_IMAGE_DIR)

# Path to the original .keras model
KERAS_MODEL_PATH = os.path.join('D:\\Python\\Health_AI\\training_2', 'best_emotion_model.keras')

# Updated quiz questions
quiz_questions = [
    {'text': '1. How do you feel when you wake up in the morning?', 'options': ['A. Happy', 'B. Neutral', 'C. Tired', 'D. Anxious']},
    {'text': '2. How do you typically react to an unexpected problem or challenge?', 'options': ['A. Stay calm', 'B. Get stressed', 'C. Ask for help', 'D. Avoid it']},
    {'text': '3. How do you feel about social gatherings?', 'options': ['A. Enjoy them', 'B. Indifferent', 'C. Prefer to stay alone', 'D. Feel anxious']},
    {'text': '4. How often do you feel stressed or overwhelmed?', 'options': ['A. Rarely', 'B. Sometimes', 'C. Often', 'D. Always']},
    {'text': '5. How do you handle feelings of sadness or disappointment?', 'options': ['A. Talk to someone', 'B. Distract myself', 'C. Accept and move on', 'D. Isolate myself']},
    {'text': '6. How do you typically spend your free time?', 'options': ['A. Hobbies', 'B. Watching TV', 'C. Reading', 'D. Socializing']},
    {'text': '7. How would you describe your relationship with family and friends?', 'options': ['A. Supportive', 'B. Complicated', 'C. Distant', 'D. Strained']},
    {'text': '8. How do you handle criticism or feedback?', 'options': ['A. Accept it', 'B. Get defensive', 'C. Reflect on it', 'D. Ignore it']},
    {'text': '9. How would you describe your sleep patterns?', 'options': ['A. Regular', 'B. Irregular', 'C. Insomnia', 'D. Oversleeping']},
    {'text': '10. How would you rate your overall mood today?', 'options': ['A. Excellent', 'B. Good', 'C. Fair', 'D. Poor']}
]

# Global variables for model loading
model = None

def load_model():
    global model
    model = keras.models.load_model(KERAS_MODEL_PATH)

# Face detection function
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        print("No face detected.")
        return None  # No face detected

    # Take the first detected face (assuming no multiple faces)
    x, y, w, h = faces[0]
    face_roi = image[y:y+h, x:x+w]  # Extract the face region of interest (ROI)

    # Save detected face image
    detected_face_path = os.path.join(TEMP_IMAGE_DIR, 'detected_face.jpg')
    cv2.imwrite(detected_face_path, face_roi)
    
    return face_roi

# Capture and save image from camera
def capture_image(image_name):
    cap = cv2.VideoCapture(0)  # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()
    cap.release()

    if ret:
        # Save the original frame
        image_path = os.path.join(TEMP_IMAGE_DIR, image_name)
        cv2.imwrite(image_path, frame)
        return image_path
    else:
        print("Error: Could not read frame from camera.")
        return None

# Sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Predict emotion based on the detected face
def predict_emotion(image_path):
    if model is None:
        return "Model not loaded", []

    img = cv2.imread(image_path)

    # Step 1: Detect face
    face = detect_face(img)
    if face is None:
        return "No face detected", []

    # Step 2: Sharpen the face image
    face = sharpen_image(face)

    # Step 3: Resize, convert to grayscale, and preprocess the face image
    face = cv2.resize(face, (48, 48))         # Resize to 48x48
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face = face.astype('float32') / 255.0     # Normalize
    face = np.expand_dims(face, axis=0)        # Add batch dimension

    # Step 4: Predict emotion using the face region
    prediction = model.predict(face)
    predicted_class = np.argmax(prediction, axis=1)

    # Emotion mapping
    emotion_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
    return emotion_map.get(int(predicted_class[0]), "Unknown"), prediction[0]

@app.route('/video_feed')  # Route for live video feed
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', quiz_questions=quiz_questions)

@app.route('/submit_quiz', methods=['POST'])
def submit_quiz():
    image_name = "captured_image.jpg"
    captured_image_path = capture_image(image_name)

    responses = request.form.to_dict()
    mental_state = assess_mental_state(responses)

    if captured_image_path:
        predicted_emotion, emotion_probabilities = predict_emotion(captured_image_path)
    else:
        return jsonify({'error': 'Failed to capture image'}), 500

    emotion_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
    emotion_labels = list(emotion_map.values())
    emotion_values = emotion_probabilities.tolist() 

    final_assessment = combine_assessments(predicted_emotion, mental_state)

    generate_bar_graph(emotion_labels, emotion_values)

    return jsonify({
        'message': final_assessment,
        'predictedEmotion': predicted_emotion,
        'mentalState': mental_state,
        'emotionLabels': emotion_labels,
        'emotionValues': emotion_values,
        'graphPath': url_for('static', filename='emotion_graph.png')
    })

def assess_mental_state(responses):
    scoring = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    total_score = sum(scoring.get(response, 0) for response in responses.values())
    average_score = total_score / len(responses)
    if average_score >= 2.5:
        return "You might be experiencing significant emotional challenges."
    elif average_score >= 1.5:
        return "You might have some emotional challenges."
    else:
        return "You seem to be in a good mental state."

def combine_assessments(predicted_emotion, mental_state):
    emotional_weight = {"Happy": 1, "Sad": -1, "Angry": -2, "Neutral": 0, "Surprise": 0.5, "Fear": -1, "Disgust": -1}
    emotion_score = emotional_weight.get(predicted_emotion, 0)
    if "significant emotional challenges" in mental_state:
        overall_score = emotion_score - 1
    elif "emotional challenges" in mental_state:
        overall_score = emotion_score - 0.5
    else:
        overall_score = emotion_score + 0.5
    if overall_score < -1:
        return f"Warning: You may be facing significant emotional difficulties. Current Emotion: {predicted_emotion}. Please consider seeking help."
    elif overall_score < 0:
        return f"Alert: You might be experiencing some emotional challenges. Current Emotion: {predicted_emotion}."
    elif overall_score > 1:
        return f"Positive: You seem to be in a good mental state. Current Emotion: {predicted_emotion}."
    else:
        return f"Neutral: Your mental state is stable, and you are feeling {predicted_emotion}."

def generate_bar_graph(emotion_labels, emotion_values):
    plt.figure(figsize=(10, 5))
    plt.bar(emotion_labels, emotion_values)
    plt.xlabel('Emotions')
    plt.ylabel('Probabilities')
    plt.title('Emotion Prediction Probabilities')
    plt.savefig(os.path.join(TEMP_IMAGE_DIR, 'emotion_graph.png'))
    plt.close()

if __name__ == "__main__":
    load_model()  # Load model at startup
    app.run(debug=True)
