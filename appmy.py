import os
import logging
import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for, flash, session
from PIL import Image
from torchvision import transforms
import base64
import mysql.connector

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management and flash messages

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",  # Replace with your MySQL username
    password="devaki@123456",  # Replace with your MySQL password
    database="plat"
)
cursor = conn.cursor()

logger.info("MySQL connected successfully")

# Set up image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjusted to match saved weights
        self.fc2 = nn.Linear(512, 9)  # Updated to match the saved model's output classes
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Model
def load_model(model, model_path="modelmsql.pth"):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    return model

# Predict Image
def predict_image(model, image_path, transform):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
        _, predicted_class = torch.max(output, 1)
        class_labels = [
            'eggplantspot', 'healthy', 'potatospot', 'powdery', 'rust',
            'tomatospot', 'bananaspot', 'chilispot', 'cottonspot'
        ]
        logger.info(f"Prediction: {class_labels[predicted_class.item()]}")
        return class_labels[predicted_class.item()]
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

# Initialize model
model = CNNModel()
model = load_model(model, "modelmsql.pth")

# Routes
@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    logger.info("Accessed the Home page")
    return render_template('home.html')

@app.route('/about')
def about():
    logger.info("Accessed the About page")
    return render_template('about.html')




@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        try:
            cursor.execute("""
                INSERT INTO contact_details (name, email, message)
                VALUES (%s, %s, %s)
            """, (name, email, message))
            conn.commit()
            logger.info("Contact form data saved to database")
        except Exception as e:
            logger.error(f"Error saving contact form data: {e}")

        return redirect('/')

    logger.info("Accessed the Contact page")
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check user credentials (for demonstration, hardcoded check)
        if username == 'plant' and password == '12345':  # Replace with actual authentication logic
            session['username'] = username  # Store username in session
            cursor.execute("""
                INSERT INTO log_details (username, password, status) 
                VALUES (%s, %s, %s)
            """, (username, password, 'success'))  # Insert the username, password, and status
            conn.commit()
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('home'))
        else:
            cursor.execute("""
                INSERT INTO log_details (username, password, status) 
                VALUES (%s, %s, %s)
            """, (username, password, 'failure'))  # Insert the username, password, and status
            conn.commit()
            logger.warning(f"Failed login attempt for {username}")
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/predict')
def predict():
    logger.info("Accessed the Predict page")
    return render_template('result.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return redirect(request.url)

        file = request.files['file']
        if file and file.filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}:
            img_bytes = file.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            image_path = "temp_image.jpg"
            with open(image_path, 'wb') as f:
                f.write(img_bytes)

            prediction = predict_image(model, image_path, transform)

            cursor.execute("""
                SELECT symptoms, medicine, cure FROM disease_details WHERE label = %s;
            """, (prediction,))
            result = cursor.fetchone()

            os.remove(image_path)

            if result:
                details = {
                    "Symptoms": result[0],
                    "Medicine": result[1],
                    "Cure": result[2]
                }
            else:
                details = {
                    "Symptoms": "Not Available",
                    "Medicine": "Not Available",
                    "Cure": "Not Available"
                }

            logger.info(f"Image uploaded and prediction made: {prediction}")
            return render_template('result.html', prediction=prediction, image_data=img_base64, details=details)

        logger.warning("Invalid file type uploaded")
        return redirect(request.url)
    except Exception as e:
        logger.error(f"Error during image upload and prediction: {e}")
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
