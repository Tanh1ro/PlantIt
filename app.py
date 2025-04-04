"""
Farmers Assistant PWA - A Progressive Web Application for Farmers

This application provides various services to farmers including:
- Soil type classification
- Crop prediction
- Weather information
- Government scheme recommendations
- Agricultural news updates
"""

import os
import time
import requests
import joblib
import pandas as pd
import google.generativeai as genai
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import feedparser
from newspaper import Article
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# Database configuration
# Using PostgreSQL database for storing user data and crop information
engine = create_engine("postgresql://postgres:Hanuman#30@localhost:5432/postgres")
db = scoped_session(sessionmaker(bind=engine))

# File upload configuration
UPLOAD_FOLDER = 'uploaded_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# API configurations
# Gemini API for generating scheme recommendations
genai.configure(api_key="AIzaSyBjVjiv-WB5PdSsH2yri2ap26GR-1JHoB8")

# News API configuration
NEWS_API_KEY = "9486ec6418b046d08213528320b97651"
NEWS_URL = "https://newsapi.org/v2/top-headlines"

def fetch_news():
    """
    Fetches agricultural news from NewsAPI
    Returns a list of news articles related to agriculture and farming
    """
    params = {
        "q": "agriculture OR farming OR crops OR rural",
        "country": "in",
        "language": "en",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(NEWS_URL, params=params)
    
    if response.status_code != 200:
        return []

    data = response.json()
    return data.get("articles", [])

# Load trained ML models for crop prediction
rf_model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask application
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    Args:
        filename: Name of the file to check
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_scheme_recommendations(user_input):
    """
    Generate government scheme recommendations using Gemini AI
    Args:
        user_input: Dictionary containing user preferences for different scheme types
    Returns:
        str: AI-generated recommendations for suitable government schemes
    """
    prompt = f"""
    Based on the following user inputs, recommend the most suitable Indian government schemes for farmers:
    {user_input}
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def fetch_google_news():
    """
    Fetches agricultural news from Google News RSS feed
    Returns a list of news articles with titles, URLs, and summaries
    """
    feed_url = "https://news.google.com/rss/search?q=indian+farmers&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(feed_url)

    articles = []
    for entry in feed.entries[:5]:  # Fetch top 5 news articles
        article_url = entry.link
        try:
            article = Article(article_url)
            article.download()
            article.parse()
            article.nlp()
            summary = article.summary
        except Exception:
            summary = "Summary not available"

        articles.append({
            "title": entry.title,
            "url": article_url,
            "summary": summary
        })
        time.sleep(1)  # Avoid getting blocked

    return articles

@app.route('/schemes', methods=['GET', 'POST'])
def schemes():
    """
    Handle government scheme recommendations
    GET: Display the schemes page
    POST: Process user inputs and generate scheme recommendations
    """
    recommendations = None
    if request.method == 'POST':
        user_input = {
            "income_support": request.form.get("income_support", "no").lower(),
            "insurance": request.form.get("insurance", "no").lower(),
            "credit": request.form.get("credit", "no").lower(),
            "irrigation": request.form.get("irrigation", "no").lower(),
            "soil_health": request.form.get("soil_health", "no").lower(),
            "livestock": request.form.get("livestock", "no").lower(),
            "market_access": request.form.get("market_access", "no").lower(),
            "organic_farming": request.form.get("organic_farming", "no").lower()
        }
        recommendations = get_scheme_recommendations(user_input)
    return render_template('schemes.html', recommendations=recommendations)

@app.route('/')
def upload():
    """
    Render the home page
    """
    return render_template('index.html')

@app.route('/news')
def news():
    """
    Display agricultural news from multiple sources
    """
    try:
        # Try NewsAPI first
        articles = fetch_news()
        if not articles:
            # Fallback to Google News if NewsAPI fails
            articles = fetch_google_news()
    except Exception as e:
        print(f"Error fetching news: {e}")
        articles = []
    return render_template('news.html', articles=articles)

@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    """
    Handle crop prediction based on soil and weather conditions
    GET: Display the crop prediction form
    POST: Process the form data and predict suitable crops
    """
    if request.method == 'GET':
        return render_template('crop_prediction.html')
    
    if request.method == 'POST':
        try:
            # Check if the request is JSON
            if not request.is_json:
                return jsonify({
                    'error': 'Invalid Content-Type',
                    'details': 'Content-Type must be application/json'
                }), 415

            data = request.get_json()
            
            # Validate required fields
            required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'soilPH', 'rainfall']
            for field in required_fields:
                if field not in data:
                    return jsonify({
                        'error': 'Missing required field',
                        'details': f'Field {field} is required'
                    }), 400

            N = float(data['nitrogen'])
            P = float(data['phosphorus'])
            K = float(data['potassium'])
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            ph = float(data['soilPH'])
            rainfall = float(data['rainfall'])

            input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
            
            prediction = rf_model.predict(input_data)
            predicted_crop = label_encoder.inverse_transform(prediction)[0]

            # Get detailed information about the recommended crop
            crop_info_prompt = f"""
            Provide detailed information about growing {predicted_crop} with the following conditions:
            - Temperature: {temperature}°C
            - Soil pH: {ph}
            - Rainfall: {rainfall}mm/year
            - Soil Nutrients: N={N}mg/kg, P={P}mg/kg, K={K}mg/kg
            - Humidity: {humidity}%

            Please provide the following information in a structured format:
            1. Optimal growing conditions:
               - Temperature range: [range]
               - Soil pH range: [range]
               - Rainfall requirements: [requirements]
               - Soil type: [type]
            2. Growing tips:
               - Planting season: [season]
               - Harvest time: [time]
               - Water requirements: [requirements]
               - Fertilizer needs: [needs]
            3. Alternative crops (list 3-5):
               - [Crop name]: [score]% suitable, [temperature range], [pH range], [rainfall requirements]
            """
            
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(crop_info_prompt)
            crop_info = response.text

            # Parse the response to extract structured information
            try:
                # Split the response into sections
                sections = crop_info.split('\n\n')
                
                # Extract optimal conditions
                conditions_section = next(s for s in sections if 'Optimal growing conditions' in s)
                conditions = {
                    'temperature_range': extract_value(conditions_section, 'temperature range'),
                    'ph_range': extract_value(conditions_section, 'soil pH range'),
                    'rainfall_requirements': extract_value(conditions_section, 'rainfall requirements'),
                    'soil_type': extract_value(conditions_section, 'soil type')
                }
                
                # Extract growing tips
                tips_section = next(s for s in sections if 'Growing tips' in s)
                tips = {
                    'planting_season': extract_value(tips_section, 'planting season'),
                    'harvest_time': extract_value(tips_section, 'harvest time'),
                    'water_requirements': extract_value(tips_section, 'water requirements'),
                    'fertilizer_needs': extract_value(tips_section, 'fertilizer needs')
                }
                
                # Extract alternative crops
                alternatives_section = next(s for s in sections if 'Alternative crops' in s)
                alternatives = parse_alternatives(alternatives_section)
                
                response_data = {
                    'predicted_crop': predicted_crop,
                    'temperature_range': conditions['temperature_range'],
                    'ph_range': conditions['ph_range'],
                    'rainfall_requirements': conditions['rainfall_requirements'],
                    'soil_type': conditions['soil_type'],
                    'planting_season': tips['planting_season'],
                    'harvest_time': tips['harvest_time'],
                    'water_requirements': tips['water_requirements'],
                    'fertilizer_needs': tips['fertilizer_needs'],
                    'alternative_crops': alternatives
                }
                
                return jsonify(response_data)
                
            except Exception as e:
                print(f"Error parsing crop information: {str(e)}")
                return jsonify({
                    'error': 'Failed to parse crop information',
                    'details': str(e)
                }), 500
                
        except Exception as e:
            return jsonify({
                'error': 'Failed to process request',
                'details': str(e)
            }), 400

    return render_template('crop_prediction.html')

def extract_value(text, key):
    """Extract a value from text based on a key"""
    try:
        lines = text.split('\n')
        for line in lines:
            if key.lower() in line.lower():
                return line.split(':')[-1].strip()
        return 'Not specified'
    except:
        return 'Not specified'

def parse_alternatives(text):
    """Parse alternative crops from text"""
    alternatives = []
    try:
        lines = text.split('\n')
        for line in lines:
            if '%' in line and ':' in line:
                parts = line.split(':')
                name = parts[0].strip()
                details = parts[1].strip()
                
                # Extract score
                score = next((int(s.strip('%')) for s in details.split() if '%' in s), 0)
                
                # Extract other details
                details_parts = details.split(',')
                temp_range = next((p.strip() for p in details_parts if '°C' in p), 'Not specified')
                ph_range = next((p.strip() for p in details_parts if 'pH' in p), 'Not specified')
                rainfall = next((p.strip() for p in details_parts if 'mm' in p), 'Not specified')
                
                alternatives.append({
                    'name': name,
                    'score': score,
                    'temperature_range': temp_range,
                    'ph_range': ph_range,
                    'rainfall_requirements': rainfall
                })
    except Exception as e:
        print(f"Error parsing alternatives: {str(e)}")
    
    return alternatives[:5]  # Return at most 5 alternatives

@app.route('/success', methods=['POST'])
def success():
    """
    Handle successful file upload for soil classification
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
        if f and allowed_file(f.filename):
            filename = secure_filename(f.filename)
            filename = filename.split(".")
            filename = filename[0] + str(time.time()) + "." + filename[1]
            f.save(os.path.join(UPLOAD_FOLDER, filename))
            global name
            name = UPLOAD_FOLDER + filename
            global city
            city = request.form.get("city")
            return redirect(url_for('predict'))
        else:
            return "File format not supported. Please upload a PNG, JPG, or JPEG file.", 400
    else:
        return "Method not allowed", 405

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    Predict soil type and recommend crops based on uploaded image
    """
    import tensorflow as tf
    import numpy as np
    from keras.preprocessing import image
    import h5py

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = tf.keras.models.load_model('soil_classifier.h5')

    global name
    image_path = name
    img = image.load_img(image_path, target_size=(128, 128))
    plt.imshow(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict_classes(img)
    if result[0]== 2:
        soil = "Red soil"
    elif result[0]== 1:
        soil = "Black soil"
    else:
        soil="Alluvial soil"
    plt.title(soil)
    path='static/predicted_images/' + "predicted" + str(time.time()) + ".jpeg"
    plt.savefig(path)

    global city

    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=271d1234d3f497eed5b1d80a07b3fcd1'

    r = requests.get(url.format(city)).json()
    global weather_data

    weather = {
        'city': city,
        'temperature': r['main']['temp'],
        'description': r['weather'][0]['description'],
        'icon': r['weather'][0]['icon'],
    }

    global min_temp
    global max_temp

    min_temp = r['main']['temp_min']
    max_temp = r['main']['temp_max']


    crop_types = db.execute("select crop,temp_min,temp_max,rainfall from soildb where soil_type = :id1 and temp_min <= :id2 and temp_max >= :id3",{"id1": soil, "id2": min_temp, "id3": max_temp}).fetchall()


    return render_template('results.html',soil=soil,crop_types=crop_types,predicted_path = path,weather_data=weather,city=city)

@app.route('/about-us')
def about_us():
    """
    Display the about us page
    """
    return render_template('about_us.html')

@app.route('/red')
def red():
    """
    Redirect to the file upload page
    """
    return render_template('upload_file.html')

if __name__ == '__main__':
    app.run(debug=True)
