"""
Farmers Assistant PWA - A Progressive Web Application for Farmers

This application provides various services to farmers including:
- Soil type classification
- Crop prediction
- Weather information
- Government scheme recommendations
- Agricultural news updates
- Market price predictions

The application uses:
- Flask for the web framework
- TensorFlow for machine learning models
- Google Gemini API for AI-powered recommendations
- PostgreSQL for data storage
- Various APIs for weather, news, and market data
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
# Replace with your database credentials
engine = create_engine("postgresql://postgres:Hanuman#30@localhost:5432/postgres")
db = scoped_session(sessionmaker(bind=engine))

# File upload configuration
UPLOAD_FOLDER = 'uploaded_images/'  # Directory to store uploaded soil images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed image file extensions

# API configurations
# Gemini API for generating scheme recommendations
# Replace with your API key
genai.configure(api_key="AIzaSyBjVjiv-WB5PdSsH2yri2ap26GR-1JHoB8")

# News API configuration
NEWS_API_KEY = "9486ec6418b046d08213528320b97651"  # Replace with your NewsAPI key
NEWS_URL = "https://newsapi.org/v2/top-headlines"

def fetch_news():
    """
    Fetches agricultural news from NewsAPI
    
    Returns:
        list: A list of news articles related to agriculture and farming
        Each article contains title, description, url, and publishedAt
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
rf_model = joblib.load("rf_model.pkl")  # Random Forest model for crop prediction
label_encoder = joblib.load("label_encoder.pkl")  # Label encoder for crop names

# Initialize Flask application
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_scheme_recommendations(user_input):
    """
    Generate government scheme recommendations using Gemini AI
    
    Args:
        user_input (dict): Dictionary containing user preferences for different scheme types
        
    Returns:
        str: AI-generated recommendations for suitable government schemes
    """
    # Define scheme templates with detailed information
    scheme_templates = {
        "income_support": {
            "name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
            "description": "Direct income support scheme providing ₹6,000 per year to small and marginal farmers",
            "eligibility": "Small and marginal farmers with landholding up to 2 hectares",
            "benefits": "₹6,000 per year in three equal installments",
            "application": "Apply through PM-KISAN portal or nearest Common Service Centre"
        },
        "insurance": {
            "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
            "description": "Crop insurance scheme to protect farmers against crop loss due to natural calamities",
            "eligibility": "All farmers growing notified crops in notified areas",
            "benefits": "Premium subsidy up to 90%, comprehensive coverage against natural calamities",
            "application": "Apply through PMFBY portal or nearest insurance company office"
        },
        "credit": {
            "name": "Kisan Credit Card (KCC) Scheme",
            "description": "Provides farmers with timely access to credit for agricultural needs",
            "eligibility": "All farmers including tenant farmers and sharecroppers",
            "benefits": "Interest subvention, flexible repayment options, credit limit up to ₹3 lakh",
            "application": "Apply at any commercial bank, cooperative bank, or regional rural bank"
        },
        "irrigation": {
            "name": "Per Drop More Crop (PDMC)",
            "description": "Micro-irrigation scheme to promote water use efficiency",
            "eligibility": "All farmers including small and marginal farmers",
            "benefits": "55% subsidy for small/marginal farmers, 45% for others",
            "application": "Apply through State Agriculture Department or online portal"
        },
        "soil_health": {
            "name": "Soil Health Card Scheme",
            "description": "Provides soil health information and recommendations to farmers",
            "eligibility": "All farmers across India",
            "benefits": "Free soil testing, nutrient recommendations, and advisory services",
            "application": "Register at nearest Soil Testing Laboratory or online portal"
        },
        "livestock": {
            "name": "National Livestock Mission",
            "description": "Comprehensive scheme for livestock development and dairy farming",
            "eligibility": "Individual farmers, SHGs, and FPOs in livestock sector",
            "benefits": "Subsidy up to 50% for various livestock activities",
            "application": "Apply through State Animal Husbandry Department"
        },
        "market_access": {
            "name": "e-NAM (National Agricultural Market)",
            "description": "Online trading platform for agricultural commodities",
            "eligibility": "All farmers, traders, and FPOs",
            "benefits": "Better price discovery, transparent trading, reduced market fees",
            "application": "Register on e-NAM portal or through APMC market"
        },
        "organic_farming": {
            "name": "Paramparagat Krishi Vikas Yojana (PKVY)",
            "description": "Promotes organic farming through cluster approach",
            "eligibility": "Farmers willing to practice organic farming",
            "benefits": "₹50,000 per hectare for 3 years, certification support",
            "application": "Apply through State Agriculture Department"
        }
    }

    # Generate recommendations based on user input
    recommendations = []
    for scheme_type, value in user_input.items():
        if value == "yes" and scheme_type in scheme_templates:
            scheme = scheme_templates[scheme_type]
            recommendation = f"{scheme['name']}\n{scheme['description']}\nEligibility: {scheme['eligibility']}\nBenefits: {scheme['benefits']}\nHow to Apply: {scheme['application']}"
            recommendations.append(recommendation)

    # If no specific schemes selected, return all schemes
    if not recommendations:
        for scheme in scheme_templates.values():
            recommendation = f"{scheme['name']}\n{scheme['description']}\nEligibility: {scheme['eligibility']}\nBenefits: {scheme['benefits']}\nHow to Apply: {scheme['application']}"
            recommendations.append(recommendation)

    return "\n\n".join(recommendations)

def fetch_google_news():
    """
    Fetches agricultural news from Google News RSS feed
    
    Returns:
        list: A list of news articles with titles, URLs, and summaries
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
    
    Returns:
        template: Renders schemes.html with recommendations
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
    
    Returns:
        template: Renders index.html
    """
    return render_template('index.html')

@app.route('/news')
def news():
    """
    Display agricultural news from multiple sources
    
    Returns:
        template: Renders news.html with articles
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
    Handle crop prediction based on soil and weather data
    
    GET: Display the crop prediction form
    POST: Process the form data and return predictions
    
    Returns:
        template: Renders crop_prediction.html with results
    """
    if request.method == 'POST':
        try:
            # Get data from request
            data = request.get_json()
            
            # Validate required fields
            required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
                if not isinstance(data[field], (int, float)):
                    return jsonify({'error': f'Invalid value for {field}'}), 400

            # Get crop prediction
            prediction = get_crop_prediction(data)
            
            # Get crop information
            crop_info = get_crop_information(prediction)
            
            # Get alternative crops
            alternative_crops = get_alternative_crops(data, prediction)
            
            # Prepare response
            response = {
                'crop': prediction,
                'temperature_range': crop_info.get('temperature_range', 'N/A'),
                'ph_range': crop_info.get('ph_range', 'N/A'),
                'rainfall_requirements': crop_info.get('rainfall_requirements', 'N/A'),
                'soil_type': crop_info.get('soil_type', 'N/A'),
                'planting_season': crop_info.get('planting_season', 'N/A'),
                'harvest_time': crop_info.get('harvest_time', 'N/A'),
                'water_requirements': crop_info.get('water_requirements', 'N/A'),
                'fertilizer_needs': crop_info.get('fertilizer_needs', 'N/A'),
                'alternative_crops': alternative_crops
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return render_template('crop_prediction.html')

def get_crop_prediction(data):
    """
    Get crop prediction based on soil and environmental parameters
    """
    try:
        # Prepare input data in the correct order
        input_data = np.array([
            data['nitrogen'],
            data['phosphorus'],
            data['potassium'],
            data['temperature'],
            data['humidity'],
            data['ph'],
            data['rainfall']
        ]).reshape(1, -1)
        
        # Make prediction using the Random Forest model
        prediction = rf_model.predict(input_data)
        
        # Convert prediction to crop name using label encoder
        crop_name = label_encoder.inverse_transform(prediction)[0]
        
        return crop_name
    except Exception as e:
        print(f"Error in crop prediction: {e}")
        return "Unknown"

def get_crop_information(crop):
    """
    Get detailed information about the recommended crop
    """
    # Your existing crop information logic here
    # This is a placeholder - replace with your actual crop information logic
    return {
        'temperature_range': '20-30°C',
        'ph_range': '5.5-7.0',
        'rainfall_requirements': '1000-2000mm',
        'soil_type': 'Loamy',
        'planting_season': 'June-July',
        'harvest_time': 'October-November',
        'water_requirements': 'Regular irrigation',
        'fertilizer_needs': 'NPK 100:50:50 kg/ha'
    }

def get_alternative_crops(data, main_crop):
    """
    Get alternative crop recommendations
    """
    # Your existing alternative crops logic here
    # This is a placeholder - replace with your actual alternative crops logic
    return [
        {
            'name': 'Wheat',
            'match_score': 85,
            'temperature': '15-25°C',
            'ph_range': '6.0-7.5',
            'rainfall': '500-1000mm'
        },
        {
            'name': 'Maize',
            'match_score': 80,
            'temperature': '18-27°C',
            'ph_range': '5.5-7.0',
            'rainfall': '500-1000mm'
        }
    ]

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

@app.route('/price-prediction', methods=['GET', 'POST'])
def price_prediction():
    import pickle
    from datetime import datetime

    prediction_result = None

    # Load all models (consider loading once globally for optimization)
    with open("all_models.pkl", "rb") as f:
        models = pickle.load(f)

    if request.method == 'POST':
        try:
            commodity = request.form.get('commodity', '').strip().lower()
            date_input = request.form.get('date', '')  # Expecting DD/MM/YYYY format

            if not commodity:
                prediction_result = {
                    'quintal': '0.00',
                    'kilo': '0.00',
                    'error': 'Please enter a commodity name'
                }
            elif not date_input:
                prediction_result = {
                    'quintal': '0.00',
                    'kilo': '0.00',
                    'error': 'Please select a date'
                }
            elif commodity not in models:
                prediction_result = {
                    'quintal': '0.00',
                    'kilo': '0.00',
                    'error': f"No model found for '{commodity}'"
                }
            else:
                model_data = models[commodity]
                model = model_data["model"]
                start_date = model_data["start_date"]
                future_date = datetime.strptime(date_input, "%d/%m/%Y")
                days = (future_date - start_date).days
                predicted_price_quintal = model.predict([[days]])[0]
                
                # Convert quintal to kilo (1 quintal = 100 kg)
                predicted_price_kilo = predicted_price_quintal / 100
                
                prediction_result = {
                    'commodity': commodity.capitalize(),
                    'quintal': f"{round(predicted_price_quintal, 2)}",
                    'kilo': f"{round(predicted_price_kilo, 2)}",
                    'date': date_input
                }
        except Exception as e:
            prediction_result = {
                'quintal': '0.00',
                'kilo': '0.00',
                'error': f"Error: {str(e)}"
            }

    return render_template('price_prediction.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
