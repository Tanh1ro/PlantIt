# import os
# import time
# import requests
# from flask import Flask, flash, request, redirect, url_for
# from matplotlib import pyplot as plt
# from werkzeug.utils import secure_filename
# from sqlalchemy import create_engine
# from sqlalchemy.orm import scoped_session, sessionmaker
# from flask import *

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# engine = create_engine("postgresql://postgres:Hanuman#30@localhost:5432/postgres")
# db = scoped_session(sessionmaker(bind=engine))
# UPLOAD_FOLDER = 'uploaded_images/'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# name = None
# min_temp=int()
# max_temp=int()
# city = ""


# app = Flask(__name__, template_folder='templates')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/', methods = ['GET','POST'])
# def upload():
#     return render_template('index.html')

# @app.route('/service-worker.js')
# def sw():
#     return app.send_static_file('service-worker.js'), 200, {'Content-Type': 'text/javascript'}

# @app.route('/manifest.json')
# def manf():
#     return app.send_static_file('manifest.json')

# @app.route('/news', methods = ['GET'])
# def news():
#     return render_template('news.html')

# @app.route('/schemes', methods = ['GET'])
# def schemes():
#     return render_template('schemes.html')

# @app.route("/red")
# def red():
#     return render_template("upload_file.html")

# @app.route('/success', methods = ['POST'])
# def success():
#     if request.method == 'POST':
#         f = request.files['file']
#         global city
#         city=request.form.get("city")
#         if f and allowed_file(f.filename):
#             filename = secure_filename(f.filename)
#             filename = filename.split(".")
#             filename = filename[0] + str(time.time()) + "." + filename[1]
#             f.save(os.path.join(UPLOAD_FOLDER,filename))
#             global name
#             name = UPLOAD_FOLDER + filename
#             return redirect(url_for('predict'))
#         else:
#             return "File format not supported"
#     else:
#         return "Get request not supported"

# @app.route('/predict', methods=['GET','POST'])
# def predict():
#     import tensorflow as tf
#     import numpy as np
#     from keras.preprocessing import image
#     import h5py

#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#     model = tf.keras.models.load_model('soil_classifier.h5')

#     global name
#     image_path = name
#     img = image.load_img(image_path, target_size=(128, 128))
#     plt.imshow(img)
#     img = np.expand_dims(img, axis=0)
#     result = model.predict_classes(img)
#     if result[0]== 2:
#         soil = "Red soil"
#     elif result[0]== 1:
#         soil = "Black soil"
#     else:
#         soil="Alluvial soil"
#     plt.title(soil)
#     path='static/predicted_images/' + "predicted" + str(time.time()) + ".jpeg"
#     plt.savefig(path)

#     global city

#     url = 'http://api.openweathermap.org/data/2.5/weather?q={}&units=metric&appid=271d1234d3f497eed5b1d80a07b3fcd1'

#     r = requests.get(url.format(city)).json()
#     global weather_data

#     weather = {
#         'city': city,
#         'temperature': r['main']['temp'],
#         'description': r['weather'][0]['description'],
#         'icon': r['weather'][0]['icon'],
#     }

#     global min_temp
#     global max_temp

#     min_temp = r['main']['temp_min']
#     max_temp = r['main']['temp_max']


#     crop_types = db.execute("select crop,temp_min,temp_max,rainfall from soildb where soil_type = :id1 and temp_min <= :id2 and temp_max >= :id3",{"id1": soil, "id2": min_temp, "id3": max_temp}).fetchall()


#     return render_template('prediction.html',soil=soil,crop_types=crop_types,predicted_path = path,weather_data=weather,city=city)

# if __name__ == '__main__':
#     app.run(debug = True)




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
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import feedparser
from newspaper import Article


# Database setup
engine = create_engine("postgresql://postgres:Hanuman#30@localhost:5432/postgres")
db = scoped_session(sessionmaker(bind=engine))

UPLOAD_FOLDER = 'uploaded_images/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure Gemini API
genai.configure(api_key="AIzaSyAk8AJy0Emha5FHsFVXpcZpgQGX9ndkM-8")

NEWS_API_KEY = "9486ec6418b046d08213528320b97651"
NEWS_URL = "https://newsapi.org/v2/top-headlines"

def fetch_news():
    params = {
    "q": "agriculture OR farming OR crops OR rural",
    "country": "in",
    "language": "en",
    "apiKey": NEWS_API_KEY
}
    response = requests.get(NEWS_URL, params=params)
    
    print(response.json())
    
    if response.status_code != 200:
        return []

    data = response.json()
    return data.get("articles", [])


# Load trained ML models
rf_model = joblib.load("rf_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_scheme_recommendations(user_input):
    prompt = f"""
    Based on the following user inputs, recommend the most suitable Indian government schemes for farmers:
    {user_input}
    """
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def fetch_google_news():
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
    return render_template('index.html')

@app.route('/news')
def news():
    articles = fetch_news()  # Try NewsAPI first
    if not articles:  
        articles = fetch_google_news()  # Use Google RSS if NewsAPI fails
    return render_template('news.html', articles=articles)

# @app.route('/news')
# def news():
#     articles = newsapi.get_everything(q="agriculture OR farming OR crops", language="en", sort_by="publishedAt", page_size=10)
    
#     if not articles['articles']:
#         return render_template('news.html', articles=None)

#     return render_template('news.html', articles=articles['articles'])

# @app.route('/schemes', methods=['GET'])
# def schemes():
#     return render_template('schemes.html')

@app.route('/crop-prediction', methods=['GET', 'POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        prediction = rf_model.predict(input_data)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        soil_data = {"N": N, "P": P, "K": K, "temperature": temperature, 
                     "humidity": humidity, "ph": ph, "rainfall": rainfall}
        
        prompt = f"""
        The predicted crop based on soil and climate conditions is "{predicted_crop}". 
        The soil conditions are: {soil_data}. 
        Is this the best crop to grow in this environment? 
        If not, suggest a better alternative based on sustainable farming practices.
        """
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        gemini_recommendation = response.text

        return render_template('crop_prediction.html', predicted_crop=predicted_crop, recommendation=gemini_recommendation)
    
    return render_template('crop_input.html')

@app.route('/about-us')
def about_us():
    return render_template('about_us.html')

if __name__ == '__main__':
    app.run(debug=True)
