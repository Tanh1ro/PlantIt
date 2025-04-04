# Crop Prediction Flask App

A machine learning-powered crop recommendation system using Flask, TensorFlow, Google Gemini API, and web scraping for real-time agricultural insights.

## Features

✅ *Soil Classification*: Uses an image of the soil to classify its type.  
✅ *Weather Integration*: Fetches real-time weather data to assess climatic conditions.  
✅ *Crop Recommendation*: Suggests the best crops based on soil type and weather conditions.  
✅ *Google Gemini AI*: Enhances predictions with LLM-based insights.  
✅ *News Scraping*: Retrieves the latest agricultural news for farmers.  
✅ *Government Schemes*: Provides farmers with information about available schemes.

## Tech Stack

- *Frontend*: HTML, CSS, JavaScript  
- *Backend*: Flask  
- *Machine Learning*: TensorFlow, Keras, Google Gemini API  
- *Database*: SQLAlchemy (for managing data)  
- *Web Scraping*: BeautifulSoup, Newspaper3k, Feedparser  

## Libraries Used

| Library | Purpose |
|---------|---------|
| os, time | System operations and time management |
| requests | Handling HTTP requests |
| joblib | Model persistence and loading |
| pandas | Data manipulation and analysis |
| google.generativeai | Google Gemini AI integration |
| tensorflow, keras | Machine learning and deep learning models |
| numpy | Numerical computing |
| bs4 (BeautifulSoup) | Web scraping |
| flask | Web framework |
| werkzeug.utils | Secure file handling |
| sqlalchemy | Database ORM |
| feedparser | Parsing RSS feeds |
| newspaper3k | Extracting articles from news sources |

## Setup Instructions

### 1. Clone the Repository
bash
git clone https://github.com/your-username/crop-prediction-flask.git
cd crop-prediction-flask


### 2. Install Dependencies
bash
pip install -r requirements.txt


### 3. Add Google AI Studio API Key
- Get your API key from Google AI Studio.
- Store it in a .env file or update your code with the key.

### 4. Run the Flask App
bash
python app.py


### 5. Access the Application
Open a browser and go to:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Future Enhancements

- Implement a mobile-friendly UI.  
- Add multilingual support for farmers.  
- Expand crop disease detection using deep learning.  


