# PlantIt - A Progressive Web Application (PWA) for Farmers

**PlantIt** is a Progressive Web Application (PWA) designed to assist farmers with various agricultural needs using machine learning and AI technologies.

## Features

### 1. Soil Classification & Crop Prediction
- Upload soil images for classification.
- Get crop recommendations based on soil type.
- View detailed crop information and alternatives.
- Real-time weather data integration.

### 2. Government Schemes
- Personalized scheme recommendations.
- Detailed information about various government schemes.
- Eligibility criteria and application process.
- Benefits and subsidies information.

### 3. Agricultural News
- Real-time news updates from multiple sources.
- NewsAPI and Google News integration.
- Categorized agricultural news.
- News summaries and direct links.

### 4. Price Prediction
- Market price predictions for crops.
- Historical price data analysis.
- Market trends and insights.

### 5. Weather Information
- Real-time weather updates.
- Weather forecasts.
- Climate data integration.

---

## Technology Stack

- **Backend:** Flask (Python)
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** TensorFlow, scikit-learn
- **AI Integration:** Google Gemini API
- **Database:** PostgreSQL
- **APIs:** NewsAPI, Weather API
- **Other Tools:** BeautifulSoup, Newspaper3k

---

## Setup Instructions

### 1. Clone the repository:
```sh
git clone [repository-url]
cd PlantIt
```

### 2. Create a virtual environment:
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies:
```sh
pip install -r requirements.txt
```

### 4. Set up environment variables:
- Create a `.env` file.
- Add your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_news_api_key
```

### 5. Initialize the database:
```sh
python init_db.py
```

### 6. Run the application:
```sh
python app.py
```

### 7. Access the application:
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## Project Structure

```
Farmers_assistant-PWA/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── uploaded_images/      # User uploaded soil images
├── models/              # Machine learning models
└── README.md            # Project documentation
```

---

## API Keys Required

- Google Gemini API Key
- NewsAPI Key
- Weather API Key

---

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---


