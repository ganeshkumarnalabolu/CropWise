# CropWise — Intelligent Crop Recommendation System

## Project Description
CropWise is a machine learning web application that recommends a suitable crop based on soil nutrient values and environmental conditions.

## Technologies Used
- Python 3.11
- Flask
- Scikit-learn
- Random Forest Classifier
- Pandas and NumPy
- HTML and CSS
- Docker and Gunicorn
- Git and GitHub

## Features
- Predicts a recommended crop using N, P, K, temperature, humidity, pH, and rainfall.
- Provides a web interface and prediction endpoint.
- Includes input validation and a health endpoint.
- Packages the application and trained model in Docker.
- Includes automated tests.

## Project Files
- `app.py` — Flask web application
- `train_model.py` — Model training script
- `model.pkl` — Trained model
- `Crop_recommendation.csv` — Crop dataset
- `templates/` — HTML templates
- `static/` — CSS styles
- `tests/` — Automated tests
- `Dockerfile` — Docker image instructions
- `.dockerignore` — Files excluded from Docker builds

## Run Locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Run with Docker
Build the image:

```bash
docker build -t cropwise:1.0 .
```

Run the container:

```bash
docker run -d --name cropwise-app -p 5000:5000 cropwise:1.0
```

Health check:

```bash
curl http://localhost:5000/health
```

## Tests
Run the tests:

```bash
pytest -v
```

## Deployment
- AWS EC2 URL: To be added after deployment.
- Google Cloud Compute Engine URL: To be added after deployment.

## GitHub Repository
https://github.com/ganeshkumarnalabolu/CropWise
