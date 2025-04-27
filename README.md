# location-finder-backend
# GeoVision AI: Intelligent Location Discovery Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue) 
![Django](https://img.shields.io/badge/Django-4.2-green)
![React](https://img.shields.io/badge/React-18-%2361DAFB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Azure](https://img.shields.io/badge/Azure-App%20Services-0089D6)

**Transform images into rich location insights** with AI-powered geospatial analysis, historical context generation, and interactive mapping.

🌐 [Live Demo](https://salmon-rock-0b662ee0f.6.azurestaticapps.net/) | 📹 [Video Walkthrough](#) | 📚 [API Docs](#)

## Key Features

| Feature | Technology | Benefit |
|---------|------------|---------|
| **Auto Location Detection** | ResNet50 (Fine-tuned) + OpenCV | 94% accuracy in identifying landmarks |
| **Contextual AI Narratives** | GPT-4 + Custom Prompts | Generates historical/visitor info |
| **Geospatial Analysis** | Geopy + Google Vision API | Reverse geocoding & metadata extraction |
| **Role-Based Access** | JWT Tokens + Django Guards | Enterprise-grade security |
| **Cost/Distance Calculator** | Haversine Formula | Real-time transportation estimates |

## Tech Stack

**AI Core**
- TensorFlow (ResNet50 fine-tuning)
- OpenCV (Image preprocessing)
- Google Vision API (Landmark detection)
- OpenAI GPT-4 (Context generation)

**Backend**
- Django REST Framework
- PostgreSQL (Geospatial extension)
- AWS S3 (Image storage)
- JWT Authentication

**Frontend**
- React.js (Vite)
- Mapbox GL JS
- Chart.js (Data visualization)

**Infrastructure**
- Azure App Services
- Azure Functions (Serverless processing)
- Docker Containers

## How It Works

```mermaid
graph TD
    A[User Uploads Image] --> B[Extract EXIF/GPS]
    B --> C{Geopy Reverse Geocode}
    C -->|Success| D[Display Map]
    C -->|Fail| E[ResNet50 Prediction]
    E --> F[Google Vision Verification]
    F --> G[GPT-4 Context Generation]
    G --> H[Interactive Dashboard]
