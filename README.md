# Veritas - AI-Powered Fake News Detection

<div align="center">

![Veritas Logo](images/logo.jpg)

**A Chrome extension with hybrid ML/LLM backend for detecting fake news in real-time**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#features) • [Architecture](#architecture) • [Installation](#installation) • [API Documentation](#api-documentation) • [Testing](#testing)

</div>

---

## 🎯 Overview

Veritas is a comprehensive fake news detection system that combines traditional machine learning with modern LLM technology. It provides real-time analysis of news articles through a Chrome extension, helping users identify misinformation and make informed decisions.

### Key Highlights

- ✅ **Hybrid Detection System**: Combines fast ML predictions with explainable LLM analysis
- ✅ **Real-Time Analysis**: Chrome extension provides instant feedback while browsing
- ✅ **Multiple Detection Methods**: ML-only, LLM-only, or intelligent hybrid approach
- ✅ **Production-Ready**: Comprehensive testing, error handling, and deployment configuration
- ✅ **Explainable AI**: LLM provides reasoning and identifies red flags

---

## ✨ Features

### Detection Methods

1. **ML Detection (Fast & Free)**

   - TF-IDF vectorization + LinearSVC classifier
   - ~10-50ms response time
   - Runs locally, no API costs
   - Accuracy: ~85-90%

2. **LLM Detection (Accurate & Explainable)**

   - LangChain + OpenAI GPT models
   - Provides reasoning and red flags
   - Better understanding of context and nuance
   - Accuracy: ~90-95%

3. **Hybrid Detection (Best of Both)**
   - Uses ML for fast predictions
   - Automatically uses LLM for low-confidence cases
   - Optimizes for both speed and accuracy
   - Cost-effective (only uses LLM when needed)

### Chrome Extension

- One-click article analysis
- Automatic page content extraction
- Manual text input option
- Confidence scores and explanations
- Clean, intuitive UI

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐
│ Chrome Extension │
│  (Frontend)     │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   Flask API     │
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│   ML   │ │  LLM   │
│Detector│ │Detector│
└────────┘ └────────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ Hybrid Detector │
│  (Orchestrator) │
└─────────────────┘
```

### Component Details

#### Backend (`app.py`)

- Flask REST API with multiple endpoints
- Lazy loading of detectors for performance
- Comprehensive error handling
- Health checks and statistics

#### ML Detector (`detectors/ml_detector.py`)

- TF-IDF feature extraction
- LinearSVC classification
- Model persistence (pickle)
- Automatic retraining capability

#### LLM Detector (`detectors/llm_detector.py`)

- LangChain integration
- OpenAI GPT models
- Structured output parsing (Pydantic)
- Error handling and fallbacks

#### Hybrid Detector (`detectors/hybrid_detector.py`)

- Intelligent routing logic
- Confidence-based switching
- Combines ML and LLM results
- Graceful degradation

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Chrome/Chromium browser
- OpenAI API key (optional, for LLM features)

### Backend Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/veritas-fake-news.git
   cd veritas-fake-news
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY (optional)
   ```

4. **Run the server**
   ```bash
   python app.py
   ```
   Server will start on `http://localhost:8000`

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the project directory
5. The Veritas extension icon should appear in your toolbar

### Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=detectors --cov=app
```

---

## 📡 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `POST /predict`

Predict if text is fake news (uses hybrid by default)

**Request:**

```json
{
  "text": "Your article text here...",
  "method": "auto" // optional: "auto", "ml", or "llm"
}
```

**Response:**

```json
{
  "is_fake": false,
  "confidence": 0.85,
  "label": "REAL",
  "method": "ml",
  "text_length": 150,
  "reasoning": "...", // only for LLM/hybrid
  "red_flags": [] // only for LLM/hybrid
}
```

#### `POST /predict/ml`

Force ML-only prediction

**Request:**

```json
{
  "text": "Your article text here..."
}
```

#### `POST /predict/llm`

Force LLM-only prediction (requires API key)

**Request:**

```json
{
  "text": "Your article text here..."
}
```

#### `GET /health`

Health check endpoint

**Response:**

```json
{
  "status": "healthy",
  "ml_available": true,
  "llm_available": false
}
```

#### `GET /stats`

Get detector statistics

**Response:**

```json
{
  "ml_available": true,
  "llm_available": false,
  "confidence_threshold": 0.7,
  "fallback_enabled": true
}
```

#### `GET /evaluate`

Get model evaluation metrics

**Response:**

```json
{
  "accuracy": 0.89,
  "precision": 0.87,
  "recall": 0.91,
  "f1_score": 0.89,
  "confusion_matrix": {
    "true_negative": 450,
    "false_positive": 50,
    "false_negative": 40,
    "true_positive": 460
  },
  "test_samples": 1000,
  "model_type": "ML (TF-IDF + LinearSVC)"
}
```

#### `POST /retrain`

Retrain the ML model (requires API key if `RETRAIN_API_KEY` is set)

**Request Headers:**

```
X-API-Key: your-secret-key
```

---

## 🧪 Testing

The project includes comprehensive test coverage:

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Mock Tests**: LLM detector testing without API calls

### Test Structure

```
tests/
├── test_ml_detector.py      # ML detector unit tests
├── test_llm_detector.py     # LLM detector unit tests
├── test_hybrid_detector.py  # Hybrid detector tests
└── test_api.py              # API integration tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=detectors --cov=app --cov-report=html

# Run specific test file
pytest tests/test_ml_detector.py
```

---

## 📊 Model Performance

### ML Model Metrics

- **Accuracy**: ~85-90%
- **Precision**: ~87%
- **Recall**: ~91%
- **F1 Score**: ~89%
- **Latency**: 10-50ms per prediction

### LLM Model Metrics

- **Accuracy**: ~90-95%
- **Explainability**: High (provides reasoning)
- **Latency**: 500ms-2s per prediction
- **Cost**: ~$0.0001-0.001 per prediction

### Hybrid Approach

- **Best of both worlds**: Fast ML + Accurate LLM
- **Cost optimization**: Only uses LLM when needed
- **User experience**: Fast responses with explanations when needed

---

## 🛠️ Tech Stack

### Backend

- **Python 3.9+**: Core language
- **Flask**: Web framework
- **scikit-learn**: Machine learning
- **LangChain**: LLM integration
- **OpenAI API**: GPT models
- **Pydantic**: Data validation

### Frontend

- **JavaScript**: Extension logic
- **HTML/CSS**: UI components
- **Chrome Extension API**: Browser integration

### DevOps

- **pytest**: Testing framework
- **Gunicorn**: Production server

---

## 📁 Project Structure

```
veritas-fake-news/
├── app.py                    # Main Flask application
├── detectors/                 # Detection modules
│   ├── __init__.py
│   ├── ml_detector.py        # ML-based detector
│   ├── llm_detector.py       # LLM-based detector
│   └── hybrid_detector.py   # Hybrid orchestrator
├── tests/                    # Test suite
│   ├── test_ml_detector.py
│   ├── test_llm_detector.py
│   ├── test_hybrid_detector.py
│   └── test_api.py
├── scripts/                  # Chrome extension scripts
│   ├── content.js
│   └── popup.js
├── images/                   # Extension icons
├── popup.html                # Extension UI
├── manifest.json             # Extension manifest
├── requirements.txt          # Python dependencies
├── Procfile                  # Deployment config
└── README.md                 # This file
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# Flask Configuration
FLASK_DEBUG=False
PORT=8000

# Model Paths
MODEL_PATH=model.pkl
DATA_PATH=fake_or_real_news.csv

# OpenAI API Key (optional, for LLM features)
OPENAI_API_KEY=sk-your-key-here

# Hybrid Settings
HYBRID_CONFIDENCE_THRESHOLD=0.7
FALLBACK_TO_LLM=true

# Security
RETRAIN_API_KEY=your-secret-key
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---


<div align="center">


</div>
