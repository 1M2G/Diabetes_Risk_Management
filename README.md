# Diabetes Risk Management review document
# Insulin Management System - Full Stack Application

A comprehensive web-based insulin management system with ML-powered insights, designed for both patients and healthcare providers.

## Features

- **Patient Dashboard**: Log glucose levels, food intake, physical activity, and view personalized insights
- **Doctor Dashboard**: Monitor patients, receive automated alerts, review ML suggestions, and manage patient care
- **ML-Powered Insights**: Pattern recognition, risk alerts, and dosage trend analysis with explainability
- **Real-time Alerts**: WebSocket-based notifications for critical events
- **Secure & Compliant**: JWT authentication, encrypted data, role-based access control
- **Modern UI**: Responsive, accessible, and user-friendly interface

## Tech Stack

### Backend
- Flask (Python) - REST API
- SQLAlchemy - ORM
- PostgreSQL/SQLite - Database
- JWT - Authentication
- WebSockets - Real-time communication
- scikit-learn, SHAP - ML models and explainability

### Frontend
- React - UI Framework
- Material-UI - Component Library
- Axios - HTTP Client
- Socket.io-client - Real-time updates
- Chart.js - Data visualization

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── routes.py
│   │   ├── ml_service.py
│   │   ├── auth.py
│   │   └── utils.py
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.js
│   ├── package.json
│   └── public/
├── ml_models/
│   └── train_model.py
├── docker-compose.yml
└── README.md
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL (optional, SQLite used by default)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python run.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Docker Setup
```bash
docker-compose up -d
```

## Environment Variables

Create `.env` files in backend/ and frontend/ directories:

**Backend (.env)**
```
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///insulin_system.db
JWT_SECRET_KEY=your-jwt-secret
FLASK_ENV=development
```

## Usage

1. Register as Patient or Doctor
2. Patients can log daily data (glucose, food, activity)
3. Doctors can view patient summaries and ML-generated insights
4. System provides automated alerts for high-risk situations
5. All ML suggestions include explanations and require doctor approval

## Safety & Compliance

- All ML outputs are advisory only
- Doctor oversight required for all critical decisions
- Data encryption at rest and in transit
- HIPAA/GDPR considerations built-in
- Audit logging for all actions
