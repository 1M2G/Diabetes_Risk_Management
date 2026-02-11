# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Using Docker (Recommended)

```bash
# 1. Start all services
docker-compose up -d

# 2. Wait for services to start (30-60 seconds)
# 3. Open your browser
#    Frontend: http://localhost:3000
#    Backend: http://localhost:5000
```

### Manual Setup

```bash
# Terminal 1 - Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py

# Terminal 2 - Frontend
cd frontend
npm install
npm start
```

## 📝 First Steps

1. **Register** as a Patient or Doctor at http://localhost:3000/register
2. **Login** with your credentials
3. **Patients**: Log your glucose data and view insights
4. **Doctors**: Assign patients and monitor their data

## 🎯 Key Features

- ✅ AI-powered insulin dosage insights
- ✅ Real-time alerts for critical situations
- ✅ Pattern recognition and trend analysis
- ✅ Secure patient-doctor communication
- ✅ Modern, responsive UI

## 📚 Documentation

- **SETUP.md** - Detailed setup instructions
- **DEPLOYMENT.md** - Production deployment guide
- **README.md** - Complete feature overview

## 🆘 Need Help?

Check the troubleshooting section in SETUP.md or review the logs:
```bash
# Docker logs
docker-compose logs -f

# Backend logs (manual)
# Check terminal output

# Frontend logs (manual)
# Check terminal output
```

