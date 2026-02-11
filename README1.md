# Setup Instructions

## Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- PostgreSQL (optional, SQLite used by default)
- Docker and Docker Compose (optional, for containerized deployment)

## Quick Start

### Option 1: Docker (Easiest)

1. **Clone and navigate to the project**
   ```bash
   cd testing_101_2026_Final_Year_Models
   ```

2. **Start all services**
   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

### Option 2: Manual Setup

#### Step 1: Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file**
   Create a `.env` file in the `backend/` directory:
   ```env
   SECRET_KEY=your-secret-key-change-in-production
   DATABASE_URL=sqlite:///insulin_system.db
   JWT_SECRET_KEY=your-jwt-secret-change-in-production
   FLASK_ENV=development
   PORT=5000
   ```

5. **Initialize database and train ML model**
   ```bash
   python run.py
   ```
   The first run will:
   - Create the database
   - Train the ML model (this may take a few minutes)
   - Start the Flask server on http://localhost:5000

#### Step 2: Frontend Setup

1. **Open a new terminal and navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install Node dependencies**
   ```bash
   npm install
   ```

3. **Create environment file (optional)**
   Create a `.env` file in the `frontend/` directory:
   ```env
   REACT_APP_API_URL=http://localhost:5000/api
   ```

4. **Start the development server**
   ```bash
   npm start
   ```
   The frontend will open at http://localhost:3000

## First Time Setup

### Create Test Accounts

1. **Register as a Patient**
   - Go to http://localhost:3000/register
   - Select "Patient" as account type
   - Fill in your information
   - Example:
     - Email: patient@example.com
     - Password: password123
     - Age: 45
     - Gender: Male
     - BMI: 25
     - HbA1c: 7.0
     - Diabetes Type: Type 2

2. **Register as a Doctor**
   - Go to http://localhost:3000/register
   - Select "Doctor" as account type
   - Fill in your information
   - Example:
     - Email: doctor@example.com
     - Password: password123
     - License Number: MD12345
     - Specialization: Endocrinology

3. **Link Patient to Doctor**
   - Login as doctor
   - Go to Dashboard
   - Use "Assign Patient" feature with patient's email

## Testing the System

### As a Patient:

1. **Log Data**
   - Navigate to "Log Data"
   - Enter glucose level, insulin dosage, food intake, etc.
   - Submit to see AI-powered insights

2. **View History**
   - Check "View History" to see trends
   - Review charts and patterns

3. **View Dashboard**
   - See summary statistics
   - Check for alerts

### As a Doctor:

1. **View Patients**
   - See list of assigned patients
   - View patient summaries

2. **Review Alerts**
   - Check critical and high-priority alerts
   - Acknowledge alerts

3. **View Patient Details**
   - Click on a patient to see detailed view
   - Review ML analysis and patterns
   - View glucose trends

## Troubleshooting

### Backend Issues

**Port already in use:**
```bash
# Change PORT in .env file or kill the process using port 5000
```

**Database errors:**
- Ensure SQLite file permissions are correct
- For PostgreSQL, verify connection string

**ML Model not training:**
- Check that all dependencies are installed
- Ensure write permissions for `ml_models/` directory

### Frontend Issues

**Cannot connect to backend:**
- Verify backend is running on port 5000
- Check `REACT_APP_API_URL` in `.env`
- Check CORS settings in backend

**Build errors:**
- Delete `node_modules/` and `package-lock.json`
- Run `npm install` again

### Docker Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs

# Rebuild containers
docker-compose up -d --build
```

**Port conflicts:**
- Change ports in `docker-compose.yml`
- Stop conflicting services

## Development Tips

1. **Hot Reload**: Both frontend and backend support hot reload during development
2. **API Testing**: Use Postman or curl to test API endpoints
3. **Database Inspection**: Use SQLite browser or pgAdmin for database inspection
4. **Logs**: Check console output for debugging information

## Next Steps

- Review the main README.md for feature overview
- Check DEPLOYMENT.md for production deployment
- Customize ML model parameters in `backend/app/ml_service.py`
- Modify UI themes in `frontend/src/App.js`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error logs
3. Consult the documentation files
4. Open an issue in the repository

