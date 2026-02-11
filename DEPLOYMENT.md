# Deployment Guide

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd testing_101_2026_Final_Year_Models
   ```

2. **Set environment variables**
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your-secret-key-here
   JWT_SECRET_KEY=your-jwt-secret-key-here
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000
   - Database: localhost:5432

### Manual Setup

#### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   Create a `.env` file:
   ```env
   SECRET_KEY=your-secret-key
   DATABASE_URL=sqlite:///insulin_system.db
   JWT_SECRET_KEY=your-jwt-secret
   FLASK_ENV=development
   PORT=5000
   ```

5. **Run the backend**
   ```bash
   python run.py
   ```

#### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set environment variables**
   Create a `.env` file:
   ```env
   REACT_APP_API_URL=http://localhost:5000/api
   ```

4. **Run the frontend**
   ```bash
   npm start
   ```

## Production Deployment

### Backend (Flask)

#### Option 1: Using Gunicorn

1. **Install Gunicorn**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 "app:create_app()"
   ```

#### Option 2: Using uWSGI

1. **Install uWSGI**
   ```bash
   pip install uwsgi
   ```

2. **Create uwsgi.ini**
   ```ini
   [uwsgi]
   module = app:create_app()
   callable = app
   http = 0.0.0.0:5000
   processes = 4
   threads = 2
   ```

3. **Run uWSGI**
   ```bash
   uwsgi uwsgi.ini
   ```

### Frontend (React)

1. **Build the application**
   ```bash
   npm run build
   ```

2. **Serve with Nginx**
   - Copy `build/` contents to `/var/www/html/`
   - Configure Nginx (see `nginx.conf` example)

### Database Setup (PostgreSQL)

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql

   # macOS
   brew install postgresql
   ```

2. **Create database**
   ```sql
   CREATE DATABASE insulin_db;
   CREATE USER insulin_user WITH PASSWORD 'insulin_pass';
   GRANT ALL PRIVILEGES ON DATABASE insulin_db TO insulin_user;
   ```

3. **Update DATABASE_URL**
   ```env
   DATABASE_URL=postgresql://insulin_user:insulin_pass@localhost:5432/insulin_db
   ```

## Cloud Deployment

### Heroku

1. **Install Heroku CLI**
   ```bash
   heroku login
   ```

2. **Create apps**
   ```bash
   heroku create insulin-backend
   heroku create insulin-frontend
   ```

3. **Deploy backend**
   ```bash
   cd backend
   heroku git:remote -a insulin-backend
   git push heroku main
   ```

4. **Deploy frontend**
   ```bash
   cd frontend
   heroku git:remote -a insulin-frontend
   git push heroku main
   ```

### AWS (EC2 + RDS)

1. **Launch EC2 instance**
2. **Install dependencies**
3. **Set up RDS PostgreSQL**
4. **Configure security groups**
5. **Deploy using Docker or directly**

### DigitalOcean App Platform

1. **Connect GitHub repository**
2. **Configure build settings**
3. **Set environment variables**
4. **Deploy**

## Environment Variables Reference

### Backend
- `SECRET_KEY`: Flask secret key (required)
- `DATABASE_URL`: Database connection string
- `JWT_SECRET_KEY`: JWT token secret (required)
- `FLASK_ENV`: Environment (development/production)
- `PORT`: Port number (default: 5000)

### Frontend
- `REACT_APP_API_URL`: Backend API URL

## Security Checklist

- [ ] Change all default secrets
- [ ] Use HTTPS in production
- [ ] Enable CORS only for trusted domains
- [ ] Set up database backups
- [ ] Configure firewall rules
- [ ] Enable rate limiting
- [ ] Set up monitoring and logging
- [ ] Regular security updates

## Monitoring

### Health Check Endpoint
```bash
curl http://localhost:5000/api/health
```

### Logs
```bash
# Docker
docker-compose logs -f

# Manual
tail -f logs/app.log
```

## Troubleshooting

### Backend won't start
- Check database connection
- Verify environment variables
- Check port availability

### Frontend can't connect to backend
- Verify API URL in `.env`
- Check CORS settings
- Verify backend is running

### Database errors
- Check connection string
- Verify database exists
- Check user permissions

## Support

For issues or questions, please refer to the main README.md or open an issue in the repository.

