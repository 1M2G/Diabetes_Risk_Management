from app import create_app, socketio
import os
import sys

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    # SocketIO async mode is set in __init__.py based on platform
    if sys.platform == 'win32':
        print("Running on Windows - using threading mode for SocketIO")
        socketio.run(
            app, 
            host='127.0.0.1', 
            port=port, 
            debug=debug,
            allow_unsafe_werkzeug=True,
            use_reloader=False  # Disable reloader to avoid port conflicts
        )
    else:
        # On Linux/Mac, use eventlet
        socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)

