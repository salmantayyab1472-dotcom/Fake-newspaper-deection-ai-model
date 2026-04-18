from flask import Flask, render_template

# Explicitly set the template folder to avoid path issues
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Enable debug mode for auto‑reload during development
    app.run(host='0.0.0.0', port=5000, debug=True)
