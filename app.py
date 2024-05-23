from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import get_answer

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.form  # This extracts the form data sent by the client
    openai_key = data['openai_key']
    question = data['question']
    answer = get_answer(openai_key,question)  # Call the get_answer function from main.py
    return jsonify({"response": answer})

if __name__ == '__main__':
    pass
    # app.run(host='0.0.0.0', port=8080)  # Runs the server on port 8080, development only