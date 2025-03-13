from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('plagiarism.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, content TEXT)''')
    return conn

def check_plagiarism(text, existing_texts):
    if not existing_texts:  # If no existing texts, return 0 similarity
        return 0.0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text] + existing_texts)
    if vectors.shape[0] < 2:  # Ensure there are at least two vectors to compare
        return 0.0
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    return max(similarities) if similarities.size > 0 else 0.0

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/check', methods=['POST'])
def check_plagiarism():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Text input is missing"}), 400

        input_text = data["text"].strip()

        # Dummy plagiarism detection logic
        # Example: If the text contains "Machine learning", assume it is plagiarized
        similarity = 0.9 if "Machine learning" in input_text else 0.2  # Example similarity score

        return jsonify({
            "similarity": similarity,
            "is_plagiarized": similarity >= 0.8  # If similarity is 0.8 or higher, mark as plagiarized
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
