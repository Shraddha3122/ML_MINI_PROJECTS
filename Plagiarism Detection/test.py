from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample reference texts (You can replace these with a database or dataset)
reference_texts = [
    "This is an example of original content.",
    "Machine learning is a field of artificial intelligence.",
    "Deep learning models are used for image recognition."
]

# Function to check plagiarism
def check_plagiarism(input_text):
    texts = reference_texts + [input_text]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    similarity_matrix = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    max_similarity = max(similarity_matrix[0])

    return max_similarity  # Returns similarity score (0 to 1)

@app.route('/check', methods=['POST'])
def check():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    input_text = data['text']
    similarity_score = check_plagiarism(input_text)

    # If similarity < 0.8, it's not plagiarized
    is_plagiarized = similarity_score >= 0.8

    return jsonify({
        "similarity": round(similarity_score, 2),
        "is_plagiarized": is_plagiarized
    })

if __name__ == '__main__':
    app.run(debug=True)
