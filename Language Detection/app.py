from flask import Flask, request, jsonify
from langdetect import detect, detect_langs, DetectorFactory

app = Flask(__name__)
DetectorFactory.seed = 0  # Ensures consistent results

# Dictionary to map language codes to full names
LANGUAGE_MAP = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "hi": "Hindi",
    "zh-cn": "Chinese",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "ja": "Japanese",
    "ko": "Korean"
}

@app.route('/detect_language', methods=['POST'])
def detect_language():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        detected_languages = detect_langs(text)
        response = {
            "detected_languages": [
                {
                    "code": str(lang).split(":")[0],
                    "name": LANGUAGE_MAP.get(str(lang).split(":")[0], "Unknown"),
                    "confidence": str(lang).split(":")[1]
                }
                for lang in detected_languages
            ]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)