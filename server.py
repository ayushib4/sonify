from flask import Flask, request, jsonify, send_from_directory
import whisper
import tempfile
import os
import speech_recognition as sr
from together import Together
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

model = whisper.load_model("base")
client = Together(api_key=os.environ.get("TOGETHER_AI_KEY"))
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.5
recognizer.energy_threshold = 1000
recognizer.dynamic_energy_threshold = True
recognizer.dynamic_energy_adjustment_damping = 0.15
recognizer.dynamic_energy_adjustment_ratio = 2.5

mappings = {}


@app.route("/")
def index():
    return send_from_directory("", "index.html")


@app.route("/styles.css")
def styles():
    return send_from_directory("", "styles.css")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request"}), 400
    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        tmp_filename = tmp.name

    result = model.transcribe(tmp_filename)
    text = result["text"]

    os.remove(tmp_filename)

    prompt = f"""
    You are an expert in proofreading and correcting texts, specifically focusing on proper nouns such as names of people, places, organizations, and brands. Given the following transcribed text, identify any proper nouns that may be misspelled or incorrectly transcribed, and provide a corrected version of the text with the proper nouns corrected while keeping the rest of the text unchanged. 

    To assist you, the user has provided the following mapping of misspelled transcriptions to the corrected versions: {', '.join([f'{m} => {mappings[m]}' for m in mappings])}

    Transcribed Text: {text}

    PLEASE ONLY RETURN THE CORRECTED TEXT WITHOUT ANY ANNOTATIONS. RETURN NOTHING IF YOU HAVE NOTHING TO CORRECT.
    """
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"],
    )
    response = response.choices[0].message.content
    return jsonify({"text": response})


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json()
    original_text = data.get("original_text")
    suggested_text = data.get("suggested_text")

    if not original_text or not suggested_text:
        return jsonify({"error": "Missing original_text or suggested_text"}), 400

    mappings[original_text] = suggested_text

    return jsonify({"message": "Thank you for your suggestion!"})


if __name__ == "__main__":
    app.run(debug=True)
