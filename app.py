from flask import Flask, request, jsonify
import werkzeug
import deepspeech
import soundfile as sf
import librosa
import re
import Levenshtein
import os

app = Flask(__name__)

# Load the DeepSpeech model
model_path = 'resources/deepspeech-0.9.3-models.tflite'
model = deepspeech.Model(model_path)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'result': 'Welcome'})

@app.route('/check_pronunciation', methods=['POST'])
def check_pronunciation():
    try:
        # Get the audio data from the request
        if 'file' not in request.files:
            return jsonify({'code': 400, 'message': 'No file attached'}), 400
        file = request.files['file']

        if file.content_type.split('/')[0] != 'audio':
            return jsonify({'code': 400, 'message': 'Invalid file type'}), 400

        # Save the audio file
        directory = "assets"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, werkzeug.utils.secure_filename(file.filename))
        file.save(file_path)

        # Load the audio data
        audio_data, sample_rate = librosa.load(file_path)

        # Transcribe the audio using DeepSpeech
        text = model.stt(audio_data)

        # Preprocess the text
        text = re.sub('[^a-zA-Z ]+', '', text.lower())

        # Get the word from the request
        word = request.form['word']

        # Define the phonemes
        phonemes = {
            'a': 'a',
            'b': 'b',
            'ch': 'ʧ',
            'd': 'd',
            'dh': 'ð',
            'e': 'e',
            'f': 'f',
            'g': 'ɡ',
            'gh': 'ɣ',
            'h': 'h',
            'i': 'i',
            'j': 'dʒ',
            'k': 'k',
            'kh': 'x',
            'l': 'l',
            'm': 'm',
            'n': 'n',
            'ng': 'ŋ',
            'ny': 'ɲ',
            'o': 'o',
            'p': 'p',
            'r': 'ɾ',
            's': 's',
            'sh': 'ʃ',
            't': 't',
            'th': 'θ',
            'u': 'u',
            'v': 'v',
            'w': 'w',
            'y': 'j',
            'z': 'z',
            'zh': 'ʒ'
        }

        # Convert the text to phonemes
        phoneme_text = ''.join([phonemes.get(letter, letter) for letter in text])

        # Convert the word to phonemes
        phoneme_word = ''.join([phonemes.get(letter, letter) for letter in word])

        # Calculate the Levenshtein distance between the two sequences
        distance = Levenshtein.distance(phoneme_text, phoneme_word)

        # Define a threshold for the distance
        threshold = 3

        # Return True if the distance is below the threshold, indicating correct pronunciation
        if distance <= threshold:
            return jsonify({'result': True, 'phoneme_text': phoneme_text, 'phoneme_word': phoneme_word})
        else:
            return jsonify({'result': False, 'phoneme_text': phoneme_text, 'phoneme_word': phoneme_word})

    except Exception as e:
        return jsonify({'code': 500, 'message': 'An error occurred', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
