from flask import Flask, request, jsonify
import werkzeug
import deepspeech
import numpy as np
import soundfile as sf
import librosa
import re
import Levenshtein
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
  return jsonify({'result': 'Welcome'})

@app.route('/check_pronunciation', methods=['POST'])
def check_pronunciation():
    # Load the DeepSpeech model
    model_path = 'resources/deepspeech-0.9.3-models.tflite'
    model = deepspeech.Model(model_path)

    # Get the audio data from the request
    if 'file' not in request.files:
        return jsonify({'code': 400, 'message': 'No file attached'}), 400
    file = request.files['file']

    if file.content_type.split('/')[0] != 'audio':
        return jsonify({'code': 400, 'message': 'Invalid file type'}), 400

    fileName = werkzeug.utils.secure_filename(file.filename)
    directory = "assets"
    if not os.path.exists(directory):
      os.makedirs(directory)
    file.save("assets/" + fileName)
    # audio_data = request.files['audio'].read()

    # Load the audio data
    # audio_data, sample_rate = sf.read(str("assets/" + fileName))
    audio_data, sample_rate = librosa.load(str("assets/" + fileName))


    # Convert the audio data to the correct format
    if audio_data.dtype != 'int16':
        audio_data = np.int16(audio_data * (2 ** 15 - 1))

    # Get the word from the request
    word = request.form['word']

    # Transcribe the audio using DeepSpeech
    text = model.stt(audio_data)

    # Preprocess the text
    text = re.sub('[^a-zA-Z ]+', '', text.lower())

    # Define the Kiswahili phonemes
    # phonemes = {
    #     'a': 'ɑ',
    #     'b': 'b',
    #     'd': 'd',
    #     'e': 'ɛ',
    #     'f': 'f',
    #     'g': 'ɡ',
    #     'h': 'h',
    #     'i': 'i',
    #     'j': 'dʒ',
    #     'k': 'k',
    #     'l': 'l',
    #     'm': 'm',
    #     'n': 'n',
    #     'o': 'ɔ',
    #     'p': 'p',
    #     'r': 'r',
    #     's': 's',
    #     't': 't',
    #     'u': 'u',
    #     'v': 'v',
    #     'w': 'w',
    #     'y': 'j',
    #     'z': 'z',
    #     'ng': 'ŋ',
    #     'ny': 'ɲ',
    #     'sh': 'ʃ',
    #     'ch': 'tʃ',
    #     'kh': 'x',
    #     'gh': 'ɣ',
    #     'ph': 'pʰ',
    #     'th': 'tʰ',
    # }
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
    phoneme_text = ''
    for letter in text:
        if letter == ' ':
            phoneme_text += ' '
        elif letter in phonemes:
            phoneme_text += phonemes[letter]
        else:
            phoneme_text += letter

    # Convert the word to phonemes
    phoneme_word = ''
    for letter in word:
        if letter == ' ':
            phoneme_word += ' '
        elif letter in phonemes:
            phoneme_word += phonemes[letter]
        else:
            phoneme_word += letter

    # Calculate the Levenshtein distance between the two sequences
    distance = Levenshtein.distance(phoneme_text, phoneme_word)

    # Define a threshold for the distance
    threshold = 3

    # Return True if the distance is below the threshold, indicating correct pronunciation
    if distance <= threshold:
        return jsonify({'result': True, 'phoneme_text': phoneme_text, 'phoneme_word': phoneme_word})
    else:
        return jsonify({'result': False, 'phoneme_text': phoneme_text, 'phoneme_word': phoneme_word})

if __name__ == '__main__':
    app.run(debug=True)