from cmath import exp
import json
from statistics import mean
from flask import Flask, render_template, flash, redirect, request, jsonify
from werkzeug.utils import secure_filename
import os
import process_video_files as vf
import process_audio_files as af
import process_text_files as tf

from datetime import datetime

dirname = os.path.dirname(__file__)
UPLOAD_FOLDER = dirname + '/static/uploads/'
app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -y -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}'.mp4".format(
        input=avi_file_path, output=UPLOAD_FOLDER + output_name)).read()
    os.popen("rm '{input}'".format(input=avi_file_path))
    return True


@app.route("/", methods=['POST', 'GET'])
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )


@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if filename.split('.')[1] != 'mp4':
            convert_avi_to_mp4(os.path.join(
                app.config['UPLOAD_FOLDER'], filename), filename.split('.')[0])
            filename = filename.split('.')[0] + '.mp4'
    img = '{:06}.png'.format(0)
    max = vf.video_length - 1
    sentence, explanation, audio_explanation, audio_fluency, fluency_explanation = tf.get_features(
        0)
    return render_template('dashboard.html', filename=filename, img=img, max=max, sentence=sentence, explanation=explanation, audio_explanation=audio_explanation, audio_fluency=audio_fluency, fluency_explanation=fluency_explanation)


@app.route('/get_features', methods=['GET', 'POST'])
def get_features():
    if request.method == "POST":
        counter = request.form['counter']
        counter = counter.replace('"', '')
        print(counter)
        vf.set_counter(int(counter))
        return jsonify(counter=vf.counter)
    else:
        img, end_of_video = vf.get_image()
        sentence, explanation, audio_explanation, audio_fluency, fluency_explanation = tf.get_features(
            vf.counter)
        return jsonify(img=img, end_of_video=end_of_video, counter=vf.counter-1, sentence=sentence, explanation=explanation, audio_explanation=audio_explanation, audio_fluency=audio_fluency, fluency_explanation=fluency_explanation)


@app.route("/video", methods=['POST', 'GET'])
def video():
    feedbacks = vf.get_conclusion()
    return render_template(
        'video_detail.html', feedbacks=feedbacks
    )


@app.route("/audio", methods=['POST', 'GET'])
def audio():
    features = af.get_features()
    feedbacks = af.get_audio_conclusion()
    return render_template(
        'audio_detail.html',
        features=features,
        feedbacks=feedbacks
    )


@app.route("/text", methods=['POST', 'GET'])
def text():
    feedbacks = tf.get_text_conclusion()
    return render_template(
        'text_detail.html',
        feedbacks=feedbacks
    )


@app.route("/general_feedback", methods=['POST', 'GET'])
def general_feedback():
    all_feedbacks = []
    all_feedbacks.extend(vf.get_conclusion())
    all_feedbacks.extend(tf.get_text_conclusion())
    all_feedbacks.extend(af.get_audio_conclusion())
    score = int(mean([af.get_audio_score(),
                      tf.get_text_score(), vf.get_video_score()]) * 10)
    return render_template(
        'general_feedback.html',
        all_feedbacks=all_feedbacks,
        score=score
    )
