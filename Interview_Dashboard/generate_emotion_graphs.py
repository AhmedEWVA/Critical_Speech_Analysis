import pandas as pd
import os
import ast
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

matplotlib.use('TkAgg')
font = {'family': 'normal',
        'size': 24}
matplotlib.rc('font', **font)
color = mcolors.TABLEAU_COLORS['tab:blue']

dirname = os.path.dirname(__file__)
TEXT_OUTPUT_FOLDER = dirname + '/static/text_output/'
AUDIO_OUTPUT_FOLDER = dirname + '/static/audio_output/'
TEXT_AND_AUDIO_FOLDER = dirname + '/static/text_and_audio/'
VIDEO_OUTPUT_FOLDER = dirname + '/static/video_output/'
VIDEO_FACES_FOLDER = dirname + '/static/video_output/faces/'

features = pd.read_csv(TEXT_AND_AUDIO_FOLDER + 'text_and_audio.csv')
features = features.drop(
    columns=['Unnamed: 0', 'Unnamed: 0.1.1', 'Unnamed: 0.1', 'Participant', 'Feedback'])
features['predictions_audio_sentiments'] = features['predictions_audio_sentiments'].apply(
    lambda x: x.replace('array', ''))
features['predictions_audio_fluency'] = features['predictions_audio_fluency'].apply(
    lambda x: x.replace('array', ''))
features = features.apply(
    lambda x: ast.literal_eval(str(x[0])), axis=0)
df = pd.read_csv(VIDEO_FACES_FOLDER + 'keypoints.csv')

detects = df['detect'].values
original_id_2_detected_id = []
prev_i = -1
for d in detects:
    if d == 1.0:
        prev_i += 1
    original_id_2_detected_id.append(max(0, prev_i))

video_emotions = pd.read_csv(
    VIDEO_OUTPUT_FOLDER + 'video_emotions.csv').to_numpy()[original_id_2_detected_id]

CATEGORIES = {'EXPR_text': ['Happiness', 'Sadness', 'Fear', 'Anger', 'Neutral'],
              'EXPR_video': ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral'],
              'EXPR_audio': ['Anger', 'Calm', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']}
counter = 0
video_length = len(os.listdir(VIDEO_OUTPUT_FOLDER + 'AUs/'))

emotion_audio = [0, 0, 0, 0, 0, 0, 0]
emotion_text = [0, 0, 0, 0, 0]


def get_text_features(counter):
    global emotion_text
    current_features = features.loc[features['new_time_stamps'].map(
        lambda x: counter >= min(x) and counter <= max(x))]
    if len(current_features) != 0:
        emotions = list(current_features['prediction'])
        emotion_text = emotions[0]
    make_EXPR(emotion_text, counter, TEXT_OUTPUT_FOLDER)


def get_audio_features(counter):
    global emotion_audio
    current_features = features.loc[features['new_time_stamps'].map(
        lambda x: counter >= min(x) and counter <= max(x))]
    if len(current_features) != 0:
        emotions = list(current_features['predictions_audio_sentiments'])
        emotion_audio = emotions[0]
    make_EXPR(emotion_audio, counter, AUDIO_OUTPUT_FOLDER)


def make_EXPR(emotion, counter, dir):
    ax.set_xlim([0, 1])
    ax.set_title("Expressions")
    ax.barh(pos, emotion,
            align='center',
            height=0.5,
            tick_label=label_list,
            color=color)
    des = os.path.join(dir + 'EXPRs/', '{:06}.png'.format(counter))
    plt.savefig(des, dpi=60,
                bbox_inches='tight')
    ax.clear()


if __name__ == '__main__':
    label_list = CATEGORIES['EXPR_text']
    fig = plt.figure(2, figsize=(8, 9))
    ax = fig.add_subplot(111)
    pos = np.arange(len(label_list))
    im = ax.barh(pos, [0]*len(label_list),
                 align='center',
                 height=0.5,
                 tick_label=label_list)
    while counter < video_length:
        get_text_features(counter)
        counter += 1
    fig.clf()

    counter = 0
    label_list = CATEGORIES['EXPR_video']
    fig = plt.figure(2, figsize=(8, 9))
    ax = fig.add_subplot(111)
    pos = np.arange(len(label_list))
    im = ax.barh(pos, [0]*len(label_list),
                 align='center',
                 height=0.5,
                 tick_label=label_list)
    while counter < len(video_emotions):
        make_EXPR(video_emotions[counter], counter, VIDEO_OUTPUT_FOLDER)
        counter += 1
    fig.clf()

    label_list = CATEGORIES['EXPR_audio']
    fig = plt.figure(2, figsize=(8, 9))
    ax = fig.add_subplot(111)
    pos = np.arange(len(label_list))
    im = ax.barh(pos, [0]*len(label_list),
                 align='center',
                 height=0.5,
                 tick_label=label_list)
    while counter < video_length:
        get_audio_features(counter)
        counter += 1
