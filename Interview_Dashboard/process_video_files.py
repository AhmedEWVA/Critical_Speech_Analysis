import os
from posixpath import split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import pandas as pd
from statistics import mean


matplotlib.use('TkAgg')
font = {'family': 'normal',
        'size': 24}
matplotlib.rc('font', **font)
color = mcolors.TABLEAU_COLORS['tab:blue']


CATEGORIES = {'EXPR': ['Anger', 'Disgust', 'Fear',
                       'Happiness', 'Sadness', 'Surprise', 'Neutral']}
dirname = os.path.dirname(__file__)
VIDEO_OUTPUT_FOLDER = dirname + '/static/video_output/'
VIDEO_FRAMES_FOLDER = dirname + '/static/video_output/frames/'
VIDEO_FACES_FOLDER = dirname + '/static/video_output/faces/'
mean_stdev = pd.read_csv(VIDEO_OUTPUT_FOLDER + 'analysis.csv')
mean_stdev = mean_stdev.set_index('emotion').to_dict()

counter = 0
video_length = len(os.listdir(VIDEO_FRAMES_FOLDER))
df = pd.read_csv(VIDEO_FACES_FOLDER + 'keypoints.csv')

detects = df['detect'].values
original_id_2_detected_id = []
prev_i = -1
for d in detects:
    if d == 1.0:
        prev_i += 1
    original_id_2_detected_id.append(max(0, prev_i))


def set_counter(value):
    global counter
    counter = value


def get_image():
    global counter
    if counter != video_length:
        img = '{:06}.png'.format(counter)
        get_face()
        counter += 1
    if counter == video_length:
        counter = 1
    return img, counter == video_length


def parse_txt(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = lines[1:]
        lines = [l.split(',') for l in lines]
        lines = [[float(d) for d in l] for l in lines]
    return np.array(lines)


video_emotions = pd.read_csv(
    VIDEO_OUTPUT_FOLDER + 'video_emotions.csv').to_numpy()[original_id_2_detected_id]


def get_emotion():
    return CATEGORIES['EXPR'][np.argmax(video_emotions[counter])]


def decode_string(string):
    x = string[1:-1].split(',')
    x = [int(s) for s in x]
    return x


def plot_rectange_cropped(image):
    emotion = get_emotion()
    if emotion == 'Happiness':
        # Chin and mouth
        cv2.rectangle(image, (0, 33), (111, 63), (0, 255, 0), 2)
        cv2.rectangle(image, (0, 70), (111, 100), (0, 255, 0), 2)
    elif emotion == 'Neutral':
        # Eye lid
        cv2.rectangle(image, (0, 12), (111, 38), (0, 255, 0), 2)
    else:
        # Eye brow
        cv2.rectangle(image, (0, 0), (111, 25), (0, 255, 0), 2)
    return image


imgs = np.array([f for f in os.listdir(VIDEO_FACES_FOLDER)
                if f.split('.')[1] == 'jpg'])
imgs = np.sort(imgs)
imgs = imgs[original_id_2_detected_id]

# Currently can come to delay


def get_face():
    f = cv2.imread(VIDEO_FACES_FOLDER + imgs[counter])
    #f = cv2.imread(VIDEO_FACES_FOLDER + '{:06}.png'.format(counter + 1))
    kpts = df.iloc[counter]
    prev_kpts = kpts
    if kpts['detect'] == 1.0:
        f = plot_rectange_cropped(f)
        prev_kpts = kpts
    else:
        f = plot_rectange_cropped(f)
    cv2.imwrite(dirname + '/static/video_output/faces_marked/' +
                '{:06}.png'.format(counter), f)


emotions = pd.read_csv(VIDEO_OUTPUT_FOLDER + 'video_emotions.csv')
max_emotions = pd.DataFrame(emotions.idxmax(axis=1))
max_emotions.rename(columns={0: 'Emotion'}, inplace=True)
max_emotions_count = max_emotions.groupby('Emotion').size(
).reset_index().rename(columns={0: 'Emotion_count'})
max_emotions_count['Emotion_count'] = max_emotions_count['Emotion_count'] / \
    len(emotions)


def get_conclusion():
    happy = max_emotions_count['Emotion_count'][2]
    neutral = max_emotions_count['Emotion_count'][3]
    less_happy = happy < mean_stdev['mean']['happy'] - \
        mean_stdev['stdev']['happy']
    more_neutral = neutral > mean_stdev['mean']['neutral'] + \
        mean_stdev['stdev']['neutral']
    happy = "{:.0%}".format(happy)
    neutral = "{:.0%}".format(neutral)
    feedbacks = []
    feedbacks.append("Your facial expression were happy {happy} of the time and neutral {neutral} of the time. ".format(
        happy=happy, neutral=neutral))
    if less_happy:
        feedbacks.append(
            "You have showed less happy expressions compared to good interviewees. Pay attention to the facial action units important for happiness next time!")
    if more_neutral:
        feedbacks.append(
            "You have showed more neutral expressions compared to good interviewees. Pay attention to the facial action unit 5 next time!")
    if len(feedbacks) == 1:
        feedbacks.append(
            "Your interview is good compared to others. Keep it like this!")
    return feedbacks


def get_video_score():
    emo_dict = {}
    max_emotions_count.apply(lambda x: emo_dict.update({x[0]: x[1]}), axis=1)
    scores = []
    features = ['neutral', 'happy', 'angry',
                'fear', 'sad', 'disgust', 'surprise']
    for feature in features:
        if not feature in emo_dict.keys():
            emo_dict[feature] = 0
        if emo_dict[feature] > mean_stdev['mean'][feature] + mean_stdev['stdev'][feature] or emo_dict[feature] < mean_stdev['mean'][feature] - mean_stdev['stdev'][feature]:
            scores.append(0)
        else:
            scores.append(1)
    return mean(scores)
