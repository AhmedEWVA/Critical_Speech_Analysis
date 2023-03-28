import pandas as pd
import os
import ast
from statistics import mean

dirname = os.path.dirname(__file__)
AUDIO_OUTPUT_FOLDER = dirname + '/static/audio_output/'

audio_features = pd.read_csv(AUDIO_OUTPUT_FOLDER + 'audio_features.csv')
audio_features_dict = audio_features.to_dict()
thresholds = {'rate_of_speech': [3, 4], 'articulation_rate': [
    4, 5], 'balance': [0.7, 0.8]}
mean_stdev = pd.read_csv(AUDIO_OUTPUT_FOLDER + 'audio_statistic.csv')
mean_stdev = mean_stdev.set_index('emotion').to_dict()
threshold_fluency = 0.0319
fluency_low = ast.literal_eval(
    audio_features_dict['Fluency_percentage'][0])['low']
fluency = ast.literal_eval(audio_features_dict['Fluency_percentage'][0])
sentiments = ast.literal_eval(audio_features_dict['Sentiments_percentage'][0])


def get_features():
    features = []
    audio_features_dict['Fluency_percentage'][0] = max(
        fluency, key=fluency.get)
    for audio_feature in audio_features_dict.keys():
        if audio_feature != 'Sentiments_percentage':
            features.append(audio_feature + ': ' +
                            str(audio_features_dict[audio_feature][0]))
    return features


def get_audio_conclusion():
    speaking_rate = audio_features_dict['rate_of_speech'][0]
    articulation_rate = audio_features_dict['articulation_rate'][0]
    pauses = audio_features_dict['balance'][0]
    feedbacks = []
    if (speaking_rate > thresholds['rate_of_speech'][1]):
        feedback = "Your speaking rate is too high, maybe try to speak slower."
        feedbacks.append(feedback)
    elif (speaking_rate < thresholds['rate_of_speech'][0]):
        feedback = "You should speak faster."
        feedbacks.append(feedback)
    if (articulation_rate > thresholds['articulation_rate'][1]):
        feedback = "Your should articulate more."
        feedbacks.append(feedback)
    elif (articulation_rate < thresholds['articulation_rate'][0]):
        feedback = "You should articulate quicker."
        feedbacks.append(feedback)
    if (pauses > thresholds['balance'][1]):
        feedback = "You reduce the number of pauses."
        feedbacks.append(feedback)
    elif (pauses < thresholds['balance'][0]):
        feedback = "You should have more pauses."
        feedbacks.append(feedback)
    if len(feedbacks) == 0:
        feedbacks.append(
            "Your interview is good compared to others. Keep it like this!")
    return feedbacks


def get_audio_score():
    scores = []
    features = ['calm', 'happy', 'angry',
                'fear', 'sad', 'disgust', 'surprise']
    for feature in features:
        if sentiments[feature] > mean_stdev['mean'][feature] + mean_stdev['stdev'][feature] or sentiments[feature] < mean_stdev['mean'][feature] - mean_stdev['stdev'][feature]:
            scores.append(0)
        else:
            scores.append(1)
    for feature in thresholds.keys():
        if audio_features_dict[feature][0] > thresholds[feature][1] or audio_features_dict[feature][0] < thresholds[feature][0]:
            scores.append(0)
        else:
            scores.append(1)
    if fluency_low > threshold_fluency:
        scores.append(0)
    else:
        scores.append(1)
    return mean(scores)
