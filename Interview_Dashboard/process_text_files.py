from statistics import mean
from matplotlib.pyplot import axis
import pandas as pd
import os
import ast
import process_video_files as vf


dirname = os.path.dirname(__file__)
TEXT_AND_AUDIO_FOLDER = dirname + '/static/text_and_audio/'
TEXT_OUTPUT_FOLDER = dirname + '/static/text_output/'
FLUENCY = ['High', 'Intermediate', 'Low']

features = pd.read_csv(TEXT_AND_AUDIO_FOLDER + 'text_and_audio.csv')
feedback = features['Feedback'].iloc[0]
features = features.drop(
    columns=['Unnamed: 0', 'Unnamed: 0.1.1', 'Unnamed: 0.1', 'Participant', 'Feedback'])
features['predictions_audio_sentiments'] = features['predictions_audio_sentiments'].apply(
    lambda x: x.replace('array', ''))
features['predictions_audio_fluency'] = features['predictions_audio_fluency'].apply(
    lambda x: x.replace('array', ''))
features = features.apply(
    lambda x: ast.literal_eval(str(x[0])), axis=0)
mean_stdev = pd.read_csv(TEXT_OUTPUT_FOLDER + 'final_page_mean_stdev.csv')
mean_stdev = mean_stdev.set_index('emotion').to_dict()
analysis = pd.read_csv(TEXT_OUTPUT_FOLDER + 'final_page_analysis.csv')
analysis = analysis.to_dict()


def reset_features():
    global sentences, audio_fluencies, text_explanations, audio_explanations, fluency_explanations
    sentences = [""]
    audio_fluencies = [""]
    text_explanations = [""]
    audio_explanations = [""]
    fluency_explanations = [""]


reset_features()


def get_features(counter):
    global sentences, audio_fluencies, text_explanations, audio_explanations, fluency_explanations
    current_features = features.loc[features['new_time_stamps'].map(
        lambda x: counter >= min(x) and counter <= max(x))]
    if len(current_features) != 0:
        sentences = list(current_features["tokenized_sentences"])
        audio_fluencies = list(
            current_features["predictions_audio_fluency"].apply(lambda x: FLUENCY[x.index(max(x))]))
        text_explanations = list(current_features["explanations"])
        audio_explanations = list(current_features["explanations_sentiments"])
        fluency_explanations = list(current_features["explanations_fluency"])
    if vf.counter == 1:
        reset_features()
    audio_explanation = audio_explanations[0]
    fluency_explanation = fluency_explanations[0]
    return ''.join(sentences), text_explanations[0], audio_explanation, audio_fluencies[0], fluency_explanation


def get_text_conclusion():
    feedbacks = []
    emotions = {'joy': 'joyful', 'anger': 'angry', 'fear': 'fearful'}
    feedbacks.append(
        "Your lexical expression were joyful {joy} of the time,  angry {anger} of the time and fearful {fear} of the time ".format(joy="{:.0%}".format(analysis['joy'][0]), anger="{:.0%}".format(analysis['anger'][0]), fear="{:.0%}".format(analysis['fear'][0])))
    for emotion in emotions.keys():
        if analysis[emotion][0] >= mean_stdev['mean'][emotion] + mean_stdev['stdev'][emotion]:
            feedbacks.append("You have used more {emotion_adj} sentences compared to good interviewees. Pay attention to the emotional message of your sentences.".format(
                emotion_adj=emotions[emotion]))
        elif analysis[emotion][0] <= mean_stdev['mean'][emotion] - mean_stdev['stdev'][emotion]:
            feedbacks.append("You have used less {emotion_adj} sentences compared to good interviewees. Pay attention to the emotional message of your sentences.".format(
                emotion_adj=emotions[emotion]))
    if analysis['combined_filler_nonflu'][0] > 0.1:
        feedbacks.append(
            "You have used some filler words. Pay attention to filler words in your answers. They make your interviews sound less professional")
    if len(feedbacks) == 1:
        feedbacks.append(
            "Your interview is good compared to others. Keep it like this!")
    return feedbacks


def get_text_score():
    scores = []
    features = ['neutral', 'joy', 'anger', 'fear', 'sadness']
    for feature in features:
        if analysis[feature][0] > mean_stdev['mean'][feature] + mean_stdev['stdev'][feature] or analysis[feature][0] < mean_stdev['mean'][feature] - mean_stdev['stdev'][feature]:
            scores.append(0)
        else:
            scores.append(1)

    return mean(scores)
