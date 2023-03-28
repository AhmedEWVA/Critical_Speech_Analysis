import os
import pandas as pd
import numpy as np
import cv2
import argparse

emotions = {"anger": [4, 5, 7, 23],
            "happy": [6, 12],
            "sad": [1, 4, 15],
            "disgust": [4, 9, 10, 17],
            "contempt": [12],
            "anxiety": [1, 2, 4, 5, 7, 20, 26],
            "surprise": [1, 2, 5, 26]}


def convert_number_to_au(number):
    if number < 10:
        return "AU0" + str(number) + "_c"
    else:
        return "AU" + str(number) + "_c"


def get_emotions(face_features):
    timestamp_emotion = {}
    for index, row in face_features.iterrows():
        timestamp_emotion[index] = []
        for emotion in emotions:
            aus = emotions[emotion]
            if all(row[convert_number_to_au(au)] == 1 for au in aus):
                timestamp_emotion[index].append(emotion)
    return timestamp_emotion


def get_emotion_frequency(timestamp_emotions):
    emotions = sum(list(timestamp_emotions.values()), [])
    return {x: emotions.count(x)/len(timestamp_emotions) for x in emotions}


if __name__ == '__main__':
    # os.system('openFace/OpenFace/build/bin/FeatureExtraction -f MIT_Dataset/Videos/P1.avi -aus')
    parser = argparse.ArgumentParser(description='Video number')
    parser.add_argument('--video', dest='video',
                        type=str, help='Number of video')
    args = parser.parse_args()
    i = args.video if args.video else "1"
    face_features_1 = pd.read_csv('../processed/P' + i + '.csv')
    timestamp_emotions_full = get_emotions(face_features_1)
    timestamp_emotions = {k: v for k,
                          v in timestamp_emotions_full.items() if len(v) != 0}

    cap = cv2.VideoCapture('../MIT_Dataset/Videos/P' + i + '.avi')

    for timestamp in timestamp_emotions.keys():

        cap.set(cv2.CAP_PROP_POS_FRAMES, timestamp * cv2.CAP_PROP_FPS)
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Cut the video extension to have the name of the video
        #my_video_name = video_name.split(".")[0]

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (255, 0, 0)
        thickness = 2
        emotions = ' '.join(timestamp_emotions[timestamp])
        image = cv2.putText(frame, emotions, org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Second ' + str(timestamp), frame)

        cv2.waitKey(0)

        # Store this frame to an image
        # cv2.imwrite(my_video_name+'_second_'+str(timestamp)+'.jpg',gray)

    cap.release()
    cv2.destroyAllWindows()

    emotion_frequency = get_emotion_frequency(timestamp_emotions_full)
    for x in emotion_frequency.keys():
        print("You are " + x +
              " for {:.2%} of the time".format(emotion_frequency[x]))
