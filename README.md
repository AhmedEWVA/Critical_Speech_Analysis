# Critical Speech-Analysis with Explanations

We present an **A**utomatic **F**eedback generation **F**ramework for job **I**nterviews **AFFI**. We mainly rely on [MIT interview dataset](https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances) to give an interviewee valuable insights into his/her performance on categories such as fluency, word choice and emotional expressions. AFFI works by seperatly analyzing the video, voice and the spoken content. The different models that are used for analyzing interviews are given in the following.

## Lexical
The lexical part of the interview analysis is split up into the following parts.
### Sentiment analysis with BERT
The sentiment analysis with BERT was done with [ktrain](https://github.com/amaiya/ktrain) which also takes care of the explanation for sentiments. The notebook [Sentiment_analysis_with_BERT](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/lexical/Sentiment_analysis_with_BERT.ipynb) contains both the finetuning of BERT and the prediction part for the MIT dataset. The results are save in the [data folder](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/lexical/data).

### Word count like features
Word count like features have been create in the notebook [Textual preprocessing and simple feature extraction](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/lexical/Textual%20preprocessing%20and%20simple%20feature%20extraction.ipynb) after some basic preprocessing.

### Timestamp creation
To match the interview transcripts to the right moment in time for the dashboard it was necessary to create timestamps for each sentence. In the notebook [Timestamps](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/lexical/Timestamps.ipynb) these are created.

### Analysis
A detailed analysis of each interview is created in the notebook [Analysis_of_lexical_features](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/lexical/Analysis_of_lexical_features.ipynb). Here also the Graphs are generated that can later on be seen in the Dashboard. 

## Video
### Facial Action Units
The facial action units are extracted with [OpenFace 2.0](https://github.com/TadasBaltrusaitis/OpenFace). Follow the [instruction](https://github.com/TadasBaltrusaitis/OpenFace/wiki) to install and build OpenFace. Then, with [extract_all_faces.py](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/video/extract_all_faces.py), one can extract facial action unit from all videos from a folder. In our work, we did some modification to OpenFace, so only one frame from the video is extracted each second. The result for all 138 MIT videos is saved as [action_units.csv](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/video/result/action_units.csv).

### Emotion Recognition
In the notebook [Emotion_Recognition](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/video/Emotion_Recognition.ipynb), emotions are extracted. The result for all 138 MIT videos is saved as [emotion_output.csv](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/video/result/emotion_output.csv). Further, relationships between emotions, different MIT labels and facial action units are investigated. Here also the graphs are generated that can later on be seen in the Dashboard.

### Failed Attempts
- [get_emotions.py](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/video/get_emotions.py): A rule-based approach for emotion recognition with a visual feedback.
- [SmileDetection.ipynb](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/video/SmileDetection.ipynb): Smile detection in a video.

## Audio

In the audio part there are three folders:

### Data preparation
This folder contains the notebooks for data preparation before applying the models:
 - MIT_data_preparation: chunking audios and extracting features
 - Separating_speakers: seperating the speakers in the audio of an interview
 - Data_preparation_sentiments: preprocessing and extracting features from the sentiments classification dataset
 - Data_preparation_fluency: preprocessing and extracting features from the fluency classification dataset
    
### Models 
This folder contains the notebooks for the models that were used
 - Sentiment_analysis_model: training, testing and analyzing sentiments classification models
 - Fluency_classification: training, testing and analyzing fluency classification models
 - High_level_feature_extraction: exctration of high level features from the MIT dataset
 - High_level_features_analysis: analysis of the high level features
 - Audio_clustering_and_regression: performing clustering and regression of the raw features generated from the MIT dataset

### Csv files
This folder contains the csv files generated with previous notebooks
 - Audio_mit_data_correct: features and labels for mit interviews
 - Features_fluency: features and labels for the fluency dataset
 - Features_sentiments: features and labels for the sentiment classification dataset
 - High_level_features: high level features that are extracted from the interview of the MIT dataset
 - Csv_audio_files: contain a csv for each interview where each row contains features for on chunk of the interview

## Dashboard
The Dashboard is located in the folder [Interview Dashboard](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/Interview_Dashboard). To run the code, you first need to decompress the 4 zip files in the [static folder](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/Interview_Dashboard/static). Then you need to go to the root directory of the dashboard and run 
```bash
pip install requirements.txt
python3 -m flask run
```
Finally you can select to upload the [sample video](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/tree/main/Interview_Dashboard/video/PP89.avi) and see the dashboard displaying.


Currently the Dashboard doesn't run on the fly, so all the features to be displayed need to be pre-generated.
The file [generate_emotion_graphs.py](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/Interview_Dashboard/generate_emotion_graphs.py) generates the displayed emotion graphs with emotion output files for all 3 domains. The cropped face, the facial action units graph and valence and arousal graphs are currently generated with code from [Multitask-Emotion-Recognition-with-Incomplete-Labels](https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/blob/master/emotion_demo.py) by running `emotion_demo.py` with several modification so only one frame per second is extracted.

The explanation graphs for lexical and prosodic features are generated in the corresponding notebooks.

The displayed distribution graphs are generated within the notebooks for emotion recognition for each domain.

You can have a look on our [demo video](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/blob/main/screenshot_and_demo/dashboard_demo.m4v) and below a screenshot of the overview page:

![alt text](https://gitlab.lrz.de/lab-courses/xai-lab-ws-2022/tobias/critical-speech-analysis-with-explanations/-/raw/main/screenshot_and_demo/db_overview.png)
