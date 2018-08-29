# Goal
The goal of this pipeline is to take a dataset of voice transcriptions (obtained from patient surveys and logs) and perform topic analysis on it to find out what patients call in and talk about. These clusters of topics will be then used to help generate categories that will be used in Speech2Classify for classifying patient calls in real-time.

## Use Case(s)
 * Generate categories to classify patient calls

## Dataset(s)
 * Danita's 50k Medica dataset

## Approach
We combined deep learning and traditional NLP approaches to embed and cluster the transcriptions in the dataset. Below are details of each step that was implemented:

 1. Preprocessing the transcriptions:
 :  After compiling all the transcriptions into one file, we decided to embed per sentence, so we split each transcription by periods and put these sentences into a text file, each on its own line. Then we tokenized each sentence into a list of words.
 :  Other approaches:
 :  :  
