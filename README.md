# drug-classifier
This repository contains a neural network model and naive bayes model to classify drugs based on their effects on the user.
The dataset used here was orignally taken from https://api.psychonautwiki.org/ and modified to work with the models used here.

The aim of this project was for me to teach myself more about machine learning models and to gain experience in data processing. As such, I decided not to use popular data processing libraries such as Pandas because I wanted to learn from carrying out these data processing tasks myself.

I decided on the hyper-parameters used here as I found that they yielded the best results, after experimenting with various configurations. Although all the code for the models is written in Python, I wrote some Javascript code to 'one-hot encode' the data as the originally data I got from the API was in JSON format.

An algorithm of this type could see potential uses in helping emergency services workers to identify what type of drug an individual may have taken. Additionally, further exploratory data analysis could be done on this data to identify patterns such as finding out which effects are likely to make a drug belong to a certain class over other classes.
