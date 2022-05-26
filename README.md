# 2ndYearProject-NLP
2nd Year Projects - Group 16
by Gergo Gyori,  Constantin-Bogdan Craciun,   Jacob Andreas Sneding Rohde,   Nicki Andersen

{gegy, cocr, jacro, nican}@itu.dk

## Project description
In this project we train a LSTM and Bert model and each one is trained and run on multiple sets of data, with the difference between each data set being the preprocessing done on the original data.
The goal of this project is to examine how different combinations of preprocessing methods affect the sentiment
analysis accuracy through a simple RNN model and how it compares to a Hugging Face transformers model without heavy preprocessing.

## How to run
In order to reproduce our results you must run the "LSTM model", "Bert Uncased" notebooks in order to create, train and test the respective models, and save a csv file for the LSTM and 1 for the Bert model containing the test results inside the "results" folder.

Afterwards you need to change the "path" value in the "Result Viz" notebook to your newly created csv file and then simply run the "Result Viz" notebook in order to view the result figures.