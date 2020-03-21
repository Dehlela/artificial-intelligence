# Predicting Part-of-Speech of Words
This program (Main.py) learns and predicts part-of-speech (POS) of each word in a given sentence. 
It learns from NLTK's brown corpora, which has predefined POS tagsets, using the Viterbi algorithm.
Given an input sentence, it then predicts the POS of each of the input sentence's words.

# Training and Testing
_**Training**_ is done using first 10000 sentences from  the NLTK Brown Corpus.

_**Testing**_ is done using the next 500 sentences.

The program finally outputs the accuracy of correct tagging during testing.
Resultant accuracy is 95.21%

# UNK Tagging
The program also has an option to implement "UNK" tagging, in which all unknown/infrequent words are replaced by the word "UNK".
This increases the combined chance of these infrequent words being tagged correctly.
- **_Gerunds_** (ending with "ing") are replaced by "UNK-ing"
- **_Proper Nouns_** (words starting with capital letter and not at start of sentence) are replaced with "PN-UNK"

An accuracy of 95.38% is received after adding UNK-tagging.

# POS Tagging in another language - Hindi
The program "Hindi.py" analyses and learns POS tagging in the Indian language Hindi, including UNK-tagging.
Accuracy without UNK tags: 80.95%
Accuracy with UNK tags: 80.97%

# Running the program
The program uses Python 3 and is made to run on Linux systems. It requires the following installations before execution:
- nltk
- numpy
- pandas
- re