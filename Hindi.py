from nltk.corpus import indian
from nltk import FreqDist, WittenBellProbDist
from nltk.corpus import brown
from nltk.util import ngrams
import pandas as pd
import numpy as np
import re


def print_heading(heading):
    print("\n")
    print("------------------------------------------------------")
    print(heading)
    print("------------------------------------------------------")


print("\nChoose from below: ")
print("1. Viterbi Algorithm\n2. Viterbi with UNK tagging")
choice = input("\nEnter your choice (1 or 2): ")
while choice != '1' and choice != '2':
    print("Incorrect Choice. Please try again.")
    choice = input("\nEnter your choice (1 or 2): ")

# ------------------------------------------------------ Splitting -----------------------------------------------------
print_heading("Splitting")

# Total 540 sentences
sentences = indian.tagged_sents('hindi.pos')

# First 440 as training set
train_sentences = sentences[:440]
print("Length of training set: ", len(train_sentences))

# Last 100 as testing set
test_sentences = sentences[440:540]
print("Length of testing set: ", len(test_sentences))


# --------------------------------------------- Pre-processing (UNK-Tagging) -------------------------------------------


def setup_unk_tagging(all_sentences):
    start_words_of_set = []
    all_words_of_set = []
    # Count of all words in training set
    for sentence in all_sentences:
        start_words_of_set.append(sentence[0])
        for (w, t) in sentence:
            all_words_of_set.append(w)

    return start_words_of_set, all_words_of_set


def get_infrequent(all_sentences):
    # List of words occurring once or twice
    infrequent_words_of_set = []
    infrequent_count = 0
    start_words_of_set, all_words_of_set = setup_unk_tagging(all_sentences)
    for (word, freq) in FreqDist(all_words_of_set).items():
        if freq <= 1:
            # print(word, freq)
            infrequent_count = infrequent_count + 1
            infrequent_words_of_set.append(word)

    return infrequent_words_of_set


def search_for_pattern(pattern, words_list):
    return_list = []
    for word in words_list:
        match = re.search(pattern, str(word))
        if match:
            # print(word)
            return_list.append(word)

    return return_list


# This function replaces words in words_list that occur in train_sentences with unk-tags
# Returns new list of word_tag_pairs with replaced UNK-tags
def replace_with_unk(words_list, replace_tag, all_sentences):
    replaced_count = 0
    new_train_sentences = list([])

    for sentence in all_sentences:
        component_sentences = []
        for (w, t) in sentence:
            if w not in words_list:
                component_sentences.append((w, t))
            else:
                component_sentences.append((replace_tag, t))
                replaced_count = replaced_count + 1
        new_train_sentences.append(component_sentences)

    print("Replaced Count:")
    print(replaced_count)
    return new_train_sentences


# UNK-tagging on train set ---------------------------------------------------------------------------------------------

def train_preprocessing(train_sentences):
    print("\nTrain set pre-processing...")
    start_words_train_set, all_words_train_set = setup_unk_tagging(train_sentences)
    infrequent_words_train_set = get_infrequent(train_sentences)

    print("\nInfrequent words:")
    print(len(infrequent_words_train_set))

    # Checking for -ing pattern (gerund) among infrequent words
    print("\nPlural Nouns:")
    train_plural_nouns = search_for_pattern(".+ों$", infrequent_words_train_set)

    # Replacing gerunds with "UNK-ing"
    train_sentences = replace_with_unk(train_plural_nouns, "UNK-o", train_sentences)
    return train_sentences, all_words_train_set


# UNK-tagging on test set ----------------------------------------------------------------------------------------------

def get_infrequent_for_test(all_words_test_set, all_words_train_set):
    infrequent_words_test_set = []
    for (word, freq) in FreqDist(all_words_train_set).items():
        if word in set(all_words_test_set) and freq <= 1:
            infrequent_words_test_set.append(word)

    # Finding words that don't occur in training set
    counter = 0
    for word in set(all_words_test_set):
        if word not in all_words_train_set and word not in infrequent_words_test_set:
            infrequent_words_test_set.append(word)
            counter = counter + 1

    return infrequent_words_test_set


def test_preprocessing(test_sentences, all_words_train_set):
    print("\nTest set pre-processing...")
    start_words_test_set, all_words_test_set = setup_unk_tagging(test_sentences)
    infrequent_words_test_set = get_infrequent_for_test(all_words_test_set, all_words_train_set)

    print("Infrequent or 0-occurring words: ")
    print(len(infrequent_words_test_set))

    # Checking for proper noun pattern among infrequent words
    print("\nPlural Nouns:")
    test_plural_nouns = search_for_pattern(".+ों$", infrequent_words_test_set)

    # Replacing o-ending with "UNK-o"
    test_sentences = replace_with_unk(test_plural_nouns, "UNK-o", test_sentences)
    return test_sentences


if choice == '2':
    print_heading("Preprocessing")
    train_sentences, all_words_train_set = train_preprocessing(train_sentences)
    test_sentences = test_preprocessing(test_sentences, all_words_train_set)

# ----------------------------------------- Preparation for Hidden Markov Model ----------------------------------------
print_heading("Preparation for Training ")
word_tag_pairs = []
all_tags = []

# Count of all words in training set
word_count = 0
for sentence in train_sentences:
    for (w, t) in sentence:
        word_tag_pairs.append((w, t))
        all_tags.append(t)
        word_count = word_count + 1

# Counting occurrences of each tag
tags_freq = FreqDist(all_tags)

print("Adding <s> and </s> tags for start and end of sentences...")
# Starting-words of each sentence
start_words = []
for sentence in train_sentences:
    start_words.append(sentence[0])

print("Creating bigrams for use in smoothing...")
# Creating ("<s>", "start word's tag") pairs for later use (Smoothing)
start_word_bigrams = []
for (w, t) in start_words:
    start_word_bigrams.append(('<s>', t))

# Ending-words of each sentence
end_words = []
i = 0
for sentence in train_sentences:
    end_words.append(sentence[-1])

# Fetching tag bigrams
tag_pairs_per_sentence = []
tag_pairs = []
i = 0
for sentence in train_sentences:
    tag_pairs_per_sentence.append([(t1, t2) for ((w1, t1), (w2, t2)) in ngrams(sentence, 2)])
    for pair in tag_pairs_per_sentence[i]:
        tag_pairs.append(pair)
    i = i + 1

print("Counting occurrence of 1 POS following another...")
# counting occurrence of 1 pos following another
tag_pair_freq = FreqDist(tag_pairs)

print("Counting occurrence of word-tag pairs...")
# counting occurrence of words with pos
word_tag_freq = FreqDist(word_tag_pairs)

# ----------------------------------------- Relative Frequencies with Smoothing ----------------------------------------
print_heading("Smoothing & Training")

smoothed_transition_prob = {}

# Preparing to also add smoothing for <s> and </s>
tag_pairs_with_start_words = tag_pairs.copy()
for pair in start_word_bigrams:
    tag_pairs_with_start_words.append(pair)

# Smoothing transition probabilities
# all transitions (q1, q2) are stored in tag_pairs
tags2 = set([t2 for (_, t2) in tag_pairs])
for tag in tags2:
    tags1 = [t1 for (t1, t2) in tag_pairs_with_start_words if t2 == tag]
    smoothed_transition_prob[tag] = WittenBellProbDist(FreqDist(tags1), bins=1e5)

# Smoothed probabilities for </s> as t2
end_tags = [t for (w, t) in end_words]
smoothed_transition_prob['</s>'] = WittenBellProbDist(FreqDist(end_tags), bins=1e5)

# Smoothing emission probabilities
# all emissions (q, w) are stored in word_tag_pairs
smoothed_emission_prob = {}
tags = set([t for (_, t) in word_tag_pairs])
for tag in tags:
    words = [w for (w, t) in word_tag_pairs if t == tag]
    smoothed_emission_prob[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

# -------------------------------------------- Training: Viterbi Algorithm ---------------------------------------------

unique_tags = np.unique(all_tags)


def get_tag(index):
    return unique_tags[index]


def predict_pos(input_words):
    viterbi = pd.DataFrame(data=None, index=unique_tags, columns=range(0, len(input_words)))

    # Back pointer to store final result
    back_pointer = []

    # initializing Viterbi table
    first_word = input_words[0]
    for tag in unique_tags:
        alpha = smoothed_transition_prob[tag].prob('<s>')
        beta = smoothed_emission_prob[tag].prob(first_word)
        viterbi.loc[tag, 0] = alpha * beta
        back_pointer.append(('<s>', tag, 0))

    # filling up Viterbi table for each input word
    word_counter = 1
    for word in input_words[1:]:
        for word_tag in unique_tags:
            viterbi_options = []
            # emission value
            beta = smoothed_emission_prob[word_tag].prob(word)
            for previous_tag in unique_tags:
                alpha = smoothed_transition_prob[word_tag].prob(previous_tag)
                viterbi_options.append(alpha * beta * viterbi.loc[previous_tag, word_counter - 1])
            # choose max from viterbi_options
            max_val = max(viterbi_options)
            viterbi.loc[word_tag, word_counter] = max_val
            back_pointer.append((get_tag(viterbi_options.index(max_val)), word_tag, word_counter))
        word_counter = word_counter + 1

    # final iteration
    viterbi_options = []
    for previous_tag in unique_tags:
        alpha = smoothed_transition_prob['</s>'].prob(previous_tag)
        viterbi_options.append(alpha * viterbi.loc[previous_tag, len(input_words) - 1])
    # choosing max
    max_val = max(viterbi_options)
    back_pointer.append((get_tag(viterbi_options.index(max_val)), '</s>', len(input_words)))

    # print(back_pointer)
    back_pointer.reverse()
    final_pos = []
    chosen_word_number = len(input_words)
    chosen_prev_tag = '</s>'

    for (previous_tag, word_tag, word_number) in back_pointer:
        if chosen_word_number == word_number and chosen_prev_tag == word_tag:
            chosen_word_number = chosen_word_number - 1
            chosen_prev_tag = previous_tag
            final_pos.append(word_tag)

    final_pos.pop(0)
    final_pos.reverse()
    return final_pos


# ----------------------------------------------------- Evaluation -----------------------------------------------------
print_heading("Evaluating")

total_tests = len(test_sentences)
correct_predictions = 0
test_counter = 1
incorrect_count = 0
acc_sum = 0
for sentence in test_sentences:
    print("Testing Sentence:", test_counter)
    incorrect_count = 0
    input_words = [w for (w, t) in sentence]
    actual_pos = [t for (w, t) in sentence]
    predicted_pos = predict_pos(input_words)
    i = 0
    for a, p in zip(actual_pos, predicted_pos):
        if a != p:
            incorrect_count = incorrect_count + 1
    correct_predictions = len(actual_pos) - incorrect_count
    accuracy = correct_predictions / len(actual_pos)
    # print(accuracy)
    acc_sum = acc_sum + accuracy
    test_counter = test_counter + 1

print_heading("Final Result")
print("Accuracy= ", (acc_sum / total_tests) * 100.0)
