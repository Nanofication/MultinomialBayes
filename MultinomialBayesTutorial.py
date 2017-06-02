"""

Following NLTK tutorial on classifying sentences and intents

"""

import nltk
from nltk.stem.lancaster import LancasterStemmer

#Word Stemmer. Reduce words to the root forms. Better for classifying
stemmer = LancasterStemmer()

# 3 classes of training data. Play around with this
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})
training_data.append({"class":"greeting", "sentence":"what's up?"})
training_data.append({"class":"greeting", "sentence":"hi"})
training_data.append({"class":"greeting", "sentence":"how are you doing?"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})


# Capture unique stemmed words
corpus_words = {}
class_words = {}

# Turn a list into a set of unique items and then a list again to remove duplications
classes = list(set([a['class'] for a in training_data]))

for c in classes:
    class_words[c] = []

# Loop through each sentence in our training data
for data in training_data:
    # Tokenize each sentence into words
    for word in nltk.word_tokenize(data['sentence']):
        # ignore some things
        if word not in ["?", "'s"]:
            stemmed_word = stemmer.stem(word.lower())
            # Have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
            # Add the word to our words in class list
            class_words[data['class']].extend([stemmed_word])

print("Corpus words and counts: {0}").format(corpus_words)

# Class words
print("Class words: {0}").format(class_words)

def calculate_class_score(sentence, class_name, show_details = True):
    score = 0

    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # Treat each word with relative weight
            score += (1.0 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1.0 / corpus_words[stemmer.stem(word.lower())]))
    return score

sentence = "How are you doing today?"

for c in class_words.keys():
    print("Class: {0} Score: {1}").format(c, calculate_class_score(sentence, c))