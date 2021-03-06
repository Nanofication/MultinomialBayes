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
training_data.append({"class":"greeting", "sentence":"what's new?"})
training_data.append({"class":"greeting", "sentence":"how's life?"})
training_data.append({"class":"greeting", "sentence":"how are you doing today?"})
training_data.append({"class":"greeting", "sentence":"good to see you"})
training_data.append({"class":"greeting", "sentence":"nice to see you"})
training_data.append({"class":"greeting", "sentence":"long time no see"})
training_data.append({"class":"greeting", "sentence":"it's been a while"})
training_data.append({"class":"greeting", "sentence":"nice to meet you"})
training_data.append({"class":"greeting", "sentence":"pleased to meet you"})
training_data.append({"class":"greeting", "sentence":"how do you do"})
training_data.append({"class":"greeting", "sentence":"yo"})
training_data.append({"class":"greeting", "sentence":"howdy"})
training_data.append({"class":"greeting", "sentence":"sup"})
# 20 training data


training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"peace"})
training_data.append({"class":"goodbye", "sentence":"catch you later"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"farewell"})
training_data.append({"class":"goodbye", "sentence":"have a good day"})
training_data.append({"class":"goodbye", "sentence":"take care"})
# 10 training datas
training_data.append({"class":"goodbye", "sentence":"bye!"})
training_data.append({"class":"goodbye", "sentence":"have a good one"})
training_data.append({"class":"goodbye", "sentence":"so long"})
training_data.append({"class":"goodbye", "sentence":"i'm out"})
training_data.append({"class":"goodbye", "sentence":"smell you later"})
training_data.append({"class":"goodbye", "sentence":"talk to you later"})
training_data.append({"class":"goodbye", "sentence":"take it easy"})
training_data.append({"class":"goodbye", "sentence":"i'm off"})
training_data.append({"class":"goodbye", "sentence":"until next time"})
training_data.append({"class":"goodbye", "sentence":"it was nice seeing you"})

training_data.append({"class":"goodbye", "sentence":"it's been real"})
training_data.append({"class":"goodbye", "sentence":"im out of here"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})

training_data.append({"class":"email", "sentence":"what's your email address?"})
training_data.append({"class":"email", "sentence":"may I get your email?"})
training_data.append({"class":"email", "sentence":"can I have your email?"})
training_data.append({"class":"email", "sentence":"what's your email?"})
training_data.append({"class":"email", "sentence":"let me get your email"})
training_data.append({"class":"email", "sentence":"give me your email"})
training_data.append({"class":"email", "sentence":"i'll take your email address"})
training_data.append({"class":"email", "sentence":"can I have your business email?"})
training_data.append({"class":"email", "sentence":"your email address?"})
training_data.append({"class":"email", "sentence":"email please?"})
training_data.append({"class":"email", "sentence":"may I have your email?"})
training_data.append({"class":"email", "sentence":"can I get your email?"})



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

# We may have to weigh each word differently. What if I'm asking for your email.
def calculate_class_score(sentence, class_name, show_details = True):
    score = 0

    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # Treat each word with relative weight
            score += (1.0 / corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1.0 / corpus_words[stemmer.stem(word.lower())]))
    return score

sentence = ""

for c in class_words.keys():
    print("Class: {0} Score: {1}").format(c, calculate_class_score(sentence, c))