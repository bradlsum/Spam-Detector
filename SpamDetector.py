from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import collections
import os
import string
import numpy as np
import pickle

cdir = os.path.abspath("")
hdir = cdir + "/ham/"
sdir = cdir + "/spam/"
tdir = cdir + "/target/"

top_n = 500

# Generate word dictionary and file contents for each file
def generate_words(files_dir, ones=False):
    words = []
    file_words = []
    if ones:
        y = np.ones(len(files_dir))
    else:
        y = np.zeros(len(files_dir))
    for f in files_dir:
        with open(f, 'r', encoding='unicode_escape', errors='ignore') as file:
            file_contents = []
            # Skip header for now
            next(file)
            # For each word in the file, tokenize it
            for line in file:
                for word in line.split():
                    if word not in string.punctuation:
                        words.append(word)
                        file_contents.append(word)
            file_words.append(file_contents)
    # Make a counter and have them store # of occurences for each word.
    words = collections.Counter(words)
    fw = []
    for f in file_words:
        fw.append(collections.Counter(f))
    return words, fw, y


"""
Create matrix, with words as the "features", and each file as the row.
Values represent that file's word count for a specific word
(Ex, A value of 4 for File 5 for the word "the" would mean that
the word "the" appears 4 times in File 5
"""


def create_matrix(file_words, num_files, words, n=0, inter = 20):
    print("Creating matrix...")
    per_count = 1
    if n > 0:
        matrix = np.zeros((num_files, n + 1))
        top_words = [w[0] for w in words.most_common(n)]
    else:
        matrix = np.zeros((num_files, len(words)))
        top_words = words
    for file_num in range(num_files):
        for file_word in file_words[file_num]:
            word_num = 0
            found = False
            for word in top_words:
                if file_word == word:
                    found = True
                    matrix[file_num][word_num] = file_words[file_num].get(file_word)
                    break
                word_num += 1
            if not found:
                matrix[file_num][word_num] += file_words[file_num].get(file_word)
        if file_num / num_files >= per_count/inter:
            print(per_count/inter*100, "%")
            per_count += 1
    print(100, "%")

    f = open("words", "wb")  # remember to open the file in binary mode
    pickle.dump(top_words, f)
    f.close()

    print("Top words:", top_words)
    return matrix


def matrix_from_words(file_words, num_files, words, n=0, inter = 20):
    f = open("words", "rb")  # remember to open the file in binary mode
    top_words = pickle.load(f)
    f.close()

    per_count = 1
    if n > 0:
        matrix = np.zeros((num_files, n + 1))
        # top_words = [w[0] for w in words.most_common(n)]
    else:
        matrix = np.zeros((num_files, len(words)))
        # top_words = words
    for file_num in range(num_files):
        for file_word in file_words[file_num]:
            word_num = 0
            found = False
            for word in top_words:
                if file_word == word:
                    found = True
                    matrix[file_num][word_num] = file_words[file_num].get(file_word)
                    break
                word_num += 1
            if not found:
                matrix[file_num][word_num] += file_words[file_num].get(file_word)
        if file_num / num_files >= per_count/inter:
            per_count += 1

    return matrix


choice = ""
while choice != "3":
    choice = input("Welcome to spam detector:\n"
                   "1. Detect target emails\n"
                   "2. Train model\n"
                   "3. Quit\n")

    # Train model
    if choice == "2":
        # List of all ham & spam files respectively.
        ham = [hdir + f for f in os.listdir(hdir) if os.path.isfile(hdir + f)]
        spam = [sdir + f for f in os.listdir(sdir) if os.path.isfile(sdir + f)]

        # Load from file
        num_files = len(ham) + len(spam)
        words1, files1, y1 = generate_words(ham)
        words2, files2, y2 = generate_words(spam, True)
        words = words1 + words2
        file_words = files1 + files2
        y = np.concatenate([y1, y2])

        # Create matrix, split data into train and test, and train the data.
        matrix = create_matrix(file_words, num_files, words, n=top_n)
        xtrain, xtest, ytrain, ytest = train_test_split(matrix, y,
                                                        test_size=0.20, random_state=42)

        model = GaussianNB()
        model.fit(xtrain, ytrain)

        # Use model to predict which of the test emails are spam or not and print accuracy.
        ypred = model.predict(xtest)
        print("Accuracy:", metrics.accuracy_score(ytest, ypred))

        f = open("model", "wb")  # remember to open the file in binary mode
        pickle.dump(model, f)
        f.close()

    # Detect spam
    elif choice == "1":
        f = open("model", "rb")
        model = pickle.load(f)
        f.close()

        # List of all target files.
        target = [tdir + f for f in os.listdir(tdir) if os.path.isfile(tdir + f)]

        # Load from file
        num_files = len(target)
        words, files, y = generate_words(target)

        # Create matrix
        matrix = matrix_from_words(files, num_files, words, n=top_n)

        # Use model to predict which of the test emails are spam or not and print accuracy.
        ypred = model.predict(matrix)

        print("\nID\tType\t Email")
        for i in range(len(files)):
            if ypred[i] == 1:
                print(i, "\tSpam:\t", target[i][target[i].find('.') + 1:])
            else:
                print(i, "\tHam: \t", target[i][target[i].find('.') + 1:])

    else:
        print("Closing...")

    print()

