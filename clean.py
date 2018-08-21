from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

additional_words = {'thank', 'thanks', '(unclear)', 'bye', 'goodbye', 'good-bye', 'ok', 'okay'}
additional_words.union(ENGLISH_STOP_WORDS)

with open("clean.txt", 'r') as f:
    list = []
    text = f.read().splitlines()
    for s in text:
        list.append(s.split(' '))

with open('nostop.txt', 'w') as f:
    tokens = []
    for sentence in list:
        s = []
        for w in sentence:
            if w not in additional_words:
                s.append(w)
        if len(s) > 2:
            f.write(' '.join(s) + '\n')
