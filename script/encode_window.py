import pickle
import numpy as np
import gensim, os

def encode_dictionary(tokens): 
    dictionary = ["<unk>"]
    docs = []
    document = []
    for doc in tokens:
        for tok in doc:
            if tok not in dictionary:
                dictionary.append(tok)
            document.append(dictionary.index(tok))
        docs.append(document)
        document = []
    return dictionary, docs



def encode_window(tokens, anchors, dit):
    windows, window, labels = [], [], []
    j = 0
    for doc in tokens:
        for tok in np.arange(len(doc)):
            for i in np.arange(-15, 16):
                if i + tok < 0 or i + tok >= len(doc):
                    window.append(0)
                else:
                    window.append(dit.index(doc[i + tok]))
            windows.append(window)
            labels.append(anchors[j][tok])
            window = []
        j += 1
    return windows, labels

def load_bin_vec(fname, vocab):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
    word_vecs = np.zeros((len(vocab), 300))
    count = 0
    vocab_bin = gensim.models.word2vec.Word2Vec.load_word2vec_format(
        os.path.join(os.path.dirname(__file__), fname), binary=True)
    for word in vocab:
        if word in vocab_bin:
            count += 1
            word_vecs[vocab.index(word)]=(vocab_bin[word])
        else:
            word_vecs[vocab.index(word)] = (np.random.uniform(-0.25, 0.25, 300))
        print("found %d" %count)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__ == "__main__":
    tokens = pickle.load(open("tokens.bin", "rb"))
    anchors = pickle.load(open("anchors.bin", "rb"))
    dictionary, tokens = encode_dictionary(tokens)
    word_vecs = load_bin_vec('GoogleNews-vectors-negative300.bin', dictionary)
    print(len(dictionary))
    tokens = pickle.load(open("tokens1.bin", "rb"))
    anchors = pickle.load(open("anchors1.bin", "rb"))
    windows, labels = encode_window(tokens, anchors, dictionary)
    pickle.dump(windows, open("windows1.bin", "wb"))
    pickle.dump(labels, open("labels1.bin", "wb"))
    tokens = pickle.load(open("tokens2.bin", "rb"))
    anchors = pickle.load(open("anchors2.bin", "rb"))
    windows, labels = encode_window(tokens, anchors, dictionary)
    pickle.dump(windows, open("windows2.bin", "wb"))
    pickle.dump(labels, open("labels2.bin", "wb"))
    tokens = pickle.load(open("tokens3.bin", "rb"))
    anchors = pickle.load(open("anchors3.bin", "rb"))
    windows, labels = encode_window(tokens, anchors, dictionary)
    pickle.dump(windows, open("windows3.bin", "wb"))
    pickle.dump(labels, open("labels3.bin", "wb"))
    tokens = pickle.load(open("tokens4.bin", "rb"))
    anchors = pickle.load(open("anchors4.bin", "rb"))
    windows, labels = encode_window(tokens, anchors, dictionary)
    pickle.dump(windows, open("windows4.bin", "wb"))
    pickle.dump(labels, open("labels4.bin", "wb"))
    pickle.dump(word_vecs, open("vector.bin", "wb"))
