import pickle, random
import numpy as np

def load_data(window, label):
    vectors = pickle.load(open("vector.bin", 'rb'))
    sents = pickle.load(open(window, 'rb'))
    anchor = pickle.load(open(label, 'rb'))
    return vectors, sents, anchor

def data_iterator(orig_X, orig_y=None, batch_size=50, label_size=2, shuffle=True, split=1700):
    if split:
        y_label = np.argmax(orig_y, axis=1)
        X = list(orig_X[y_label != 0])
        y = list(orig_y[y_label != 0])
        rand = []
        for x in range(split):
            i = random.randint(0, len(y_label) - 1)
            while y_label[i] != 0 or i in rand:
                i = random.randint(0,len(y_label) - 1)
                y.append(orig_y[i])
                X.append(orig_X[i])
    X = np.array(X)
    y = np.array(y)
  # Optionally shuffle the data before training
    if shuffle: # dao thu tu
        indices = np.random.permutation(len(X))
        data_X = X[indices]
        data_y = y[indices]
    else:
        data_X = X
        data_y = y
  ###
    total_processed_examples = 0
    total_steps = int(np.ceil(len(data_X) / float(batch_size)))
    for step in range(total_steps):
        # Create the batch by selecting up to batch_size elements
        batch_start = step * batch_size
        x = data_X[batch_start:batch_start + batch_size]
        y = data_y[batch_start:batch_start + batch_size]
        yield x, y
        total_processed_examples += len(x)
  # Sanity check to make sure we iterated over all the dataset as intended
    assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), 
                                                                                          total_processed_examples)

def data_evaluate(x_dev, y_dev, label_size=2, shuffle=False):
    if shuffle:  # dao thu tu
        indices = np.random.permutation(len(x_dev))
        x = x_dev[indices]
        y_indices = y_dev[indices] if np.any(y_dev) else None
    else:
        x = x_dev
        y = y_dev
    return x, y
