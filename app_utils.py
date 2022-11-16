"""
Utility function(s) for skimlit_app.py
"""
import tensorflow as tf
import numpy as np
import pandas as pd

# Make a Preprocessing function for the abstract
def preprocess_and_predict_abstract(abstract:str,
                                    model,
                                    classes=['BACKGROUND',
                                             'CONCLUSIONS',
                                             'METHODS',
                                             'OBJECTIVE',
                                             'RESULTS'],
                                    correct_class_order=["BACKGROUND",
                                                         "OBJECTIVE",
                                                         "METHODS",
                                                         "RESULTS",
                                                         "CONCLUSIONS"]):
    """Preprocess PubMed Medical Abstract and Predict  using a TensorFlow Model on it.

    Args:
        abstract (str): PubMed Abstract Data
        model (tf.keras.Model): TensorFLow model to predict with
        classes (list, optional): class names model trained on. Defaults to ['BACKGROUND', 'CONCLUSIONS', 'METHODS', 'OBJECTIVE', 'RESULTS'].
        correct_class_order (list, optional): class order if sequential dependency. Defaults to ["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"].

    Returns:
        [type]: [description]
    """
    # Get lines
    abstract_lines = abstract.split(".")
    # Get chars
    chars = np.array([" ".join(list(i)) for i in abstract_lines])
    # Get tokens
    sentences = np.array([i.strip() for i in abstract_lines])
    # Get line numbers
    line_numbers = np.array(list(range(len(abstract_lines))))
    line_numbers_one_hot = tf.one_hot(line_numbers, depth=15)
    # Get total lines
    total_lines = np.array([len(abstract_lines) - 1])
    total_lines_one_hot = [tf.one_hot(total_lines, depth=20) for i in range(len(abstract_lines))]
    total_lines_one_hot = tf.squeeze(total_lines_one_hot)
    # Create tf.data Dataset
    abstract_dataset = tf.data.Dataset.from_tensor_slices(((line_numbers_one_hot,
                                                           total_lines_one_hot,
                                                           sentences,
                                                           chars),
                                                          tf.ones([len(abstract_lines)]))).batch(len(abstract_lines))

    # Predict and get classes
    abstract_prediction_probs = model.predict(abstract_dataset)
    abstract_preds = tf.argmax(abstract_prediction_probs, axis=1)
    pred_classes = [classes[i] for i in abstract_preds]

    # Abstractify
    abstractified = []

    for pred, sentence in zip(pred_classes, abstract_lines):
        abstractified.append({pred: sentence})

    # Delete for next set of iterations
    del pred
    del sentence

    # Decompose "abstractify"
    decomposed = [[] for i in range(len(correct_class_order))]

    for dict_ in abstractified:
        for pred, sentence in dict_.items():
            for i in range(len(correct_class_order)):
                if pred == correct_class_order[i]:
                    decomposed[i].append(sentence)
    
    # Turn Decomposed data into Abstract Chunks
    
    for class_ in range(len(decomposed)):
        # Turn into abstract chunks and append the class and a period at the end
        decomposed[class_] = "**" + correct_class_order[class_] + "**" + ":  " + ". ".join(decomposed[class_]) + "\n"
    
    skimlitted_abstract = "\n".join(decomposed)
    return skimlitted_abstract