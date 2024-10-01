import os
import pickle
import tkinter as tk
from tkinter import messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Define paths as raw strings to handle backslashes properly
tokenizer_path_en = r'E:\KANISHK\projects_null_class\English to French and Hindi words Translator project(3)\tokenizer_en.pkl'
tokenizer_path_fr = r'E:\KANISHK\projects_null_class\English to French and Hindi words Translator project(3)\tokenizer_fr.pkl'
tokenizer_path_hi = r'E:\KANISHK\projects_null_class\English to French and Hindi words Translator project(3)\tokenizer_hi.pkl'


model_fr = load_model('english_to_french_model.h5')
model_hi = load_model('english_to_hindi_model.h5')

# Initialize tokenizers
tokenizer_en = Tokenizer()
tokenizer_fr = Tokenizer()
tokenizer_hi = Tokenizer()

# Sample English training data (replace with actual data)
english_training_data = [
    "This is a sample sentence.",
    "Here is another example sentence.",
    "This dataset can be large for better translation."
]

# Step 1: Train and save the English tokenizer if it doesn't exist
if not os.path.exists(tokenizer_path_en):
    tokenizer_en.fit_on_texts(english_training_data)
    with open(tokenizer_path_en, 'wb') as f:
        pickle.dump(tokenizer_en, f)
    print("English tokenizer has been trained and saved.")
else:
    # Load the tokenizer if it already exists
    with open(tokenizer_path_en, 'rb') as f:
        tokenizer_en = pickle.load(f)
    print("English tokenizer loaded from file.")

# Step 2: Load French and Hindi tokenizers
try:
    with open(tokenizer_path_fr, 'rb') as f:
        tokenizer_fr = pickle.load(f)
    print("French tokenizer loaded from file.")
except Exception as e:
    print(f"Error loading French tokenizer: {e}")

try:
    with open(tokenizer_path_hi, 'rb') as f:
        tokenizer_hi = pickle.load(f)
    print("Hindi tokenizer loaded from file.")
except Exception as e:
    print(f"Error loading Hindi tokenizer: {e}")


def translate(text):
    # Tokenize and pad the input
    input_seq = tokenizer_en.texts_to_sequences([text])
    max_length = 20  # Adjust the max length according to your model requirements
    input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')

    # Predict French
    fr_pred = model_fr.predict(input_pad)
    fr_word_index = np.argmax(fr_pred, axis=-1)

    # Check if the prediction is scalar or array and handle accordingly
    if isinstance(fr_word_index, np.ndarray) and fr_word_index.ndim > 1:
        fr_word = tokenizer_fr.index_word.get(fr_word_index[0][0], "Unknown")
    else:
        fr_word = tokenizer_fr.index_word.get(fr_word_index[0], "Unknown")

    # Predict Hindi
    hi_pred = model_hi.predict(input_pad)
    hi_word_index = np.argmax(hi_pred, axis=-1)

    # Check if the prediction is scalar or array and handle accordingly
    if isinstance(hi_word_index, np.ndarray) and hi_word_index.ndim > 1:
        hi_word = tokenizer_hi.index_word.get(hi_word_index[0][0], "Unknown")
    else:
        hi_word = tokenizer_hi.index_word.get(hi_word_index[0], "Unknown")

    return fr_word, hi_word


class DualLanguageTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Language Translator")

        # Create input label and entry
        self.label = tk.Label(root, text="Enter English Word (10+ letters):")
        self.label.pack()

        self.entry = tk.Entry(root, width=50)
        self.entry.pack()

        # Create Translate button
        self.translate_button = tk.Button(root, text="Translate", command=self.perform_translation)
        self.translate_button.pack()

        # Create output labels
        self.fr_label = tk.Label(root, text="French Translation:")
        self.fr_label.pack()

        self.fr_output = tk.Label(root, text="")
        self.fr_output.pack()

        self.hi_label = tk.Label(root, text="Hindi Translation:")
        self.hi_label.pack()

        self.hi_output = tk.Label(root, text="")
        self.hi_output.pack()

    def perform_translation(self):
        text = self.entry.get()
        if len(text) < 10:
            messagebox.showerror("Error", "Please enter a word with 10 or more letters.")
            return

        french_translation, hindi_translation = translate(text)

        # Display translations
        self.fr_output.config(text=french_translation)
        self.hi_output.config(text=hindi_translation)


# Create the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = DualLanguageTranslator(root)
    root.mainloop()
