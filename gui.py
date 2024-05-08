import tkinter as tk
from tkinter import scrolledtext
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Load necessary files
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\\chatbot\\intents.json', encoding="utf-8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    message = entry.get()
    if message.strip() != "":
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, "Te: " + message + "\n")
        chat_history.see(tk.END)
        entry.delete(0, tk.END)
        ints = predict_class(message)
        res = get_response(ints, intents)
        chat_history.insert(tk.END, "Csevegőrobot: " + res + "\n\n")
        chat_history.see(tk.END)
        chat_history.config(state=tk.DISABLED)

# Create main window
root = tk.Tk()
root.title("Csevegőrobot")

# Create chat history widget
chat_history = scrolledtext.ScrolledText(root, width=50, height=20, wrap=tk.WORD, state=tk.DISABLED)
chat_history.pack(padx=10, pady=10)

# Create entry widget for typing messages
entry = tk.Entry(root, width=50)
entry.pack(padx=10, pady=(0, 10))
entry.bind("<Return>", send_message)

# Create send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=(0, 10))

# Run the GUI
root.mainloop()