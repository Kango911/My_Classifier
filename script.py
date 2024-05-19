import numpy as np
import re
from Stemmer import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import telebot

from core import BOT_TOKEN


# Очистка текста с помощью regexp, приведение слов в инфинитив и нижний регистр, замена цифр

def text_cleaner(text):
    text = text.lower()  # Приведение в lowercase
    stemmer = Stemmer('russian')
    text = ' '.join(stemmer.stemWords(text.split()))
    text = re.sub(r'\b\d+\b', ' digit ', text)  # Замена цифр
    return text

# Загрузка данных из файла model.txt

def load_data():
    data = {'text': [], 'tag': []}
    with open('model.txt', encoding='utf-8') as file:
        for line in file:
            if not ('#' in line):
                row = line.split("@")
                data['text'] += [row[0]]
                data['tag'] += [row[1]]
    return data

# Обучение

def train_test_split(data, validation_split=0.1):
    sz = len(data['text'])
    indices = np.arange(sz)
    np.random.shuffle(indices)

    X = [data['text'][i] for i in indices]
    Y = [data['tag'][i] for i in indices]
    nb_validation_samples = int(validation_split * sz)

    return {
        'train': {'x': X[:-nb_validation_samples], 'y': Y[:-nb_validation_samples]},
        'test': {'x': X[-nb_validation_samples:], 'y': Y[-nb_validation_samples:]}
    }

data = load_data()
D = train_test_split(data)

text_clf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SGDClassifier(loss='hinge')),
])

text_clf.fit(D['train']['x'], D['train']['y'])

bot = telebot.TeleBot(BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
 bot.reply_to(message, "Hi! I'm K9 classifier bot.")

@bot.message_handler(func=lambda message: True)
def reply_to_message(message):
    z = message.text
    zz = []
    zz.append(z)
    predicted = text_clf.predict(zz)
    bot.reply_to(message, predicted[0])

bot.polling()
