from flask import Flask, request, jsonify
from keras.models import load_model
import pickle
import numpy as np
import nltk
import random
import json
import requests
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load model and necessary data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json', encoding='utf-8').read())
lemmatizer = nltk.stem.WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
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
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list



def get_movie_api():
    movie_titles = []
    # Gọi API để lấy danh sách các bộ phim đang chiếu
    movie_data = requests.get('http://172.20.10.4:8080/api/v1/movies?page=1').json()
    # Kiểm tra xem dữ liệu phim có tồn tại không
    for movie in movie_data['data']['result']:
        movie_titles.append(movie['title'])
    print(movie_titles)
    return movie_titles



def get_movie_title(sentence):
    movie_titles = get_movie_api()  # Lấy danh sách các tiêu đề phim từ API
    for title in movie_titles:
        if title.lower() in sentence.lower():
            print(title)
            return title
    return None



def get_response(intents_list, intents_json, sentence):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if tag == 'movies':
                # Gọi API để lấy danh sách phim đang chiếu và giờ chiếu của nó
                movie_data = requests.get('http://172.20.10.4:8080/api/v1/movies?page=1').json()
                # Xây dựng câu trả lời từ dữ liệu nhận được từ API
                movies_response = "Dưới đây là danh sách các bộ phim đang chiếu: "
                print(movie_data)
                for movie in movie_data['data']['result']:
                    movies_response += f"{movie['title']}, "
                result = movies_response
            elif tag == 'showtimes':
                # Lấy tên phim từ câu hỏi
                movie_title = get_movie_title(sentence)
                if movie_title:
                    # Gọi API để lấy thông tin lịch chiếu của phim cụ thể
                    showtimes_data = requests.get(f'http://172.20.10.4:8080/api/v1/showtime/find-by-name?movieName={movie_title}&dateShow=26-2-2024&status=1').json()
                    # Xây dựng câu trả lời từ dữ liệu nhận được từ API
                    if 'data' in showtimes_data and 'result' in showtimes_data['data']:
                        showtimes_response = f"Lịch chiếu của phim {movie_title}:\n"
                        times = [f"{showtime['timeStart']} - {showtime['timeEnd']}" for showtime in showtimes_data['data']['result']]
                        result = ', '.join(times)
                    else:
                        result = "Xin lỗi, không tìm thấy thông tin lịch chiếu cho phim này."
                else:
                    result = "Xin lỗi, không tìm thấy tên phim trong câu của bạn."
            else:
                result = random.choice(i['responses'])
            break
    return result

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form.get('message', '')  # Trích xuất câu nhập từ trường 'message' trong dữ liệu form
    ints = predict_class(message)
    res = get_response(ints, intents, message)  # Truyền câu nhập vào hàm get_response
    return jsonify({'response': res})

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='172.20.10.9', debug=True)
