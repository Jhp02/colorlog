from flask import Flask, request, render_template
import cv2
import matplotlib.pyplot as plt
import numpy as np
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)


# 메인 함수
def emotion_analysis(img1, img2, img3):
    # 각 색의 RGB값 array
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    yellow = np.array([255 ,255 ,0])
    gray = np.array([128, 128, 128])
    green = np.array([0, 128, 0])
    purple = np.array([128, 0, 128])
    orange = np.array([255, 165, 0])
    turquoise = np.array([64, 224, 208])
    brown = np.array([165, 42,42])
    pink = np.array([255, 192, 203])

    color_rgb_array = np.array([black, white, red, blue, yellow, gray, green, purple, orange, turquoise, brown, pink])
    
    # 색깔 순서 리스트
    color_list = ["black", "white", "red", "blue", "yellow", "gray", "green", "purple", "orange", "turquoise", "brown", "pink"]
    
    # 각 색깔이 갖는 감정의 비율
    black_emotion = {"sadness" : 0.51,
                 "fear" : 0.48,
                 "hate": 0.41,
                 "anger": 0.32,
                 "guilt": 0.30}

    white_emotion = {"relief": 0.43,
                     "contentment": 0.30}
    red_emotion = {"love" : 0.68,
                   "anger": 0.51,
                   "pleasure": 0.33,
                   "hate": 0.29}
    blue_emotion = {"relief": 0.35,
                    "contentment": 0.34,
                    "interest": 0.27}
    yellow_emotion = {"joy": 0.52,
                      "amusement": 0.40,
                      "pleasure": 0.33}
    gray_emotion = {"sadness": 0.48,
                    "disappointment": 0.41,
                    "regret": 0.31}
    green_emotion = {"contentment": 0.39,
                     "joy": 0.34,
                     "pleasure": 0.34,
                     "relief": 0.33,
                     "interest": 0.31}
    purple_emotion = {"pleasure": 0.25,
                      "interest": 0.24,
                      "pride": 0.24,
                      "admiration": 0.24}
    orange_emotion = {"joy": 0.44,
                      "amusement": 0.42,
                      "pleasure": 0.33}
    turquoise_emotion = {"please": 0.35,
                         "relief": 0.34,
                         "joy": 0.32,
                         "contentmemt": 0.31}
    brown_emotion = {"disgust": 0.36}
    pink_emotion = {"love": 0.50,
                    "joy": 0.41,
                    "pleasure": 0.40,
                    "amusement": 0.36}


    color_emotion_dic = {"black" : black_emotion,
                     "white" : white_emotion,
                     "red" : red_emotion,
                     "blue": blue_emotion,
                     "yellow": yellow_emotion,
                     "gray": gray_emotion,
                     "green": green_emotion,
                     "purple" : purple_emotion,
                     "orange": orange_emotion,
                     "turquoise" : turquoise_emotion,
                     "brown": brown_emotion,
                     "pink": pink_emotion
                     }
    
    # 감정 딕셔너리
    #love, anger, pleasure, hate, joy, amusement, contentment, relief, interest, pride,admiration,sadness,disappointment,regret,disgust,fear,guilt
    emotion_dic = {"love": 0,
           "anger": 0,
           "pleasure": 0,
           "hate":0,
           "joy": 0,
           "amusement": 0,
           "contentment": 0,
           "relief": 0,
           "interest": 0,
           "interest": 0,
           "pride": 0,
           "admiration": 0,
           "sadness": 0,
           "disappointment": 0,
           "regret": 0,
           "disgust": 0,
           "fear": 0,
           "guilt": 0}

    emotion_list = ["love", "anger", "pleasure", "hate", "joy", "amusement", "contentment", "relief", "interest", "pride", "admiration", "sadness", "disappointment", "regret", "disgust", "fear", "guilt"]
    
    #감정 분석 코드
    img = img_resize(img1, img2, img3)
    ret, label, center = clustering(img)
    cluster_color_match = cluster_color_matching(center, color_rgb_array)
    cluster_pixel_num = calculation_cluster_pixel_num(label, img)
    color_pixel_num = calculation_color_pixel(cluster_color_match, cluster_pixel_num)
    percentage = pixel_percentage(color_pixel_num, color_list, img)
    emotion_dic_ = calculation_emotion(emotion_dic, color_emotion_dic, percentage)
    first_emotion, first_emo_percentage, second_emotion, second_emo_percentage = find_emotion(emotion_dic_, emotion_list)
    return first_emotion, first_emo_percentage, second_emotion, second_emo_percentage


    
# 이미지 크기 및 차원 변경 함수
def img_resize(img1, img2, img3):
    
    #크기 변경
    img1_resize = cv2.resize(img1, dsize = (500, 500), interpolation = cv2.INTER_AREA)
    img2_resize = cv2.resize(img2, dsize = (500, 500), interpolation = cv2.INTER_AREA)
    img3_resize = cv2.resize(img3, dsize = (500, 500), interpolation = cv2.INTER_AREA)
    
    #하나로 합치기
    img = np.concatenate((img1_resize, img2_resize, img3_resize))
    
    #차원 변경
    img = img.reshape(-1, 3)
    return img



# k-means 함수
def clustering(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    ret, label, center = cv2.kmeans(img.astype(np.float32), 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return ret, label, center



# 각 군집과 색을 매칭하는 함수
def cluster_color_matching(center, color_rgb_array):
    cluster_color_match = np.array([]).astype('uint8')
    for i in np.arange(12):
        cluster_color_match = np.append(cluster_color_match, np.argmin(np.abs(np.mean((center[i] - color_rgb_array), axis = 1))))
    return cluster_color_match


# 각 군집의 픽셀 개수 계산 함수
def calculation_cluster_pixel_num(label,img):
    cluster_pixel_num = np.zeros(12).astype('uint32')
    for i in np.arange(img.shape[0]):
        if(label[i] == 0):
            cluster_pixel_num[0] += 1
        elif(label[i] == 1):
            cluster_pixel_num[1]+= 1
        elif(label[i] == 2):
            cluster_pixel_num[2]+= 1
        elif(label[i] == 3):
            cluster_pixel_num[3]+= 1
        elif(label[i] == 4):
            cluster_pixel_num[4]+= 1
        elif(label[i] == 5):
            cluster_pixel_num[5]+= 1
        elif(label[i] == 6):
            cluster_pixel_num[6]+= 1
        elif(label[i] == 7):
            cluster_pixel_num[7]+= 1
        elif(label[i] == 8):
            cluster_pixel_num[8]+= 1
        elif(label[i] == 9):
            cluster_pixel_num[9]+= 1
        elif(label[i] == 10):
            cluster_pixel_num[10]+= 1
        elif(label[i] == 11):
            cluster_pixel_num[11]+= 1
            
    return cluster_pixel_num



# 색깔 별 픽셀 개수 계산 함수
def calculation_color_pixel(cluster_color_match, cluster_pixel_num):
    color_pixel_num = np.zeros(12).astype('uint32')

    for i in np.arange(12):
        for j in np.arange(12):
            if(cluster_color_match[j] == i):
                color_pixel_num[i] += cluster_pixel_num[j]
    return color_pixel_num



# 각 색의 픽셀 개수를 백분율로 환산하는 함수
def pixel_percentage(color_pixel_num, color_list, img):
    percentage = {"black" : 0,
                 "white" : 0,
                 "red" : 0,
                 "blue": 0,
                 "yellow": 0,
                 "gray": 0,
                 "green": 0,
                 "purple" : 0,
                 "orange": 0,
                 "turquoise" : 0,
                 "brown": 0,
                 "pink": 0}
    
    percentage_  = ((color_pixel_num/img.shape[0])*100)

    for i, j in zip(color_list, np.arange(12)):
        percentage[i] = percentage_[j]
    return percentage


# 각 감정이 갖는 값 계산 함수
def calculation_emotion(emotion_dic, color_emotion_dic, percentage):
    
    emotion_dic["love"] = color_emotion_dic['red']['love']  * percentage['red'] + color_emotion_dic['pink']['love'] * percentage['pink']

    emotion_dic["anger"] =  color_emotion_dic['black']['anger']  * percentage['black'] + color_emotion_dic['red']['anger'] * percentage['red']

    emotion_dic["pleasure"] = color_emotion_dic['green']['pleasure'] * percentage['green'] + color_emotion_dic['orange']['pleasure'] * percentage['orange'] + color_emotion_dic['pink']['pleasure'] * percentage['pink'] + color_emotion_dic['purple']['pleasure'] * percentage['purple'] + color_emotion_dic['red']['pleasure'] * percentage['red'] + color_emotion_dic['yellow']['pleasure'] * percentage['yellow']

    emotion_dic["hate"] = color_emotion_dic['black']['hate'] * percentage['black'] + color_emotion_dic['red']['hate'] * percentage['red'] 

    emotion_dic["joy"]  =  color_emotion_dic['green']['joy'] * percentage['green'] +  color_emotion_dic['orange']['joy'] * percentage['orange']  +  color_emotion_dic['pink']['joy'] * percentage['pink'] +  color_emotion_dic['turquoise']['joy'] * percentage['turquoise'] +  color_emotion_dic['yellow']['joy'] * percentage['yellow'] 

    emotion_dic["amusement"] = color_emotion_dic['orange']['amusement'] * percentage['orange'] + color_emotion_dic['pink']['amusement'] * percentage['pink'] + color_emotion_dic['yellow']['amusement'] * percentage['yellow']

    emotion_dic["contentment"] = color_emotion_dic['blue']['contentment'] * percentage['blue'] + color_emotion_dic['green']['contentment'] * percentage['green'] + color_emotion_dic['white']['contentment'] * percentage['white']

    emotion_dic["relief"] = color_emotion_dic['blue']['relief'] * percentage['blue'] + color_emotion_dic['green']['relief'] * percentage['green'] + color_emotion_dic['turquoise']['relief'] * percentage['turquoise'] + color_emotion_dic['white']['relief'] * percentage['white']

    emotion_dic["interest"] = color_emotion_dic['blue']['interest'] * percentage['blue'] + color_emotion_dic['green']['interest'] * percentage['green'] + color_emotion_dic['purple']['interest'] * percentage['purple']

    emotion_dic["pride"] = color_emotion_dic['purple']['pride'] * percentage['purple']

    emotion_dic["admiration"] = color_emotion_dic['purple']['admiration'] * percentage['purple']

    emotion_dic["sadness"] = color_emotion_dic['black']['sadness'] * percentage['black'] + color_emotion_dic['gray']['sadness'] * percentage['gray']

    emotion_dic["disappointment"] = color_emotion_dic['gray']['disappointment'] * percentage['gray']

    emotion_dic["regret"] = color_emotion_dic['gray']['regret'] * percentage['gray']

    emotion_dic["disgust"] = color_emotion_dic['brown']['disgust'] * percentage['brown']

    emotion_dic["fear"] =  color_emotion_dic['black']['fear'] * percentage['black']

    emotion_dic["guilt"] = color_emotion_dic['black']['guilt'] * percentage['black']
    
    return emotion_dic 



# 주 감정 추출 함수
def find_emotion(emotion_dic, emotion_list):
    first_emotion = ""
    second_emotion = ""
    first_percentage = 0.0
    second_percentage = 0.0

    for emo in emotion_list:
        if(emotion_dic[emo] > first_percentage):
            second_percentage = first_percentage
            second_emotion = first_emotion
            first_percentage = emotion_dic[emo]
            first_emotion  = emo
        elif(emotion_dic[emo] > second_percentage):
            second_percentage = emotion_dic[emo]
            second_emotion = emo

    return first_emotion, first_percentage, second_emotion, second_percentage




@app.route("/", methods = ["GET", 'POST'])

def main():
    if request.method == "GET":
        return render_template("page1.html")
    else:
        img1 = request.files['img1']
        img2 = request.files['img2']
        img3 = request.files['img3']
        
        
        img1_name = './image/' + str(time.time()) + ".jpg"
        img2_name = './image/' + str(time.time() + 1) + ".jpg"
        img3_name = './image/' + str(time.time() + 2) + ".jpg"
        
        img1.save(img1_name)
        img2.save(img2_name)
        img3.save(img3_name)
        
        img1 = cv2.imread(img1_name, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img2_name, cv2.IMREAD_COLOR)
        img3 = cv2.imread(img3_name, cv2.IMREAD_COLOR)
        

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        results = emotion_analysis(img1,img2,img3)
        return render_template("result.html", results = results)
    
if __name__ == "__main__":
    app.run(debug = True)
