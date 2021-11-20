# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import datetime
import requests
from sklearn.metrics import mean_squared_error

def error_made1(im1, im2):
    mse = (np.square(im1-im2)).mean(axis=None)
    return mse

back = 0
th = 30    # 差分画像の閾値
flag = 0
Ids = 0
todays_in = 0
todays_out = 0
con = 0
#person = [[0]*2]*10だと [0,0] == [0,0] == ...となってしまう
tmp = [int(i) for i in [0]*10]
sa = []
face_remember = []
person = []
face_pre = []
val = []
val1 = []

# 動画書き出し用のオブジェクトを生成
fmt = cv2.VideoWriter_fourcc('m','p','4','v')
fps = 20.0
size = (1280, 720)
writer = cv2.VideoWriter('test1.m4v', fmt, fps, size) # --- (*1)

# 動画ファイルのキャプチャ
cap = cv2.VideoCapture(0)
wCam = 1280
hCam = 720
cap.set(3,wCam)
cap.set(4,hCam)
# 最初のフレームを背景画像に設定
ret, bg = cap.read()
# bg = cv2.flip(bg,1)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
bg = cv2.GaussianBlur(bg, (7,7), 0)

dt_now = datetime.datetime.now()
pretime = dt_now.hour

while(cap.isOpened()):
    dt_now = datetime.datetime.now()
    # フレームの取得
    ret, frame = cap.read()
    save_frame = frame.copy()
    flame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)
    cv2.rectangle(frame,(wCam//4-100,0),(wCam//2-20,hCam),(255,255,0),thickness=20)
    cv2.rectangle(frame,(wCam//2+20,0),(3*+wCam//4+100,hCam),(255,255,0),thickness=20)
    # cv2.rectangle(frame,(wCam//2-10,0),(wCam//2+10,hCam),(255,0,0),thickness=-1)    

    # 差分の絶対値を計算
    mask = cv2.absdiff(gray, bg)

    # 差分画像を二値化してマスク画像を算出
    mask[mask < th] = 0
    mask[mask >= th] = 255

    if cv2.waitKey(1) & 0xFF == ord('a'):
        todays_in = 0
        todays_out = 0
    if cv2.waitKey(1) & 0xFF == ord('i'):
        todays_in -= 1
    if cv2.waitKey(1) & 0xFF == ord('o'):
        todays_out -= 1

    
    if (flag == 1):
        #opening
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (10,10))
        flag = 2
    # #輪郭抽出
    mode = cv2.RETR_LIST #RETR_EXTERNAL,RETR_CCOMP,RETR_FREE
    contours, hierarchy = cv2.findContours(mask,mode,cv2.CHAIN_APPROX_SIMPLE)
    person = []
    new_person = []
    face_remember = []
    for j in contours:
        x,y,w,h = cv2.boundingRect(j)
        if w >= 150 and h >= 140:
            centX = x+w//2
            centY = y+h//2
            person.append([centX,centY])
            # print(person)
            if Ids >= 1 and (abs(person[Ids-1][0]-person[Ids][0]) <= w/2+10 and abs(person[Ids-1][1]-person[Ids][1]) <= h/2+10):
                person.pop(Ids)
                # print("remove")
                continue
            face_remember.append(save_frame[y:y+h,x:x+w])
            cv2.imwrite('detect_data/face'+str(Ids)+'.jpg', face_remember[Ids])
            face_remember[Ids] = np.resize(face_remember[Ids],(64,64))
            Ids += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255),thickness=15)
            # 各オブジェクトのラベル番号と面積に黄文字で表示
            cv2.putText(frame, "Ids: " +str(Ids), (x, y + -15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            # 各オブジェクトの重心座標をに黄文字で表示
            cv2.putText(frame, "W: " + str(int(w)), (x+w - 10, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            cv2.putText(frame, "H: " + str(int(h)), (x+w - 10, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # #前回フレームで見つかった移動物体が今回フレームで見つかった移動物体にどのように関連しているか
    for i in range(len(tmp)):
        # if i < len(face_pre):
        if i < len(face_pre) and i < len(face_remember):
            for j in range(len(face_pre)):
                val.append(error_made1(face_remember[i],face_pre[j]))
            # print("val=", min(val))
            if min(val) <= 80:
                new_person.append(tmp[np.argmin(val)])
            else:
                new_person.append(0)
            val = []
        else:
            new_person.append(tmp[i])
    face_pre = face_remember

    for id in range(len(person)):
        if (person[id][0] >= 3*wCam//4+100) or (person[id][0] <= wCam//4-100):
            new_person[id] = 0
        if new_person[id] == 0 and person[id][0] >= wCam//4-100 and person[id][0] <= wCam//2-40:
            new_person[id] = 1
        if (new_person[id] == 3 or new_person[id] == 1) and person[id][0] >= wCam//2+40 and person[id][0] <= 3*wCam//4+100:
            new_person[id] = 0
            todays_out += 1
        #out_section
        if new_person[id] == 0 and person[id][0] >= wCam//2+40 and person[id][0] <= 3*wCam//4+100:
            new_person[id] = 2
        if (new_person[id] == 3 or new_person[id] == 2) and person[id][0] >= wCam//4-100 and person[id][0] <= wCam//2-40:
            new_person[id] = 0
            todays_in += 1
        #senter_section
        if person[id][0] > wCam//2-40 and person[id][0] < wCam//2+40:
            new_person[id] = 3

    tmp = new_person
    Ids = 0
    fr = frame.copy()

    cv2.rectangle(frame,(0,0),(wCam,hCam),(255,255,255),thickness=-1)
    cv2.putText(frame,str(todays_in),(3*wCam//4-30,3*hCam//4),cv2.FONT_HERSHEY_PLAIN,7,(0,255,0),10)
    cv2.putText(frame,str(todays_out),(wCam//4-30,3*hCam//4),cv2.FONT_HERSHEY_PLAIN,7,(0,255,0),10)
    cv2.putText(frame,"Number of people in here",(30,50),cv2.FONT_HERSHEY_PLAIN,5,(255,255,0),5)
    cv2.putText(frame,str(todays_in - todays_out),(wCam//2-30,hCam//4),cv2.FONT_HERSHEY_PLAIN,10,(0,255,0),5)
    cv2.putText(frame,"IN",(3*wCam//4-3,350),cv2.FONT_HERSHEY_PLAIN,5,(255,255,0),7)
    cv2.putText(frame,"OUT",(wCam//4-30,350),cv2.FONT_HERSHEY_PLAIN,5,(255,255,0),7)

    cv2.putText(fr,str(tmp),(wCam//4-30,350),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
    cv2.putText(fr,str(todays_in),(3*wCam//4-30,3*hCam//4),cv2.FONT_HERSHEY_PLAIN,4,(0,255,0),4)
    cv2.putText(fr,str(todays_out),(wCam//4-30,3*hCam//4),cv2.FONT_HERSHEY_PLAIN,4,(0,255,0),4)

    # フレームとマスク画像を表示
    # cv2.imshow("Background", bg)
    cv2.imshow("img",frame)
    cv2.imshow("mask",mask)
    cv2.imshow("fr",fr)

    # 待機(0.03sec)
    time.sleep(0.03)

    # 背景画像の更新（一定間隔）
    if(cv2.waitKey(1) & 0xFF == ord('r')):
        flag = 1
        ret, bg = cap.read()
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        writer.write(save_frame)
    # qキーが押されたら途中終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()
cv2.destroyAllWindows()