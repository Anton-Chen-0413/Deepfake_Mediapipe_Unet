# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 09:26:40 2023

@author: user
"""
import time

import cv2
import numpy as np       # 載入 numpy 函式庫
import mediapipe as mp

def face_Deep():
    mp_drawing = mp.solutions.drawing_utils             # mediapipe 繪圖方法
    mp_drawing_styles = mp.solutions.drawing_styles     # mediapipe 繪圖樣式
    mp_face_mesh = mp.solutions.face_mesh               # mediapipe 人臉網格方法
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)  # 繪圖參數設定
    ta = time.time()
    cap = cv2.VideoCapture(0)
    # 啟用人臉網格偵測，設定相關參數
    with mp_face_mesh.FaceMesh(
        max_num_faces=10,       # 一次偵測最多幾個人臉
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
            
        
        n = 0
        while True:
            print("ta", ta)
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 顏色 BGR 轉換為 RGB
            img2 = cv2.resize(img1, (320, 256))#312 256
            results = face_mesh.process(img2)             # 取得人臉網格資訊
            output = np.zeros((256,320,3), dtype='uint8')   # 繪製 320x256 的黑色畫布
            # print(results.multi_face_landmarks)
            # print("================================")
            # print(mp_face_mesh.FACEMESH_TESSELATION)
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 繪製網格
                    mp_drawing.draw_landmarks(
                        image=output,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    
                    # 繪製輪廓
                    # mp_drawing.draw_landmarks(
                    #     image=output,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_contours_style())
                    
                    # 繪製眼睛
                    mp_drawing.draw_landmarks(
                        image=output,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            tb = time.time()
            print("tb", tb)
            
            if tb - ta > 0.1 :
                print("tb-ta", tb-ta)
                n += 1
                cv2.imwrite("D:/Face_Off/Face_B/{}_mask_0_r.jpg".format(n), img2)
                cv2.imwrite("D:/Face_Off/Face_A/{}.jpg".format(n), output)
                ta = tb
            
            cv2.imshow('oxxostudio_1', img2)               
            if cv2.waitKey(5) == ord('q'):
                break    # 按下 q 鍵停止
            cv2.imshow('oxxostudio_2', output)
            if cv2.waitKey(5) == ord('q'):
                break    # 按下 q 鍵停止    
                
    cap.release()
    cv2.destroyAllWindows()


'''
def deep():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    
    cap = cv2.VideoCapture(0)
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img,(480,320))                 # 調整影像尺寸為 480x320
            output = np.zeros((320,480,3), dtype='uint8')   # 繪製 480x320 的黑色畫布
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img2)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 繪製網格
                    mp_drawing.draw_landmarks(
                        image=output,     # 繪製到 output
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    # 繪製輪廓
                    mp_drawing.draw_landmarks(
                        image=output,     # 繪製到 output
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    # 繪製眼睛
                    mp_drawing.draw_landmarks(
                        image=output,     # 繪製到 output
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
    
            cv2.imshow('oxxostudio', output)     # 顯示 output
            if cv2.waitKey(5) == ord('q'):
                break    # 按下 q 鍵停止
    cap.release()
    cv2.destroyAllWindows()
'''   
face_Deep()

