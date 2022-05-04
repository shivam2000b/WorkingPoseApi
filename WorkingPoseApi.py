import time
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response


app=Flask(__name__)

def gen():
    mpDraw = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    cap = cv2.VideoCapture(0)
    counter=0
    stage=None
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(imgRGB)
        
        if result.pose_landmarks:
            mpDraw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
            
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            a = np.array(shoulder) 
            b = np.array(elbow) 
            c = np.array(wrist) 
            d = np.array(hip)
            ang = 0
            ang1 = 0
            radians= np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            ang = np.abs(radians*180/np.pi)
            if ang >180.0:
                ang = 360-ang
            
            radians1= np.arctan2(b[1]-a[1], b[0]-a[0]) - np.arctan2(d[1]-a[1], d[0]-a[0])
            ang1 = np.abs(radians1*180/np.pi)
            if ang1 >180.0:
                ang1 = 360-ang1
            
            
            cv2.putText(img, str(ang), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            cv2.putText(img, str(ang1), 
                           tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            
            
            
            
            
            if ang1<50 and 109<ang:
                stage="DOWN"
            
            if ang1<50 and ang<70 and stage=='DOWN':
                stage="UP"
                counter=counter+1
                
            
            else:    
                mpDraw.draw_landmarks(img, result.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS,
                mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
            
            cv2.putText(img, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.putText(img, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(img, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            #cv2.imshow("Pose detection", img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if cv2.waitKey(20) & 0xFF == ord('q') :
                break
            

           
            
            
@app.route('/api',methods=['GET'])
def video_feed():
    return Response(gen(),mimetype ='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=False)



