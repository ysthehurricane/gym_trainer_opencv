import cv2
import mediapipe as mp
import time
import math

class poseDetector:
    def __init__(self, static_mode=False, upbbody=False, smooth=True, detection_confident= 0.5, tracking_confident=0.5):
        self.static_mode = static_mode
        self.upbbody = upbbody
        self.smooth = smooth
        self.detection_confident = detection_confident
        self.tracking_confident = tracking_confident
        
        self.mppose = mp.solutions.pose
        self.pose = self.mppose.Pose(self.static_mode, self.upbbody, self.smooth, self.detection_confident, self.tracking_confident) 
        self.mpdraw = mp.solutions.drawing_utils
        
        
    def findpose(self, frame, draw_landmark=True):
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        
        if self.results.pose_landmarks:
            if draw_landmark:
                self.mpdraw.draw_landmarks(frame, self.results.pose_landmarks, self.mppose.POSE_CONNECTIONS)
        
        return frame
    
    def findlocation(self, frame, draw_landmark=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            mypose = self.results.pose_landmarks
            
            for idx,lm in enumerate(mypose.landmark):
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                
                self.lmlist.append([idx,cx,cy])
                if draw_landmark:
                    cv2.circle(frame, (cx,cy), 5, (255,0,255), cv2.FILLED)

        return self.lmlist
    
    def findangle(self, frame, p1, p2, p3, draw=True):
        
        #Get landmarks position
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
        
        #Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2)) 
       
        if angle < 0:
            angle += 360
        
        #Draw landmarks
        if draw:
            cv2.line(frame, (x1,y1), (x2,y2), (255,255,0), 5)
            cv2.line(frame, (x3,y3), (x2,y2), (255,255,0), 5)
            cv2.circle(frame, (x1, y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x1, y1), 15, (0,0,255), 2)
            cv2.circle(frame, (x2, y2), 10, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 15, (0,0,255), 2)
            cv2.circle(frame, (x3, y3), 10, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x3, y3), 15, (0,0,255), 2)
            cv2.putText(frame, str(int(angle)), (x2-50, y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            
        return angle
