import cv2
import torch
import glob
import camera_realworldxyz
import numpy as np

cameraXYZ=camera_realworldxyz.camera_realtimeXYZ()


# define a video capture object
url="http://192.168.1.100:4747/video"
# Neu dung web cam tren ung dung DroidCam thi thay url vao dong duoi vid = cv2.VideoCapture(url)
vid = cv2.VideoCapture(url)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/model_detect2D.pt') # Khai báo model đc train
classes = model.names  # name of objects
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def score_frame(frame):
    # Dự đoán
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord

def class_to_label(x):
    return classes[int(x)]

# Hàm vẽ box
def plot_boxes(results, frame):

    labels, cord = results
    print("labels", labels)
    print("cord", cord[:, :-1])
    clas = 14
    if len(labels) != 0:
        print("list is not empty")
        for label in labels:
            if label == clas:
                print("send objects")
            else:
                print("wrong objects")
    else:
        print("list is empty")
        print("no objects")
    x_c=0
    y_c=0
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, "Card" + " " + str(round(row[4], 2)), (x1-2, y1-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)     
            
            # Tâm vật trên ảnh
            x_c=int((x2-x1)/2)+x1
            y_c=int((y2-y1)/2)+y1
   
    return frame,x_c,y_c


while True:

    ret, frame = vid.read()
    frame = cv2.medianBlur(frame,5) # Lọc trung vị nhằm loại bỏ nhiễu

    cv2.line(frame, (320,0), (320,480), (0,0,255), 1) # Vẽ trục Oy 
    cv2.line(frame, (0,240), (640,240), (0,0,255), 1) # Vẽ trục Ox
    results = score_frame(frame)
    frame,x,y = plot_boxes(results, frame)

    # resize lại ảnh về 700x480
    frame=cv2.resize(frame,dsize=(700,410))
    x=int(x*700/640)
    y=int(y*410/480)
    # Tinh toán lại tọa độ tâm của vật so với trục tọa độ gốc có tâm O ở chính giữa bức ảnh
    x_0=x-350  
    y_0=205-y  
    cXYZ=cameraXYZ.calculate_XYZ(x,y)
    cv2.putText(frame, "(x,y,z)="+"("+str(int(cXYZ[0,0]))+","+str(int(cXYZ[1,0]))+","+str(int(cXYZ[2,0]))+")", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),1)
    cv2.putText(frame, "("+str(x) + "," + str(y)+")", (x, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.circle(frame,(x,y), 4, (0,0,255),-1)

    # Hiển thị ảnh 
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()
