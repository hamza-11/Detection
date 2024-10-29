from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import threading
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يمكنك تحديد نطاقات معينة بدلاً من "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تهيئة متغيرات عامة
camera = None
output_frame = None
lock = threading.Lock()
detected_objects = []

# تحديد الفئات
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# تهيئة COLORS عشوائياً
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# تحميل نموذج Caffe
print("[INFO] تحميل النموذج...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

def detect_motion(frame):
    detected = []
    (h, w) = frame.shape[:2]
    
    # تحويل الإطار إلى blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                (300, 300), 127.5)
    
    # تمرير البلوب عبر الشبكة العصبية
    net.setInput(blob)
    detections = net.forward()
    
    # التكرار على الكشوفات
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            # استخراج مؤشر الفئة
            idx = int(detections[0, 0, i, 1])
            
            # حساب إحداثيات المربع
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # إضافة الكشف إلى القائمة
            detected.append({
                'type': CLASSES[idx],
                'confidence': float(confidence),
                'box': [startX, startY, endX - startX, endY - startY]
            })
    
    return detected

def generate_frames():
    global camera, output_frame, lock, detected_objects
    
    while True:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                logging.error("Could not open camera")
                break
        
        success, frame = camera.read()
        if not success:
            break
        
        detected_objects = detect_motion(frame)
        
        # رسم المربعات حول الكائنات المكتشفة
        for obj in detected_objects:
            startX, startY, w, h = obj['box']
            endX = startX + w
            endY = startY + h
            
            # اختيار لون للفئة
            idx = CLASSES.index(obj['type'])
            color = [int(c) for c in COLORS[idx]]
            
            # رسم المربع والتسمية
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            label = f"{obj['type']}: {obj['confidence']:.2f}%"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        with lock:
            output_frame = frame.copy()
        
        if detected_objects:
            # إرسال إشعار بالكائنات المكتشفة
            print(f"Objects detected: {[obj['type'] for obj in detected_objects]}")
        
        # تحويل الإطار إلى JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/get_detections')
async def get_detections():
    global detected_objects
    return detected_objects

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000, debug=True)
