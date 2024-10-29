from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import numpy as np
import time

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

def detect_objects():
    # إعداد نموذج الكشف عن الكائنات
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # افتح كاميرا الويب
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # قم بإعداد الإطار للكشف
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward(output_layers)

        for detection in detections:
            for result in detection:
                confidence = result[2]
                if confidence > 0.5:  # يمكن تعديل هذه القيمة لتقليل أو زيادة الحساسية
                    # يمكن إضافة كود لتحليل الكشف هنا
                    socketio.emit('alert', {'message': 'Object detected!'})
                    break  # للخروج من الحلقة بعد اكتشاف كائن

        time.sleep(1)  # تأخير لمحاكاة الكشف

        # أضف شرط للخروج من الحلقة
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    socketio.start_background_task(detect_objects)  # بدء عملية الكشف في الخلفية
    socketio.run(app, host='0.0.0.0', port=5000)
