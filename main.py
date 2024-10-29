import cv2
from flask import Flask, jsonify, Response
from flask_cors import CORS

# تهيئة التطبيق والصوت
app = Flask(__name__)
CORS(app)
# ضع مسار صوت الإنذار هنا

# تحميل نموذج MobileNet-SSD
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_objects():
    cap = cv2.VideoCapture(0)
    detected_objects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # تمرير الصورة عبر الشبكة
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        detected_objects.clear()  # إعادة ضبط الكائنات المكتشفة
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] in ["person", "car", "bus", "bicycle", "motorbike"]:
                    detected_objects.append(CLASSES[idx])

                    # تشغيل الإنذار إذا لم يكن قيد التشغيل
                    

        # كسر الحلقة بعد التشغيل لتجنب الإفراط في استخدام الكاميرا
        break

    cap.release()
    return detected_objects

@app.route('/detect', methods=['GET'])
def detect():
    detected = detect_objects()
    return jsonify({"detected_objects": detected})

if __name__ == "__main__":
    app.run(debug=True)
