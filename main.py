from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO

app = FastAPI()

# تحميل نموذج YOLO مدرب على كشف الأشخاص، السيارات، والدراجات
model = YOLO("yolov5s.pt")  # يمكنك اختيار نموذج مختلف إذا لزم الأمر

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # تحقق من نوع الملف
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # قراءة محتويات الملف
    image_data = await file.read()
    image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # تشغيل الكشف باستخدام YOLO
    results = model.predict(image)
    
    # التحقق من وجود شخص، سيارة أو دراجة
    alert = False
    alert_classes = {"person", "car", "motorcycle"}
    detected_classes = set()

    for result in results:
        for detection in result.boxes:
            label = model.names[int(detection.cls)]
            if label in alert_classes:
                detected_classes.add(label)
                alert = True

    # إذا تم الكشف عن الجسم المطلوب، إرسال إنذار
    if alert:
        return {"alert": True, "detected_classes": list(detected_classes)}
    else:
        return {"alert": False, "detected_classes": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
