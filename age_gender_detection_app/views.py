import cv2
import os
from django.shortcuts import render, redirect
from .forms import ImageUploadForm
from .models import UploadedImage
from django.conf import settings
from django.core.files.storage import default_storage


# Paths to pre-trained models
MODEL_DIR = os.path.join(settings.BASE_DIR, 'models')
FACE_DETECTION_CONFIG = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_DETECTION_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_CONFIG = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_CONFIG = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_CATEGORIES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_CATEGORIES = ['Male', 'Female']

def upload_result(request):
    upload_form = ImageUploadForm()
    result_image = None
    result_message = None
    gender = None
    age_category = None
    confidence = None

    if request.method == 'POST':
        upload_form = ImageUploadForm(request.POST, request.FILES)
        if upload_form.is_valid():
            # Clear old images before saving the new one
            clear_uploaded_images()
            image_instance = upload_form.save(commit=False)
            image_instance.save()
            result_image, result_message, gender, age_category, confidence = process_image(image_instance)
    
    context = {
        'upload_form': upload_form,
        'result_image': result_image,
        'result_message': result_message,
        'gender': gender,
        'age_category': age_category,
        'confidence': confidence,
    }
    return render(request, 'upload_result.html', context)

def process_image(image_instance):
    image_path = os.path.join(settings.MEDIA_ROOT, image_instance.image.name)
    image = cv2.imread(image_path)
    face_net = cv2.dnn.readNet(FACE_DETECTION_MODEL, FACE_DETECTION_CONFIG)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_CONFIG)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_CONFIG)

    # Face detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    result_message = "No face detected"
    result_image = None
    gender = None
    age_category = None
    confidence = None  # Initialize the confidence variable

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Extract the confidence score
        if confidence > 0.7:
            fr_h, fr_w = image.shape[:2]
            x1 = int(detections[0, 0, i, 3] * fr_w)
            y1 = int(detections[0, 0, i, 4] * fr_h)
            x2 = int(detections[0, 0, i, 5] * fr_w)
            y2 = int(detections[0, 0, i, 6] * fr_h)
            face = image[y1:y2, x1:x2]
            blob_face = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender prediction
            gender_net.setInput(blob_face)
            gender_preds = gender_net.forward()
            gender = GENDER_CATEGORIES[gender_preds[0].argmax()]

            # Age prediction
            age_net.setInput(blob_face)
            age_preds = age_net.forward()
            age_category = AGE_CATEGORIES[age_preds[0].argmax()]

            # Draw rectangle and labels on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), int(round(fr_h/150)), 8)
            label = f'Gender: {gender}, Age: {age_category}'
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4, cv2.LINE_AA)

            result_message = "Age and gender detected"
            result_image = image_instance.image.name

    return result_image, result_message, gender, age_category, confidence


def delete_image(request):
    UploadedImage.objects.all().delete()
    return redirect('upload_result')

def clear_uploaded_images():
    all_images = UploadedImage.objects.all()
    
    if len(all_images) > 1:
        # Get the most recent image
        most_recent_image = all_images.latest('id')
        
        # Delete all images except the most recent one
        for image in all_images.exclude(id=most_recent_image.id):
            image_path = os.path.join(settings.MEDIA_ROOT, str(image.image))
            default_storage.delete(image_path)
            image.delete()

