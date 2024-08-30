from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import easyocr
import pytesseract
import os
import tempfile
import fitz
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = 'uploads/'

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update to your Tesseract path

model = models.resnet18(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        recognized_text = ocr_scan(file_path)
        return render_template('result.html', text=recognized_text, file_url=url_for('uploaded_file', filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        classification_result = classify(file_path)
        return render_template('classification_result.html', result=classification_result, file_url=url_for('uploaded_file', filename=filename))

def classify(image_path):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    with open("imagenet_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]
    
    results = [(categories[catid], prob.item()) for catid, prob in zip(top5_catid, top5_prob)]
    return results

def ocr_scan(file_path):
    if file_path.lower().endswith('.pdf'):
        image_paths = extract_images_from_pdf(file_path)
        recognized_text = ""
        for img_path in image_paths:
            recognized_text += ocr_scan(img_path) + "\n\n"
        return recognized_text.strip()
    else:
        preprocessed_image_path = preprocess_image(file_path)
        easyocr_text = perform_easyocr(preprocessed_image_path)
        tesseract_text = perform_tesseract_ocr(preprocessed_image_path)
        recognized_text = f"EasyOCR: {easyocr_text}\nTesseract: {tesseract_text}"
        return recognized_text

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = ImageOps.invert(image)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    preprocessed_image_path = f"{tempfile.mkdtemp()}/{os.path.basename(image_path)}_preprocessed.jpg"
    image.save(preprocessed_image_path, "JPEG")
    return preprocessed_image_path

def perform_easyocr(image_path):
    result = reader.readtext(image_path)
    recognized_text = " ".join(elem[1] for elem in result)
    return recognized_text

def perform_tesseract_ocr(image_path):
    text = pytesseract.image_to_string(image_path)
    return text

def extract_images_from_pdf(pdf_path):
    image_paths = []
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                if image.mode == 'RGBA':
                    image = image.convert('RGB')  # Convert RGBA to RGB
                image_path = f"{tempfile.mkdtemp()}/{os.path.basename(pdf_path)}_page{page_num}_img{img_index}.jpg"
                image.save(image_path, "JPEG")
                image_paths.append(image_path)
    return image_paths

if __name__ == '__main__':  
    app.run(debug=True)
