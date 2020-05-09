# Text Extraction from Images using Pytesseract
 
A Python wrapper for Google Tesseract that can automatically orient images horizontally and extract textual data using Tesseract OCR, OpenCV, and CLD2.

## Requirements:

* OpenCV:
pip install opencv-python

* Pillow:
pip install Pillow==2.2.2

* Scipy:
pip install scipy

* CLD2:
pip install pycld2

* PyTesseract:

1. Install tesseract using windows installer available at: https://github.com/UB-Mannheim/tesseract/wiki

2. Note the tesseract path from the installation.Default installation path at the time the time of this edit was: C:\Users\USER\AppData\Local\Tesseract-OCR. It may change so please check the installation path.

3. pip install pytesseract

4. Set the tesseract path in the script before calling image_to_string:

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

## Procedure:

python extract.py -i "image_path.jpg" -p {default=thresh}

or

python extract.py --image {"image_path.jpg"} --preprocess {default=thresh}

or just simply,

python extract.py --image {"image_path.jpg}