import os
import re

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OMP error

import cv2
import easyocr
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder upload
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Inisialisasi model (pastikan file model sudah ada di folder yang sama)
model = YOLO("best.pt")  # Model deteksi plat nomor
reader = easyocr.Reader(["en"], gpu=True)  # OCR


# Fungsi-fungsi helper
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_plate_color(plate_img):
    """Mendeteksi warna dominan dari plat (putih atau hitam)."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return "white" if mean_intensity > 127 else "black"


def correct_perspective(plate_img):
    """Meluruskan plat nomor agar fit dengan ukuran gambar."""

    # **Pastikan gambar dalam format BGR sebelum mengubah ke grayscale**
    if len(plate_img.shape) == 2:  # Jika sudah grayscale (1 channel)
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Thresholding untuk mendapatkan bentuk plat
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Cari kontur terbesar (area plat)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate kontur agar mendapatkan bentuk segiempat
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(approx) == 4:
            # Urutkan titik menjadi TL, TR, BR, BL
            approx = approx.reshape(4, 2)
            sorted_points = sorted(approx, key=lambda x: x[0])  # Urutkan berdasarkan X

            if sorted_points[0][1] < sorted_points[1][1]:
                tl, bl = sorted_points[:2]  # Kiri atas, kiri bawah
            else:
                bl, tl = sorted_points[:2]

            if sorted_points[2][1] < sorted_points[3][1]:
                tr, br = sorted_points[2:]  # Kanan atas, kanan bawah
            else:
                br, tr = sorted_points[2:]

            # Tentukan titik referensi untuk perspektif transform
            src_pts = np.array([tl, tr, br, bl], dtype="float32")
            dst_pts = np.array(
                [[0, 0], [300, 0], [300, 100], [0, 100]], dtype="float32"
            )

            # Hitung matriks perspektif transform
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Transformasi gambar
            warped = cv2.warpPerspective(plate_img, matrix, (300, 100))

            return warped

    return plate_img  # Jika gagal, kembalikan gambar asli


def remove_small_contours(binary_img, min_area=50):
    """Menghapus kontur kecil seperti baut atau noise dari plat nomor."""
    contours, _ = cv2.findContours(
        binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.ones(binary_img.shape, dtype="uint8") * 255  # Mask putih

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(
                mask, [cnt], -1, 0, thickness=cv2.FILLED
            )  # Hapus kontur kecil

    return cv2.bitwise_and(binary_img, mask)


def preprocess_for_ocr(plate_img, plate_color):
    """Preprocessing dengan noise removal, eliminasi garis tepi, dan perspektif transformasi."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Noise removal menggunakan Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Pilihan thresholding berdasarkan warna plat nomor
    if plate_color == "white":
        thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]
    else:  # Black plate
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Hilangkan noise kecil (seperti baut)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Pastikan ukuran gambar cukup besar untuk OCR
    cleaned = cv2.resize(cleaned, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return cleaned


def detect_and_crop_plate(image_path):
    """Deteksi dan crop plat nomor"""
    image = cv2.imread(image_path)
    results = model(image)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_plate = image[y1:y2, x1:x2]
        return cropped_plate
    return None


def recognize_license_plate(plate_img):
    """Mengenali teks dari plat nomor dengan EasyOCR, hanya membaca plat utama."""
    plate_color = detect_plate_color(plate_img)
    processed_img = preprocess_for_ocr(plate_img, plate_color)
    results = reader.readtext(processed_img, detail=0, paragraph=True)

    raw_text = " ".join(results).upper()
    plate_matches = re.findall(r"[A-Za-z]{1,2}\s?\d{1,4}\s?[A-Za-z]{1,3}", raw_text)

    plate_text = max(results, key=len, default="")

    # Ambil hanya baris pertama dari hasil OCR (agar tidak membaca bulan/tahun berlaku)

    return plate_text


# Route untuk halaman utama
@app.route("/")
def index():
    return render_template("index.html")


# Route untuk API deteksi plat nomor
@app.route("/detect", methods=["POST"])
def detect_plate():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        try:
            # Simpan file upload
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Proses deteksi plat
            cropped_plate = detect_and_crop_plate(filepath)

            if cropped_plate is None:
                return jsonify({"error": "No license plate detected"}), 400

            # Simpan hasil crop (untuk ditampilkan di web)
            plate_filename = f"plate_{filename}"
            plate_path = os.path.join(app.config["UPLOAD_FOLDER"], plate_filename)
            cv2.imwrite(plate_path, cropped_plate)

            # OCR plat nomor
            plate_text = recognize_license_plate(cropped_plate)

            return jsonify(
                {
                    "status": "success",
                    "original_image": filename,
                    "plate_image": plate_filename,
                    "license_plate": plate_text,
                }
            )

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
