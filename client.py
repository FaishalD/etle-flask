import requests

url = "http://localhost:5000/detect"  # Ganti dengan URL server jika di-deploy

# File gambar yang akan dikirim
file_path = "static\uploads"  # Ganti dengan path gambar Anda

try:
    with open(file_path, 'rb') as f:
        files = {'file': (file_path.split('/')[-1], f, 'image/jpeg')}
        response = requests.post(url, files=files)
    
    # Cek respons
    if response.status_code == 200:
        result = response.json()
        print("Deteksi berhasil!")
        print(f"Gambar asli: {result['original_image']}")
        print(f"Gambar plat: {result['plate_image']}")
        print(f"Teks plat: {result['license_plate']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

except Exception as e:
    print(f"Terjadi error: {str(e)}")