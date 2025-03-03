import os
import requests

def download_file(url, path):
    print(f"Downloading file from {url} to {path}...")
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully: {path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

if __name__ == "__main__":

    if not os.path.exists('database'):
        os.makedirs('database')
        print("Created 'database' folder.")

    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/'
    cifar = 'edm-cifar10-32x32-cond-vp.pkl'
    ffhq = 'edm-ffhq-64x64-uncond-vp.pkl'
    ffhqv2 = 'edm-afhqv2-64x64-uncond-vp.pkl'
    imagenet = 'edm-imagenet-64x64-cond-adm.pkl'

    download_file(model_root + cifar, os.path.join('database', cifar))
    download_file(model_root + ffhq, os.path.join('database', ffhq))
    download_file(model_root + ffhqv2, os.path.join('database', ffhqv2))
    download_file(model_root + imagenet, os.path.join('database', imagenet))

