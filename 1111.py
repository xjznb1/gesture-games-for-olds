import requests

url = "http://129.28.37.144:8080/upload"

files = {'file': open(r"training_data\user_20260303_000426.json", 'rb')}
r = requests.post(url, files=files)

print(r.text)