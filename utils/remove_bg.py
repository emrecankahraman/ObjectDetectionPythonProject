import requests
import os

REMOVE_BG_API_KEY = os.getenv("REMOVE_BG_API_KEY")  # Ortam değişkeni olarak sakla

def remove_background(image_path, output_path="output_no_bg.png"):
    with open(image_path, 'rb') as file:
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': file},
            data={'size': 'auto'},
            headers={'X-Api-Key': REMOVE_BG_API_KEY}
        )

    if response.status_code == requests.codes.ok:
        with open(output_path, 'wb') as out:
            out.write(response.content)
        return output_path
    else:
        raise Exception(f"Remove.bg error: {response.status_code} - {response.text}")
