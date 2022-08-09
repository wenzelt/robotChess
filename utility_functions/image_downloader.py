import requests

from apiCheckApp.services import EchoService

LOCAL = False


def download_image(url="https://lab.bpm.in.tum.de/img/high/url") -> bytes:
    if LOCAL:
        with open(
            "full_boards/chess_full2022-06-28 13:56:05.803477.png", "rb"
        ) as image:
            f = image.read()
        return f
    try:
        EchoService.echo("sending request")
        url_endpoint_response = requests.get(url, params={})
        img_endpoint_response = requests.get(url_endpoint_response.content, params={})
        EchoService.echo("response received")
        if img_endpoint_response.status_code == 200:
            EchoService.echo(f"status_code == {img_endpoint_response.status_code}")
            image_bytes = img_endpoint_response.content
            return image_bytes
        elif img_endpoint_response.status_code == 502:
            EchoService.echo(f"status_code == {img_endpoint_response.status_code}")
            raise SystemExit()
        else:
            raise requests.exceptions.HTTPError
    except requests.exceptions.Timeout:
        print("Timeout exception")

