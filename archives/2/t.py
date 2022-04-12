import requests

def main():
    with open("image.jpeg", "rb") as f:
        r = requests.post("https://cat-or-dog-api.tunaisyummy.repl.co/api", data=f.read())
        print(r.json())

main()