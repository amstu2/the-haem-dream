import requests


def mlflow_server_alive(tracking_uri: str):
    res = requests.get(tracking_uri + "/version")
    if res.status_code == 200:
        return True
    else:
        print(f"No mlflow server response! Content: {res.content}")
        return False
