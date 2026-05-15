import zipfile
from pathlib import Path

import requests


def mlflow_server_alive(tracking_uri: str):
    res = requests.get(tracking_uri + "/version")
    if res.status_code == 200:
        return True
    else:
        print(f"No mlflow server response! Content: {res.content}")
        return False


def download_gcs_file(
    bucket_name: str, source_blob_name: str, destination_file_path: str
):
    # https://oneuptime.com/blog/post/2026-02-17-how-to-use-the-google-cloud-storage-python-library-to-upload-and-download-files-from-cloud-storage-buckets/view
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)

    print(
        f"File gs://{bucket_name}/{source_blob_name} downloaded to {destination_file_path}"
    )


def unzip_file(zip_path: Path, extract_path: Path):
    with zipfile.ZipFile(str(zip_path), "r") as f:
        for member in f.infolist():
            target = extract_path / member.filename.rstrip("/")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                f.extract(member, extract_path)

    print(f"File {zip_path} extracted to {extract_path}")


def initialise_worker_datasets():
    zip_path = Path("./pbc_dataset.zip")
    if not zip_path.exists():
        download_gcs_file(
            "the-haem-dream", "cell-dataset/pbc_dataset.zip", str(zip_path)
        )
    unzip_file(zip_path, Path("./"))
