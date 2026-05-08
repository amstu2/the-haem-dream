from pathlib import Path
import zipfile

def download_gcs_file(bucket_name: str, source_blob_name: str, destination_file_path: str):
    # https://oneuptime.com/blog/post/2026-02-17-how-to-use-the-google-cloud-storage-python-library-to-upload-and-download-files-from-cloud-storage-buckets/view
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_path)

    print(f"File gs://{bucket_name}/{source_blob_name} downloaded to {destination_file_path}")

def unzip_file(zip_path: Path, extract_path:Path):
    with zipfile.ZipFile(str(zip_path), 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    
    print(f"File {zip_path} extracted to {extract_path}")