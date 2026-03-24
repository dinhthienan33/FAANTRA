import argparse
import mimetypes
import os
import socket
import time
from pathlib import Path

import dotenv
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    ReadTimeout,
    Timeout,
)

dotenv.load_dotenv()

REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")

CHUNK_UP = 128 * (1 << 20)
RETRIES_API = 5


def get_drive():
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def _retry_sleep(i: int):
    base = min(60, 2**i)
    jitter = min(60, base * 0.1)
    time.sleep(base + (jitter * (os.urandom(1)[0] / 255.0)))


def create_drive_folder(drive, name: str, parent_id: str) -> str:
    body = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    resp = drive.files().create(body=body, fields="id", supportsAllDrives=True).execute()
    return resp["id"]


def upload_file(drive, local_path: Path, parent_id: str, name: str | None = None) -> str:
    name = name or local_path.name
    mime, _ = mimetypes.guess_type(str(local_path))
    for i in range(RETRIES_API):
        try:
            media = MediaFileUpload(
                str(local_path),
                mimetype=mime or "application/octet-stream",
                resumable=True,
                chunksize=CHUNK_UP,
            )
            body = {"name": name, "parents": [parent_id]}
            req = drive.files().create(
                body=body,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            resp = None
            while resp is None:
                try:
                    _, resp = req.next_chunk()
                except (
                    ConnectionResetError,
                    BrokenPipeError,
                    socket.timeout,
                    RequestsConnectionError,
                    ReadTimeout,
                    Timeout,
                    ChunkedEncodingError,
                ):
                    _retry_sleep(i)
                    continue
            return resp["id"]
        except HttpError:
            if i == RETRIES_API - 1:
                raise
            _retry_sleep(i)
    raise RuntimeError("upload_file: unreachable")


def upload_folder_recursive(drive, local_folder: Path, parent_drive_id: str):
    # Tạo folder hiện tại trên Drive
    drive_folder_id = create_drive_folder(drive, local_folder.name, parent_drive_id)
    print(f"Created folder: {local_folder.name}")

    for item in local_folder.iterdir():
        if item.is_dir():
            upload_folder_recursive(drive, item, drive_folder_id)
        else:
            print(f"Uploading file: {item.name}")
            upload_file(drive, item, drive_folder_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", 
        type=Path, 
        default=Path("/workspace/evaluation_engine/folder_vivos_hifigan_inf")
    )
    parser.add_argument(
        "--folder-id",
        default="1AXx6CwC8OdGAv8PnIgPrWkJ_JK1i4xYv",
    )
    args = parser.parse_args()

    if not all([REFRESH_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise SystemExit("Missing Drive credentials in environment.")

    path = args.path.resolve()
    if not path.exists():
        raise SystemExit(f"Path not found: {path}")

    drive = get_drive()

    if path.is_dir():
        print(f"Starting recursive upload of folder: {path}")
        upload_folder_recursive(drive, path, args.folder_id)
    else:
        print(f"Uploading single file: {path}")
        upload_file(drive, path, args.folder_id)

    print("Done.")


if __name__ == "__main__":
    main()