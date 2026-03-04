"""
Upload toàn bộ thư mục /workspace/FAANTRA/data lên Google Drive
Cần: pip install google-api-python-client google-auth google-auth-oauthlib
"""

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Scope cần thiết để upload file
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Thư mục nguồn và tên thư mục trên Drive
SOURCE_DIR = "/workspace/FAANTRA/data"
DRIVE_FOLDER_NAME = "FAANTRA_data"  # Tên thư mục tạo trên Drive


def get_drive_service():
    """Xác thực và trả về Drive API service."""
    creds = None
    token_path = "token.json"
    credentials_path = "credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    "Cần file credentials.json. Tải từ Google Cloud Console:\n"
                    "1. Tạo project → APIs & Services → Enable Drive API\n"
                    "2. Credentials → Create OAuth 2.0 Client ID (Desktop app)\n"
                    "3. Tải JSON, đổi tên thành credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)


def create_or_get_folder(service, name, parent_id=None):
    """Tạo hoặc lấy ID thư mục trên Drive."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def upload_file(service, local_path, parent_id, filename=None):
    """Upload một file lên Drive."""
    filename = filename or os.path.basename(local_path)
    file_metadata = {"name": filename, "parents": [parent_id]}
    media = MediaFileUpload(local_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields="id, name").execute()
    return file.get("id")


def upload_directory(service, source_dir, parent_id, base_path=None):
    """Upload đệ quy toàn bộ thư mục."""
    base_path = base_path or source_dir
    items = sorted(Path(source_dir).iterdir())
    for item in items:
        rel_path = os.path.relpath(item, base_path)
        if item.is_file():
            try:
                upload_file(service, str(item), parent_id, item.name)
                print(f"  [OK] {rel_path}")
            except Exception as e:
                print(f"  [FAIL] {rel_path}: {e}")
        else:
            folder_id = create_or_get_folder(service, item.name, parent_id)
            print(f"  [DIR] {rel_path}/")
            upload_directory(service, str(item), folder_id, base_path)


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"Thư mục không tồn tại: {SOURCE_DIR}")
        return

    print(f"Đang upload {SOURCE_DIR} lên Google Drive...")
    service = get_drive_service()
    root_folder_id = create_or_get_folder(service, DRIVE_FOLDER_NAME)
    print(f"Thư mục đích: {DRIVE_FOLDER_NAME} (ID: {root_folder_id})")
    upload_directory(service, SOURCE_DIR, root_folder_id)
    print("Hoàn tất!")


if __name__ == "__main__":
    main()