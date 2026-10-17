from __future__ import annotations

import re
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

import requests

from .config import Settings
from .ingestion import IngestionError

DRIVE_API = "https://www.googleapis.com/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
SLIDES_MIME = "application/vnd.google-apps.presentation"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
PDF_MIME = "application/pdf"


def parse_folder_url(url: str) -> tuple[str, str | None]:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise IngestionError("Enter an https://drive.google.com/drive/folders/... URL.")
    parts = parsed.path.strip("/").split("/")
    try:
        folder_id = parts[parts.index("folders") + 1]
    except (ValueError, IndexError):
        raise IngestionError("Enter a Google Drive folder link, not a file link.") from None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise IngestionError("The Google Drive folder link is invalid.")
    resource_key = parse_qs(parsed.query).get("resourcekey", [None])[0]
    return folder_id, resource_key


def _request(
    url: str, params: dict, api_key: str, resource_key: str | None = None,
    resource_id: str | None = None, *, stream=False,
):
    headers = (
        {"X-Goog-Drive-Resource-Keys": f"{resource_id}/{resource_key}"}
        if resource_key and resource_id else {}
    )
    try:
        response = requests.get(
            url, params={**params, "key": api_key}, headers=headers,
            timeout=(10, 120), stream=stream,
        )
    except requests.RequestException as e:
        raise IngestionError(f"Google Drive request failed ({type(e).__name__}).") from None
    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        response.close()
        suffix = f": {detail}" if detail else ""
        raise IngestionError(
            f"Google Drive returned HTTP {response.status_code}{suffix}. Check that the Drive API is enabled and the folder is shared with anyone who has the link."
        )
    return response


def _safe_name(name: str) -> str:
    name = PurePosixPath(name.replace("\\", "/")).name
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).rstrip(" .")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {
        f"LPT{i}" for i in range(1, 10)
    }
    if name.upper().split(".")[0] in reserved:
        name = f"_{name}"
    return name if name not in {"", ".", ".."} else "unnamed"


def _items(folder_id: str, resource_key: str | None, api_key: str):
    folders = [(folder_id, resource_key, ())]
    while folders:
        current_id, current_key, parent_path = folders.pop()
        page_token = None
        while True:
            params = {
                "q": f"'{current_id}' in parents and trashed = false",
                "fields": "nextPageToken,files(id,name,mimeType,size,resourceKey)",
                "pageSize": 1000,
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            response = _request(
                f"{DRIVE_API}/files", params, api_key, current_key, current_id
            )
            data = response.json()
            response.close()
            for item in data.get("files", []):
                item_id = item.get("id", "")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", item_id):
                    continue
                name = _safe_name(item.get("name", "unnamed"))
                path = (*parent_path, name)
                if item.get("mimeType") == FOLDER_MIME:
                    folders.append((item_id, item.get("resourceKey"), path))
                elif item.get("mimeType") in (PDF_MIME, PPTX_MIME, SLIDES_MIME):
                    yield item, "/".join(path)
            page_token = data.get("nextPageToken")
            if not page_token:
                break


def create_folder_zip(folder_url: str, zip_path: Path, settings: Settings) -> Path:
    """Download every supported document in a public Drive folder tree into a ZIP."""
    if not settings.google_drive_api_key:
        raise IngestionError("Set GOOGLE_DRIVE_API_KEY to enable Google Drive folder evaluation.")
    folder_id, resource_key = parse_folder_url(folder_url)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = total_bytes = 0
    used_names: set[str] = set()

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item, archive_name in _items(folder_id, resource_key, settings.google_drive_api_key):
                count += 1
                if count > settings.max_files:
                    raise IngestionError(f"Drive folder contains more than the {settings.max_files} file limit.")
                file_id = item["id"]
                mime = item["mimeType"]
                suffix = ".pdf" if mime == PDF_MIME else ".pptx"
                if mime == SLIDES_MIME:
                    url = f"{DRIVE_API}/files/{file_id}/export"
                    params = {"mimeType": PPTX_MIME}
                else:
                    url = f"{DRIVE_API}/files/{file_id}"
                    params = {"alt": "media"}
                key = item.get("resourceKey")
                response = _request(
                    url, params, settings.google_drive_api_key, key, file_id, stream=True
                )
                display = archive_name
                if Path(display).suffix.lower() != suffix:
                    display += suffix
                if display.lower() in used_names:
                    display = f"{Path(display).stem}-{file_id[:8]}{suffix}"
                used_names.add(display.lower())
                info = zipfile.ZipInfo(display)
                info.compress_type = zipfile.ZIP_DEFLATED
                expected = int(response.headers.get("Content-Length", "0") or 0)
                if expected and total_bytes + expected > settings.max_uncompressed_mb * 1024 * 1024:
                    response.close()
                    raise IngestionError(
                        f"Drive downloads exceed the {settings.max_uncompressed_mb} MB uncompressed limit."
                    )
                with response, archive.open(info, "w") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        total_bytes += len(chunk)
                        if total_bytes > settings.max_uncompressed_mb * 1024 * 1024:
                            raise IngestionError(
                                f"Drive downloads exceed the {settings.max_uncompressed_mb} MB uncompressed limit."
                            )
                        target.write(chunk)
        if count == 0:
            raise IngestionError("Drive folder contains no supported PDF or PowerPoint files.")
        if zip_path.stat().st_size > settings.max_zip_mb * 1024 * 1024:
            raise IngestionError(f"Drive download exceeds the {settings.max_zip_mb} MB ZIP limit.")
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise
    return zip_path
