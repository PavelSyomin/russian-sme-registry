import pathlib
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests
import tqdm

from ..utils.enums import SourceDatasets, Storages


class Downloader:
    API_ENDPOINTS = {
        "resources": "disk/resources",
        "upload": "disk/resources/upload",
    }
    HOST = "https://cloud-api.yandex.net/v1/"
    OPENDATA_URLS = {
        SourceDatasets.sme.value: "https://www.nalog.gov.ru/opendata/7707329152-rsmp/",
        SourceDatasets.revexp.value: "https://www.nalog.gov.ru/opendata/7707329152-revexp/",
        SourceDatasets.empl.value: "https://www.nalog.gov.ru/opendata/7707329152-sshr2019/",
    }
    STORAGES = [s.value for s in Storages]
    YDISK_DOWNLOAD_TIMEOUT = 60

    def __init__(self, storage: str = Storages.local.value,
                 token: Optional[str] = None):
        if storage not in self.STORAGES:
            raise RuntimeError(
                f"Unknown storage {storage}, expected one of {self.STORAGES}")

        if storage in (Storages.ydisk.value,) and token is None:
            raise RuntimeError("Token is required to use ydisk storage")

        self._token = token
        self._storage = storage

    def __call__(self, source_dataset: str, download_dir: Optional[str] = None):
        if source_dataset not in self.OPENDATA_URLS:
            raise RuntimeError(f"Unknown source dataset {source_dataset}, "
                f"expected one of {list(self.OPENDATA_URLS.keys())}")

        data_urls = self._get_data_urls(source_dataset)

        if download_dir is None:
            download_dir = self._get_default_download_dir(source_dataset)
        self._create_download_dir(download_dir)

        if self._storage == Storages.local.value:
            self._download_to_local(data_urls, download_dir)
        elif self._storage == Storages.ydisk.value:
            self._download_to_ydisk(data_urls, download_dir)
        elif self._storage == Storages.ydisk_public.value:
            raise RuntimeError(
                "Download from public Yandex Disk storage is not supported yet. "
                "Please run extract stage (and next stages) directly "
                "or run process command with default --no-download flag"
            )

    def _create_download_dir(self, download_dir: str):
        if self._storage == Storages.local.value:
            path = pathlib.Path(download_dir)
            if not path.exists():
                path.mkdir(parents=True)
            else:
                print(f"Directory {download_dir} already exists")
        else:
            self._make_ydisk_dir(download_dir)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"OAuth {self._token}",
            "Content-Type": "application/json",
        }

    def _get_default_download_dir(self, source_dataset: str) -> str:
        return f"russian-sme-registry-data/download/{source_dataset}"

    def _get_data_urls(self, source_dataset: str) -> List[str]:
        url = self.OPENDATA_URLS[source_dataset]

        # Get the page source
        print(f"Scraping {url}")
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Cannot get the page")
            return None

        # Make the soup
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get the table of interest
        table = soup.find("table", class_="border_table")
        if table is None:
            print("Cannot find table in data source")
            return []

        # Parse table
        rows = table("tr")
        row_index = {}
        for row in rows:
            cells = row("td")
            if len(cells) != 3:
                continue

            number = cells[0].get_text(strip=True)
            if number.isdigit():
                number = int(number)
            else:
                print("Unexpected non-numeric row number")
                continue

            name = cells[1].get_text(strip=True)
            value = cells[2]

            row_index[number] = dict(name=name, value=value)

        # Check whether the table contains all necessary items
        expected_rows = {
            8: "Гиперссылка (URL) на набор",
            16: "Гиперссылки (URL) на предыдущие релизы набора данных",
        }
        for expected_number, expected_name in expected_rows.items():
            name = row_index.get(expected_number, {}).get("name")
            if name != expected_name:
                print("Something is wrong with data source: "
                      f"expected {expected_name} on position {expected_number}, found {name}")
                return []

        # Extract URLs of data files
        data_urls = []
        data_links = row_index[8]["value"]("a") + row_index[16]["value"]("a")
        for link in data_links:
            url = link.get("href")
            if url is not None:
                data_urls.append(url.strip())

        print(f"Found {len(data_urls)} data url(s)")

        return data_urls

    def _make_ydisk_dir(self, path: str):
        print(f"Trying to create {path} on Yandex Disk")

        headers = self._get_headers()

        if path[0] != "/":
            path = "/" + path

        parts = path[1:].split("/")
        for i, part in enumerate(parts):
            current_path = "/" + "/".join(parts[:i+1])
            params = {
                "path": current_path,
            }
            url = urljoin(self.HOST, self.API_ENDPOINTS["resources"])
            resp = requests.put(url, headers=headers, params=params)

            if resp.status_code == 201:
                print(f"Folder {current_path} created")
            elif resp.json().get("error") == "DiskPathPointsToExistentDirectoryError":
                print(f"Folder {current_path} exists")
            else:
                print("Error while creating folder, see details below")
                print(resp.json())

    def _get_existing_files(self, path: str) -> List[str]:
        print(f"Trying to get items list for {path}")
        result = []
        headers = self._get_headers()
        params = {
            "path": path,
            "fields": "_embedded.items.path,_embedded.items.type",
            "limit": 500,
        }
        url = urljoin(self.HOST, self.API_ENDPOINTS["resources"])

        resp = requests.get(url, headers=headers, params=params)
        if resp.status_code != 200:
            print("Cannot get path medatata, see error message below")
            print(resp.json())
            print()
            return result

        for item in resp.json().get("_embedded", {}).get("items", []):
            if item.get("type") == "file":
                result.append(item.get("path"))

        print(f"Found {len(result)} files")

        return result

    def _get_existing_files_local(self, path: str) -> List[str]:
        return [f.name for f in pathlib.Path(path).glob("*")]

    def _check_existing(self, link: str, existing_files: List[str]):
        # Checking by name (the last part of link or path)
        _, _, link_name = link.rpartition("/")
        for existing_file in existing_files:
            _, _, file_name = existing_file.rpartition("/")
            if link_name == file_name:
                return True

        return False

    def _save_remote_file_to_local(self, file_url: str, download_dir: str):
        filename = self._extract_filename_from_url(file_url)
        download_path = pathlib.Path(download_dir) / filename
        resp = requests.get(file_url, stream=True)
        if resp.status_code != 200:
            print("Cannot download file")
            return

        with open(download_path, "wb") as f:
            for chunk in tqdm.tqdm(resp.iter_content(2**20)): # chunk size is 1 Mib
                f.write(chunk)

    def _save_remote_file_to_ydisk(self, file_url: str, path: str) -> Optional[str]:
        print(f"Uploading {file_url} to {path}")

        _, _, file_name = file_url.rpartition("/")
        dest = f"{path}/{file_name}"

        headers = self._get_headers()
        params = {
            "url": file_url,
            "path": dest,
        }
        url = urljoin(self.HOST, self.API_ENDPOINTS["upload"])

        resp = requests.post(url, headers=headers, params=params)
        if resp.status_code == 202:
            print("Upload task created successfully")
            info = resp.json().get("href")
        else:
            print("Error while creating upload task")
            print(resp.json())
            info = None

        return info

    def _check_ydisk_download_status(self, info_url: Optional[str]) -> str:
        if info_url is None:
            return "cannot check (no status URL)"

        headers = self._get_headers()

        try:
            resp = requests.get(info_url, headers=headers, timeout=3)
        except Exception as e:
            return "check error"

        if resp.status_code == 200:
            return resp.json().get("status")
        else:
            return "check error"

    def _download_to_ydisk(self, data_urls: str, download_dir: str):
        existing_data_files = self._get_existing_files(download_dir)
        for url in data_urls:
            if self._check_existing(url, existing_data_files):
                continue

            info_url = self._save_remote_file_to_ydisk(url, download_dir)
            if info_url is None:
                print("Failed to create download task")
                continue

            print("Upload task created")

            print("Waiting for download to complete")
            interval = 5
            for _ in range(0 + self.YDISK_DOWNLOAD_TIMEOUT // interval):
                time.sleep(interval)
                status = self._check_ydisk_download_status(info_url)
                if status == "success":
                    print("Downloaded")
                    break
                if status == "failed":
                    print("Failed to download")
                    break
            else:
                print("Download timeout exceeded, result is unknown")

    def _download_to_local(self, data_urls: List[str], download_dir: str):
        existing_files = self._get_existing_files_local(download_dir)
        for url in data_urls:
            if self._check_existing(url, existing_files):
                continue
            self._save_remote_file_to_local(url, download_dir)

    def _extract_filename_from_url(self, url: str) -> str:
        *_, filename = url.rpartition("/")

        return filename
