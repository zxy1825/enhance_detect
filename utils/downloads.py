# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Download utils."""

import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch

"""
功能：判断给定的字符串是否为有效的URL地址。如果check参数为真，还会检查该URL在线上是否存在。
参数：
    url：待检查的字符串。
    check：是否检查URL在线上是否存在。
返回值：如果是有效的URL并且（如果check为真）在线上存在，则返回True；否则返回False。
"""
def is_url(url, check=True):
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False

"""
功能：使用gsutil du命令返回Google Cloud Storage URL上文件的大小（字节为单位）。
参数：
    url：Google Cloud Storage的URL。
返回值：文件的大小（字节为单位），如果命令失败或输出为空，则返回0。
"""
def gsutil_getsize(url=""):
    output = subprocess.check_output(["gsutil", "du", url], shell=True, encoding="utf-8")
    return int(output.split()[0]) if output else 0

"""
功能：返回给定URL的可下载文件大小（字节为单位）；如果未找到，则默认为-1。
参数：
    url：文件的URL地址。
返回值：文件的大小（字节为单位），如果未找到则为-1。
"""
def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """Returns the size in bytes of a downloadable file at a given URL; defaults to -1 if not found."""
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get("content-length", -1))

"""
功能：使用curl从URL下载文件到指定的文件名。
参数：
    url：文件的URL地址。
    filename：保存文件的路径和名称。
    silent：是否在下载时不显示进度条和其他消息。
返回值：如果下载成功，则返回True；否则返回False。
"""
def curl_download(url, filename, *, silent: bool = False) -> bool:
    """Download a file from a url to a filename using curl."""
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0

"""
功能：从URL（或备用URL）下载文件到指定路径，如果文件大小超过最小字节数，则执行此操作。如果下载不完整，则会移除下载的部分。
参数：
    file：文件路径。
    url：主URL地址。
    url2：备用URL地址。
    min_bytes：文件的最小字节数。
    error_msg：错误消息模板。
行为：尝试从主URL下载；如果失败，尝试从备用URL下载。检查下载文件的大小，并处理错误情况。
"""
def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")

"""
功能：如果本地不存在文件，则尝试从GitHub发布资产或直接URL下载文件，支持备用版本。
参数：
    file：文件路径。
    repo：GitHub仓库名称。
    release：GitHub发行版本。
行为：检查文件是否已存在；如果不存在，则尝试从GitHub或直接URL下载。支持从特定版本或最新版本的GitHub仓库中下载文件。
"""
def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    from utils.general import LOGGER

    def github_assets(repository, version="latest"):
        # Return GitHub repo tag (i.e. 'v7.0') and assets (i.e. ['yolov5s.pt', 'yolov5m.pt', ...])
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)
