import os
import platform
import requests
import zipfile
import argparse
import json
from packaging import version
import stat


def log(message, verbose=False):
    if verbose:
        print(message)


def get_latest_release(verbose=False):
    api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
    log(f"Fetching latest release info from: {api_url}", verbose)
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to fetch release info. Status code: {response.status_code}"
        )


def get_appropriate_asset(assets, verbose=False):
    system = platform.system().lower()
    machine = platform.machine().lower()
    processor = platform.processor()
    log(f"System: {system}", verbose)
    log(f"Machine: {machine}", verbose)
    log(f"Processor: {processor}", verbose)
    if system == "windows":
        if "arm" in machine:
            asset = next((a for a in assets if "win-arm64" in a["name"]), None)
        else:
            if "avx512" in processor:
                asset = next((a for a in assets if "win-avx512-x64" in a["name"]), None)
            elif "avx2" in processor:
                asset = next((a for a in assets if "win-avx2-x64" in a["name"]), None)
            elif "avx" in processor:
                asset = next((a for a in assets if "win-avx-x64" in a["name"]), None)
            else:
                asset = next((a for a in assets if "win-noavx-x64" in a["name"]), None)
    elif system == "darwin":
        if "arm" in machine:
            asset = next((a for a in assets if "macos-arm64" in a["name"]), None)
        else:
            asset = next((a for a in assets if "macos-x64" in a["name"]), None)
    elif system == "linux":
        asset = next((a for a in assets if "ubuntu-x64" in a["name"]), None)
    else:
        asset = None
    log(f"Selected asset: {asset['name'] if asset else None}", verbose)
    return asset


def set_executable(file_path):
    current_mode = os.stat(file_path).st_mode
    os.chmod(file_path, current_mode | stat.S_IEXEC)


def download_and_unzip(url, asset_name, cache_dir, verbose=False):
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, asset_name)
    log(f"Downloading from: {url}", verbose)
    response = requests.get(url)
    if response.status_code == 200:
        with open(zip_path, "wb") as file:
            file.write(response.content)
        log(f"Downloaded: {asset_name}", verbose)
        extract_dir = os.path.join(cache_dir, "llama_cpp")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        log(f"Extracted to: {extract_dir}", verbose)
        # Set execute permissions for all extracted files
        if platform.system() != "Windows":
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    set_executable(file_path)
            log("Set execute permissions for extracted files", verbose)
        else:
            log("Skipping permission setting on Windows", verbose)
        print(f"Successfully downloaded and extracted {asset_name}")
        return True
    else:
        print(f"Failed to download {asset_name}")
        return False


def check_cache(release_info, asset, cache_dir, verbose=False):
    cache_info_path = os.path.join(cache_dir, "cache_info.json")
    if os.path.exists(cache_info_path):
        with open(cache_info_path, "r") as f:
            cache_info = json.load(f)
        if (
            cache_info.get("tag_name") == release_info["tag_name"]
            and cache_info.get("asset_name") == asset["name"]
        ):
            log("Latest version already downloaded.", verbose)
            return True
    return False


def update_cache_info(release_info, asset, cache_dir):
    cache_info = {"tag_name": release_info["tag_name"], "asset_name": asset["name"]}
    cache_info_path = os.path.join(cache_dir, "cache_info.json")
    with open(cache_info_path, "w") as f:
        json.dump(cache_info, f)


def unzip_asset(asset_name, cache_dir, verbose=False):
    zip_path = os.path.join(cache_dir, asset_name)
    extract_dir = os.path.join(cache_dir, "llama_cpp")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        log(f"Extracted to: {extract_dir}", verbose)
        # Set execute permissions for all extracted files
        if platform.system() != "Windows":
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    set_executable(file_path)
            log("Set execute permissions for extracted files", verbose)
        else:
            log("Skipping permission setting on Windows", verbose)
        print(f"Successfully extracted {asset_name}")
        return True
    else:
        print(f"Zip file not found: {asset_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract llama.cpp binaries"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument("-c", "--cache", default="./cache", help="Cache directory")
    args = parser.parse_args()
    try:
        release_info = get_latest_release(args.verbose)
        assets = release_info["assets"]
        appropriate_asset = get_appropriate_asset(assets, args.verbose)
        if appropriate_asset:
            asset_name = appropriate_asset["name"]
            if check_cache(release_info, appropriate_asset, args.cache, args.verbose):
                print("Latest version already downloaded. Extracting cached version.")
                unzip_asset(asset_name, args.cache, args.verbose)
            else:
                download_url = appropriate_asset["browser_download_url"]
                if download_and_unzip(
                    download_url, asset_name, args.cache, args.verbose
                ):
                    update_cache_info(release_info, appropriate_asset, args.cache)
        else:
            print("No appropriate binary found for your system.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
