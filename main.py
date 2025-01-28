import os
import platform
import requests
import zipfile
import json
import subprocess
import threading
import stat
import time
import socket


class LlamaCppServer:
    def __init__(
        self, llama_cpp_path=None, gguf_path=None, cache_dir="./cache", verbose=False
    ):
        self.verbose = verbose
        self.cache_dir = cache_dir
        self.llama_cpp_path = llama_cpp_path
        self.gguf_path = gguf_path
        self.server_process = None
        self._server_url = None
        self._server_thread = None
        self.port = None

        # Fetch or validate llama path
        if llama_cpp_path is None:
            self.llama_cpp_path = self._install_llama_cpp_binaries()
        elif not os.path.exists(llama_cpp_path):
            raise FileNotFoundError(
                f"Specified llama_cpp_path not found: {llama_cpp_path}"
            )

        # Start the server if gguf_path is provided
        if gguf_path:
            self._start_server_in_thread()

    @property
    def url(self):
        """Return the URL where the server is running."""
        if self._server_url is None:
            raise ValueError(
                "Server is not running. Start the server with a valid GGUF path."
            )
        return self._server_url

    def kill(self):
        """Kill the server process and clean up."""
        if self.server_process and self.server_process.poll() is None:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            self._server_url = None
            self.port = None
            self._log("Llama server successfully killed.")
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join()

    def _start_server_in_thread(self):
        """Start the server in a separate thread."""

        def target():
            try:
                self._start_server()
            except Exception as e:
                self._log(f"Failed to start server: {e}")

        self._server_thread = threading.Thread(target=target, daemon=True)
        self._server_thread.start()

    def _start_server(self):
        """Start the llama-server."""
        if not self.gguf_path or not os.path.exists(self.gguf_path):
            raise ValueError(
                f"GGUF model path is not specified or invalid: {self.gguf_path}"
            )

        server_binary = os.path.join(
            self.llama_cpp_path, "build", "bin", "llama-server"
        )
        if not os.path.exists(server_binary):
            raise FileNotFoundError(f"Server binary not found: {server_binary}")

        # Ensure the binary is executable
        self._set_executable(server_binary)

        # Find an available port
        self.port = self._find_available_port(start_port=10000)
        if self.port is None:
            raise RuntimeError("No available port found between 10000 and 11000.")

        self._log(f"Starting server with binary: {server_binary}")
        self._log(f"Using GGUF path: {self.gguf_path}")
        self._log(f"Using port: {self.port}")

        self.server_process = subprocess.Popen(
            [server_binary, "-m", self.gguf_path, "--port", str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Wait for the server to confirm it is ready by monitoring its output
        self._server_url = None
        for line in iter(self.server_process.stdout.readline, ""):
            self._log(line.strip())
            if "Listening on" in line:
                self._server_url = f"http://localhost:{self.port}"
                self._log(f"Server is now accessible at {self._server_url}")
                break

        if not self._server_url:
            raise RuntimeError("Failed to confirm server is running.")

    def _find_available_port(self, start_port=10000, end_port=11000):
        """Find an available port between `start_port` and `end_port`."""
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("localhost", port)) != 0:
                    return port
        return None

    def _set_executable(self, file_path):
        """Ensure the file at `file_path` is executable."""
        if platform.system() != "Windows":
            current_mode = os.stat(file_path).st_mode
            os.chmod(
                file_path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
            )

    def _log(self, message):
        """Print a log message if verbosity is enabled."""
        if self.verbose:
            print(f"[LlamaCppServer] {message}")

    def _install_llama_cpp_binaries(self):
        """Download and install llama.cpp binaries."""
        self._log("Installing llama.cpp binaries...")
        release_info = self._get_latest_release()
        assets = release_info["assets"]
        asset = self._get_appropriate_asset(assets)
        if not asset:
            raise RuntimeError("No appropriate binary found for your system.")
        asset_name = asset["name"]
        if self._check_cache(release_info, asset):
            self._log("Using cached llama.cpp binaries.")
        else:
            self._download_and_unzip(asset["browser_download_url"], asset_name)
            self._update_cache_info(release_info, asset)
        return os.path.join(self.cache_dir, "llama_cpp")

    def _get_latest_release(self):
        """Fetch the latest release of llama.cpp from GitHub."""
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(
                f"Failed to fetch release info. Status code: {response.status_code}"
            )

    def _get_appropriate_asset(self, assets):
        """Select the appropriate binary asset for the current system."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        processor = platform.processor()
        if system == "windows":
            if "arm" in machine:
                return next((a for a in assets if "win-arm64" in a["name"]), None)
            elif "avx512" in processor:
                return next((a for a in assets if "win-avx512-x64" in a["name"]), None)
            elif "avx2" in processor:
                return next((a for a in assets if "win-avx2-x64" in a["name"]), None)
            elif "avx" in processor:
                return next((a for a in assets if "win-avx-x64" in a["name"]), None)
            else:
                return next((a for a in assets if "win-noavx-x64" in a["name"]), None)
        elif system == "darwin":
            if "arm" in machine:
                return next((a for a in assets if "macos-arm64" in a["name"]), None)
            else:
                return next((a for a in assets if "macos-x64" in a["name"]), None)
        elif system == "linux":
            return next((a for a in assets if "ubuntu-x64" in a["name"]), None)
        return None

    def _check_cache(self, release_info, asset):
        """Check whether the latest binaries are already cached."""
        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        if os.path.exists(cache_info_path):
            with open(cache_info_path, "r") as f:
                cache_info = json.load(f)
            if (
                cache_info.get("tag_name") == release_info["tag_name"]
                and cache_info.get("asset_name") == asset["name"]
            ):
                return True
        return False

    def _download_and_unzip(self, url, asset_name):
        """Download and extract llama.cpp binaries."""
        os.makedirs(self.cache_dir, exist_ok=True)
        zip_path = os.path.join(self.cache_dir, asset_name)
        self._log(f"Downloading binary from: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(zip_path, "wb") as file:
                file.write(response.content)
            self._log(f"Successfully downloaded: {asset_name}")
        else:
            raise RuntimeError(f"Failed to download binary: {url}")
        extract_dir = os.path.join(self.cache_dir, "llama_cpp")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        self._log(f"Extracted binaries to: {extract_dir}")

    def _update_cache_info(self, release_info, asset):
        """Update cache metadata with the downloaded release info."""
        cache_info = {"tag_name": release_info["tag_name"], "asset_name": asset["name"]}
        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        with open(cache_info_path, "w") as f:
            json.dump(cache_info, f)
