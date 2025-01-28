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


class LlamaCpp:
    def __init__(
        self, models_dir, cache_dir="./cache", verbose=False, timeout_minutes=5
    ):
        """
        Initialize the LlamaCpp class.

        Args:
            models_dir (str): Directory where GGUF models are stored.
            cache_dir (str): Directory to store llama.cpp binaries and related assets.
            verbose (bool): Whether to enable verbose logging.
            timeout_minutes (int): Timeout for shutting down idle servers.
        """
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.timeout_minutes = timeout_minutes
        self.llama_cpp_path = (
            self._install_llama_cpp_binaries()
        )  # Handle binaries installation
        self.servers = (
            {}
        )  # Maintain a mapping of model names to LlamaCppServer instances

    def list_models(self):
        """
        List all GGUF models available in the `models_dir`.

        Returns:
            list: A list of model names (files ending in ".gguf").
        """
        if not os.path.exists(self.models_dir):
            self._log(f"Models directory does not exist: {self.models_dir}")
            return []
        models = [f for f in os.listdir(self.models_dir) if f.endswith(".gguf")]
        self._log(f"Available models: {models}")
        return models

    def chat_completion(self, body):
        """
        Handle chat completion requests.

        Args:
            body (dict): The payload for the chat completion request. It must contain the "model" key.

        Returns:
            dict or generator: Response from the server (non-streaming or streaming mode).
        """
        if "model" not in body:
            raise ValueError("The request body must contain a 'model' key.")

        model_name = body["model"]
        gguf_path = os.path.join(self.models_dir, model_name)

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"Model file not found: {gguf_path}")

        # Check if the server for this model is already running
        if model_name not in self.servers or not self.servers[model_name]._server_url:
            self._log(f"Initializing a new server for model: {model_name}")
            self.servers[model_name] = self._create_server(gguf_path)

        server = self.servers[model_name]
        return server.chat_completion(body)

    def _create_server(self, gguf_path):
        """
        Create a new LlamaCppServer instance for the given model.

        Args:
            gguf_path (str): Path to the GGUF model file.

        Returns:
            LlamaCppServer: A new server instance.
        """
        return LlamaCppServer(
            llama_cpp_path=self.llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
            timeout_minutes=self.timeout_minutes,
        )

    def _install_llama_cpp_binaries(self):
        """
        Download and install llama.cpp binaries.

        Returns:
            str: Path to the installed llama.cpp binaries.
        """
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
        """
        Fetch the latest release of llama.cpp from GitHub.

        Returns:
            dict: Release information.
        """
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        response = requests.get(api_url)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(
                f"Failed to fetch release info. Status code: {response.status_code}"
            )

    def _get_appropriate_asset(self, assets):
        """
        Select the appropriate binary asset for the current system.

        Args:
            assets (list): List of asset metadata from the release.

        Returns:
            dict or None: Matching asset metadata, or None if no match found.
        """
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
        """
        Check whether the latest binaries are already cached.

        Args:
            release_info (dict): Metadata of the latest release.
            asset (dict): Metadata of the selected asset.

        Returns:
            bool: True if the cached binary matches the latest release, False otherwise.
        """
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
        """
        Download and extract llama.cpp binaries.

        Args:
            url (str): URL of the asset to download.
            asset_name (str): Name of the asset file.
        """
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
        """
        Update cache metadata with the downloaded release info.

        Args:
            release_info (dict): Metadata of the latest release.
            asset (dict): Metadata of the downloaded asset.
        """
        cache_info = {"tag_name": release_info["tag_name"], "asset_name": asset["name"]}
        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        with open(cache_info_path, "w") as f:
            json.dump(cache_info, f)

    def _log(self, message):
        """
        Print a log message if verbosity is enabled.

        Args:
            message (str): Log message to print.
        """
        if self.verbose:
            print(f"[LlamaCpp] {message}")


class LlamaCppServer:
    def __init__(
        self,
        llama_cpp_path=None,
        gguf_path=None,
        cache_dir="./cache",
        hugging_face=False,
        verbose=False,
        timeout_minutes=5,
    ):
        """
        Initialize the LlamaCppServer.

        Args:
            llama_cpp_path (str): Path to the llama.cpp binaries.
            gguf_path (str): Path to the GGUF model file.
            cache_dir (str): Directory to store llama.cpp binaries and related files.
            hugging_face (bool): Whether the model is hosted on Hugging Face.
            verbose (bool): Enable verbose logging.
            timeout_minutes (int): Timeout duration for shutting down idle servers.
        """
        self.verbose = verbose
        self.hugging_face = hugging_face
        self.cache_dir = cache_dir
        self.llama_cpp_path = llama_cpp_path
        self.gguf_path = gguf_path
        self.server_process = None
        self._server_url = None
        self._server_thread = None
        self.port = None
        self.last_used = time.time()  # Tracks the last time the server was used
        self.timeout_minutes = timeout_minutes
        self._auto_terminate_thread = None

        # Validate llama_cpp_path
        if llama_cpp_path is None:
            raise ValueError("llama_cpp_path must be provided.")
        elif not os.path.exists(llama_cpp_path):
            raise FileNotFoundError(
                f"Specified llama_cpp_path not found: {llama_cpp_path}"
            )

        # Validate gguf_path
        if gguf_path and not os.path.exists(gguf_path) and not hugging_face:
            raise FileNotFoundError(f"Specified gguf_path not found: {gguf_path}")

        # Start the server if gguf_path is provided
        if gguf_path:
            self._start_server_in_thread()
            self._start_auto_terminate_thread()

    @property
    def url(self):
        """Return the URL where the server is running."""
        if self._server_url is None:
            # If the server URL is not available, ensure the server spins up again
            self._log("Server is off. Restarting the server...")
            self._start_server_in_thread()
            self._start_auto_terminate_thread()
            # Wait for the thread to start the server
            while self._server_url is None:
                time.sleep(1)

        # Update the last-used timestamp whenever this property is accessed
        self.last_used = time.time()
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

        if self._auto_terminate_thread and self._auto_terminate_thread.is_alive():
            self._auto_terminate_thread.join()

    def chat_completion(self, payload):
        """
        Send a chat completion request to the server.

        Args:
            payload (dict): Payload for the chat completion request.

        Returns:
            dict or generator: Response from the server (non-streaming or streaming mode).
        """
        if self._server_url is None:
            self._log(
                "Server is off. Restarting the server before making the request..."
            )
            self._start_server_in_thread()
            self._start_auto_terminate_thread()
            # Wait for the thread to start the server
            while self._server_url is None:
                time.sleep(1)

        # Reset the last-used timestamp
        self.last_used = time.time()
        endpoint = f"{self._server_url}/v1/chat/completions"
        self._log(f"Sending chat completion request to {endpoint}...")

        # Check if streaming is enabled in the payload
        if payload.get("stream", False):
            self._log(f"Streaming mode enabled. Returning a generator.")
            response = requests.post(endpoint, json=payload, stream=True)
            if response.status_code == 200:
                # Return a generator for streaming responses
                def stream_response():
                    for line in response.iter_lines(decode_unicode=True):
                        yield line

                return stream_response()
            else:
                self._log(
                    f"Request failed with status code: {response.status_code} - {response.text}"
                )
                response.raise_for_status()
        else:
            # Non-streaming mode
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                self._log("Request successful.")
                return response.json()
            else:
                self._log(
                    f"Request failed with status code: {response.status_code} - {response.text}"
                )
                response.raise_for_status()

    def _start_server_in_thread(self):
        """Start the server in a separate thread."""

        def target():
            try:
                self._start_server()
            except Exception as e:
                self._log(f"Failed to start server: {e}")

        self._server_thread = threading.Thread(target=target, daemon=True)
        self._server_thread.start()

    def _start_auto_terminate_thread(self):
        """Start the auto-terminate thread that monitors idle time."""

        def monitor_idle_time():
            while True:
                time.sleep(10)
                if (
                    self.server_process and self.server_process.poll() is None
                ):  # Server is running
                    elapsed_time = time.time() - self.last_used
                    if elapsed_time > self.timeout_minutes * 60:
                        self._log(
                            "Server has been idle for too long. Auto-terminating..."
                        )
                        self.kill()
                        break

        self._auto_terminate_thread = threading.Thread(
            target=monitor_idle_time, daemon=True
        )
        self._auto_terminate_thread.start()

    def _start_server(self):
        """Start the llama-server."""
        if not self.gguf_path or (
            not self.hugging_face and not os.path.exists(self.gguf_path)
        ):
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

        commands = [server_binary]
        if self.hugging_face:
            commands.extend(["-hf", self.gguf_path, "--port", str(self.port)])
        else:
            commands.extend(["-m", self.gguf_path, "--port", str(self.port)])

        self.server_process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Wait for the server to confirm it is ready by monitoring its output
        self._server_url = None
        for line in iter(self.server_process.stdout.readline, ""):
            self._log(line.strip())
            if "listening on" in line:
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
