# 🦙 llama-cpp-runner

`llama-cpp-runner` is the ultimate Python library for running [llama.cpp](https://github.com/ggerganov/llama.cpp) with zero hassle. It automates the process of downloading prebuilt binaries from the upstream repo, keeping you always **up to date** with the latest developments. All while requiring no complicated setups—everything works **out-of-the-box**.

---

## Key Features 🌟

1. **Always Up-to-Date**: Automatically fetches the latest prebuilt binaries from the upstream llama.cpp GitHub repo. No need to worry about staying current.
2. **Zero Dependencies**: No need to manually install compilers or build binaries. Everything is handled for you during installation.
3. **Model Flexibility**: Seamlessly load and serve **GGUF** models stored locally or from Hugging Face with ease.
4. **Built-in HTTP Server**: Automatically spins up a server for chat interactions and manages idle timeouts to save resources.
5. **Cross-Platform Support**: Works on **Windows**, **Linux**, and **macOS** with automatic detection for AVX/AVX2/AVX512/ARM architectures.

---

## Why Use `llama-cpp-runner`?

- **Out-of-the-box experience**: Forget about setting up complex environments for building. Just install and get started! 🛠️  
- **Streamlined Model Serving**: Effortlessly manage multiple models and serve them with an integrated HTTP server.
- **Fast Integration**: Use prebuilt binaries from upstream so you can spend more time building and less time troubleshooting.

---

## Installation 🚀

Installing `llama-cpp-runner` is quick and easy! Just use pip:

```bash
pip install llama-cpp-runner
```

---

## Usage 📖

### Initialize the Runner

```python
from llama_cpp_runner import LlamaCpp

llama_runner = LlamaCpp(models_dir="path/to/models", verbose=True)

# List all available GGUF models
models = llama_runner.list_models()
print("Available Models:", models)
```

### Chat Completion

```python
response = llama_runner.chat_completion({
    "model": "your-model-name.gguf",
    "messages": [{"role": "user", "content": "Hello, Llama!"}],
    "stream": False
})

print(response)
```

---

## How It Works 🛠️

1. Automatically detects your system architecture (e.g., AVX, AVX2, ARM) and platform.
2. Downloads and extracts the prebuilt llama.cpp binaries from the official repo.
3. Spins up a lightweight HTTP server for chat interactions.

---

## Advantages 👍

- **Hassle-Free**: No need to compile binaries or manage system-specific dependencies.  
- **Latest Features, Always**: Stay up to date with llama.cpp’s improvements with every release.  
- **Optimized for Your System**: Automatically fetches the best binary for your architecture.

---

## Supported Platforms 🖥️

- Windows
- macOS
- Linux

---

## Contributing 💻

We’d love your contributions! Bug reports, feature requests, and pull requests are all welcome. 

---

## License 📜

This library is open-source and distributed under the MIT license.  

Happy chatting with llama.cpp! 🚀