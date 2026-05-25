# Getting Started on Snapdragon X Elite — Install Commands

Congrats on the new dev box! Snapdragon X Elite is an ARM64 device, so a few of the install steps differ from typical x64 Windows setups. Here's a clean order of operations to run before doing anything else. Run all of these in **PowerShell** (the native ARM64 build, not x86 emulation).

> Tip: Open PowerShell and confirm you're on ARM64 first:
> ```powershell
> $env:PROCESSOR_ARCHITECTURE   # should print "ARM64"
> ```

---

## 1. Confirm Python 3.11 is the ARM64 build

You said Python 3.11 is already installed, but on Snapdragon you want the **native ARM64** build, not the x64 emulated one — that's critical for ONNX Runtime / QNN performance.

```powershell
python --version
python -c "import platform; print(platform.machine())"
```

The second command should print `ARM64`. If it prints `AMD64`, you're running the x64 build under emulation — uninstall it and grab the **Windows ARM64** installer from https://www.python.org/downloads/windows/ (look for "Windows installer (ARM64)").

---

## 2. Install Git (if not already there)

```powershell
winget install --id Git.Git -e --source winget
```

Close and reopen PowerShell so `git` is on PATH.

---

## 3. Install Visual Studio Build Tools (ARM64)

Some Python packages (and the QNN tooling) need MSVC + the Windows SDK. Install the Build Tools with ARM64 components:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
```

Then run the Visual Studio Installer, click **Modify** on Build Tools, and add:
- **Desktop development with C++** workload
- Individual components: **MSVC v143 - VS 2022 C++ ARM64 build tools**, **Windows 11 SDK (latest)**, **C++ CMake tools for Windows**

---

## 4. Install `uv` (recommended Python package manager)

`uv` is fast and handles virtualenvs cleanly on ARM64:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Reopen PowerShell so `uv` is on PATH. Verify:
```powershell
uv --version
```

(If you'd rather stick with stock `pip`, that's fine — just substitute `python -m pip install ...` for the `uv pip install ...` commands below.)

---

## 5. Create a project virtualenv

```powershell
mkdir C:\dev\winml-work
cd C:\dev\winml-work
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

You should now see `(.venv)` in your prompt.

---

## 6. Install ONNX Runtime with the QNN execution provider

This is the package that actually lets you target the **Hexagon NPU** on Snapdragon X Elite:

```powershell
uv pip install --upgrade pip
uv pip install onnxruntime-qnn
```

`onnxruntime-qnn` is the Qualcomm-flavored build of ORT and ships the QNN HTP backend DLLs. Don't install vanilla `onnxruntime` and `onnxruntime-qnn` side-by-side — pick one. For NPU work on this device, you want `onnxruntime-qnn`.

Verify the QNN EP is visible:
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `QNNExecutionProvider` in the list.

---

## 7. Install the rest of the model-prep toolchain

```powershell
uv pip install onnx onnxruntime-tools numpy
uv pip install optimum[onnxruntime] transformers
uv pip install huggingface_hub
```

This gets you:
- `onnx` — model IR manipulation
- `optimum` + `transformers` — exporting Hugging Face models to ONNX
- `huggingface_hub` — pulling models down

---

## 8. (Optional but useful) Install the WinML / DirectML stack

If you also want to fall back to GPU via DirectML on the Adreno GPU:

```powershell
uv pip install onnxruntime-directml
```

Again — only install this in a **separate** venv from `onnxruntime-qnn` to avoid DLL conflicts. Keep one venv per EP.

---

## 9. Sanity check: pull a small ONNX model and run on NPU

```powershell
python -c @"
import onnxruntime as ort
print('Providers:', ort.get_available_providers())
print('QNN available:', 'QNNExecutionProvider' in ort.get_available_providers())
"@
```

If that prints `QNN available: True`, you're set to start loading models onto the Hexagon NPU.

---

## TL;DR — minimum commands to copy-paste

```powershell
# (Make sure your Python 3.11 is the ARM64 build first!)
winget install --id Git.Git -e --source winget
winget install --id Microsoft.VisualStudio.2022.BuildTools -e
# ...then add "Desktop development with C++" + ARM64 MSVC via the VS Installer UI

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

mkdir C:\dev\winml-work; cd C:\dev\winml-work
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1

uv pip install --upgrade pip
uv pip install onnxruntime-qnn onnx numpy
uv pip install optimum[onnxruntime] transformers huggingface_hub

python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Once `QNNExecutionProvider` shows up in that final print, your environment is ready and you can move on to exporting / running models on the NPU.
