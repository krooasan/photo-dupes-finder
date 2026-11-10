@echo off
rem Builds dist\PhotoDupesFinder\ and dist\PhotoDupesFinder-win64.zip
rem Needs Python 3.11+ and models\sscd_disc_mixup.onnx (see tools\export_onnx.py)
setlocal
cd /d "%~dp0"

if not exist models\sscd_disc_mixup.onnx (
    echo models\sscd_disc_mixup.onnx not found. Run tools\export_onnx.py first.
    exit /b 1
)

if not exist .venv-build (
    py -3.11 -m venv .venv-build || exit /b 1
)
call .venv-build\Scripts\activate.bat
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt pyinstaller || exit /b 1

pyinstaller --noconfirm --clean --windowed --name PhotoDupesFinder --icon assets\icon.ico ^
    --add-data "models\sscd_disc_mixup.onnx;models" ^
    --add-data "assets\fonts;assets\fonts" ^
    --add-data "assets\icon.ico;assets" ^
    --collect-binaries onnxruntime ^
    --collect-submodules pillow_heif ^
    --collect-data customtkinter ^
    --exclude-module torch --exclude-module torchvision --exclude-module matplotlib ^
    run_app.py || exit /b 1

powershell -NoProfile -Command "Compress-Archive -Force -Path dist\PhotoDupesFinder -DestinationPath dist\PhotoDupesFinder-win64.zip"
echo Done: dist\PhotoDupesFinder\PhotoDupesFinder.exe
