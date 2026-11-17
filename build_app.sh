#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PROJECT_DIR}/.venv-tk/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Dieser Build ist für macOS vorgesehen." >&2
    exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python-Umgebung fehlt: ${PYTHON}" >&2
    echo "Lege zuerst .venv-tk an und installiere requirements.txt." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
"${PYTHON}" -m pip install -r requirements-build.txt
ICONSET_DIR="${PROJECT_DIR}/build/Sound Library Tagger.iconset"
ICON_PNG="${PROJECT_DIR}/build/sound-library-icon.svg.png"
ICON_FILE="${PROJECT_DIR}/build/Sound Library Tagger.icns"
mkdir -p "${ICONSET_DIR}"
qlmanage -t -s 1024 \
    -o "${PROJECT_DIR}/build" \
    "${PROJECT_DIR}/assets/sound-library-icon.svg" >/dev/null

for entry in \
    "icon_16x16.png:16" \
    "icon_16x16@2x.png:32" \
    "icon_32x32.png:32" \
    "icon_32x32@2x.png:64" \
    "icon_128x128.png:128" \
    "icon_128x128@2x.png:256" \
    "icon_256x256.png:256" \
    "icon_256x256@2x.png:512" \
    "icon_512x512.png:512" \
    "icon_512x512@2x.png:1024"
do
    filename="${entry%%:*}"
    size="${entry##*:}"
    sips -z "${size}" "${size}" "${ICON_PNG}" \
        --out "${ICONSET_DIR}/${filename}" >/dev/null
done
iconutil -c icns -o "${ICON_FILE}" "${ICONSET_DIR}"

"${PYTHON}" -m PyInstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "Sound Library Tagger" \
    --icon "${ICON_FILE}" \
    --osx-bundle-identifier "com.tuarek.soundlibrarytagger" \
    --collect-all tkinterdnd2 \
    --add-data "assets/ucs/ucs-8.2.1.tsv:assets/ucs" \
    app.py

"${PYTHON}" - <<'PY'
import plistlib
from pathlib import Path

info_path = Path("dist/Sound Library Tagger.app/Contents/Info.plist")
with info_path.open("rb") as info_file:
    info = plistlib.load(info_file)

info["CFBundleDocumentTypes"] = [
    {
        "CFBundleTypeName": "WAV Audio",
        "CFBundleTypeExtensions": ["wav"],
        "CFBundleTypeRole": "Editor",
        "LSHandlerRank": "Alternate",
        "LSItemContentTypes": ["com.microsoft.waveform-audio"],
    },
    {
        "CFBundleTypeName": "MP3 Audio",
        "CFBundleTypeExtensions": ["mp3"],
        "CFBundleTypeRole": "Editor",
        "LSHandlerRank": "Alternate",
        "LSItemContentTypes": ["public.mp3"],
    },
]

with info_path.open("wb") as info_file:
    plistlib.dump(info, info_file)
PY

echo
echo "App erstellt: ${PROJECT_DIR}/dist/Sound Library Tagger.app"
