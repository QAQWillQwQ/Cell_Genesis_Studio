#!/bin/zsh
set -euo pipefail

APP_NAME="Cell Genesis Studio"
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_ROOT="$(cd "${APP_DIR}/.." && pwd)"
BUNDLE_DIR="${APP_DIR}/Dist/${APP_NAME}.app"
CONTENTS_DIR="${BUNDLE_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
LAUNCHER="${MACOS_DIR}/CellGenesisStudio"
PLIST="${CONTENTS_DIR}/Info.plist"
LOG_FILE="/tmp/cell_genesis_studio.log"

rm -rf "${BUNDLE_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cat > "${LAUNCHER}" <<EOF
#!/bin/zsh
set -u

APP_DIR="${APP_DIR}"
LOG_FILE="${LOG_FILE}"

export PYTHONPATH="${APP_DIR}/src:\${PYTHONPATH:-}"
export XDG_CACHE_HOME="/tmp/cell_genesis_studio_cache"
export NUMBA_CACHE_DIR="/tmp/cell_genesis_studio_numba_cache"
mkdir -p "\${XDG_CACHE_HOME}" "\${NUMBA_CACHE_DIR}"
cd "${APP_DIR}" || exit 1

find_conda() {
  for candidate in \
    "\${CONDA_EXE:-}" \
    "/opt/anaconda3/bin/conda" \
    "/opt/homebrew/bin/conda" \
    "/usr/local/bin/conda" \
    "\${HOME}/miniconda3/bin/conda" \
    "\${HOME}/anaconda3/bin/conda" \
    "/opt/homebrew/Caskroom/miniconda/base/bin/conda"
  do
    if [ -n "\${candidate}" ] && [ -x "\${candidate}" ]; then
      echo "\${candidate}"
      return 0
    fi
  done
  return 1
}

{
  echo "[\$(date '+%Y-%m-%d %H:%M:%S')] launching Cell Genesis Studio"
  if [ -n "\${CELLUNIVERSE_PYTHON:-}" ] && [ -x "\${CELLUNIVERSE_PYTHON}" ]; then
    echo "[launcher] using CELLUNIVERSE_PYTHON=\${CELLUNIVERSE_PYTHON}"
    exec "\${CELLUNIVERSE_PYTHON}" -m celluniverse_napari_app.local_app
  fi

  CONDA_BIN="\$(find_conda || true)"
  if [ -n "\${CONDA_BIN}" ]; then
    echo "[launcher] using conda=\${CONDA_BIN} env=napari"
    exec "\${CONDA_BIN}" run -n napari env XDG_CACHE_HOME="\${XDG_CACHE_HOME}" NUMBA_CACHE_DIR="\${NUMBA_CACHE_DIR}" python -m celluniverse_napari_app.local_app
  fi

  LOGIN_CONDA="\$(/bin/zsh -lc 'command -v conda' 2>/dev/null || true)"
  if [ -n "\${LOGIN_CONDA}" ]; then
    echo "[launcher] using login shell conda=\${LOGIN_CONDA} env=napari"
    exec /bin/zsh -lc "cd '${APP_DIR}' && export PYTHONPATH='${APP_DIR}/src:'\"\${PYTHONPATH:-}\" && export XDG_CACHE_HOME='/tmp/cell_genesis_studio_cache' && export NUMBA_CACHE_DIR='/tmp/cell_genesis_studio_numba_cache' && conda run -n napari env XDG_CACHE_HOME='/tmp/cell_genesis_studio_cache' NUMBA_CACHE_DIR='/tmp/cell_genesis_studio_numba_cache' python -m celluniverse_napari_app.local_app"
  fi

  echo "[launcher] conda was not found; falling back to python3"
  exec python3 -m celluniverse_napari_app.local_app
} >> "${LOG_FILE}" 2>&1
EOF
chmod +x "${LAUNCHER}"

cat > "${PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleDisplayName</key>
  <string>${APP_NAME}</string>
  <key>CFBundleExecutable</key>
  <string>CellGenesisStudio</string>
  <key>CFBundleIdentifier</key>
  <string>com.celluniverse.cellgenesisstudio</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${APP_NAME}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

for icon in \
  "${PROJECT_ROOT}/CellUniverse_Electron/resources/cell_universe_icon.png" \
  "${PROJECT_ROOT}/CellUniverse_MacLiquidGlass/Cell Universe.png"
do
  if [ -f "${icon}" ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    ICONSET="${RESOURCES_DIR}/CellGenesisStudio.iconset"
    mkdir -p "${ICONSET}"
    sips -z 16 16 "${icon}" --out "${ICONSET}/icon_16x16.png" >/dev/null
    sips -z 32 32 "${icon}" --out "${ICONSET}/icon_16x16@2x.png" >/dev/null
    sips -z 32 32 "${icon}" --out "${ICONSET}/icon_32x32.png" >/dev/null
    sips -z 64 64 "${icon}" --out "${ICONSET}/icon_32x32@2x.png" >/dev/null
    sips -z 128 128 "${icon}" --out "${ICONSET}/icon_128x128.png" >/dev/null
    sips -z 256 256 "${icon}" --out "${ICONSET}/icon_128x128@2x.png" >/dev/null
    sips -z 256 256 "${icon}" --out "${ICONSET}/icon_256x256.png" >/dev/null
    sips -z 512 512 "${icon}" --out "${ICONSET}/icon_256x256@2x.png" >/dev/null
    sips -z 512 512 "${icon}" --out "${ICONSET}/icon_512x512.png" >/dev/null
    sips -z 1024 1024 "${icon}" --out "${ICONSET}/icon_512x512@2x.png" >/dev/null
    if iconutil -c icns "${ICONSET}" -o "${RESOURCES_DIR}/CellGenesisStudio.icns" >/dev/null 2>&1; then
      /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string CellGenesisStudio" "${PLIST}" >/dev/null
      rm -rf "${ICONSET}"
      break
    fi
    rm -rf "${ICONSET}"
  fi
done

echo "Created: ${BUNDLE_DIR}"
echo "Runtime log: ${LOG_FILE}"
