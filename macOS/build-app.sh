#!/bin/sh
set -eu
app_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$app_root"
swift build -c release
binary_dir=$(swift build -c release --show-bin-path)
bundle="${WHISPERX_APP_OUTPUT:-$app_root/output/WhisperX.app}"
mkdir -p "$bundle/Contents/MacOS"
mkdir -p "$bundle/Contents/Resources"
cp "$binary_dir/WhisperXMac" "$bundle/Contents/MacOS/WhisperXMac.next"
mv -f "$bundle/Contents/MacOS/WhisperXMac.next" "$bundle/Contents/MacOS/WhisperXMac"
ditto "$binary_dir/WhisperXMac_WhisperXMac.bundle" "$bundle/Contents/Resources/WhisperXMac_WhisperXMac.bundle"
cp "$app_root/Worker/transcribe_worker.py" "$bundle/Contents/Resources/transcribe_worker.py"
ditto "$app_root/../whisperx" "$bundle/Contents/Resources/whisperx"
cat > "$bundle/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>WhisperX</string>
<key>CFBundleDisplayName</key><string>WhisperX</string>
<key>CFBundleIdentifier</key><string>it.whisperx.mac</string>
<key>CFBundleExecutable</key><string>WhisperXMac</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.6.0</string>
<key>CFBundleVersion</key><string>7</string>
<key>CFBundleLocalizations</key><array><string>it</string><string>en</string></array>
<key>CFBundleDevelopmentRegion</key><string>en</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$bundle"
printf 'App creata: %s\n' "$bundle"
