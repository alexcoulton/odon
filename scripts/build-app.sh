#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a macOS .app bundle for odon.

Usage:
  ./scripts/build-app.sh [--debug|--release] [--open]

Options:
  --debug    Build the debug binary and bundle it
  --release  Build the release binary and bundle it (default)
  --open     Open the resulting .app in Finder after bundling

Environment overrides:
  APP_NAME   Bundle name / executable name (default: odon)
  BUNDLE_ID  macOS bundle identifier (default: org.odon.odon)
EOF
}

profile="release"
open_after=0

while (($# > 0)); do
  case "$1" in
    --debug)
      profile="debug"
      ;;
    --release)
      profile="release"
      ;;
    --open)
      open_after=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_name="${APP_NAME:-odon}"
bundle_id="${BUNDLE_ID:-org.odon.odon}"
version="$(awk -F '"' '/^version = / { print $2; exit }' "$root_dir/Cargo.toml")"
target_dir="$root_dir/target/$profile"
binary_path="$target_dir/$app_name"
app_dir="$target_dir/bundle/osx/${app_name}.app"
contents_dir="$app_dir/Contents"
macos_dir="$contents_dir/MacOS"
resources_dir="$contents_dir/Resources"

if [[ "$profile" == "release" ]]; then
  cargo build --release
else
  cargo build
fi

if [[ ! -x "$binary_path" ]]; then
  echo "Built binary not found at $binary_path" >&2
  exit 1
fi

rm -rf "$app_dir"
mkdir -p "$macos_dir" "$resources_dir"

cp "$binary_path" "$macos_dir/$app_name"
chmod +x "$macos_dir/$app_name"

if [[ -d "$root_dir/assets" ]]; then
  mkdir -p "$resources_dir/assets"
  cp -R "$root_dir/assets/." "$resources_dir/assets/"
fi

icon_src=""
for candidate in \
  "$root_dir/assets/${app_name}.icns" \
  "$root_dir/assets/icon.icns"
do
  if [[ -f "$candidate" ]]; then
    icon_src="$candidate"
    break
  fi
done

icon_block=""
if [[ -n "$icon_src" ]]; then
  icon_name="$(basename "$icon_src")"
  cp "$icon_src" "$resources_dir/$icon_name"
  icon_block="  <key>CFBundleIconFile</key>
  <string>${icon_name}</string>"
fi

cat > "$contents_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>${app_name}</string>
  <key>CFBundleIdentifier</key>
  <string>${bundle_id}</string>
$icon_block
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>${app_name}</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${version}</string>
  <key>CFBundleVersion</key>
  <string>${version}</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
EOF

printf 'APPL????' > "$contents_dir/PkgInfo"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$app_dir" >/dev/null 2>&1 || true
fi

echo "Built app bundle:"
echo "  $app_dir"

if [[ "$open_after" -eq 1 ]]; then
  open "$app_dir"
fi
