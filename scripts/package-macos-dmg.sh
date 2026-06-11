#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a macOS DMG for odon.

Usage:
  ./scripts/package-macos-dmg.sh [--debug|--release]

Environment overrides:
  APP_NAME   Bundle name / executable name (default: odon)
EOF
}

profile="release"

while (($# > 0)); do
  case "$1" in
    --debug)
      profile="debug"
      ;;
    --release)
      profile="release"
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
bundle_dir="$root_dir/target/$profile/bundle/osx"
app_dir="$bundle_dir/${app_name}.app"
dmg_stage="$bundle_dir/dmg-stage"
dmg_path="$bundle_dir/${app_name}-macos.dmg"

"$root_dir/scripts/build-app.sh" "--$profile"

rm -rf "$dmg_stage"
mkdir -p "$dmg_stage"
cp -R "$app_dir" "$dmg_stage/"
ln -s /Applications "$dmg_stage/Applications"

rm -f "$dmg_path"
hdiutil create \
  -volname "Odon" \
  -srcfolder "$dmg_stage" \
  -ov \
  -format UDZO \
  "$dmg_path"

echo "$dmg_path"
