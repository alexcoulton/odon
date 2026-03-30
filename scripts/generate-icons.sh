#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_png="$root_dir/docs/assets/images/logo.upscaled.white.cropped.png"
out_png="$root_dir/assets/odon.png"
out_ico="$root_dir/assets/odon.ico"
out_icns="$root_dir/assets/odon.icns"

if ! command -v magick >/dev/null 2>&1; then
  echo "ImageMagick ('magick') is required." >&2
  exit 1
fi

if [[ ! -f "$src_png" ]]; then
  echo "Source logo not found: $src_png" >&2
  exit 1
fi

mkdir -p "$root_dir/assets"

# Crop away the wordmark so the icon uses the same dragonfly mark as the docs
# logo, then center it on a transparent square canvas for desktop icon use.
magick "$src_png" \
  -gravity north \
  -crop 1748x620+0+0 +repage \
  -background none \
  -gravity center \
  -resize 900x900 \
  -extent 1024x1024 \
  "$out_png"

magick "$out_png" -define icon:auto-resize=256,128,64,48,32,16 "$out_ico"
magick "$out_png" "$out_icns"

echo "Generated:"
echo "  $out_png"
echo "  $out_ico"
echo "  $out_icns"
