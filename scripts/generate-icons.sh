#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_png="$root_dir/docs/assets/images/logo.white.transparent.png"
out_png="$root_dir/assets/odon.png"
out_ico="$root_dir/assets/odon.ico"
out_icns="$root_dir/assets/odon.icns"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

if ! command -v magick >/dev/null 2>&1; then
  echo "ImageMagick ('magick') is required." >&2
  exit 1
fi

if [[ ! -f "$src_png" ]]; then
  echo "Source logo not found: $src_png" >&2
  exit 1
fi

mkdir -p "$root_dir/assets"

background_png="$tmp_dir/background.png"
logo_png="$tmp_dir/logo.png"
mask_png="$tmp_dir/mask.png"
composite_png="$tmp_dir/composite.png"

magick -size 1024x1024 gradient:'#1a1a1a-#000000' \
  -rotate 90 \
  "$background_png"

magick "$src_png" \
  -background none \
  -gravity center \
  -resize 900x900 \
  -extent 1024x1024 \
  "$logo_png"

magick "$background_png" "$logo_png" \
  -compose over \
  -composite \
  "$composite_png"

magick -size 1024x1024 xc:none \
  -fill white \
  -draw "roundrectangle 0,0 1023,1023 180,180" \
  "$mask_png"

magick "$composite_png" "$mask_png" \
  -alpha off \
  -compose copyopacity \
  -composite \
  "$out_png"

magick "$out_png" -define icon:auto-resize=256,128,64,48,32,16 "$out_ico"
magick "$out_png" "$out_icns"

echo "Generated:"
echo "  $out_png"
echo "  $out_ico"
echo "  $out_icns"
