#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_name="${APP_NAME:-odon}"
mcp_name="${MCP_NAME:-odon_mcp}"
arch="$(uname -m)"
bundle_dir="$root_dir/target/release/bundle/linux"
stage_dir="$bundle_dir/${app_name}-linux-${arch}"
archive_path="$bundle_dir/${app_name}-linux-${arch}.tar.gz"

cargo build --release --bin "$app_name" --bin "$mcp_name"

rm -rf "$stage_dir"
mkdir -p "$stage_dir"
mkdir -p "$bundle_dir"

cp "$root_dir/target/release/$app_name" "$stage_dir/$app_name"
chmod +x "$stage_dir/$app_name"
cp "$root_dir/target/release/$mcp_name" "$stage_dir/$mcp_name"
chmod +x "$stage_dir/$mcp_name"

if [[ -d "$root_dir/assets" ]]; then
  mkdir -p "$stage_dir/assets"
  cp -R "$root_dir/assets/." "$stage_dir/assets/"
fi

mkdir -p "$stage_dir/examples"
cp -R "$root_dir/fixtures/synthetic_5ch.ome.zarr" "$stage_dir/examples/synthetic_5ch.ome.zarr"
cp "$root_dir/fixtures/synthetic_5ch.project.json" "$stage_dir/examples/synthetic_5ch.project.json"
cp "$root_dir/fixtures/odon-deep-link-test.html" "$stage_dir/examples/odon-deep-link-test.html"

cp "$root_dir/README.md" "$stage_dir/README.md"
cp "$root_dir/LICENSE" "$stage_dir/LICENSE"

tar -C "$bundle_dir" -czf "$archive_path" "$(basename "$stage_dir")"

echo "$archive_path"
