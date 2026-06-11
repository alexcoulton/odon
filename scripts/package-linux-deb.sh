#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_name="${APP_NAME:-odon}"
mcp_name="${MCP_NAME:-odon_mcp}"
version="$(awk -F '"' '/^version = / { print $2; exit }' "$root_dir/Cargo.toml")"
machine="$(uname -m)"
case "$machine" in
  x86_64)
    deb_arch="amd64"
    ;;
  aarch64|arm64)
    deb_arch="arm64"
    ;;
  *)
    deb_arch="$machine"
    ;;
esac

bundle_dir="$root_dir/target/release/bundle/linux"
package_root="$bundle_dir/deb-root"
deb_path="$bundle_dir/${app_name}_${version}_${deb_arch}.deb"

cargo build --release --bin "$app_name" --bin "$mcp_name"

rm -rf "$package_root"
mkdir -p \
  "$package_root/DEBIAN" \
  "$package_root/usr/bin" \
  "$package_root/usr/lib/$app_name" \
  "$package_root/usr/share/applications" \
  "$package_root/usr/share/doc/$app_name" \
  "$package_root/usr/share/icons/hicolor/256x256/apps"

cp "$root_dir/target/release/$app_name" "$package_root/usr/lib/$app_name/$app_name"
cp "$root_dir/target/release/$mcp_name" "$package_root/usr/lib/$app_name/$mcp_name"
chmod 0755 "$package_root/usr/lib/$app_name/$app_name" "$package_root/usr/lib/$app_name/$mcp_name"

if [[ -d "$root_dir/assets" ]]; then
  mkdir -p "$package_root/usr/lib/$app_name/assets"
  cp -R "$root_dir/assets/." "$package_root/usr/lib/$app_name/assets/"
fi

cp "$root_dir/README.md" "$package_root/usr/share/doc/$app_name/README.md"
cp "$root_dir/LICENSE" "$package_root/usr/share/doc/$app_name/copyright"
cp "$root_dir/assets/odon.png" "$package_root/usr/share/icons/hicolor/256x256/apps/odon.png"

ln -s "../lib/$app_name/$app_name" "$package_root/usr/bin/$app_name"

cat > "$package_root/usr/share/applications/odon.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Odon
Comment=Spatial omics image viewer
Exec=/usr/lib/odon/odon %U
Icon=odon
Terminal=false
Categories=Science;Graphics;Viewer;
MimeType=x-scheme-handler/odon;
EOF

depends="libc6"
if command -v dpkg-shlibdeps >/dev/null 2>&1; then
  generated_depends="$(
    dpkg-shlibdeps -O \
      "$package_root/usr/lib/$app_name/$app_name" \
      "$package_root/usr/lib/$app_name/$mcp_name" \
      2>/dev/null || true
  )"
  generated_depends="$(printf '%s\n' "$generated_depends" | sed -n 's/^shlibs:Depends=//p')"
  if [[ -n "$generated_depends" ]]; then
    depends="$generated_depends"
  fi
fi

cat > "$package_root/DEBIAN/control" <<EOF
Package: odon
Version: $version
Section: science
Priority: optional
Architecture: $deb_arch
Maintainer: Odon Developers <noreply@example.com>
Depends: $depends
Description: Fast desktop viewer for multiplexed spatial omics data
 Odon is a native desktop viewer for OME-Zarr spatial proteomics and spatial
 transcriptomics data. The package includes the optional odon_mcp helper for
 MCP clients.
EOF

cat > "$package_root/DEBIAN/postinst" <<'EOF'
#!/usr/bin/env sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor >/dev/null 2>&1 || true
fi
EOF
chmod 0755 "$package_root/DEBIAN/postinst"

cat > "$package_root/DEBIAN/postrm" <<'EOF'
#!/usr/bin/env sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor >/dev/null 2>&1 || true
fi
EOF
chmod 0755 "$package_root/DEBIAN/postrm"

find "$package_root" -type d -exec chmod 0755 {} +

rm -f "$deb_path"
dpkg-deb --build "$package_root" "$deb_path"

echo "$deb_path"
