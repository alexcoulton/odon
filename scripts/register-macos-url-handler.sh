#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Register a macOS development handler for odon:// links.

The generated handler is a tiny native Swift .app that receives odon:// URLs
from LaunchServices. It logs every URL, forwards it directly to a running Odon
process through the local development IPC socket, and falls back to launching
this checkout with:

  cargo run -- '<url>'

Usage:
  ./scripts/register-macos-url-handler.sh [--scheme odon] [--test-url URL]

Options:
  --scheme URL_SCHEME  URL scheme to register (default: odon)
  --test-url URL       After registration, open this URL through the handler

Environment overrides:
  APP_NAME             Handler app name (default: Odon Dev URL Handler)
  CARGO                Cargo executable (default: cargo)
EOF
}

scheme="odon"
test_url=""

while (($# > 0)); do
  case "$1" in
    --scheme)
      scheme="${2:-}"
      shift
      ;;
    --scheme=*)
      scheme="${1#--scheme=}"
      ;;
    --test-url)
      test_url="${2:-}"
      shift
      ;;
    --test-url=*)
      test_url="${1#--test-url=}"
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

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper is macOS-only." >&2
  exit 1
fi

if [[ -z "$scheme" ]]; then
  echo "--scheme must not be empty" >&2
  exit 1
fi

if ! command -v /usr/bin/swiftc >/dev/null 2>&1; then
  echo "Could not find /usr/bin/swiftc. Install Xcode Command Line Tools and rerun." >&2
  exit 1
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_name="${APP_NAME:-Odon Dev URL Handler}"
cargo_bin="${CARGO:-$(command -v cargo || true)}"
if [[ -z "$cargo_bin" ]]; then
  echo "Could not find cargo. Set CARGO=/absolute/path/to/cargo and rerun." >&2
  exit 1
fi
bundle_id="org.odon.dev-url-handler.${scheme}"
handler_root="$root_dir/target/dev-url-handler"
app_dir="$handler_root/${app_name}.app"
contents_dir="$app_dir/Contents"
macos_dir="$contents_dir/MacOS"
resources_dir="$contents_dir/Resources"
handler_src="$handler_root/OdonDevUrlHandler.swift"
handler_bin="$macos_dir/OdonDevUrlHandler"
log_path="$handler_root/handler.log"
socket_user="$(id -un)"
socket_path="/tmp/odon-${socket_user}.deeplink.sock"

escape_swift_string() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

xml_escape() {
  printf '%s' "$1" |
    sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g; s/'"'"'/\&apos;/g'
}

mkdir -p "$handler_root"
rm -rf "$app_dir"
mkdir -p "$macos_dir" "$resources_dir"

cat > "$handler_src" <<EOF
import AppKit
import Darwin
import Foundation

let odonRepo = "$(escape_swift_string "$root_dir")"
let odonCargo = "$(escape_swift_string "$cargo_bin")"
let odonLog = "$(escape_swift_string "$log_path")"
let odonSocket = "$(escape_swift_string "$socket_path")"

func appendLog(_ message: String) {
    let stamp = ISO8601DateFormatter().string(from: Date())
    let line = "[\\(stamp)] \\(message)\\n"
    let data = Data(line.utf8)
    let url = URL(fileURLWithPath: odonLog)
    try? FileManager.default.createDirectory(
        at: url.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    if !FileManager.default.fileExists(atPath: odonLog) {
        FileManager.default.createFile(atPath: odonLog, contents: nil)
    }
    if let handle = try? FileHandle(forWritingTo: url) {
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: data)
        try? handle.close()
    }
}

func sendToRunningOdon(_ rawUrl: String) -> Bool {
    let fd = socket(AF_UNIX, SOCK_STREAM, 0)
    guard fd >= 0 else {
        appendLog("socket() failed: \\(String(cString: strerror(errno)))")
        return false
    }
    defer { close(fd) }
    var socketTimeout = timeval(tv_sec: 0, tv_usec: 250_000)
    withUnsafePointer(to: &socketTimeout) { timeoutPtr in
        timeoutPtr.withMemoryRebound(to: UInt8.self, capacity: MemoryLayout<timeval>.size) { rawPtr in
            _ = setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, rawPtr, socklen_t(MemoryLayout<timeval>.size))
            _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, rawPtr, socklen_t(MemoryLayout<timeval>.size))
        }
    }

    var addr = sockaddr_un()
    addr.sun_family = sa_family_t(AF_UNIX)

    let maxPathBytes = MemoryLayout.size(ofValue: addr.sun_path)
    guard odonSocket.utf8CString.count <= maxPathBytes else {
        appendLog("Socket path too long: \\(odonSocket)")
        return false
    }

    _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
        ptr.withMemoryRebound(to: CChar.self, capacity: maxPathBytes) { dest in
            odonSocket.withCString { src in
                strncpy(dest, src, maxPathBytes)
            }
        }
    }

    let connected = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            connect(fd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
        }
    }
    guard connected == 0 else {
        appendLog("connect(\\(odonSocket)) failed: \\(String(cString: strerror(errno)))")
        return false
    }

    let bytes = Array((rawUrl + "\\n").utf8)
    var written = 0
    while written < bytes.count {
        let n = bytes.withUnsafeBytes { rawBuffer in
            write(fd, rawBuffer.baseAddress!.advanced(by: written), bytes.count - written)
        }
        guard n > 0 else {
            appendLog("write() failed: \\(String(cString: strerror(errno)))")
            return false
        }
        written += n
    }

    appendLog("Forwarded URL to running Odon over \\(odonSocket).")
    return true
}

func launchOdon(url rawUrl: String?) {
    appendLog("Launching Odon via cargo at \\(odonCargo).")

    let logUrl = URL(fileURLWithPath: odonLog)
    if !FileManager.default.fileExists(atPath: odonLog) {
        FileManager.default.createFile(atPath: odonLog, contents: nil)
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: odonCargo)
    process.currentDirectoryURL = URL(fileURLWithPath: odonRepo)
    if let rawUrl {
        process.arguments = ["run", "--", rawUrl]
    } else {
        process.arguments = ["run"]
    }

    if let output = try? FileHandle(forWritingTo: logUrl) {
        _ = try? output.seekToEnd()
        process.standardOutput = output
        process.standardError = output
    }

    do {
        try process.run()
    } catch {
        appendLog("Failed to launch cargo: \\(error)")
    }
}

func handleDeepLink(_ rawUrl: String) {
    let started = Date()
    appendLog("Received URL: \\(rawUrl)")
    if !sendToRunningOdon(rawUrl) {
        appendLog("No running Odon IPC listener; falling back to cargo.")
        launchOdon(url: rawUrl)
    }
    appendLog(String(format: "URL handling returned after %.3fs.", Date().timeIntervalSince(started)))
}

if CommandLine.arguments.count >= 3 && CommandLine.arguments[1] == "--send" {
    let rawUrl = CommandLine.arguments[2]
    appendLog("Direct --send URL: \\(rawUrl)")
    exit(sendToRunningOdon(rawUrl) ? 0 : 1)
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var handledUrl = false
    private var launchTimer: Timer?

    override init() {
        super.init()
        NSAppleEventManager.shared().setEventHandler(
            self,
            andSelector: #selector(handleGetUrlEvent(_:withReplyEvent:)),
            forEventClass: AEEventClass(kInternetEventClass),
            andEventID: AEEventID(kAEGetURL)
        )
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        appendLog("Handler launched. pid=\\(getpid()) args=\\(CommandLine.arguments)")
        launchTimer = Timer.scheduledTimer(withTimeInterval: 0.75, repeats: false) { [weak self] _ in
            guard let self, !self.handledUrl else { return }
            appendLog("No URL event received after launch; starting Odon without URL.")
            launchOdon(url: nil)
            NSApp.terminate(nil)
        }
    }

    func application(_ application: NSApplication, open urls: [URL]) {
        for url in urls {
            handle(url.absoluteString)
        }
    }

    @objc func handleGetUrlEvent(_ event: NSAppleEventDescriptor, withReplyEvent reply: NSAppleEventDescriptor) {
        guard let rawUrl = event.paramDescriptor(forKeyword: keyDirectObject)?.stringValue else {
            appendLog("Received kAEGetURL event without a URL payload.")
            return
        }
        handle(rawUrl)
    }

    private func handle(_ rawUrl: String) {
        handledUrl = true
        launchTimer?.invalidate()
        handleDeepLink(rawUrl)
        DispatchQueue.main.async {
            NSApp.terminate(nil)
        }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
EOF

/usr/bin/swiftc "$handler_src" -o "$handler_bin" -framework AppKit
chmod +x "$handler_bin"

cat > "$contents_dir/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>OdonDevUrlHandler</string>
  <key>CFBundleIdentifier</key>
  <string>$(xml_escape "$bundle_id")</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>$(xml_escape "$app_name")</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>CFBundleURLTypes</key>
  <array>
    <dict>
      <key>CFBundleURLName</key>
      <string>$(xml_escape "${bundle_id}.url")</string>
      <key>CFBundleURLSchemes</key>
      <array>
        <string>$(xml_escape "$scheme")</string>
      </array>
    </dict>
  </array>
  <key>LSMinimumSystemVersion</key>
  <string>10.13</string>
  <key>LSUIElement</key>
  <true/>
</dict>
</plist>
EOF

if command -v codesign >/dev/null 2>&1; then
  codesign --force --deep --sign - "$app_dir" >/dev/null 2>&1 || true
fi

lsregister="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
"$lsregister" -f "$app_dir"

if command -v /usr/bin/swift >/dev/null 2>&1; then
  swift_src="$(mktemp "${TMPDIR:-/tmp}/odon-url-handler.XXXXXX.swift")"
  cat > "$swift_src" <<'EOF'
import CoreServices
import Foundation

let scheme = CommandLine.arguments[1] as CFString
let bundleID = CommandLine.arguments[2] as CFString
let status = LSSetDefaultHandlerForURLScheme(scheme, bundleID)
if status != noErr {
  FileHandle.standardError.write(Data("LSSetDefaultHandlerForURLScheme failed: \(status)\n".utf8))
  exit(1)
}
let current = LSCopyDefaultHandlerForURLScheme(scheme)?.takeRetainedValue() as String?
print(current ?? "")
EOF
  current_handler="$(/usr/bin/swift "$swift_src" "$scheme" "$bundle_id" 2>/dev/null || true)"
  rm -f "$swift_src"
else
  current_handler=""
fi

echo "Registered ${scheme}:// handler:"
echo "  $app_dir"
if [[ -n "$current_handler" ]]; then
  echo "Default ${scheme}:// handler bundle id:"
  echo "  $current_handler"
else
  echo "Default handler could not be confirmed automatically."
fi
echo
echo "Development workflow:"
echo "  1. Keep this checkout as your active Odon source tree."
echo "  2. Click ${scheme}:// links, or test one explicitly with:"
echo "     open '${scheme}://open?...'"
echo "  3. If Odon is already running, the URL is handed to that process over:"
echo "     $socket_path"
echo "  4. Handler output is appended to:"
echo "     $log_path"
echo "  5. Direct IPC testing can use:"
echo "     \"$handler_bin\" --send '${scheme}://open?...'"

if [[ -n "$test_url" ]]; then
  open "$test_url"
fi
