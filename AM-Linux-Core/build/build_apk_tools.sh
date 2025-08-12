#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APK_TOOLS_DIR="${APK_TOOLS_DIR:-$SCRIPT_DIR/../apk-tools}"
OUTPUT_DIR="${OUTPUT_DIR:-$APK_TOOLS_DIR/out}"

if [ ! -d "$APK_TOOLS_DIR" ]; then
  echo "APK tools directory not found: $APK_TOOLS_DIR" >&2
  exit 1
fi

make -C "$APK_TOOLS_DIR" clean
make -C "$APK_TOOLS_DIR" static
make -C "$APK_TOOLS_DIR" DESTDIR="$OUTPUT_DIR" SBINDIR=/usr/bin install

BIN="$OUTPUT_DIR/usr/bin/apk"
strip --strip-unneeded "$BIN" 2>/dev/null || true
rm -rf "${OUTPUT_DIR:?}/usr/share"

# Print path to built apk binary
echo "$BIN"
