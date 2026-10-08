#!/usr/bin/env bash
# build.sh <target> [output]
set -euo pipefail

target="${1:?target is required}"
output="${2:-dist}"
ffmpeg_version=8.1.3
ffmpeg_source=7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3
ffmpeg_release=autobuild-2026-10-06-13-06
ffmpeg_build=ffmpeg-n$ffmpeg_version-14-g330caae0c1
x264_commit=b35605ace3ddf7c1a5d67a2eb553f034aef41d55
moltenvk_version=v1.4.2
moltenvk_package=f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e

case "$target" in
  x86_64-unknown-linux-gnu)
    prefix=lib suffix=.so
    archive="$ffmpeg_build-linux64-gpl-shared-8.1.tar.xz"
    hash=a6d0ea7dfef6ef85d86b8acf1c0a2d5288a05bac42c2cb16914e26830da3d344
    ;;
  x86_64-pc-windows-msvc)
    prefix='' suffix=.dll
    archive="$ffmpeg_build-win64-gpl-shared-8.1.zip"
    hash=751c56e0b63426426487ab4048031b0166281c59c0a7e33ef7dd7428495e1d8a
    ;;
  aarch64-apple-darwin | x86_64-apple-darwin)
    prefix=lib suffix=.dylib
    export MACOSX_DEPLOYMENT_TARGET=12.0
    ;;
  *)
    echo 'unsupported target' >&2
    exit 1
    ;;
esac

root="$(pwd -P)"
command -v cygpath >/dev/null 2>&1 && root="$(cygpath -m "$root")"
work="$root/target"
output="$root/$output"
mkdir -p "$work" "$output/bin/rife" "$output/licenses"
sha() { sha256sum "$@" 2>/dev/null || shasum -a 256 "$@"; }
# gets an archive, checks it against its pinned hash and unpacks it into the work folder
unpack() {
  curl -fsSL "$1" -o "$work/download"
  if [[ "$(sha "$work/download" | cut -d' ' -f1)" != "$2" ]]; then
    echo "$1 does not have the pinned hash" >&2
    exit 1
  fi
  case "$1" in
    *.zip) unzip -q "$work/download" -d "$work" ;;
    *) tar -xf "$work/download" -C "$work" ;;
  esac
  rm "$work/download"
}
SOURCE_DATE_EPOCH="$(git -c safe.directory="$root" show -s --format=%ct HEAD)"
export SOURCE_DATE_EPOCH
export TZ=UTC
export LC_ALL=C
jobs="$(getconf _NPROCESSORS_ONLN)"

cmake -S rife -B "$work/rife" -G Ninja
cmake --build "$work/rife"
cp -r "$work/rife/models/rife-v4.6" "$output/bin/rife/"
cp "$work/rife/${prefix}interpolini_rife$suffix" "$output/bin/"
cp "$work/rife/source/LICENSE" "$output/licenses/rife-ncnn-vulkan.txt"
cp "$work/rife/_deps/ncnn-src/LICENSE.txt" "$output/licenses/ncnn.txt"
if [[ "$target" != *-apple-darwin ]]; then
  # macos has no tensorrt
  cp "$work/rife/${prefix}interpolini_rife_trt$suffix" "$output/bin/"
  cp "$work/rife/_deps/tensorrt-src/LICENSE" "$output/licenses/tensorrt-headers.txt"

  # btbn deletes its daily builds, so the one that is pinned stays here
  unpack "https://github.com/BtbN/FFmpeg-Builds/releases/download/$ffmpeg_release/$archive" "$hash"
  folder="${archive%.zip}"
  mv "$work/${folder%.tar.xz}" "$output/ffmpeg"
else
  # apple killed vulkan so we bring our own
  unpack "https://github.com/KhronosGroup/MoltenVK/releases/download/$moltenvk_version/MoltenVK-macos.tar" "$moltenvk_package"
  cp "$work/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" "$output/bin/"
  cp "$work/MoltenVK/LICENSE" "$output/licenses/moltenvk.txt"

  # nobody builds ffmpeg for macos so we do it ourselves, lovely
  # it has the encoders of the system, and x264 inside it as the software encoder
  git init -q "$work/x264"
  git -C "$work/x264" fetch -q --depth 1 https://github.com/mirror/x264.git "$x264_commit"
  git -C "$work/x264" checkout -q FETCH_HEAD
  (
    cd "$work/x264"
    ./configure --prefix="$work/x264-built" --enable-static --enable-pic --disable-cli
    make -j"$jobs" install
  )
  unpack "https://ffmpeg.org/releases/ffmpeg-$ffmpeg_version.tar.xz" "$ffmpeg_source"
  (
    cd "$work/ffmpeg-$ffmpeg_version"
    PKG_CONFIG_PATH="$work/x264-built/lib/pkgconfig" ./configure --prefix="$output/ffmpeg" --install-name-dir=@rpath \
      --enable-shared --disable-static --disable-programs --disable-doc --disable-avdevice --disable-avfilter \
      --disable-autodetect --enable-videotoolbox --enable-audiotoolbox --enable-zlib --enable-gpl --enable-libx264
    make -j"$jobs" install
    cp COPYING.GPLv2 "$output/ffmpeg/LICENSE.txt"
  )
fi
{
  echo "Commit: $(git -c safe.directory="$root" rev-parse HEAD)"
  echo "Target: $target"
  echo "FFmpeg: $ffmpeg_version"
} > "$output/BUILD.txt"
