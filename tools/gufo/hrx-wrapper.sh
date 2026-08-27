#!/bin/sh

native_backend=0
expect_backend=0
for argument in "$@"; do
  if [ "$expect_backend" -eq 1 ]; then
    if [ "$argument" = "hrx-native" ]; then
      native_backend=1
    fi
    expect_backend=0
    continue
  fi
  case "$argument" in
    --qwen-backend)
      expect_backend=1
      ;;
    --qwen-backend=hrx-native)
      native_backend=1
      ;;
  esac
done

binary_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ "$native_backend" -eq 1 ]; then
  filtered_preload=
  saved_ifs=$IFS
  IFS=': '
  for library in ${LD_PRELOAD-}; do
    if [ "$library" = "@HRX_HIP_LIBRARY@" ]; then
      continue
    fi
    if [ -z "$filtered_preload" ]; then
      filtered_preload=$library
    else
      filtered_preload="$filtered_preload:$library"
    fi
  done
  IFS=$saved_ifs
  if [ -n "$filtered_preload" ]; then
    export LD_PRELOAD=$filtered_preload
  else
    unset LD_PRELOAD
  fi
  exec "$binary_directory/.gufo-hrx-real" "$@"
fi

export LD_PRELOAD="@HRX_HIP_LIBRARY@${LD_PRELOAD:+:$LD_PRELOAD}"
exec "$binary_directory/.gufo-hrx-real" "$@"
