#!/usr/bin/env bash
# Compile the standalone gfx1151 microbenchmarks with hipcc.
#
# hipcc invokes the raw HIP clang++, not the Nix cc wrapper, so the include and
# library search paths that `nix develop` exports through NIX_CFLAGS_COMPILE /
# NIX_LDFLAGS have to be forwarded explicitly.
#
#   nix develop -c tools/bench/build.sh [name ...]
#
# With no arguments every *.hip under tools/bench is built into /tmp.
set -euo pipefail

cd "$(dirname "$0")/../.."

# shellcheck disable=SC2206
read -r -a cflags <<<"${NIX_CFLAGS_COMPILE:-}"
read -r -a ldflags <<<"${NIX_LDFLAGS:-}"

inc=()
for f in "${cflags[@]}"; do
  case "$f" in
    -isystem | -I) ;;
    /*) inc+=("-isystem" "$f") ;;
    -I*) inc+=("$f") ;;
  esac
done
lib=()
for f in "${ldflags[@]}"; do
  case "$f" in
    -L | -rpath) ;;
    /*) lib+=("-L$f" "-Wl,-rpath,$f") ;;
    -L*) lib+=("$f") ;;
  esac
done

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  for f in tools/bench/*.hip; do
    targets+=("$(basename "$f" .hip)")
  done
fi

for t in "${targets[@]}"; do
  if [[ "$t" == *.hip ]]; then
    src="$t"
    t="$(basename "$t" .hip)"
  else
    src="tools/bench/${t}.hip"
  fi
  out="/tmp/${t}"
  extra=()
  if grep -q "hipblas" "$src"; then
    extra+=(-lhipblas -lhipblaslt -lrocblas)
  fi
  if grep -q "aotriton" "$src"; then
    extra+=(-laotriton_v2)
  fi
  echo "==> $src -> $out"
  hipcc -O3 --offload-arch=gfx1151 -std=c++20 \
    -I. "${inc[@]}" "${lib[@]}" \
    -Rpass-analysis=kernel-resource-usage \
    "$src" "${extra[@]}" -o "$out" 2>"/tmp/${t}.build.log" ||
    { tail -40 "/tmp/${t}.build.log"; exit 1; }
  grep -E "Function Name|VGPRs:|Occupancy|VGPRs Spill|LDS Size" \
    "/tmp/${t}.build.log" | sed 's/.*remark: //; s/ \[-Rpass.*//' \
    >"/tmp/${t}.res.txt" || true
done
echo "ok"
