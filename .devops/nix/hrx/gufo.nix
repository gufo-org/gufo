# Experimental: Gufo running on the HRX HIP runtime. The Gufo binaries are
# unchanged; the wrappers put HRX's drop-in libamdhip64.so.7 first on the
# library path.
{
  lib,
  runCommand,
  makeWrapper,
  gufo,
  hrx,
}:

runCommand "gufo-hrx-${gufo.version}"
  {
    nativeBuildInputs = [ makeWrapper ];
    passthru = { inherit gufo hrx; };
    meta = gufo.meta // {
      description = "Gufo on the HRX HIP runtime (experimental)";
    };
  }
  ''
    mkdir -p $out/bin
    for program in gufo gufo-server; do
      makeWrapper ${gufo}/bin/$program $out/bin/$program \
        --prefix LD_LIBRARY_PATH : ${hrx}/lib
    done
    ln -s ${hrx}/bin/hrx-info ${hrx}/bin/iree-profile $out/bin/
  ''
