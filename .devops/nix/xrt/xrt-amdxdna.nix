{
  lib,
  symlinkJoin,
  xrt,
  xrt-plugin-amdxdna,
}:

# Combined XRT + AMDXDNA plugin package
# This creates a unified package with XRT and the XDNA driver plugin
# that can be used directly without needing to set up plugin discovery paths.
symlinkJoin {
  name = "xrt-amdxdna-${xrt.version}";

  paths = [
    xrt
    xrt-plugin-amdxdna
  ];

  # Create symlinks so XRT can discover the XDNA plugin
  postBuild = ''
    cd $out/opt/xilinx/xrt/lib
    pluginLib="${xrt-plugin-amdxdna}/opt/xilinx/xrt/lib"
    ln -sf "$pluginLib/libxrt_driver_xdna.so.2" .
    ln -sf "$pluginLib/libxrt_driver_xdna.so.${xrt-plugin-amdxdna.pluginVersion}" .
    # The tool wrappers in bin/ (xclbinutil, xrt-smi) run unwrapped/loader
    # beside themselves.
    if [ ! -e "$out/bin/unwrapped" ]; then
      ln -s "$out/opt/xilinx/xrt/bin/unwrapped" "$out/bin/unwrapped"
    fi
  '';

  meta = {
    description = "Xilinx Runtime with AMD XDNA plugin for Ryzen AI NPU";
    homepage = "https://github.com/Xilinx/XRT";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
