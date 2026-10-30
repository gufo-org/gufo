# Build with: nix build --impure --file .devops/nix/image.nix \
#   --argstr source "$PWD" --argstr revision "$(git rev-parse HEAD)"
{ source, revision }:
let
  flake = builtins.getFlake source;
  system = "x86_64-linux";
  pkgs = import flake.inputs.nixpkgs { inherit system; };
  engine = flake.packages.${system}.default.override {
    revision = builtins.substring 0 7 revision;
  };
in
pkgs.dockerTools.buildLayeredImage {
  name = "localhost/gufo-image-turbo";
  tag = builtins.substring 0 12 revision;
  contents = [ engine pkgs.bashInteractive pkgs.coreutils pkgs.cacert ];
  extraCommands = ''
    mkdir -p tmp home/gufo etc
    chmod 1777 tmp
    printf 'gufo:x:1000:1000:Gufo:/home/gufo:/bin/bash\n' > etc/passwd
    printf 'gufo:x:1000:\n' > etc/group
  '';
  config = {
    User = "1000:1000";
    WorkingDir = "/home/gufo";
    Env = [ "PATH=/bin" "HOME=/home/gufo" ];
    Cmd = [ "/bin/gufo" ];
    Labels = {
      "org.opencontainers.image.source" = "https://github.com/lfthht/gufo";
      "org.opencontainers.image.revision" = revision;
      "org.opencontainers.image.description" = "Native BF16 Qwen Image 2.1 Turbo generation and editing on gfx1151";
    };
  };
}
