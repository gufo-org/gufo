# Extensions built as loadable modules for one gufo package (gufo-org/gufo#1075),
# with the same `gufo extension install` that users run, against the package's
# own SDK output.
{
  lib,
  runCommand,
  makeWrapper,
}:

rec {
  # `gufo` with its SDK: `gufo extension install` works out of the box.
  withSdk =
    gufo:
    runCommand "gufo-sdk-${gufo.version}"
      {
        nativeBuildInputs = [ makeWrapper ];
        meta.mainProgram = "gufo";
      }
      ''
        mkdir -p $out/bin
        makeWrapper ${gufo}/bin/gufo $out/bin/gufo \
          --set-default GUFO_SDK_DIR ${gufo.sdk}/lib/gufo/sdk
        ln -s gufo $out/bin/gufo-server
      '';

  # A read-only extension store holding `extensions`, pinned sources such as
  # non-flake inputs. Unvalidated code: review each source before adding it.
  store =
    {
      gufo,
      extensions,
    }:
    runCommand "gufo-extension-modules-${gufo.version}" { } ''
      export HOME=$TMPDIR
      export GUFO_SDK_DIR=${gufo.sdk}/lib/gufo/sdk
      export GUFO_EXTENSIONS_DIR=$out
      ${lib.concatMapStringsSep "\n" (
        extension: "${gufo}/bin/gufo extension install ${extension} --yes"
      ) extensions}
      rm -f $out/.lock
      ${gufo}/bin/gufo extension list
    '';

  # `gufo` that loads `extensions` as modules in every command.
  withExtensions =
    {
      gufo,
      extensions,
    }:
    let
      modules = store { inherit gufo extensions; };
    in
    runCommand "gufo-with-extensions-${gufo.version}"
      {
        nativeBuildInputs = [ makeWrapper ];
        passthru = { inherit modules; };
        meta.mainProgram = "gufo";
      }
      ''
        mkdir -p $out/bin
        makeWrapper ${gufo}/bin/gufo $out/bin/gufo \
          --set-default GUFO_EXTENSIONS_DIR ${modules}
        ln -s gufo $out/bin/gufo-server
      '';
}
