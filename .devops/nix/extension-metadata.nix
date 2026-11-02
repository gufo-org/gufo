# Match CMake discovery without a generated, hand-maintained provider inventory.
{
  lib,
  sourceRoot ? ../..,
  extensions ? [ ],
  extensionsFromLock ? false,
}:
let
  children =
    dir:
    if builtins.pathExists dir then
      map (name: dir + "/${name}") (
        builtins.attrNames (lib.filterAttrs (_: type: type == "directory") (builtins.readDir dir))
      )
    else
      [ ];
  manifestsIn =
    dir:
    lib.filter (path: builtins.pathExists path) (map (root: root + "/extension.json") (children dir));
  paths = lib.unique (
    manifestsIn (sourceRoot + "/src/models")
    ++ lib.concatMap manifestsIn (children (sourceRoot + "/src/families"))
    ++ lib.concatMap (
      root:
      if builtins.pathExists (root + "/extension.json") then
        [ (root + "/extension.json") ]
      else
        manifestsIn root
    ) extensions
  );
  manifests = map (
    path:
    (builtins.fromJSON (builtins.readFile path))
    // {
      _buildable = builtins.pathExists (builtins.dirOf path + "/extension.cmake");
    }
  ) paths;
  names = map (m: m.name) manifests;
  locked = (builtins.fromJSON (builtins.readFile (sourceRoot + "/extensions.lock"))).extensions;
  selected = lib.filter (
    m:
    (!extensionsFromLock || builtins.elem m.name locked)
    && m._buildable
    && builtins.elem "linux" m.platforms
  ) manifests;
  providers = lib.concatMap (m: m.providers or [ ]) selected;
  providerIds = map (p: p.id) providers;
in
assert lib.assertMsg (
  builtins.length names == builtins.length (lib.unique names)
) "gufo: duplicate extension names in serving metadata";
assert lib.assertMsg (
  !extensionsFromLock || lib.all (name: builtins.elem name names) locked
) "gufo: locked extension missing from serving metadata";
assert lib.assertMsg (
  builtins.length providerIds == builtins.length (lib.unique providerIds)
) "gufo: duplicate provider IDs in serving metadata";
{
  inherit providers;
  options = lib.concatMap (
    m:
    map (
      o:
      o
      // {
        providers = o.providers or (map (p: p.id) (m.providers or [ ]));
      }
    ) (m.options or [ ])
  ) selected;
  extensions = map (m: m.name) selected;
  selectionMode = if extensionsFromLock then "lock" else "discovery";
}
