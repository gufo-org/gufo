{
  lib,
  stdenv,
  fetchFromGitHub,
  rocmUpdateScript,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "hip-common";
  version = "10.0.0";

  src = fetchFromGitHub {
    owner = "ROCm";
    repo = "rocm-systems";
    rev = "therock-10.0";
    sparseCheckout = [
      "projects/hip"
      "shared"
    ];
    hash = "sha256-v30H6y9zgB5B2AzlxJsKu0mbJMxpi3q45bc3+pbbB8M=";
  };
  sourceRoot = "${finalAttrs.src.name}/projects/hip";

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out
    mv * $out

    runHook postInstall
  '';

  passthru.updateScript = rocmUpdateScript { inherit finalAttrs; };

  meta = {
    description = "C++ Heterogeneous-Compute Interface for Portability";
    homepage = "https://github.com/ROCm/rocm-systems/tree/develop/projects/hip";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [ lovesegfault ];
    teams = [ lib.teams.rocm ];
    platforms = lib.platforms.linux;
  };
})
