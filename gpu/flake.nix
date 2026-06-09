{
  description = "modArTransformer GPU tier — categorical reverse-mode AD over libtorch kernels";

  nixConfig = {
    extra-substituters = [ "https://hasktorch.cachix.org" ];
    extra-trusted-public-keys = [
      "hasktorch.cachix.org-1:wLjNS6HuFVpmzbmv01lxwjdCOtWRD8pQVR3Zr/wVoQc="
    ];
  };

  inputs = {
    hasktorch.url = "github:hasktorch/hasktorch/043552377932093d36e10d46e8b45aaa44225f2d";
  };

  outputs = { self, hasktorch }:
    let
      system = "x86_64-linux";
      ghc = "ghc984";
      # replicate hasktorch's own CPU nixpkgs instance exactly (cache hits);
      # the CUDA flavor swaps config.cudaSupport = true on the GPU machine.
      pkgs = import hasktorch.inputs.nixpkgs {
        inherit system;
        config.cudaSupport = false;
        config.allowBroken = true;
        overlays = [ hasktorch.overlays.default ];
      };
      ghcEnv = pkgs.haskell.packages.${ghc}.ghcWithPackages (p: [ p.hasktorch ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ ghcEnv ];
      };

      packages.${system}.conformance-g = pkgs.stdenv.mkDerivation {
        name = "conformance-g";
        src = ./.;
        nativeBuildInputs = [ ghcEnv ];
        buildPhase = ''
          ghc -O1 ConformanceG.hs -o conformance-g
        '';
        installPhase = ''
          mkdir -p $out/bin
          cp conformance-g $out/bin/
        '';
      };
    };
}
