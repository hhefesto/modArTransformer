{
  description = "Minimal transformer for modular arithmetic — Haskell + hmatrix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    haskell-flake.url = "github:srid/haskell-flake";
  };

  outputs = inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      imports = [
        inputs.haskell-flake.flakeModule
      ];

      perSystem = { self', pkgs, ... }: {
        haskellProjects.default = {
          basePackages = pkgs.haskellPackages;

          settings = {
            modArTransformer = {
              custom = pkg: pkg.overrideAttrs (old: {
                buildInputs = (old.buildInputs or []) ++ [
                  pkgs.blas
                  pkgs.lapack
                ];
              });
            };
          };

          devShell = {
            tools = hp: {
              inherit (hp)
                cabal-install
                ghcid
                haskell-language-server;
            };

            mkShellArgs = {
              packages = [
                pkgs.blas
                pkgs.lapack
                pkgs.pkg-config
              ];
            };
          };
        };

        packages.default = self'.packages.modArTransformer;
      };
    };
}
