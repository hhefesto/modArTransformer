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

      perSystem = { self', pkgs, config, ... }: {
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
          };
        };

        packages.default = self'.packages.modArTransformer;

        apps.default = {
          type = "app";
          program = "${self'.packages.modArTransformer}/bin/modArTransformer";
        };

        apps.diagram = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-diagram" ''
            exec ${pkgs.nodejs}/bin/npx vite "$@"
          '');
        };

        devShells.default = pkgs.lib.mkForce (pkgs.mkShell {
          name = "modArTransformer-dev";
          inputsFrom = [
            config.haskellProjects.default.outputs.devShell
          ];
          nativeBuildInputs = with pkgs; [
            blas
            lapack
            pkg-config
            esbuild
            python3
            # nodejs_20
            nodejs
          ];
        });
      };
    };
}
