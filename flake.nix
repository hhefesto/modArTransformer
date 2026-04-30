{
  description = "Minimal transformer for modular arithmetic — Haskell + hmatrix, ported to Agda with Conal-style AD";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    haskell-flake.url = "github:srid/haskell-flake";
    # PureScript tooling
    purescript-overlay = {
      url = "github:thomashoneyman/purescript-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Conal Elliott's Agda category library
    felix = {
      url = "github:conal/felix";
      flake = false;
    };
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-compat, flake-parts, haskell-flake, ... }:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      imports = [
        inputs.haskell-flake.flakeModule
      ];

      perSystem = { self', config, system, ... }:
      let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [ inputs.purescript-overlay.overlays.default ];
        };

        # ── Agda + stdlib + felix ────────────────────────────────────────────
        agdaWithStdlib = pkgs.agda.withPackages (p: [ p.standard-library ]);

        # Pre-compile felix interfaces into a writable output path so agda
        # never tries to write .agdai files into the read-only nix store.
        felixCompiled = pkgs.stdenv.mkDerivation {
          name = "felix-compiled";
          src = inputs.felix;
          nativeBuildInputs = [ agdaWithStdlib ];
          buildPhase = ''
            mkdir -p $out/src
            cp -r src/. $out/src/
            cd $out
            agda -i src -i ${pkgs.agdaPackages.standard-library}/src \
                 src/Felix/Homomorphism.agda
          '';
          installPhase = ":";
        };

        # Wrap agda so every invocation gets -i <felix>/src automatically.
        myAgda = pkgs.symlinkJoin {
          name = "agda-with-felix";
          paths = [ agdaWithStdlib ];
          buildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/agda \
              --add-flags "-i ${felixCompiled}/src"
          '';
        };
      in {
        _module.args.pkgs = pkgs;
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
              nativeBuildInputs = [
                myAgda
                pkgs.glibcLocales
              ];
              shellHook = ''
                export LOCALE_ARCHIVE="${pkgs.glibcLocales}/lib/locale/locale-archive"
                export LC_ALL="en_US.UTF-8"
                echo "Agda ready with memory guard. Type-check with:"
                echo "  ./scripts/agda-guard.sh agda modArTransformer.agda"
                echo "Compile with:"
                echo "  ./scripts/agda-guard.sh agda --compile modArTransformer.agda"
              '';
            };
          };
        };

        packages.default = self'.packages.modArTransformer;

        # ── Agda packages ───────────────────────────────────────────────────

        # Compile the full Agda port to a native executable via MAlonzo.
        packages.agda-modArTransformer =
          let stdlib = pkgs.agdaPackages.standard-library; in
          pkgs.stdenv.mkDerivation {
            name = "agda-modArTransformer";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.agda pkgs.ghc pkgs.glibcLocales ];
            LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
            LC_ALL = "en_US.UTF-8";
            buildPhase = ''
              cp -r ${inputs.felix}/src felix-src
              chmod -R u+w felix-src
              bash ./scripts/agda-guard.sh agda -i ${stdlib}/src -i felix-src --compile modArTransformer.agda
            '';
            installPhase = ''
              mkdir -p $out/bin
              cp modArTransformer $out/bin/agda-modArTransformer
            '';
          };

        # Type-check only (fast CI gate, no GHC compilation step).
        packages.agda-modArTransformer-check =
          let stdlib = pkgs.agdaPackages.standard-library; in
          pkgs.stdenv.mkDerivation {
            name = "agda-modArTransformer-check";
            src = pkgs.lib.cleanSource ./.;
            nativeBuildInputs = [ pkgs.agda pkgs.glibcLocales ];
            LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
            LC_ALL = "en_US.UTF-8";
            buildPhase = ''
              cp -r ${inputs.felix}/src felix-src
              chmod -R u+w felix-src
              bash ./scripts/agda-guard.sh agda -i ${stdlib}/src -i felix-src modArTransformer.agda
            '';
            installPhase = ''
              mkdir -p $out
              echo "modArTransformer.agda type-checked" > $out/result
            '';
          };

        apps.default = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-agda-modArTransformer-guarded" ''
            exec ${pkgs.bash}/bin/bash ${./scripts/agda-guard.sh} ${self'.packages.agda-modArTransformer}/bin/agda-modArTransformer "$@"
          '');
        };

        apps.haskell = {
          type = "app";
          program = "${self'.packages.modArTransformer}/bin/modArTransformer";
        };

        apps.agda-modArTransformer = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-agda-modArTransformer-guarded-app" ''
            exec ${pkgs.bash}/bin/bash ${./scripts/agda-guard.sh} ${self'.packages.agda-modArTransformer}/bin/agda-modArTransformer "$@"
          '');
        };

        packages.diagram = pkgs.runCommand "transformer-diagram" {} ''
          mkdir -p $out
          cp ${./purescript/index.html} $out/index.html
          cp ${./purescript/index.js} $out/index.js
        '';

        apps.diagram = {
          type = "app";
          program = toString (pkgs.writeShellScript "run-diagram-purs" ''
            echo "Starting server at http://localhost:8080"
            ${pkgs.darkhttpd}/bin/darkhttpd ${self'.packages.diagram} --port 8080
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
            # PureScript (from purescript-overlay)
            spago-unstable
            purs-unstable
            darkhttpd
            # Agda
            myAgda
            glibcLocales
          ];
          shellHook = ''
            export LOCALE_ARCHIVE="${pkgs.glibcLocales}/lib/locale/locale-archive"
            export LC_ALL="en_US.UTF-8"
          '';
        });

        checks = self'.packages;
      };
    };
}
