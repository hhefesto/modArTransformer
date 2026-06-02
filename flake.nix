{
  description = "Denotational transformer for modular arithmetic — Tai-Danae Bradley's [0,1]-enriched-category semantics as meaning, Conal Elliott's AD-as-categories (on felix) as gradient descent. Agda.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    haskell-flake.url = "github:srid/haskell-flake";
    concat.url = "github:compiling-to-categories/concat";
    # Conal Elliott's Agda category library
    felix = {
      url = "github:conal/felix";
      flake = false;
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-parts, haskell-flake, concat, ... }:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      imports = [ inputs.haskell-flake.flakeModule ];

      perSystem = { self', pkgs, system, ... }:
      let
        # ── Agda + stdlib + felix ────────────────────────────────────────────
        agdaWithStdlib = pkgs.agda.withPackages (p: [ p.standard-library ]);

        # Pre-compile felix interfaces (.agdai next to source) so agda never
        # re-type-checks felix nor writes into the read-only nix store.
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

        # Proxy importing the stdlib closure the project uses, so its interfaces
        # get precompiled once.  (stdlib ships .agdai only under its _build/
        # "library" layout, invisible to `-i <stdlib>/src` usage, which would
        # otherwise re-type-check all of stdlib on every build.)
        stdlibProxy = pkgs.writeText "ModArStdlibUses.agda" ''
          {-# OPTIONS --guardedness #-}
          module ModArStdlibUses where
          import Agda.Builtin.Float
          import Agda.Builtin.String
          import Data.Bool
          import Data.Char
          import Data.Empty
          import Data.Fin
          import Data.Fin.Properties
          import Data.List
          import Data.Maybe
          import Data.Nat
          import Data.Nat.DivMod
          import Data.Nat.Properties
          import Data.Nat.Show
          import Data.Product
          import Data.String
          import Data.Sum
          import Data.Unit
          import Data.Unit.Base
          import Data.Unit.Polymorphic.Base
          import Data.Vec.Base
          import Function
          import IO
          import IO.Base
          import IO.Primitive.Core
          import Level
          import Relation.Binary.PropositionalEquality
        '';

        # Precompile the stdlib closure to .agdai *next to source* (no .agda-lib
        # in $out/src ⇒ next-to-source interface layout, matching how the project
        # consumes it via `-i <stdlibCompiled>/src`).
        stdlibCompiled = pkgs.stdenv.mkDerivation {
          name = "agda-stdlib-compiled";
          src = pkgs.agdaPackages.standard-library;
          nativeBuildInputs = [ pkgs.agda pkgs.glibcLocales ];
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LC_ALL = "en_US.UTF-8";
          buildPhase = ''
            mkdir -p $out/src
            cp -r src/. $out/src/
            cp ${stdlibProxy} $out/src/ModArStdlibUses.agda
            cd $out
            agda -i src src/ModArStdlibUses.agda
          '';
          installPhase = ":";
        };

        # Wrap agda so every invocation reuses the precompiled stdlib + felix
        # interfaces (fast local type-checks; no withPackages, to avoid a second
        # stdlib on the search path).
        myAgda = pkgs.symlinkJoin {
          name = "agda-with-deps";
          paths = [ pkgs.agda ];
          buildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/agda \
              --add-flags "-i ${stdlibCompiled}/src -i ${felixCompiled}/src"
          '';
        };

        projectSource = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            let
              name = baseNameOf path;
              rel = pkgs.lib.removePrefix ((toString ./.) + "/") (toString path);
            in
              pkgs.lib.cleanSourceFilter path type
              && !(rel == "MAlonzo"
                || pkgs.lib.hasPrefix "MAlonzo/" rel
                || pkgs.lib.hasSuffix ".agdai" name
                || name == "ERROR_LOG"
                || name == ".#ERROR_LOG"
                || name == "checkpoint.ckpt"
                || name == "checkpoint.ckpt.bak"
                || (rel == "modArTransformer" && type != "directory"));
        };

        ctcPkgs = import concat.inputs.nixpkgs {
          inherit system;
          overlays = [
            concat.overlays.default
            (final: prev: {
              haskell = prev.haskell // {
                packages = prev.haskell.packages // {
                  ghc948 = prev.haskell.packages.ghc948.extend (hfinal: hprev: {
                    "concat-plugin" = final.haskell.lib.dontCheck hprev."concat-plugin";
                  });
                };
              };
            })
          ];
        };
      in {
        haskellProjects.default = {
          basePackages = pkgs.haskellPackages;
          # Don't let haskell-flake own devShells.default — it collides with the
          # explicit Agda dev shell below ("defined multiple times").  We fold the
          # Haskell toolchain into that single shell instead.
          autoWire = [ "packages" "checks" "apps" ];
        };

        # ── Agda packages ───────────────────────────────────────────────────

        # Default build: compile the Agda project to a native executable (MAlonzo).
        packages.default = self'.packages.agda-modArTransformer;

        packages.ctc-smoke = ctcPkgs.haskell.packages.ghc948.callCabal2nix "ctc-smoke" ./ctc-smoke { };
        packages.ctc-grad-smoke = ctcPkgs.haskell.packages.ghc948.callCabal2nix "ctc-grad-smoke" ./ctc-grad-smoke { };
        packages.ctc-train = ctcPkgs.haskell.packages.ghc948.callCabal2nix "ctc-train" ./ctc-train { };

        packages.agda-modArTransformer = pkgs.stdenv.mkDerivation {
          name = "agda-modArTransformer";
          src = projectSource;
          nativeBuildInputs = [ pkgs.agda pkgs.ghc pkgs.glibcLocales ];
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LC_ALL = "en_US.UTF-8";
          # Reuse precompiled stdlib + felix interfaces (no re-type-checking of
          # the libraries); no memory guard (ulimit -Sv throttles GHC).
          buildPhase = ''
            agda -i ${stdlibCompiled}/src -i ${felixCompiled}/src --compile modArTransformer.agda
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp modArTransformer $out/bin/agda-modArTransformer
          '';
        };

        packages.agda-modArTensorDiagnostics = pkgs.stdenv.mkDerivation {
          name = "agda-modArTensorDiagnostics";
          src = projectSource;
          nativeBuildInputs = [ pkgs.agda pkgs.ghc pkgs.glibcLocales ];
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LC_ALL = "en_US.UTF-8";
          buildPhase = ''
            agda -i ${stdlibCompiled}/src -i ${felixCompiled}/src --compile modArTensorDiagnostics.agda
          '';
          installPhase = ''
            mkdir -p $out/bin
            cp modArTensorDiagnostics $out/bin/agda-modArTensorDiagnostics
          '';
        };

        # Type-check only (fast CI gate, no GHC codegen).
        packages.agda-modArTransformer-check = pkgs.stdenv.mkDerivation {
          name = "agda-modArTransformer-check";
          src = projectSource;
          nativeBuildInputs = [ pkgs.agda pkgs.glibcLocales ];
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LC_ALL = "en_US.UTF-8";
          buildPhase = ''
            agda -i ${stdlibCompiled}/src -i ${felixCompiled}/src modArTransformer.agda
          '';
          installPhase = ''
            mkdir -p $out
            echo "modArTransformer.agda type-checked" > $out/result
          '';
        };

        apps.default = {
          type = "app";
          program = "${self'.packages.agda-modArTransformer}/bin/agda-modArTransformer";
        };
        apps.agda-modArTransformer = self'.apps.default;
        apps.tensor-diagnostics = {
          type = "app";
          program = "${self'.packages.agda-modArTensorDiagnostics}/bin/agda-modArTensorDiagnostics";
        };
        apps.cont-ad-playground = {
          type = "app";
          program = "${self'.packages.modartransformer-backend}/bin/cont-ad-playground";
        };
        apps.backend-train-epoch = {
          type = "app";
          program = "${self'.packages.modartransformer-backend}/bin/backend-train-epoch";
        };
        apps.ctc-smoke = {
          type = "app";
          program = "${self'.packages.ctc-smoke}/bin/ctc-smoke";
        };
        apps.ctc-grad-smoke = {
          type = "app";
          program = "${self'.packages.ctc-grad-smoke}/bin/ctc-grad-smoke";
        };
        apps.ctc-train = {
          type = "app";
          program = "${self'.packages.ctc-train}/bin/ctc-train";
        };

        devShells.default = pkgs.mkShell {
          name = "modArTransformer-dev";
          nativeBuildInputs = [
            myAgda          # agda preloaded with stdlib + felix interfaces
            # One GHC for both MAlonzo (`agda --compile`) and the Haskell backend
            # (`cabal`).  Includes the backend deps + text (MAlonzo FFI).
            (pkgs.haskellPackages.ghcWithPackages (p: [
              p.hmatrix p.vector p.random p.text
            ]))
            pkgs.cabal-install
            pkgs.glibcLocales
          ];
          shellHook = ''
            export LOCALE_ARCHIVE="${pkgs.glibcLocales}/lib/locale/locale-archive"
            export LC_ALL="en_US.UTF-8"
            echo "Agda ready (stdlib + felix interfaces preloaded)."
            echo "  Type-check: agda modArTransformer.agda"
            echo "  Compile:    agda --compile modArTransformer.agda"
            echo "Haskell backend: cabal build (ghc has hmatrix/vector/random)."
          '';
        };

        checks = {
          inherit (self'.packages) agda-modArTransformer agda-modArTransformer-check agda-modArTensorDiagnostics modartransformer-backend ctc-smoke ctc-grad-smoke;
        };
      };
    };
}
