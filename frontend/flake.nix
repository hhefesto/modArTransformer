{
  description = "Reflex-dom diagram for the denotational modular-arithmetic transformer";

  # Standalone flake (kept separate from the repo's flake-parts/haskell-flake +
  # Agda flake, since reflex-platform brings its own nix infrastructure — matching
  # the structure of the user's other reflex projects, e.g. ~/src/wedding-website).
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    reflex-platform = {
      url = "github:reflex-frp/reflex-platform";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, reflex-platform }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        rp = import reflex-platform { inherit system; };
        project = rp.project ({ ... }: {
          packages = { frontend = ./.; };
          shells = {
            ghc   = [ "frontend" ];
            ghcjs = [ "frontend" ];
          };
          # native dev preview (jsaddle-warp) — like ~/src/directo's `useWarp`.
          useWarp = true;
        });
        pkgs = import nixpkgs { inherit system; };
        ghcjsBuild = project.ghcjs.frontend;
      in {
        # GHCJS build -> static site (the .jsexe bundle + an index.html shell).
        packages.frontend = pkgs.runCommand "frontend-static" { } ''
          mkdir -p $out
          cp -r ${ghcjsBuild}/bin/frontend.jsexe/* $out/
          # the produced index.html loads all.js; rename if needed
          [ -f $out/index.html ] || cat > $out/index.html <<'HTML'
<!doctype html><html><head><meta charset="utf-8"><title>Transformer diagram</title></head>
<body><script language="javascript" src="all.js"></script></body></html>
HTML
        '';
        packages.default = self.packages.${system}.frontend;

        # serve the static GHCJS bundle
        apps.diagram = {
          type = "app";
          program = toString (pkgs.writeShellScript "serve-diagram" ''
            echo "Serving the reflex diagram at http://localhost:8080"
            ${pkgs.darkhttpd}/bin/darkhttpd ${self.packages.${system}.frontend} --port 8080
          '');
        };
        apps.default = self.apps.${system}.diagram;

        devShells.default = project.shells.ghc;
      });
}
