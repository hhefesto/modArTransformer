# Modular Arithmetic Transformer

A minimal transformer that learns `(a + b) mod 53` — demonstrating grokking in ~57k parameters.

## Training

```bash
# With nix
nix run

# Or from nix develop
cabal run modArTransformer
```

Saves checkpoints to `checkpoint.ckpt`.

## Interactive Diagram

```bash
# With nix
nix run .#diagram

# Or from nix develop
npm install
npx vite
```

Opens at http://localhost:5173 — click any block to explore what it does, parameter counts, tensor shapes, and the Haskell implementation.

## References

- Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*. arXiv:2201.02177 (2022). https://arxiv.org/abs/2201.02177
- Welch Labs. *The most complex model we actually understand*. https://www.youtube.com/watch?v=D8GOeCFFby4
