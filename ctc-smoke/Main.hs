{-# LANGUAGE TypeApplications #-}

module Main where

import ConCat.AltCat (toCcc)

-- Stage 0: Bool projection (structural; no numerics).
ctcFirst :: (Bool, Bool) -> Bool
ctcFirst = toCcc @(->) @(Bool, Bool) @Bool (\(x, _y) -> x)

-- Stage 1: scalar Double arithmetic — exercises NumCat/FloatingCat elaboration.
ctcAffine :: (Double, Double) -> Double
ctcAffine = toCcc @(->) @(Double, Double) @Double (\(x, y) -> x * y + 1)

-- Stage 2: a numeric kernel — a 2-D dot product (mul + add over Doubles).
ctcDot2 :: ((Double, Double), (Double, Double)) -> Double
ctcDot2 = toCcc @(->) (\((a, b), (c, d)) -> a * c + b * d)

-- Stage 3: a fixed-size matvec, the smallest transformer-shaped linear kernel.
ctcMatVec2 :: (((Double, Double), (Double, Double)), (Double, Double)) -> (Double, Double)
ctcMatVec2 = toCcc @(->) (\(((a, b), (c, d)), (x, y)) -> (a * x + b * y, c * x + d * y))

-- Stage 4: two-class softmax, exercising exp and division used by attention/loss.
ctcSoftmax2 :: (Double, Double) -> (Double, Double)
ctcSoftmax2 = toCcc @(->) (\(x, y) -> let ex = exp x; ey = exp y; s = ex + ey in (ex / s, ey / s))

-- Stage 5: a tiny negative-log-likelihood loss for class 0, exercising log.
ctcNll0 :: (Double, Double) -> Double
ctcNll0 = toCcc @(->) (\(x, y) -> let ex = exp x; ey = exp y in negate (log (ex / (ex + ey))))

-- Stage 6: fixed tiny MLP loss: affine -> sigmoid -> affine -> NLL.
ctcTinyMlpNll0 :: (Double, Double) -> Double
ctcTinyMlpNll0 = toCcc @(->) $ \(x, y) ->
  let z0 = 0.5 * x - 0.25 * y + 0.1
      z1 = (-0.3) * x + 0.8 * y - 0.2
      h0 = 1 / (1 + exp (negate z0))
      h1 = 1 / (1 + exp (negate z1))
      l0 = 1.2 * h0 - 0.7 * h1 + 0.05
      l1 = (-0.4) * h0 + 0.9 * h1 - 0.1
      e0 = exp l0
      e1 = exp l1
  in negate (log (e0 / (e0 + e1)))

-- Stage 7: fixed tiny attention readout: dot scores -> softmax -> value mix.
ctcTinyAttention2 :: (((Double, Double), ((Double, Double), (Double, Double))), ((Double, Double), (Double, Double))) -> (Double, Double)
ctcTinyAttention2 = toCcc @(->) $ \(((qx, qy), ((k0x, k0y), (k1x, k1y))), ((v0x, v0y), (v1x, v1y))) ->
  let s0 = qx * k0x + qy * k0y
      s1 = qx * k1x + qy * k1y
      e0 = exp s0
      e1 = exp s1
      z = e0 + e1
      w0 = e0 / z
      w1 = e1 / z
  in (w0 * v0x + w1 * v1x, w0 * v0y + w1 * v1y)

-- Stage 8: fixed mini transformer-block loss: attention -> residual -> layernorm -> FFN logits -> NLL.
ctcTinyBlockNll0 :: (((Double, Double), ((Double, Double), (Double, Double))), ((Double, Double), (Double, Double))) -> Double
ctcTinyBlockNll0 = toCcc @(->) $ \(((qx, qy), ((k0x, k0y), (k1x, k1y))), ((v0x, v0y), (v1x, v1y))) ->
  let s0 = qx * k0x + qy * k0y
      s1 = qx * k1x + qy * k1y
      a0 = exp s0
      a1 = exp s1
      az = a0 + a1
      w0 = a0 / az
      w1 = a1 / az
      att0 = w0 * v0x + w1 * v1x
      att1 = w0 * v0y + w1 * v1y
      r0 = qx + att0
      r1 = qy + att1
      mu = (r0 + r1) / 2
      d0 = r0 - mu
      d1 = r1 - mu
      invStd = 1 / sqrt ((d0 * d0 + d1 * d1) / 2 + 1e-5)
      n0 = d0 * invStd
      n1 = d1 * invStd
      h0 = 1 / (1 + exp (negate (0.6 * n0 - 0.2 * n1 + 0.05)))
      h1 = 1 / (1 + exp (negate ((-0.1) * n0 + 0.4 * n1 - 0.03)))
      l0 = 0.7 * h0 - 0.5 * h1 + 0.2
      l1 = (-0.3) * h0 + 0.8 * h1 - 0.1
      e0 = exp l0
      e1 = exp l1
  in negate (log (e0 / (e0 + e1)))

main :: IO ()
main = do
  check "stage0 first"  (ctcFirst (True, False))      (fst (True, False))
  check "stage1 affine" (ctcAffine (3, 4))            (3 * 4 + 1)
  check "stage2 dot2"   (ctcDot2 ((1, 2), (3, 4)))    (1 * 3 + 2 * 4)
  check "stage3 matvec2" (ctcMatVec2 (((1, 2), (3, 4)), (5, 6))) (1 * 5 + 2 * 6, 3 * 5 + 4 * 6)
  checkNear2 "stage4 softmax2" (ctcSoftmax2 (1, 2)) softmax2Direct
  checkNear "stage5 nll0" (ctcNll0 (1, 2)) nll0Direct
  checkNear "stage6 tiny-mlp-nll0" (ctcTinyMlpNll0 (0.7, -1.1)) tinyMlpNll0Direct
  checkNear2 "stage7 tiny-attention2" (ctcTinyAttention2 tinyAttentionInput) tinyAttentionDirect
  checkNear "stage8 tiny-block-nll0" (ctcTinyBlockNll0 tinyAttentionInput) tinyBlockNll0Direct
  putStrLn "ctc smoke passed"
  where
    softmax2Direct :: (Double, Double)
    softmax2Direct = let ex = exp 1; ey = exp 2; s = ex + ey in (ex / s, ey / s)

    nll0Direct :: Double
    nll0Direct = let ex = exp 1; ey = exp 2 in negate (log (ex / (ex + ey)))

    tinyMlpNll0Direct :: Double
    tinyMlpNll0Direct =
      let x = 0.7
          y = -1.1
          z0 = 0.5 * x - 0.25 * y + 0.1
          z1 = (-0.3) * x + 0.8 * y - 0.2
          h0 = 1 / (1 + exp (negate z0))
          h1 = 1 / (1 + exp (negate z1))
          l0 = 1.2 * h0 - 0.7 * h1 + 0.05
          l1 = (-0.4) * h0 + 0.9 * h1 - 0.1
          e0 = exp l0
          e1 = exp l1
      in negate (log (e0 / (e0 + e1)))

    tinyAttentionInput :: (((Double, Double), ((Double, Double), (Double, Double))), ((Double, Double), (Double, Double)))
    tinyAttentionInput = (((0.2, -0.3), ((0.5, 0.1), (-0.4, 0.7))), ((1.0, -2.0), (0.3, 0.8)))

    tinyAttentionDirect :: (Double, Double)
    tinyAttentionDirect =
      let (((qx, qy), ((k0x, k0y), (k1x, k1y))), ((v0x, v0y), (v1x, v1y))) = tinyAttentionInput
          s0 = qx * k0x + qy * k0y
          s1 = qx * k1x + qy * k1y
          e0 = exp s0
          e1 = exp s1
          z = e0 + e1
          w0 = e0 / z
          w1 = e1 / z
      in (w0 * v0x + w1 * v1x, w0 * v0y + w1 * v1y)

    tinyBlockNll0Direct :: Double
    tinyBlockNll0Direct =
      let (((qx, qy), ((k0x, k0y), (k1x, k1y))), ((v0x, v0y), (v1x, v1y))) = tinyAttentionInput
          s0 = qx * k0x + qy * k0y
          s1 = qx * k1x + qy * k1y
          a0 = exp s0
          a1 = exp s1
          az = a0 + a1
          w0 = a0 / az
          w1 = a1 / az
          att0 = w0 * v0x + w1 * v1x
          att1 = w0 * v0y + w1 * v1y
          r0 = qx + att0
          r1 = qy + att1
          mu = (r0 + r1) / 2
          d0 = r0 - mu
          d1 = r1 - mu
          invStd = 1 / sqrt ((d0 * d0 + d1 * d1) / 2 + 1e-5)
          n0 = d0 * invStd
          n1 = d1 * invStd
          h0 = 1 / (1 + exp (negate (0.6 * n0 - 0.2 * n1 + 0.05)))
          h1 = 1 / (1 + exp (negate ((-0.1) * n0 + 0.4 * n1 - 0.03)))
          l0 = 0.7 * h0 - 0.5 * h1 + 0.2
          l1 = (-0.3) * h0 + 0.8 * h1 - 0.1
          e0 = exp l0
          e1 = exp l1
      in negate (log (e0 / (e0 + e1)))

    check :: (Eq a, Show a) => String -> a -> a -> IO ()
    check name got want = do
      putStrLn (name ++ ": ctc=" ++ show got ++ " direct=" ++ show want)
      if got == want then pure () else error (name ++ " mismatch")

    checkNear :: String -> Double -> Double -> IO ()
    checkNear name got want = do
      putStrLn (name ++ ": ctc=" ++ show got ++ " direct=" ++ show want)
      if abs (got - want) <= 1e-12 then pure () else error (name ++ " mismatch")

    checkNear2 :: String -> (Double, Double) -> (Double, Double) -> IO ()
    checkNear2 name (gx, gy) (wx, wy) = do
      putStrLn (name ++ ": ctc=" ++ show (gx, gy) ++ " direct=" ++ show (wx, wy))
      if abs (gx - wx) <= 1e-12 && abs (gy - wy) <= 1e-12 then pure () else error (name ++ " mismatch")
