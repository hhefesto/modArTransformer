module AD
  ( AddFun(..)
  , idL
  , composeL
  , Dual(..)
  , composeDual
  , Cont(..)
  , composeCont
  , DDual(..)
  , DCont(..)
  , composeDDual
  , composeDCont
  ) where

newtype AddFun a b = AddFun { applyL :: a -> b }

idL :: AddFun a a
idL = AddFun id

composeL :: AddFun b c -> AddFun a b -> AddFun a c
composeL (AddFun g) (AddFun f) = AddFun (g . f)

newtype Dual a b = Dual { unDual :: AddFun b a }

composeDual :: Dual b c -> Dual a b -> Dual a c
composeDual (Dual g) (Dual f) = Dual (f `composeL` g)

newtype Cont r a b = Cont { runCont :: AddFun b r -> AddFun a r }

composeCont :: Cont r b c -> Cont r a b -> Cont r a c
composeCont (Cont g) (Cont f) = Cont (f . g)

newtype DDual a b = DDual { runDDual :: a -> (b, Dual a b) }

newtype DCont r a b = DCont { runDCont :: a -> (b, Cont r a b) }

composeDDual :: DDual b c -> DDual a b -> DDual a c
composeDDual (DDual g) (DDual f) = DDual $ \a ->
  let (b, f') = f a
      (c, g') = g b
  in (c, g' `composeDual` f')

composeDCont :: DCont r b c -> DCont r a b -> DCont r a c
composeDCont (DCont g) (DCont f) = DCont $ \a ->
  let (b, f') = f a
      (c, g') = g b
  in (c, g' `composeCont` f')
