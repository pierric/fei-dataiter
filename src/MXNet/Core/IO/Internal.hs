{-# Language TemplateHaskell #-}
module MXNet.Core.IO.Internal where

import MXNet.Core.IO.Internal.TH

$(registerDataIters)