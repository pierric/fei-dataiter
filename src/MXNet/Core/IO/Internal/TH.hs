module MXNet.Core.IO.Internal.TH where

import Data.List
import Data.Char
import Data.Bifunctor
import Text.ParserCombinators.ReadP
import Language.Haskell.TH
import MXNet.Core.Base.Internal

diInfoName (n,_,_,_,_,_) = n
diInfoDesc (_,n,_,_,_,_) = n
diInfoArgc (_,_,n,_,_,_) = n
diInfoArgN (_,_,_,n,_,_) = n
diInfoArgT (_,_,_,_,n,_) = n
diInfoArgD (_,_,_,_,_,n) = n

registerDataIters :: Q [Dec]
registerDataIters = runIO $ do
    di <- mxListDataIters
    concat <$> mapM register di
  where
    register creator = do
        info <- mxDataIterGetIterInfo creator
        let name = diInfoName info
            argn = diInfoArgN info
            argt = diInfoArgT info
            args = nub $ zip argn argt
        makeDataIter name args

makeDataIter :: String -> [(String, String)] -> IO [Dec]
makeDataIter name args = do
    let args' = map (second parseArgDesc) args
        (req, opt) = partition ((== Required) . snd . snd) args'

        sig = SigD (mkName name) $ 

data ArgType = ArgString | ArgInt | ArgLong | ArgFloat | ArgBool | ArgShape | ArgEnum [String] | ArgTuple ArgType
    deriving (Eq, Show)
data ArgOccr = Required | Optional
    deriving (Eq, Show)

parseArgDesc :: String -> (ArgType, ArgOccr)
parseArgDesc str = case readP_to_S desc str of
                     (r, _):_ -> r
                     _ -> error ("cannot parse arg desc: " ++ str)

alphaNum = many1 (satisfy isAlphaNum)
quoted = between (char '\'') (char '\'') (many $ satisfy isAlphaNum +++ choice (map char "/_-."))
boxed = between (char '[') (char ']') (quoted +++ number +++ alphaNum)
number = optional (char '-') >> many1 (satisfy isDigit)
comma = skipSpaces >> char ',' >> skipSpaces
enum = between (char '{') (char '}') (sepBy1 (alphaNum +++ quoted) comma)
typ = choice [ string "string"              >> return ArgString
             , string "int"                 >> return ArgInt
             , string "int (non-negative)"  >> return ArgInt
             , string "long"                >> return ArgLong
             , string "long (non-negative)" >> return ArgLong
             , string "boolean"             >> return ArgBool
             , string "float"               >> return ArgFloat
             , string "Shape(tuple)"        >> return ArgShape 
             , string "tuple of" >> skipSpaces >> (between (char '<') (char '>') typ >>= return . ArgTuple)
             , enum >>= (return . ArgEnum) ]
occ = choice [ string "required" >> 
               return Required 
             , string "optional" >> comma >> 
               string "default=" >> (quoted +++ boxed +++ alphaNum +++ number) >> 
               return Optional]

desc :: ReadP (ArgType, ArgOccr)
desc = do
    t <- typ
    comma
    o <- occ
    return (t, o)
