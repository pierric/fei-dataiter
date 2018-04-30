{-# Language TemplateHaskell #-}
module MXNet.Core.IO.Internal.TH where

import Data.List
import Data.Char
import Data.Bifunctor
import Text.ParserCombinators.ReadP
import Language.Haskell.TH

import MXNet.Core.Base
import MXNet.Core.Base.Internal

diInfoName (n,_,_,_,_,_) = n
diInfoDesc (_,n,_,_,_,_) = n
diInfoArgc (_,_,n,_,_,_) = n
diInfoArgN (_,_,_,n,_,_) = n
diInfoArgT (_,_,_,_,n,_) = n
diInfoArgD (_,_,_,_,_,n) = n

registerDataIters :: Q [Dec]
registerDataIters = do 
    dataiterInfo <- runIO (mxListDataIters >>= mapM info . zip [0..])
    concat <$> mapM (uncurry makeDataIter) dataiterInfo
  where
    info (idx, creator) = do
        info <- mxDataIterGetIterInfo creator
        let name = diInfoName info
            argn = diInfoArgN info
            argt = diInfoArgT info
            args = nub $ zip argn argt
        return ((idx, name), args)

makeDataIter :: (Integer, String) -> [(String, String)] -> Q [Dec]
makeDataIter (index, name) args = do
    let args' = map (second parseArgDesc) args
        dname = mkName (deCap name)
    let kvs = mkName "kvs"
        cstName = mkName $ name ++ "_Args"
        args = foldr add promotedNilT args'
        typWithArgs = if null args' then [t| IO DataIterHandle |] else [t| HMap $(varT kvs) -> IO DataIterHandle |]
    cst <- tySynD cstName [] args
    sig <- sigD dname [t| (MatchKVList $(varT kvs) $(conT cstName), ShowKV $(varT kvs)) => $(typWithArgs) |]
    let allargs = mkName "allargs"
    fun <- funD dname [clause [varP allargs] (normalB [e| do{
        args <- return (dump $(varE allargs));
        len  <- return (fromIntegral $ length args);
        (keys, vals) <- return (unzip args);
        crts <- mxListDataIters;
        checked $ mxDataIterCreateIter (crts !! $(litE $ integerL index)) len keys vals;
    } |]) []]
    return [cst, sig, fun]
  where
    deCap (x:xs) = (toLower x):xs
    toTyp ArgString = [t| String |]
    toTyp ArgInt    = [t| Int |]
    toTyp ArgLong   = [t| Integer |]
    toTyp ArgFloat  = [t| Float |]
    toTyp ArgBool   = [t| Bool |]
    toTyp ArgShape  = [t| [Int] |]
    toTyp (ArgEnum v)  = [t| String |]
    toTyp (ArgTuple t) = [t| [$(toTyp t)] |]
    app t1 t2 = [t| $(toTyp t1)  -> $(t2) |]
    add (nm,(at,_)) lst = let item = [t| $(litT (strTyLit nm)) ':= $(toTyp at) |]
                          in appT (appT promotedConsT item) lst

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
