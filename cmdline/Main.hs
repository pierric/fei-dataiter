module Main where

import Options.Applicative
import Data.Semigroup ((<>))
import Language.Haskell.Exts
import qualified Data.Text as T
import System.Log.Logger
import Control.Monad
import Control.Monad.Writer (Writer, execWriter, tell)
import Data.Either
import Data.Char (toLower, isUpper, isSpace, isAlphaNum)
import Text.Printf (printf)
import Text.ParserCombinators.ReadP
import System.FilePath
import System.Directory

import MXNet.Base.Raw

_module_ = "Main"

data Arguments = Arguments {
    output_dir :: FilePath
}

args_spec = Arguments 
         <$> strOption (long "output" <> short 'o' <> value "dataiter/src" <> metavar "OUTPUT-DIR")

main = do
    updateGlobalLogger _module_ (setLevel INFO)
    args <- execParser opts
    let base = output_dir args </> "MXNet" </> "NN" </> "DataIter"
    createDirectoryIfMissing True base

    dataitercreators  <- mxListDataIters

    infoM _module_ "Generating DataIters..."
    dataiters <- concat <$> mapM genDataIter (zip dataitercreators [0..])
    writeFile (base </> "Raw.hs") $ prettyPrint (modDataIter dataiters)
    
  where
    opts = info (args_spec <**> helper) (fullDesc <> progDesc "Generate MXNet dataiters")
    modDataIter = Module () (Just $ ModuleHead () (ModuleName () "MXNet.NN.DataIter.Raw") Nothing Nothing) [] 
                  [ simpleImport "MXNet.Base.Raw"
                  , simpleImport "MXNet.Base.Spec.Operator"
                  , simpleImport "MXNet.Base.Spec.HMap"
                  , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]

simpleImport mod = ImportDecl {
    importAnn = (),
    importModule = ModuleName () mod,
    importQualified = False,
    importSrc = False,
    importSafe = False,
    importPkg = Nothing,
    importAs = Nothing,
    importSpecs = Nothing
}

simpleImportVars mod vars = ImportDecl {
    importAnn = (),
    importModule = ModuleName () mod,
    importQualified = False,
    importSrc = False,
    importSafe = False,
    importPkg = Nothing,
    importAs = Nothing,
    importSpecs = Just $ ImportSpecList () False [IVar () $ Ident () var | var <- vars]
}

genDataIter :: (DataIterCreator, Integer) -> IO [Decl ()]
genDataIter (dataitercreator, index) = do
    (diname, didesc, argnames, argtypes, argdescs) <- mxDataIterGetIterInfo dataitercreator
    let diname_ = normalizeName diname
        (errs, scalarTypes) = execWriter $ zipWithM_ resolveHaskellType argnames argtypes

        -- parameter list
        paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) scalarTypes
        paramInst = TypeInsDecl () (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr diname))
                        (tyPromotedList paramList)

        -- signature
        cxfullfill = appA (name "Fullfilled") [tyPromotedStr diname, tyVarIdent "args"]
        tyfun = tyFun (tyApp (tyApp (tyCon $ unQual $ name "ArgsHMap") (tyPromotedStr diname)) (tyVarIdent "args")) 
                    (tyApp (tyCon $ unQual $ name "IO") (tyCon $ unQual $ name "DataIterHandle"))
        tysig = tySig [name diname_] $ tyForall [unkindedVar (name "args")] (cxSingle cxfullfill) tyfun

        -- function
        fun = sfun (name diname_) [name "args"] (UnGuardedRhs () body) Nothing
        body = letE ([
                patBind (pvar $ name "allargs") (function "catMaybes" 
                    `app` listE [
                        infixApp (infixApp (tupleSection [Just $ strE argkey, Nothing]) (op $ sym ".") (function "showValue")) (op $ sym "<$>") $ 
                            ExpTypeSig () (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey)) (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- scalarTypes])
              , patBind (pTuple [pvar $ name "keys", pvar $ name "vals"]) (app (function "unzip") $ var $ name "allargs")
              ]) (doE $ [
                  genStmt (pvar $ name "dis") $ function "mxListDataIters",
                  genStmt (pvar $ name "di") $ function "return" `app` (infixApp (var $ name "dis") (op $ sym "!!") (intE index)),
                  qualStmt $ function "mxDataIterCreateIter" `app` (var $ name "di") `app` (var $ name "keys") `app` (var $ name "vals")
              ])

    return [paramInst, tysig, fun]


  where
    normalizeName :: String -> String
    normalizeName name@(c:cs) 
        | isUpper c = '_' : name
        | otherwise = name

data ParamDesc = ParamDescItem String | ParamDescList Bool [String] deriving (Eq, Show)

type ResolvedType = (String, Type (), Type ())
resolveHaskellType :: String -> String -> Writer ([(String, String)], [ResolvedType]) ()
resolveHaskellType argname desc =
    case head fields of 
        ParamDescItem "Shape(tuple)"        -> scalar $ tyList $ tyCon $ unQual $ name "Int"
        ParamDescItem "int"                 -> scalar $ tyCon $ unQual $ name "Int"
        ParamDescItem "int (non-negative)"  -> scalar $ tyCon $ unQual $ name "Int"
        ParamDescItem "long (non-negative)" -> scalar $ tyCon $ unQual $ name "Int"
        ParamDescItem "boolean"             -> scalar $ tyCon $ unQual $ name "Bool"
        ParamDescItem "float"               -> scalar $ tyCon $ unQual $ name "Float"
        ParamDescItem "double"              -> scalar $ tyCon $ unQual $ name "Double"
        ParamDescItem "float32"             -> scalar $ tyCon $ unQual $ name "Float"
        ParamDescItem "string"              -> scalar $ tyCon $ unQual $ name "String"
        ParamDescItem "int or None"         -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Int")
        ParamDescItem "double or None"      -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Double")
        ParamDescList hasnone vs -> do
            let vsprom = map tyPromotedStr vs
                typ1 = tyApp (tyCon $ unQual $ name "EnumType") (tyPromotedList vsprom)
                typ2 = tyApp (tyCon $ unQual $ name "Maybe") typ1
            scalar $ if hasnone then typ2 else typ1

        t -> fail $ printf "unsupported arg: %s" (show t)
  where
    typedesc = do
        ds <- sepBy (skipSpaces >> (list1 +++ list2 +++ item)) (char ',')
        eof
        return ds
    list1 = ParamDescList True  <$> between (string "{None,") (char '}') (sepBy (skipSpaces >> listItem) (char ','))
    list2 = ParamDescList False <$> between (string "{") (char '}') (sepBy (skipSpaces >> listItem) (char ','))
    listItem = between (char '\'') (char '\'') (munch1 (\c -> isAlphaNum c || c `elem` "_"))
    item = ParamDescItem <$> munch1 (\c -> isAlphaNum c || c `elem` " _-()=[]<>'./+")
    runP str = case readP_to_S typedesc str of 
                    [(xs, "")] -> xs
                    other -> error ("cannot parse type description: " ++ str)

    fields = runP desc
    required = ParamDescItem "required" `elem` fields
    attr = tyCon $ unQual $ name $ if required then "AttrReq" else "AttrOpt"
    scalar hstyp = tell ([], [(argname, attr, hstyp)])
    fail msg   = tell ([(argname, msg)], [])

makeParamInst :: String -> [ResolvedType] -> Bool -> Decl ()
makeParamInst symname typs symbolapi = 
    TypeInsDecl () (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname_with_appendix))
                   (tyPromotedList paramList)
  where
    symname_with_appendix = symname ++ (if symbolapi then "(symbol)" else "(ndarray)")
    paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) typs


unQual = UnQual ()
unkindedVar = UnkindedVar ()

tyCon = TyCon ()
tyVarSymbol = TyVar () . Symbol ()
tyVarIdent = TyVar () . Ident ()
tyApp = TyApp ()
tyFun = TyFun ()
tySig names types = TypeSig () names types
tyList = TyList ()
tyVar = TyVar ()

tyPromotedInteger s = TyPromoted () (PromotedInteger () s (show s))
tyPromotedStr s     = TyPromoted () (PromotedString () s s)
tyPromotedList s    = TyPromoted () (PromotedList () True s)
tyPromotedTuple s   = TyPromoted () (PromotedTuple () s)

tyForall vars cxt typ = TyForall () vars_ cxt_ typ
  where
    vars_ = if null vars then Nothing else Just vars
    cxt_  = if cxt == CxEmpty () then Nothing else Just cxt

cxSingle = CxSingle ()
cxTuple  = CxTuple ()

appA = AppA ()

tupleSection = TupleSection () Boxed

con = Con ()