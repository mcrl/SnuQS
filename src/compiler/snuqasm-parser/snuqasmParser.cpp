
// Generated from snuqasm.g4 by ANTLR 4.12.0


#include "snuqasmListener.h"

#include "snuqasmParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct SnuqasmParserStaticData final {
  SnuqasmParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  SnuqasmParserStaticData(const SnuqasmParserStaticData&) = delete;
  SnuqasmParserStaticData(SnuqasmParserStaticData&&) = delete;
  SnuqasmParserStaticData& operator=(const SnuqasmParserStaticData&) = delete;
  SnuqasmParserStaticData& operator=(SnuqasmParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag snuqasmParserOnceFlag;
SnuqasmParserStaticData *snuqasmParserStaticData = nullptr;

void snuqasmParserInitialize() {
  assert(snuqasmParserStaticData == nullptr);
  auto staticData = std::make_unique<SnuqasmParserStaticData>(
    std::vector<std::string>{
      "mainprogram", "version", "program", "include", "statement", "decl", 
      "quantumDecl", "classicalDecl", "gatedeclStatement", "goplist", "opaqueDeclStatement", 
      "qopStatement", "uopStatement", "measureQop", "resetQop", "ifStatement", 
      "barrierStatement", "unitaryOp", "customOp", "anylist", "idlist", 
      "designatedIdentifier", "mixedlist", "arglist", "argument", "explist", 
      "exp", "binop", "unaryop"
    },
    std::vector<std::string>{
      "", "'SNUQASM'", "';'", "'include'", "'qreg'", "'['", "']'", "'creg'", 
      "'gate'", "'{'", "'}'", "'('", "')'", "'opaque'", "'measure'", "'->'", 
      "'reset'", "'if'", "'=='", "'barrier'", "'U'", "'u'", "'CX'", "','", 
      "'id'", "'h'", "'x'", "'y'", "'z'", "'sx'", "'sy'", "'s'", "'sdg'", 
      "'t'", "'tdg'", "'rx'", "'ry'", "'rz'", "'u1'", "'u2'", "'u3'", "'swap'", 
      "'cx'", "'cy'", "'cz'", "'ch'", "'crx'", "'cry'", "'crz'", "'cu1'", 
      "'cu2'", "'cu3'", "'ccx'", "'pi'", "'-'", "'+'", "'*'", "'/'", "'^'", 
      "'sin'", "'cos'", "'tan'", "'exp'", "'ln'", "'sqrt'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "StringLiteral", 
      "Real", "Integer", "Decimal", "SciSuffix", "Identifier", "Whitespace", 
      "Newline", "LineComment", "BlockComment"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,74,521,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,2,27,7,27,2,28,7,
  	28,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,2,1,2,1,2,3,2,69,8,2,1,2,1,2,1,2,1,2,
  	5,2,75,8,2,10,2,12,2,78,9,2,1,3,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,4,3,
  	4,90,8,4,1,5,1,5,3,5,94,8,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,
  	7,1,7,1,7,1,7,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,
  	1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,
  	8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,1,8,
  	3,8,159,8,8,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,3,9,169,8,9,1,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,
  	1,10,1,10,1,10,3,10,191,8,10,1,11,1,11,1,11,3,11,196,8,11,1,12,1,12,3,
  	12,200,8,12,1,13,1,13,1,13,1,13,1,13,1,13,1,14,1,14,1,14,1,14,1,15,1,
  	15,1,15,1,15,1,15,1,15,1,15,1,15,1,16,1,16,1,16,1,16,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,
  	17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,1,17,3,17,422,
  	8,17,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,
  	1,18,1,18,1,18,1,18,3,18,441,8,18,1,19,1,19,3,19,445,8,19,1,20,1,20,1,
  	20,1,20,3,20,451,8,20,1,21,1,21,1,21,1,21,1,21,1,22,1,22,1,22,1,22,1,
  	22,1,22,1,22,1,22,1,22,3,22,467,8,22,1,23,1,23,1,23,1,23,1,23,3,23,474,
  	8,23,1,24,1,24,1,24,1,24,1,24,3,24,481,8,24,1,25,1,25,1,25,1,25,1,25,
  	3,25,488,8,25,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,
  	1,26,1,26,1,26,1,26,1,26,3,26,506,8,26,1,26,1,26,1,26,1,26,5,26,512,8,
  	26,10,26,12,26,515,9,26,1,27,1,27,1,28,1,28,1,28,0,2,4,52,29,0,2,4,6,
  	8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,
  	56,0,2,1,0,54,58,1,0,59,64,561,0,58,1,0,0,0,2,61,1,0,0,0,4,68,1,0,0,0,
  	6,79,1,0,0,0,8,89,1,0,0,0,10,93,1,0,0,0,12,95,1,0,0,0,14,102,1,0,0,0,
  	16,158,1,0,0,0,18,168,1,0,0,0,20,190,1,0,0,0,22,195,1,0,0,0,24,199,1,
  	0,0,0,26,201,1,0,0,0,28,207,1,0,0,0,30,211,1,0,0,0,32,219,1,0,0,0,34,
  	421,1,0,0,0,36,440,1,0,0,0,38,444,1,0,0,0,40,450,1,0,0,0,42,452,1,0,0,
  	0,44,466,1,0,0,0,46,473,1,0,0,0,48,480,1,0,0,0,50,487,1,0,0,0,52,505,
  	1,0,0,0,54,516,1,0,0,0,56,518,1,0,0,0,58,59,3,2,1,0,59,60,3,4,2,0,60,
  	1,1,0,0,0,61,62,5,1,0,0,62,63,5,66,0,0,63,64,5,2,0,0,64,3,1,0,0,0,65,
  	66,6,2,-1,0,66,69,3,8,4,0,67,69,3,6,3,0,68,65,1,0,0,0,68,67,1,0,0,0,69,
  	76,1,0,0,0,70,71,10,2,0,0,71,75,3,8,4,0,72,73,10,1,0,0,73,75,3,6,3,0,
  	74,70,1,0,0,0,74,72,1,0,0,0,75,78,1,0,0,0,76,74,1,0,0,0,76,77,1,0,0,0,
  	77,5,1,0,0,0,78,76,1,0,0,0,79,80,5,3,0,0,80,81,5,65,0,0,81,82,5,2,0,0,
  	82,7,1,0,0,0,83,90,3,10,5,0,84,90,3,16,8,0,85,90,3,20,10,0,86,90,3,22,
  	11,0,87,90,3,30,15,0,88,90,3,32,16,0,89,83,1,0,0,0,89,84,1,0,0,0,89,85,
  	1,0,0,0,89,86,1,0,0,0,89,87,1,0,0,0,89,88,1,0,0,0,90,9,1,0,0,0,91,94,
  	3,12,6,0,92,94,3,14,7,0,93,91,1,0,0,0,93,92,1,0,0,0,94,11,1,0,0,0,95,
  	96,5,4,0,0,96,97,5,70,0,0,97,98,5,5,0,0,98,99,5,67,0,0,99,100,5,6,0,0,
  	100,101,5,2,0,0,101,13,1,0,0,0,102,103,5,7,0,0,103,104,5,70,0,0,104,105,
  	5,5,0,0,105,106,5,67,0,0,106,107,5,6,0,0,107,108,5,2,0,0,108,15,1,0,0,
  	0,109,110,5,8,0,0,110,111,5,70,0,0,111,112,3,40,20,0,112,113,5,9,0,0,
  	113,114,5,10,0,0,114,159,1,0,0,0,115,116,5,8,0,0,116,117,5,70,0,0,117,
  	118,3,40,20,0,118,119,5,9,0,0,119,120,3,18,9,0,120,121,5,10,0,0,121,159,
  	1,0,0,0,122,123,5,8,0,0,123,124,5,70,0,0,124,125,5,11,0,0,125,126,5,12,
  	0,0,126,127,3,40,20,0,127,128,5,9,0,0,128,129,5,10,0,0,129,159,1,0,0,
  	0,130,131,5,8,0,0,131,132,5,70,0,0,132,133,5,11,0,0,133,134,5,12,0,0,
  	134,135,3,40,20,0,135,136,5,9,0,0,136,137,3,18,9,0,137,138,5,10,0,0,138,
  	159,1,0,0,0,139,140,5,8,0,0,140,141,5,70,0,0,141,142,5,11,0,0,142,143,
  	3,40,20,0,143,144,5,12,0,0,144,145,3,40,20,0,145,146,5,9,0,0,146,147,
  	5,10,0,0,147,159,1,0,0,0,148,149,5,8,0,0,149,150,5,70,0,0,150,151,5,11,
  	0,0,151,152,3,40,20,0,152,153,5,12,0,0,153,154,3,40,20,0,154,155,5,9,
  	0,0,155,156,3,18,9,0,156,157,5,10,0,0,157,159,1,0,0,0,158,109,1,0,0,0,
  	158,115,1,0,0,0,158,122,1,0,0,0,158,130,1,0,0,0,158,139,1,0,0,0,158,148,
  	1,0,0,0,159,17,1,0,0,0,160,169,3,24,12,0,161,169,3,32,16,0,162,163,3,
  	24,12,0,163,164,3,18,9,0,164,169,1,0,0,0,165,166,3,32,16,0,166,167,3,
  	18,9,0,167,169,1,0,0,0,168,160,1,0,0,0,168,161,1,0,0,0,168,162,1,0,0,
  	0,168,165,1,0,0,0,169,19,1,0,0,0,170,171,5,13,0,0,171,172,5,70,0,0,172,
  	173,3,40,20,0,173,174,5,2,0,0,174,191,1,0,0,0,175,176,5,13,0,0,176,177,
  	5,70,0,0,177,178,5,11,0,0,178,179,5,12,0,0,179,180,3,40,20,0,180,181,
  	5,2,0,0,181,191,1,0,0,0,182,183,5,13,0,0,183,184,5,70,0,0,184,185,5,11,
  	0,0,185,186,3,40,20,0,186,187,5,12,0,0,187,188,3,40,20,0,188,189,5,2,
  	0,0,189,191,1,0,0,0,190,170,1,0,0,0,190,175,1,0,0,0,190,182,1,0,0,0,191,
  	21,1,0,0,0,192,196,3,24,12,0,193,196,3,26,13,0,194,196,3,28,14,0,195,
  	192,1,0,0,0,195,193,1,0,0,0,195,194,1,0,0,0,196,23,1,0,0,0,197,200,3,
  	34,17,0,198,200,3,36,18,0,199,197,1,0,0,0,199,198,1,0,0,0,200,25,1,0,
  	0,0,201,202,5,14,0,0,202,203,3,48,24,0,203,204,5,15,0,0,204,205,3,48,
  	24,0,205,206,5,2,0,0,206,27,1,0,0,0,207,208,5,16,0,0,208,209,3,48,24,
  	0,209,210,5,2,0,0,210,29,1,0,0,0,211,212,5,17,0,0,212,213,5,11,0,0,213,
  	214,5,70,0,0,214,215,5,18,0,0,215,216,5,67,0,0,216,217,5,12,0,0,217,218,
  	3,22,11,0,218,31,1,0,0,0,219,220,5,19,0,0,220,221,3,46,23,0,221,222,5,
  	2,0,0,222,33,1,0,0,0,223,224,5,20,0,0,224,225,5,11,0,0,225,226,3,50,25,
  	0,226,227,5,12,0,0,227,228,3,48,24,0,228,229,5,2,0,0,229,422,1,0,0,0,
  	230,231,5,21,0,0,231,232,5,11,0,0,232,233,3,50,25,0,233,234,5,12,0,0,
  	234,235,3,48,24,0,235,236,5,2,0,0,236,422,1,0,0,0,237,238,5,22,0,0,238,
  	239,3,48,24,0,239,240,5,23,0,0,240,241,3,48,24,0,241,242,5,2,0,0,242,
  	422,1,0,0,0,243,244,5,24,0,0,244,245,3,48,24,0,245,246,5,2,0,0,246,422,
  	1,0,0,0,247,248,5,25,0,0,248,249,3,48,24,0,249,250,5,2,0,0,250,422,1,
  	0,0,0,251,252,5,26,0,0,252,253,3,48,24,0,253,254,5,2,0,0,254,422,1,0,
  	0,0,255,256,5,27,0,0,256,257,3,48,24,0,257,258,5,2,0,0,258,422,1,0,0,
  	0,259,260,5,28,0,0,260,261,3,48,24,0,261,262,5,2,0,0,262,422,1,0,0,0,
  	263,264,5,29,0,0,264,265,3,48,24,0,265,266,5,2,0,0,266,422,1,0,0,0,267,
  	268,5,30,0,0,268,269,3,48,24,0,269,270,5,2,0,0,270,422,1,0,0,0,271,272,
  	5,31,0,0,272,273,3,48,24,0,273,274,5,2,0,0,274,422,1,0,0,0,275,276,5,
  	32,0,0,276,277,3,48,24,0,277,278,5,2,0,0,278,422,1,0,0,0,279,280,5,33,
  	0,0,280,281,3,48,24,0,281,282,5,2,0,0,282,422,1,0,0,0,283,284,5,34,0,
  	0,284,285,3,48,24,0,285,286,5,2,0,0,286,422,1,0,0,0,287,288,5,35,0,0,
  	288,289,5,11,0,0,289,290,3,50,25,0,290,291,5,12,0,0,291,292,3,48,24,0,
  	292,293,5,2,0,0,293,422,1,0,0,0,294,295,5,36,0,0,295,296,5,11,0,0,296,
  	297,3,50,25,0,297,298,5,12,0,0,298,299,3,48,24,0,299,300,5,2,0,0,300,
  	422,1,0,0,0,301,302,5,37,0,0,302,303,5,11,0,0,303,304,3,50,25,0,304,305,
  	5,12,0,0,305,306,3,48,24,0,306,307,5,2,0,0,307,422,1,0,0,0,308,309,5,
  	38,0,0,309,310,5,11,0,0,310,311,3,50,25,0,311,312,5,12,0,0,312,313,3,
  	48,24,0,313,314,5,2,0,0,314,422,1,0,0,0,315,316,5,39,0,0,316,317,5,11,
  	0,0,317,318,3,50,25,0,318,319,5,12,0,0,319,320,3,48,24,0,320,321,5,2,
  	0,0,321,422,1,0,0,0,322,323,5,40,0,0,323,324,5,11,0,0,324,325,3,50,25,
  	0,325,326,5,12,0,0,326,327,3,48,24,0,327,328,5,2,0,0,328,422,1,0,0,0,
  	329,330,5,41,0,0,330,331,3,48,24,0,331,332,5,23,0,0,332,333,3,48,24,0,
  	333,334,5,2,0,0,334,422,1,0,0,0,335,336,5,42,0,0,336,337,3,48,24,0,337,
  	338,5,23,0,0,338,339,3,48,24,0,339,340,5,2,0,0,340,422,1,0,0,0,341,342,
  	5,43,0,0,342,343,3,48,24,0,343,344,5,23,0,0,344,345,3,48,24,0,345,346,
  	5,2,0,0,346,422,1,0,0,0,347,348,5,44,0,0,348,349,3,48,24,0,349,350,5,
  	23,0,0,350,351,3,48,24,0,351,352,5,2,0,0,352,422,1,0,0,0,353,354,5,45,
  	0,0,354,355,3,48,24,0,355,356,5,23,0,0,356,357,3,48,24,0,357,358,5,2,
  	0,0,358,422,1,0,0,0,359,360,5,46,0,0,360,361,5,11,0,0,361,362,3,50,25,
  	0,362,363,5,12,0,0,363,364,3,48,24,0,364,365,5,23,0,0,365,366,3,48,24,
  	0,366,367,5,2,0,0,367,422,1,0,0,0,368,369,5,47,0,0,369,370,5,11,0,0,370,
  	371,3,50,25,0,371,372,5,12,0,0,372,373,3,48,24,0,373,374,5,23,0,0,374,
  	375,3,48,24,0,375,376,5,2,0,0,376,422,1,0,0,0,377,378,5,48,0,0,378,379,
  	5,11,0,0,379,380,3,50,25,0,380,381,5,12,0,0,381,382,3,48,24,0,382,383,
  	5,23,0,0,383,384,3,48,24,0,384,385,5,2,0,0,385,422,1,0,0,0,386,387,5,
  	49,0,0,387,388,5,11,0,0,388,389,3,50,25,0,389,390,5,12,0,0,390,391,3,
  	48,24,0,391,392,5,23,0,0,392,393,3,48,24,0,393,394,5,2,0,0,394,422,1,
  	0,0,0,395,396,5,50,0,0,396,397,5,11,0,0,397,398,3,50,25,0,398,399,5,12,
  	0,0,399,400,3,48,24,0,400,401,5,23,0,0,401,402,3,48,24,0,402,403,5,2,
  	0,0,403,422,1,0,0,0,404,405,5,51,0,0,405,406,5,11,0,0,406,407,3,50,25,
  	0,407,408,5,12,0,0,408,409,3,48,24,0,409,410,5,23,0,0,410,411,3,48,24,
  	0,411,412,5,2,0,0,412,422,1,0,0,0,413,414,5,52,0,0,414,415,3,48,24,0,
  	415,416,5,23,0,0,416,417,3,48,24,0,417,418,5,23,0,0,418,419,3,48,24,0,
  	419,420,5,2,0,0,420,422,1,0,0,0,421,223,1,0,0,0,421,230,1,0,0,0,421,237,
  	1,0,0,0,421,243,1,0,0,0,421,247,1,0,0,0,421,251,1,0,0,0,421,255,1,0,0,
  	0,421,259,1,0,0,0,421,263,1,0,0,0,421,267,1,0,0,0,421,271,1,0,0,0,421,
  	275,1,0,0,0,421,279,1,0,0,0,421,283,1,0,0,0,421,287,1,0,0,0,421,294,1,
  	0,0,0,421,301,1,0,0,0,421,308,1,0,0,0,421,315,1,0,0,0,421,322,1,0,0,0,
  	421,329,1,0,0,0,421,335,1,0,0,0,421,341,1,0,0,0,421,347,1,0,0,0,421,353,
  	1,0,0,0,421,359,1,0,0,0,421,368,1,0,0,0,421,377,1,0,0,0,421,386,1,0,0,
  	0,421,395,1,0,0,0,421,404,1,0,0,0,421,413,1,0,0,0,422,35,1,0,0,0,423,
  	424,5,70,0,0,424,425,3,38,19,0,425,426,5,2,0,0,426,441,1,0,0,0,427,428,
  	5,70,0,0,428,429,5,11,0,0,429,430,5,12,0,0,430,431,3,38,19,0,431,432,
  	5,2,0,0,432,441,1,0,0,0,433,434,5,70,0,0,434,435,5,11,0,0,435,436,3,50,
  	25,0,436,437,5,12,0,0,437,438,3,38,19,0,438,439,5,2,0,0,439,441,1,0,0,
  	0,440,423,1,0,0,0,440,427,1,0,0,0,440,433,1,0,0,0,441,37,1,0,0,0,442,
  	445,3,40,20,0,443,445,3,44,22,0,444,442,1,0,0,0,444,443,1,0,0,0,445,39,
  	1,0,0,0,446,451,5,70,0,0,447,448,5,70,0,0,448,449,5,23,0,0,449,451,3,
  	40,20,0,450,446,1,0,0,0,450,447,1,0,0,0,451,41,1,0,0,0,452,453,5,70,0,
  	0,453,454,5,5,0,0,454,455,5,67,0,0,455,456,5,6,0,0,456,43,1,0,0,0,457,
  	467,5,70,0,0,458,467,3,42,21,0,459,460,5,70,0,0,460,461,5,23,0,0,461,
  	467,3,44,22,0,462,463,3,42,21,0,463,464,5,23,0,0,464,465,3,44,22,0,465,
  	467,1,0,0,0,466,457,1,0,0,0,466,458,1,0,0,0,466,459,1,0,0,0,466,462,1,
  	0,0,0,467,45,1,0,0,0,468,474,3,48,24,0,469,470,3,48,24,0,470,471,5,23,
  	0,0,471,472,3,46,23,0,472,474,1,0,0,0,473,468,1,0,0,0,473,469,1,0,0,0,
  	474,47,1,0,0,0,475,481,5,70,0,0,476,477,5,70,0,0,477,478,5,5,0,0,478,
  	479,5,67,0,0,479,481,5,6,0,0,480,475,1,0,0,0,480,476,1,0,0,0,481,49,1,
  	0,0,0,482,488,3,52,26,0,483,484,3,52,26,0,484,485,5,23,0,0,485,486,3,
  	50,25,0,486,488,1,0,0,0,487,482,1,0,0,0,487,483,1,0,0,0,488,51,1,0,0,
  	0,489,490,6,26,-1,0,490,506,5,66,0,0,491,506,5,67,0,0,492,506,5,53,0,
  	0,493,506,5,70,0,0,494,495,3,56,28,0,495,496,5,11,0,0,496,497,3,52,26,
  	0,497,498,5,12,0,0,498,506,1,0,0,0,499,500,5,11,0,0,500,501,3,52,26,0,
  	501,502,5,12,0,0,502,506,1,0,0,0,503,504,5,54,0,0,504,506,3,52,26,1,505,
  	489,1,0,0,0,505,491,1,0,0,0,505,492,1,0,0,0,505,493,1,0,0,0,505,494,1,
  	0,0,0,505,499,1,0,0,0,505,503,1,0,0,0,506,513,1,0,0,0,507,508,10,4,0,
  	0,508,509,3,54,27,0,509,510,3,52,26,5,510,512,1,0,0,0,511,507,1,0,0,0,
  	512,515,1,0,0,0,513,511,1,0,0,0,513,514,1,0,0,0,514,53,1,0,0,0,515,513,
  	1,0,0,0,516,517,7,0,0,0,517,55,1,0,0,0,518,519,7,1,0,0,519,57,1,0,0,0,
  	20,68,74,76,89,93,158,168,190,195,199,421,440,444,450,466,473,480,487,
  	505,513
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  snuqasmParserStaticData = staticData.release();
}

}

snuqasmParser::snuqasmParser(TokenStream *input) : snuqasmParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

snuqasmParser::snuqasmParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  snuqasmParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *snuqasmParserStaticData->atn, snuqasmParserStaticData->decisionToDFA, snuqasmParserStaticData->sharedContextCache, options);
}

snuqasmParser::~snuqasmParser() {
  delete _interpreter;
}

const atn::ATN& snuqasmParser::getATN() const {
  return *snuqasmParserStaticData->atn;
}

std::string snuqasmParser::getGrammarFileName() const {
  return "snuqasm.g4";
}

const std::vector<std::string>& snuqasmParser::getRuleNames() const {
  return snuqasmParserStaticData->ruleNames;
}

const dfa::Vocabulary& snuqasmParser::getVocabulary() const {
  return snuqasmParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView snuqasmParser::getSerializedATN() const {
  return snuqasmParserStaticData->serializedATN;
}


//----------------- MainprogramContext ------------------------------------------------------------------

snuqasmParser::MainprogramContext::MainprogramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::VersionContext* snuqasmParser::MainprogramContext::version() {
  return getRuleContext<snuqasmParser::VersionContext>(0);
}

snuqasmParser::ProgramContext* snuqasmParser::MainprogramContext::program() {
  return getRuleContext<snuqasmParser::ProgramContext>(0);
}


size_t snuqasmParser::MainprogramContext::getRuleIndex() const {
  return snuqasmParser::RuleMainprogram;
}

void snuqasmParser::MainprogramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMainprogram(this);
}

void snuqasmParser::MainprogramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMainprogram(this);
}

snuqasmParser::MainprogramContext* snuqasmParser::mainprogram() {
  MainprogramContext *_localctx = _tracker.createInstance<MainprogramContext>(_ctx, getState());
  enterRule(_localctx, 0, snuqasmParser::RuleMainprogram);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(58);
    version();
    setState(59);
    program(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VersionContext ------------------------------------------------------------------

snuqasmParser::VersionContext::VersionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::VersionContext::Real() {
  return getToken(snuqasmParser::Real, 0);
}


size_t snuqasmParser::VersionContext::getRuleIndex() const {
  return snuqasmParser::RuleVersion;
}

void snuqasmParser::VersionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVersion(this);
}

void snuqasmParser::VersionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVersion(this);
}

snuqasmParser::VersionContext* snuqasmParser::version() {
  VersionContext *_localctx = _tracker.createInstance<VersionContext>(_ctx, getState());
  enterRule(_localctx, 2, snuqasmParser::RuleVersion);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(61);
    match(snuqasmParser::T__0);
    setState(62);
    match(snuqasmParser::Real);
    setState(63);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProgramContext ------------------------------------------------------------------

snuqasmParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::StatementContext* snuqasmParser::ProgramContext::statement() {
  return getRuleContext<snuqasmParser::StatementContext>(0);
}

snuqasmParser::IncludeContext* snuqasmParser::ProgramContext::include() {
  return getRuleContext<snuqasmParser::IncludeContext>(0);
}

snuqasmParser::ProgramContext* snuqasmParser::ProgramContext::program() {
  return getRuleContext<snuqasmParser::ProgramContext>(0);
}


size_t snuqasmParser::ProgramContext::getRuleIndex() const {
  return snuqasmParser::RuleProgram;
}

void snuqasmParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void snuqasmParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


snuqasmParser::ProgramContext* snuqasmParser::program() {
   return program(0);
}

snuqasmParser::ProgramContext* snuqasmParser::program(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  snuqasmParser::ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, parentState);
  snuqasmParser::ProgramContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 4;
  enterRecursionRule(_localctx, 4, snuqasmParser::RuleProgram, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(68);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__3:
      case snuqasmParser::T__6:
      case snuqasmParser::T__7:
      case snuqasmParser::T__12:
      case snuqasmParser::T__13:
      case snuqasmParser::T__15:
      case snuqasmParser::T__16:
      case snuqasmParser::T__18:
      case snuqasmParser::T__19:
      case snuqasmParser::T__20:
      case snuqasmParser::T__21:
      case snuqasmParser::T__23:
      case snuqasmParser::T__24:
      case snuqasmParser::T__25:
      case snuqasmParser::T__26:
      case snuqasmParser::T__27:
      case snuqasmParser::T__28:
      case snuqasmParser::T__29:
      case snuqasmParser::T__30:
      case snuqasmParser::T__31:
      case snuqasmParser::T__32:
      case snuqasmParser::T__33:
      case snuqasmParser::T__34:
      case snuqasmParser::T__35:
      case snuqasmParser::T__36:
      case snuqasmParser::T__37:
      case snuqasmParser::T__38:
      case snuqasmParser::T__39:
      case snuqasmParser::T__40:
      case snuqasmParser::T__41:
      case snuqasmParser::T__42:
      case snuqasmParser::T__43:
      case snuqasmParser::T__44:
      case snuqasmParser::T__45:
      case snuqasmParser::T__46:
      case snuqasmParser::T__47:
      case snuqasmParser::T__48:
      case snuqasmParser::T__49:
      case snuqasmParser::T__50:
      case snuqasmParser::T__51:
      case snuqasmParser::Identifier: {
        setState(66);
        statement();
        break;
      }

      case snuqasmParser::T__2: {
        setState(67);
        include();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(76);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(74);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ProgramContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleProgram);
          setState(70);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(71);
          statement();
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ProgramContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleProgram);
          setState(72);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(73);
          include();
          break;
        }

        default:
          break;
        } 
      }
      setState(78);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- IncludeContext ------------------------------------------------------------------

snuqasmParser::IncludeContext::IncludeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::IncludeContext::StringLiteral() {
  return getToken(snuqasmParser::StringLiteral, 0);
}


size_t snuqasmParser::IncludeContext::getRuleIndex() const {
  return snuqasmParser::RuleInclude;
}

void snuqasmParser::IncludeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInclude(this);
}

void snuqasmParser::IncludeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInclude(this);
}

snuqasmParser::IncludeContext* snuqasmParser::include() {
  IncludeContext *_localctx = _tracker.createInstance<IncludeContext>(_ctx, getState());
  enterRule(_localctx, 6, snuqasmParser::RuleInclude);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(79);
    match(snuqasmParser::T__2);
    setState(80);
    match(snuqasmParser::StringLiteral);
    setState(81);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

snuqasmParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::DeclContext* snuqasmParser::StatementContext::decl() {
  return getRuleContext<snuqasmParser::DeclContext>(0);
}

snuqasmParser::GatedeclStatementContext* snuqasmParser::StatementContext::gatedeclStatement() {
  return getRuleContext<snuqasmParser::GatedeclStatementContext>(0);
}

snuqasmParser::OpaqueDeclStatementContext* snuqasmParser::StatementContext::opaqueDeclStatement() {
  return getRuleContext<snuqasmParser::OpaqueDeclStatementContext>(0);
}

snuqasmParser::QopStatementContext* snuqasmParser::StatementContext::qopStatement() {
  return getRuleContext<snuqasmParser::QopStatementContext>(0);
}

snuqasmParser::IfStatementContext* snuqasmParser::StatementContext::ifStatement() {
  return getRuleContext<snuqasmParser::IfStatementContext>(0);
}

snuqasmParser::BarrierStatementContext* snuqasmParser::StatementContext::barrierStatement() {
  return getRuleContext<snuqasmParser::BarrierStatementContext>(0);
}


size_t snuqasmParser::StatementContext::getRuleIndex() const {
  return snuqasmParser::RuleStatement;
}

void snuqasmParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void snuqasmParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

snuqasmParser::StatementContext* snuqasmParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 8, snuqasmParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(89);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__3:
      case snuqasmParser::T__6: {
        enterOuterAlt(_localctx, 1);
        setState(83);
        decl();
        break;
      }

      case snuqasmParser::T__7: {
        enterOuterAlt(_localctx, 2);
        setState(84);
        gatedeclStatement();
        break;
      }

      case snuqasmParser::T__12: {
        enterOuterAlt(_localctx, 3);
        setState(85);
        opaqueDeclStatement();
        break;
      }

      case snuqasmParser::T__13:
      case snuqasmParser::T__15:
      case snuqasmParser::T__19:
      case snuqasmParser::T__20:
      case snuqasmParser::T__21:
      case snuqasmParser::T__23:
      case snuqasmParser::T__24:
      case snuqasmParser::T__25:
      case snuqasmParser::T__26:
      case snuqasmParser::T__27:
      case snuqasmParser::T__28:
      case snuqasmParser::T__29:
      case snuqasmParser::T__30:
      case snuqasmParser::T__31:
      case snuqasmParser::T__32:
      case snuqasmParser::T__33:
      case snuqasmParser::T__34:
      case snuqasmParser::T__35:
      case snuqasmParser::T__36:
      case snuqasmParser::T__37:
      case snuqasmParser::T__38:
      case snuqasmParser::T__39:
      case snuqasmParser::T__40:
      case snuqasmParser::T__41:
      case snuqasmParser::T__42:
      case snuqasmParser::T__43:
      case snuqasmParser::T__44:
      case snuqasmParser::T__45:
      case snuqasmParser::T__46:
      case snuqasmParser::T__47:
      case snuqasmParser::T__48:
      case snuqasmParser::T__49:
      case snuqasmParser::T__50:
      case snuqasmParser::T__51:
      case snuqasmParser::Identifier: {
        enterOuterAlt(_localctx, 4);
        setState(86);
        qopStatement();
        break;
      }

      case snuqasmParser::T__16: {
        enterOuterAlt(_localctx, 5);
        setState(87);
        ifStatement();
        break;
      }

      case snuqasmParser::T__18: {
        enterOuterAlt(_localctx, 6);
        setState(88);
        barrierStatement();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclContext ------------------------------------------------------------------

snuqasmParser::DeclContext::DeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::QuantumDeclContext* snuqasmParser::DeclContext::quantumDecl() {
  return getRuleContext<snuqasmParser::QuantumDeclContext>(0);
}

snuqasmParser::ClassicalDeclContext* snuqasmParser::DeclContext::classicalDecl() {
  return getRuleContext<snuqasmParser::ClassicalDeclContext>(0);
}


size_t snuqasmParser::DeclContext::getRuleIndex() const {
  return snuqasmParser::RuleDecl;
}

void snuqasmParser::DeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecl(this);
}

void snuqasmParser::DeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecl(this);
}

snuqasmParser::DeclContext* snuqasmParser::decl() {
  DeclContext *_localctx = _tracker.createInstance<DeclContext>(_ctx, getState());
  enterRule(_localctx, 10, snuqasmParser::RuleDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(93);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__3: {
        enterOuterAlt(_localctx, 1);
        setState(91);
        quantumDecl();
        break;
      }

      case snuqasmParser::T__6: {
        enterOuterAlt(_localctx, 2);
        setState(92);
        classicalDecl();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QuantumDeclContext ------------------------------------------------------------------

snuqasmParser::QuantumDeclContext::QuantumDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::QuantumDeclContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

tree::TerminalNode* snuqasmParser::QuantumDeclContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}


size_t snuqasmParser::QuantumDeclContext::getRuleIndex() const {
  return snuqasmParser::RuleQuantumDecl;
}

void snuqasmParser::QuantumDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumDecl(this);
}

void snuqasmParser::QuantumDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumDecl(this);
}

snuqasmParser::QuantumDeclContext* snuqasmParser::quantumDecl() {
  QuantumDeclContext *_localctx = _tracker.createInstance<QuantumDeclContext>(_ctx, getState());
  enterRule(_localctx, 12, snuqasmParser::RuleQuantumDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(95);
    match(snuqasmParser::T__3);
    setState(96);
    match(snuqasmParser::Identifier);
    setState(97);
    match(snuqasmParser::T__4);
    setState(98);
    match(snuqasmParser::Integer);
    setState(99);
    match(snuqasmParser::T__5);
    setState(100);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalDeclContext ------------------------------------------------------------------

snuqasmParser::ClassicalDeclContext::ClassicalDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::ClassicalDeclContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

tree::TerminalNode* snuqasmParser::ClassicalDeclContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}


size_t snuqasmParser::ClassicalDeclContext::getRuleIndex() const {
  return snuqasmParser::RuleClassicalDecl;
}

void snuqasmParser::ClassicalDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalDecl(this);
}

void snuqasmParser::ClassicalDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalDecl(this);
}

snuqasmParser::ClassicalDeclContext* snuqasmParser::classicalDecl() {
  ClassicalDeclContext *_localctx = _tracker.createInstance<ClassicalDeclContext>(_ctx, getState());
  enterRule(_localctx, 14, snuqasmParser::RuleClassicalDecl);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(102);
    match(snuqasmParser::T__6);
    setState(103);
    match(snuqasmParser::Identifier);
    setState(104);
    match(snuqasmParser::T__4);
    setState(105);
    match(snuqasmParser::Integer);
    setState(106);
    match(snuqasmParser::T__5);
    setState(107);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GatedeclStatementContext ------------------------------------------------------------------

snuqasmParser::GatedeclStatementContext::GatedeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::GatedeclStatementContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

std::vector<snuqasmParser::IdlistContext *> snuqasmParser::GatedeclStatementContext::idlist() {
  return getRuleContexts<snuqasmParser::IdlistContext>();
}

snuqasmParser::IdlistContext* snuqasmParser::GatedeclStatementContext::idlist(size_t i) {
  return getRuleContext<snuqasmParser::IdlistContext>(i);
}

snuqasmParser::GoplistContext* snuqasmParser::GatedeclStatementContext::goplist() {
  return getRuleContext<snuqasmParser::GoplistContext>(0);
}


size_t snuqasmParser::GatedeclStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleGatedeclStatement;
}

void snuqasmParser::GatedeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGatedeclStatement(this);
}

void snuqasmParser::GatedeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGatedeclStatement(this);
}

snuqasmParser::GatedeclStatementContext* snuqasmParser::gatedeclStatement() {
  GatedeclStatementContext *_localctx = _tracker.createInstance<GatedeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 16, snuqasmParser::RuleGatedeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(158);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(109);
      match(snuqasmParser::T__7);
      setState(110);
      match(snuqasmParser::Identifier);
      setState(111);
      idlist();
      setState(112);
      match(snuqasmParser::T__8);
      setState(113);
      match(snuqasmParser::T__9);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(115);
      match(snuqasmParser::T__7);
      setState(116);
      match(snuqasmParser::Identifier);
      setState(117);
      idlist();
      setState(118);
      match(snuqasmParser::T__8);
      setState(119);
      goplist();
      setState(120);
      match(snuqasmParser::T__9);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(122);
      match(snuqasmParser::T__7);
      setState(123);
      match(snuqasmParser::Identifier);
      setState(124);
      match(snuqasmParser::T__10);
      setState(125);
      match(snuqasmParser::T__11);
      setState(126);
      idlist();
      setState(127);
      match(snuqasmParser::T__8);
      setState(128);
      match(snuqasmParser::T__9);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(130);
      match(snuqasmParser::T__7);
      setState(131);
      match(snuqasmParser::Identifier);
      setState(132);
      match(snuqasmParser::T__10);
      setState(133);
      match(snuqasmParser::T__11);
      setState(134);
      idlist();
      setState(135);
      match(snuqasmParser::T__8);
      setState(136);
      goplist();
      setState(137);
      match(snuqasmParser::T__9);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(139);
      match(snuqasmParser::T__7);
      setState(140);
      match(snuqasmParser::Identifier);
      setState(141);
      match(snuqasmParser::T__10);
      setState(142);
      idlist();
      setState(143);
      match(snuqasmParser::T__11);
      setState(144);
      idlist();
      setState(145);
      match(snuqasmParser::T__8);
      setState(146);
      match(snuqasmParser::T__9);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(148);
      match(snuqasmParser::T__7);
      setState(149);
      match(snuqasmParser::Identifier);
      setState(150);
      match(snuqasmParser::T__10);
      setState(151);
      idlist();
      setState(152);
      match(snuqasmParser::T__11);
      setState(153);
      idlist();
      setState(154);
      match(snuqasmParser::T__8);
      setState(155);
      goplist();
      setState(156);
      match(snuqasmParser::T__9);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GoplistContext ------------------------------------------------------------------

snuqasmParser::GoplistContext::GoplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::UopStatementContext* snuqasmParser::GoplistContext::uopStatement() {
  return getRuleContext<snuqasmParser::UopStatementContext>(0);
}

snuqasmParser::BarrierStatementContext* snuqasmParser::GoplistContext::barrierStatement() {
  return getRuleContext<snuqasmParser::BarrierStatementContext>(0);
}

snuqasmParser::GoplistContext* snuqasmParser::GoplistContext::goplist() {
  return getRuleContext<snuqasmParser::GoplistContext>(0);
}


size_t snuqasmParser::GoplistContext::getRuleIndex() const {
  return snuqasmParser::RuleGoplist;
}

void snuqasmParser::GoplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGoplist(this);
}

void snuqasmParser::GoplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGoplist(this);
}

snuqasmParser::GoplistContext* snuqasmParser::goplist() {
  GoplistContext *_localctx = _tracker.createInstance<GoplistContext>(_ctx, getState());
  enterRule(_localctx, 18, snuqasmParser::RuleGoplist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(168);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(160);
      uopStatement();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(161);
      barrierStatement();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(162);
      uopStatement();
      setState(163);
      goplist();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(165);
      barrierStatement();
      setState(166);
      goplist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OpaqueDeclStatementContext ------------------------------------------------------------------

snuqasmParser::OpaqueDeclStatementContext::OpaqueDeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::OpaqueDeclStatementContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

std::vector<snuqasmParser::IdlistContext *> snuqasmParser::OpaqueDeclStatementContext::idlist() {
  return getRuleContexts<snuqasmParser::IdlistContext>();
}

snuqasmParser::IdlistContext* snuqasmParser::OpaqueDeclStatementContext::idlist(size_t i) {
  return getRuleContext<snuqasmParser::IdlistContext>(i);
}


size_t snuqasmParser::OpaqueDeclStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleOpaqueDeclStatement;
}

void snuqasmParser::OpaqueDeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOpaqueDeclStatement(this);
}

void snuqasmParser::OpaqueDeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOpaqueDeclStatement(this);
}

snuqasmParser::OpaqueDeclStatementContext* snuqasmParser::opaqueDeclStatement() {
  OpaqueDeclStatementContext *_localctx = _tracker.createInstance<OpaqueDeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 20, snuqasmParser::RuleOpaqueDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(190);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(170);
      match(snuqasmParser::T__12);
      setState(171);
      match(snuqasmParser::Identifier);
      setState(172);
      idlist();
      setState(173);
      match(snuqasmParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(175);
      match(snuqasmParser::T__12);
      setState(176);
      match(snuqasmParser::Identifier);
      setState(177);
      match(snuqasmParser::T__10);
      setState(178);
      match(snuqasmParser::T__11);
      setState(179);
      idlist();
      setState(180);
      match(snuqasmParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(182);
      match(snuqasmParser::T__12);
      setState(183);
      match(snuqasmParser::Identifier);
      setState(184);
      match(snuqasmParser::T__10);
      setState(185);
      idlist();
      setState(186);
      match(snuqasmParser::T__11);
      setState(187);
      idlist();
      setState(188);
      match(snuqasmParser::T__1);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopStatementContext ------------------------------------------------------------------

snuqasmParser::QopStatementContext::QopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::UopStatementContext* snuqasmParser::QopStatementContext::uopStatement() {
  return getRuleContext<snuqasmParser::UopStatementContext>(0);
}

snuqasmParser::MeasureQopContext* snuqasmParser::QopStatementContext::measureQop() {
  return getRuleContext<snuqasmParser::MeasureQopContext>(0);
}

snuqasmParser::ResetQopContext* snuqasmParser::QopStatementContext::resetQop() {
  return getRuleContext<snuqasmParser::ResetQopContext>(0);
}


size_t snuqasmParser::QopStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleQopStatement;
}

void snuqasmParser::QopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopStatement(this);
}

void snuqasmParser::QopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopStatement(this);
}

snuqasmParser::QopStatementContext* snuqasmParser::qopStatement() {
  QopStatementContext *_localctx = _tracker.createInstance<QopStatementContext>(_ctx, getState());
  enterRule(_localctx, 22, snuqasmParser::RuleQopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(195);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__19:
      case snuqasmParser::T__20:
      case snuqasmParser::T__21:
      case snuqasmParser::T__23:
      case snuqasmParser::T__24:
      case snuqasmParser::T__25:
      case snuqasmParser::T__26:
      case snuqasmParser::T__27:
      case snuqasmParser::T__28:
      case snuqasmParser::T__29:
      case snuqasmParser::T__30:
      case snuqasmParser::T__31:
      case snuqasmParser::T__32:
      case snuqasmParser::T__33:
      case snuqasmParser::T__34:
      case snuqasmParser::T__35:
      case snuqasmParser::T__36:
      case snuqasmParser::T__37:
      case snuqasmParser::T__38:
      case snuqasmParser::T__39:
      case snuqasmParser::T__40:
      case snuqasmParser::T__41:
      case snuqasmParser::T__42:
      case snuqasmParser::T__43:
      case snuqasmParser::T__44:
      case snuqasmParser::T__45:
      case snuqasmParser::T__46:
      case snuqasmParser::T__47:
      case snuqasmParser::T__48:
      case snuqasmParser::T__49:
      case snuqasmParser::T__50:
      case snuqasmParser::T__51:
      case snuqasmParser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(192);
        uopStatement();
        break;
      }

      case snuqasmParser::T__13: {
        enterOuterAlt(_localctx, 2);
        setState(193);
        measureQop();
        break;
      }

      case snuqasmParser::T__15: {
        enterOuterAlt(_localctx, 3);
        setState(194);
        resetQop();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UopStatementContext ------------------------------------------------------------------

snuqasmParser::UopStatementContext::UopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::UnitaryOpContext* snuqasmParser::UopStatementContext::unitaryOp() {
  return getRuleContext<snuqasmParser::UnitaryOpContext>(0);
}

snuqasmParser::CustomOpContext* snuqasmParser::UopStatementContext::customOp() {
  return getRuleContext<snuqasmParser::CustomOpContext>(0);
}


size_t snuqasmParser::UopStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleUopStatement;
}

void snuqasmParser::UopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUopStatement(this);
}

void snuqasmParser::UopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUopStatement(this);
}

snuqasmParser::UopStatementContext* snuqasmParser::uopStatement() {
  UopStatementContext *_localctx = _tracker.createInstance<UopStatementContext>(_ctx, getState());
  enterRule(_localctx, 24, snuqasmParser::RuleUopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(199);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__19:
      case snuqasmParser::T__20:
      case snuqasmParser::T__21:
      case snuqasmParser::T__23:
      case snuqasmParser::T__24:
      case snuqasmParser::T__25:
      case snuqasmParser::T__26:
      case snuqasmParser::T__27:
      case snuqasmParser::T__28:
      case snuqasmParser::T__29:
      case snuqasmParser::T__30:
      case snuqasmParser::T__31:
      case snuqasmParser::T__32:
      case snuqasmParser::T__33:
      case snuqasmParser::T__34:
      case snuqasmParser::T__35:
      case snuqasmParser::T__36:
      case snuqasmParser::T__37:
      case snuqasmParser::T__38:
      case snuqasmParser::T__39:
      case snuqasmParser::T__40:
      case snuqasmParser::T__41:
      case snuqasmParser::T__42:
      case snuqasmParser::T__43:
      case snuqasmParser::T__44:
      case snuqasmParser::T__45:
      case snuqasmParser::T__46:
      case snuqasmParser::T__47:
      case snuqasmParser::T__48:
      case snuqasmParser::T__49:
      case snuqasmParser::T__50:
      case snuqasmParser::T__51: {
        enterOuterAlt(_localctx, 1);
        setState(197);
        unitaryOp();
        break;
      }

      case snuqasmParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(198);
        customOp();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MeasureQopContext ------------------------------------------------------------------

snuqasmParser::MeasureQopContext::MeasureQopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<snuqasmParser::ArgumentContext *> snuqasmParser::MeasureQopContext::argument() {
  return getRuleContexts<snuqasmParser::ArgumentContext>();
}

snuqasmParser::ArgumentContext* snuqasmParser::MeasureQopContext::argument(size_t i) {
  return getRuleContext<snuqasmParser::ArgumentContext>(i);
}


size_t snuqasmParser::MeasureQopContext::getRuleIndex() const {
  return snuqasmParser::RuleMeasureQop;
}

void snuqasmParser::MeasureQopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMeasureQop(this);
}

void snuqasmParser::MeasureQopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMeasureQop(this);
}

snuqasmParser::MeasureQopContext* snuqasmParser::measureQop() {
  MeasureQopContext *_localctx = _tracker.createInstance<MeasureQopContext>(_ctx, getState());
  enterRule(_localctx, 26, snuqasmParser::RuleMeasureQop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(201);
    match(snuqasmParser::T__13);
    setState(202);
    argument();
    setState(203);
    match(snuqasmParser::T__14);
    setState(204);
    argument();
    setState(205);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ResetQopContext ------------------------------------------------------------------

snuqasmParser::ResetQopContext::ResetQopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::ArgumentContext* snuqasmParser::ResetQopContext::argument() {
  return getRuleContext<snuqasmParser::ArgumentContext>(0);
}


size_t snuqasmParser::ResetQopContext::getRuleIndex() const {
  return snuqasmParser::RuleResetQop;
}

void snuqasmParser::ResetQopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterResetQop(this);
}

void snuqasmParser::ResetQopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitResetQop(this);
}

snuqasmParser::ResetQopContext* snuqasmParser::resetQop() {
  ResetQopContext *_localctx = _tracker.createInstance<ResetQopContext>(_ctx, getState());
  enterRule(_localctx, 28, snuqasmParser::RuleResetQop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(207);
    match(snuqasmParser::T__15);
    setState(208);
    argument();
    setState(209);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfStatementContext ------------------------------------------------------------------

snuqasmParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::IfStatementContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

tree::TerminalNode* snuqasmParser::IfStatementContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}

snuqasmParser::QopStatementContext* snuqasmParser::IfStatementContext::qopStatement() {
  return getRuleContext<snuqasmParser::QopStatementContext>(0);
}


size_t snuqasmParser::IfStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleIfStatement;
}

void snuqasmParser::IfStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStatement(this);
}

void snuqasmParser::IfStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStatement(this);
}

snuqasmParser::IfStatementContext* snuqasmParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 30, snuqasmParser::RuleIfStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(211);
    match(snuqasmParser::T__16);
    setState(212);
    match(snuqasmParser::T__10);
    setState(213);
    match(snuqasmParser::Identifier);
    setState(214);
    match(snuqasmParser::T__17);
    setState(215);
    match(snuqasmParser::Integer);
    setState(216);
    match(snuqasmParser::T__11);
    setState(217);
    qopStatement();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- BarrierStatementContext ------------------------------------------------------------------

snuqasmParser::BarrierStatementContext::BarrierStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::ArglistContext* snuqasmParser::BarrierStatementContext::arglist() {
  return getRuleContext<snuqasmParser::ArglistContext>(0);
}


size_t snuqasmParser::BarrierStatementContext::getRuleIndex() const {
  return snuqasmParser::RuleBarrierStatement;
}

void snuqasmParser::BarrierStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrierStatement(this);
}

void snuqasmParser::BarrierStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrierStatement(this);
}

snuqasmParser::BarrierStatementContext* snuqasmParser::barrierStatement() {
  BarrierStatementContext *_localctx = _tracker.createInstance<BarrierStatementContext>(_ctx, getState());
  enterRule(_localctx, 32, snuqasmParser::RuleBarrierStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(219);
    match(snuqasmParser::T__18);
    setState(220);
    arglist();
    setState(221);
    match(snuqasmParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnitaryOpContext ------------------------------------------------------------------

snuqasmParser::UnitaryOpContext::UnitaryOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::ExplistContext* snuqasmParser::UnitaryOpContext::explist() {
  return getRuleContext<snuqasmParser::ExplistContext>(0);
}

std::vector<snuqasmParser::ArgumentContext *> snuqasmParser::UnitaryOpContext::argument() {
  return getRuleContexts<snuqasmParser::ArgumentContext>();
}

snuqasmParser::ArgumentContext* snuqasmParser::UnitaryOpContext::argument(size_t i) {
  return getRuleContext<snuqasmParser::ArgumentContext>(i);
}


size_t snuqasmParser::UnitaryOpContext::getRuleIndex() const {
  return snuqasmParser::RuleUnitaryOp;
}

void snuqasmParser::UnitaryOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnitaryOp(this);
}

void snuqasmParser::UnitaryOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnitaryOp(this);
}

snuqasmParser::UnitaryOpContext* snuqasmParser::unitaryOp() {
  UnitaryOpContext *_localctx = _tracker.createInstance<UnitaryOpContext>(_ctx, getState());
  enterRule(_localctx, 34, snuqasmParser::RuleUnitaryOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(421);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::T__19: {
        enterOuterAlt(_localctx, 1);
        setState(223);
        match(snuqasmParser::T__19);
        setState(224);
        match(snuqasmParser::T__10);
        setState(225);
        explist();
        setState(226);
        match(snuqasmParser::T__11);
        setState(227);
        argument();
        setState(228);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__20: {
        enterOuterAlt(_localctx, 2);
        setState(230);
        match(snuqasmParser::T__20);
        setState(231);
        match(snuqasmParser::T__10);
        setState(232);
        explist();
        setState(233);
        match(snuqasmParser::T__11);
        setState(234);
        argument();
        setState(235);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__21: {
        enterOuterAlt(_localctx, 3);
        setState(237);
        match(snuqasmParser::T__21);
        setState(238);
        argument();
        setState(239);
        match(snuqasmParser::T__22);
        setState(240);
        argument();
        setState(241);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__23: {
        enterOuterAlt(_localctx, 4);
        setState(243);
        match(snuqasmParser::T__23);
        setState(244);
        argument();
        setState(245);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__24: {
        enterOuterAlt(_localctx, 5);
        setState(247);
        match(snuqasmParser::T__24);
        setState(248);
        argument();
        setState(249);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__25: {
        enterOuterAlt(_localctx, 6);
        setState(251);
        match(snuqasmParser::T__25);
        setState(252);
        argument();
        setState(253);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__26: {
        enterOuterAlt(_localctx, 7);
        setState(255);
        match(snuqasmParser::T__26);
        setState(256);
        argument();
        setState(257);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__27: {
        enterOuterAlt(_localctx, 8);
        setState(259);
        match(snuqasmParser::T__27);
        setState(260);
        argument();
        setState(261);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__28: {
        enterOuterAlt(_localctx, 9);
        setState(263);
        match(snuqasmParser::T__28);
        setState(264);
        argument();
        setState(265);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__29: {
        enterOuterAlt(_localctx, 10);
        setState(267);
        match(snuqasmParser::T__29);
        setState(268);
        argument();
        setState(269);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__30: {
        enterOuterAlt(_localctx, 11);
        setState(271);
        match(snuqasmParser::T__30);
        setState(272);
        argument();
        setState(273);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__31: {
        enterOuterAlt(_localctx, 12);
        setState(275);
        match(snuqasmParser::T__31);
        setState(276);
        argument();
        setState(277);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__32: {
        enterOuterAlt(_localctx, 13);
        setState(279);
        match(snuqasmParser::T__32);
        setState(280);
        argument();
        setState(281);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__33: {
        enterOuterAlt(_localctx, 14);
        setState(283);
        match(snuqasmParser::T__33);
        setState(284);
        argument();
        setState(285);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__34: {
        enterOuterAlt(_localctx, 15);
        setState(287);
        match(snuqasmParser::T__34);
        setState(288);
        match(snuqasmParser::T__10);
        setState(289);
        explist();
        setState(290);
        match(snuqasmParser::T__11);
        setState(291);
        argument();
        setState(292);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__35: {
        enterOuterAlt(_localctx, 16);
        setState(294);
        match(snuqasmParser::T__35);
        setState(295);
        match(snuqasmParser::T__10);
        setState(296);
        explist();
        setState(297);
        match(snuqasmParser::T__11);
        setState(298);
        argument();
        setState(299);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__36: {
        enterOuterAlt(_localctx, 17);
        setState(301);
        match(snuqasmParser::T__36);
        setState(302);
        match(snuqasmParser::T__10);
        setState(303);
        explist();
        setState(304);
        match(snuqasmParser::T__11);
        setState(305);
        argument();
        setState(306);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__37: {
        enterOuterAlt(_localctx, 18);
        setState(308);
        match(snuqasmParser::T__37);
        setState(309);
        match(snuqasmParser::T__10);
        setState(310);
        explist();
        setState(311);
        match(snuqasmParser::T__11);
        setState(312);
        argument();
        setState(313);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__38: {
        enterOuterAlt(_localctx, 19);
        setState(315);
        match(snuqasmParser::T__38);
        setState(316);
        match(snuqasmParser::T__10);
        setState(317);
        explist();
        setState(318);
        match(snuqasmParser::T__11);
        setState(319);
        argument();
        setState(320);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__39: {
        enterOuterAlt(_localctx, 20);
        setState(322);
        match(snuqasmParser::T__39);
        setState(323);
        match(snuqasmParser::T__10);
        setState(324);
        explist();
        setState(325);
        match(snuqasmParser::T__11);
        setState(326);
        argument();
        setState(327);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__40: {
        enterOuterAlt(_localctx, 21);
        setState(329);
        match(snuqasmParser::T__40);
        setState(330);
        argument();
        setState(331);
        match(snuqasmParser::T__22);
        setState(332);
        argument();
        setState(333);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__41: {
        enterOuterAlt(_localctx, 22);
        setState(335);
        match(snuqasmParser::T__41);
        setState(336);
        argument();
        setState(337);
        match(snuqasmParser::T__22);
        setState(338);
        argument();
        setState(339);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__42: {
        enterOuterAlt(_localctx, 23);
        setState(341);
        match(snuqasmParser::T__42);
        setState(342);
        argument();
        setState(343);
        match(snuqasmParser::T__22);
        setState(344);
        argument();
        setState(345);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__43: {
        enterOuterAlt(_localctx, 24);
        setState(347);
        match(snuqasmParser::T__43);
        setState(348);
        argument();
        setState(349);
        match(snuqasmParser::T__22);
        setState(350);
        argument();
        setState(351);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__44: {
        enterOuterAlt(_localctx, 25);
        setState(353);
        match(snuqasmParser::T__44);
        setState(354);
        argument();
        setState(355);
        match(snuqasmParser::T__22);
        setState(356);
        argument();
        setState(357);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__45: {
        enterOuterAlt(_localctx, 26);
        setState(359);
        match(snuqasmParser::T__45);
        setState(360);
        match(snuqasmParser::T__10);
        setState(361);
        explist();
        setState(362);
        match(snuqasmParser::T__11);
        setState(363);
        argument();
        setState(364);
        match(snuqasmParser::T__22);
        setState(365);
        argument();
        setState(366);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__46: {
        enterOuterAlt(_localctx, 27);
        setState(368);
        match(snuqasmParser::T__46);
        setState(369);
        match(snuqasmParser::T__10);
        setState(370);
        explist();
        setState(371);
        match(snuqasmParser::T__11);
        setState(372);
        argument();
        setState(373);
        match(snuqasmParser::T__22);
        setState(374);
        argument();
        setState(375);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__47: {
        enterOuterAlt(_localctx, 28);
        setState(377);
        match(snuqasmParser::T__47);
        setState(378);
        match(snuqasmParser::T__10);
        setState(379);
        explist();
        setState(380);
        match(snuqasmParser::T__11);
        setState(381);
        argument();
        setState(382);
        match(snuqasmParser::T__22);
        setState(383);
        argument();
        setState(384);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__48: {
        enterOuterAlt(_localctx, 29);
        setState(386);
        match(snuqasmParser::T__48);
        setState(387);
        match(snuqasmParser::T__10);
        setState(388);
        explist();
        setState(389);
        match(snuqasmParser::T__11);
        setState(390);
        argument();
        setState(391);
        match(snuqasmParser::T__22);
        setState(392);
        argument();
        setState(393);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__49: {
        enterOuterAlt(_localctx, 30);
        setState(395);
        match(snuqasmParser::T__49);
        setState(396);
        match(snuqasmParser::T__10);
        setState(397);
        explist();
        setState(398);
        match(snuqasmParser::T__11);
        setState(399);
        argument();
        setState(400);
        match(snuqasmParser::T__22);
        setState(401);
        argument();
        setState(402);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__50: {
        enterOuterAlt(_localctx, 31);
        setState(404);
        match(snuqasmParser::T__50);
        setState(405);
        match(snuqasmParser::T__10);
        setState(406);
        explist();
        setState(407);
        match(snuqasmParser::T__11);
        setState(408);
        argument();
        setState(409);
        match(snuqasmParser::T__22);
        setState(410);
        argument();
        setState(411);
        match(snuqasmParser::T__1);
        break;
      }

      case snuqasmParser::T__51: {
        enterOuterAlt(_localctx, 32);
        setState(413);
        match(snuqasmParser::T__51);
        setState(414);
        argument();
        setState(415);
        match(snuqasmParser::T__22);
        setState(416);
        argument();
        setState(417);
        match(snuqasmParser::T__22);
        setState(418);
        argument();
        setState(419);
        match(snuqasmParser::T__1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CustomOpContext ------------------------------------------------------------------

snuqasmParser::CustomOpContext::CustomOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::CustomOpContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

snuqasmParser::AnylistContext* snuqasmParser::CustomOpContext::anylist() {
  return getRuleContext<snuqasmParser::AnylistContext>(0);
}

snuqasmParser::ExplistContext* snuqasmParser::CustomOpContext::explist() {
  return getRuleContext<snuqasmParser::ExplistContext>(0);
}


size_t snuqasmParser::CustomOpContext::getRuleIndex() const {
  return snuqasmParser::RuleCustomOp;
}

void snuqasmParser::CustomOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCustomOp(this);
}

void snuqasmParser::CustomOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCustomOp(this);
}

snuqasmParser::CustomOpContext* snuqasmParser::customOp() {
  CustomOpContext *_localctx = _tracker.createInstance<CustomOpContext>(_ctx, getState());
  enterRule(_localctx, 36, snuqasmParser::RuleCustomOp);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(440);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(423);
      match(snuqasmParser::Identifier);
      setState(424);
      anylist();
      setState(425);
      match(snuqasmParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(427);
      match(snuqasmParser::Identifier);
      setState(428);
      match(snuqasmParser::T__10);
      setState(429);
      match(snuqasmParser::T__11);
      setState(430);
      anylist();
      setState(431);
      match(snuqasmParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(433);
      match(snuqasmParser::Identifier);
      setState(434);
      match(snuqasmParser::T__10);
      setState(435);
      explist();
      setState(436);
      match(snuqasmParser::T__11);
      setState(437);
      anylist();
      setState(438);
      match(snuqasmParser::T__1);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- AnylistContext ------------------------------------------------------------------

snuqasmParser::AnylistContext::AnylistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::IdlistContext* snuqasmParser::AnylistContext::idlist() {
  return getRuleContext<snuqasmParser::IdlistContext>(0);
}

snuqasmParser::MixedlistContext* snuqasmParser::AnylistContext::mixedlist() {
  return getRuleContext<snuqasmParser::MixedlistContext>(0);
}


size_t snuqasmParser::AnylistContext::getRuleIndex() const {
  return snuqasmParser::RuleAnylist;
}

void snuqasmParser::AnylistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnylist(this);
}

void snuqasmParser::AnylistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnylist(this);
}

snuqasmParser::AnylistContext* snuqasmParser::anylist() {
  AnylistContext *_localctx = _tracker.createInstance<AnylistContext>(_ctx, getState());
  enterRule(_localctx, 38, snuqasmParser::RuleAnylist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(444);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(442);
      idlist();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(443);
      mixedlist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdlistContext ------------------------------------------------------------------

snuqasmParser::IdlistContext::IdlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::IdlistContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

snuqasmParser::IdlistContext* snuqasmParser::IdlistContext::idlist() {
  return getRuleContext<snuqasmParser::IdlistContext>(0);
}


size_t snuqasmParser::IdlistContext::getRuleIndex() const {
  return snuqasmParser::RuleIdlist;
}

void snuqasmParser::IdlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdlist(this);
}

void snuqasmParser::IdlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdlist(this);
}

snuqasmParser::IdlistContext* snuqasmParser::idlist() {
  IdlistContext *_localctx = _tracker.createInstance<IdlistContext>(_ctx, getState());
  enterRule(_localctx, 40, snuqasmParser::RuleIdlist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(450);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(446);
      match(snuqasmParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(447);
      match(snuqasmParser::Identifier);
      setState(448);
      match(snuqasmParser::T__22);
      setState(449);
      idlist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DesignatedIdentifierContext ------------------------------------------------------------------

snuqasmParser::DesignatedIdentifierContext::DesignatedIdentifierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::DesignatedIdentifierContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

tree::TerminalNode* snuqasmParser::DesignatedIdentifierContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}


size_t snuqasmParser::DesignatedIdentifierContext::getRuleIndex() const {
  return snuqasmParser::RuleDesignatedIdentifier;
}

void snuqasmParser::DesignatedIdentifierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDesignatedIdentifier(this);
}

void snuqasmParser::DesignatedIdentifierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDesignatedIdentifier(this);
}

snuqasmParser::DesignatedIdentifierContext* snuqasmParser::designatedIdentifier() {
  DesignatedIdentifierContext *_localctx = _tracker.createInstance<DesignatedIdentifierContext>(_ctx, getState());
  enterRule(_localctx, 42, snuqasmParser::RuleDesignatedIdentifier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(452);
    match(snuqasmParser::Identifier);
    setState(453);
    match(snuqasmParser::T__4);
    setState(454);
    match(snuqasmParser::Integer);
    setState(455);
    match(snuqasmParser::T__5);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MixedlistContext ------------------------------------------------------------------

snuqasmParser::MixedlistContext::MixedlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::MixedlistContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

snuqasmParser::DesignatedIdentifierContext* snuqasmParser::MixedlistContext::designatedIdentifier() {
  return getRuleContext<snuqasmParser::DesignatedIdentifierContext>(0);
}

snuqasmParser::MixedlistContext* snuqasmParser::MixedlistContext::mixedlist() {
  return getRuleContext<snuqasmParser::MixedlistContext>(0);
}


size_t snuqasmParser::MixedlistContext::getRuleIndex() const {
  return snuqasmParser::RuleMixedlist;
}

void snuqasmParser::MixedlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMixedlist(this);
}

void snuqasmParser::MixedlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMixedlist(this);
}

snuqasmParser::MixedlistContext* snuqasmParser::mixedlist() {
  MixedlistContext *_localctx = _tracker.createInstance<MixedlistContext>(_ctx, getState());
  enterRule(_localctx, 44, snuqasmParser::RuleMixedlist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(466);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(457);
      match(snuqasmParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(458);
      designatedIdentifier();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(459);
      match(snuqasmParser::Identifier);
      setState(460);
      match(snuqasmParser::T__22);
      setState(461);
      mixedlist();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(462);
      designatedIdentifier();
      setState(463);
      match(snuqasmParser::T__22);
      setState(464);
      mixedlist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArglistContext ------------------------------------------------------------------

snuqasmParser::ArglistContext::ArglistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::ArgumentContext* snuqasmParser::ArglistContext::argument() {
  return getRuleContext<snuqasmParser::ArgumentContext>(0);
}

snuqasmParser::ArglistContext* snuqasmParser::ArglistContext::arglist() {
  return getRuleContext<snuqasmParser::ArglistContext>(0);
}


size_t snuqasmParser::ArglistContext::getRuleIndex() const {
  return snuqasmParser::RuleArglist;
}

void snuqasmParser::ArglistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArglist(this);
}

void snuqasmParser::ArglistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArglist(this);
}

snuqasmParser::ArglistContext* snuqasmParser::arglist() {
  ArglistContext *_localctx = _tracker.createInstance<ArglistContext>(_ctx, getState());
  enterRule(_localctx, 46, snuqasmParser::RuleArglist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(473);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(468);
      argument();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(469);
      argument();
      setState(470);
      match(snuqasmParser::T__22);
      setState(471);
      arglist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArgumentContext ------------------------------------------------------------------

snuqasmParser::ArgumentContext::ArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::ArgumentContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

tree::TerminalNode* snuqasmParser::ArgumentContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}


size_t snuqasmParser::ArgumentContext::getRuleIndex() const {
  return snuqasmParser::RuleArgument;
}

void snuqasmParser::ArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArgument(this);
}

void snuqasmParser::ArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArgument(this);
}

snuqasmParser::ArgumentContext* snuqasmParser::argument() {
  ArgumentContext *_localctx = _tracker.createInstance<ArgumentContext>(_ctx, getState());
  enterRule(_localctx, 48, snuqasmParser::RuleArgument);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(480);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(475);
      match(snuqasmParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(476);
      match(snuqasmParser::Identifier);
      setState(477);
      match(snuqasmParser::T__4);
      setState(478);
      match(snuqasmParser::Integer);
      setState(479);
      match(snuqasmParser::T__5);
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExplistContext ------------------------------------------------------------------

snuqasmParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqasmParser::ExpContext* snuqasmParser::ExplistContext::exp() {
  return getRuleContext<snuqasmParser::ExpContext>(0);
}

snuqasmParser::ExplistContext* snuqasmParser::ExplistContext::explist() {
  return getRuleContext<snuqasmParser::ExplistContext>(0);
}


size_t snuqasmParser::ExplistContext::getRuleIndex() const {
  return snuqasmParser::RuleExplist;
}

void snuqasmParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void snuqasmParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}

snuqasmParser::ExplistContext* snuqasmParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 50, snuqasmParser::RuleExplist);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(487);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(482);
      exp(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(483);
      exp(0);
      setState(484);
      match(snuqasmParser::T__22);
      setState(485);
      explist();
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpContext ------------------------------------------------------------------

snuqasmParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqasmParser::ExpContext::Real() {
  return getToken(snuqasmParser::Real, 0);
}

tree::TerminalNode* snuqasmParser::ExpContext::Integer() {
  return getToken(snuqasmParser::Integer, 0);
}

tree::TerminalNode* snuqasmParser::ExpContext::Identifier() {
  return getToken(snuqasmParser::Identifier, 0);
}

snuqasmParser::UnaryopContext* snuqasmParser::ExpContext::unaryop() {
  return getRuleContext<snuqasmParser::UnaryopContext>(0);
}

std::vector<snuqasmParser::ExpContext *> snuqasmParser::ExpContext::exp() {
  return getRuleContexts<snuqasmParser::ExpContext>();
}

snuqasmParser::ExpContext* snuqasmParser::ExpContext::exp(size_t i) {
  return getRuleContext<snuqasmParser::ExpContext>(i);
}

snuqasmParser::BinopContext* snuqasmParser::ExpContext::binop() {
  return getRuleContext<snuqasmParser::BinopContext>(0);
}


size_t snuqasmParser::ExpContext::getRuleIndex() const {
  return snuqasmParser::RuleExp;
}

void snuqasmParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void snuqasmParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


snuqasmParser::ExpContext* snuqasmParser::exp() {
   return exp(0);
}

snuqasmParser::ExpContext* snuqasmParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  snuqasmParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  snuqasmParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 52;
  enterRecursionRule(_localctx, 52, snuqasmParser::RuleExp, precedence);

    

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(505);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqasmParser::Real: {
        setState(490);
        match(snuqasmParser::Real);
        break;
      }

      case snuqasmParser::Integer: {
        setState(491);
        match(snuqasmParser::Integer);
        break;
      }

      case snuqasmParser::T__52: {
        setState(492);
        match(snuqasmParser::T__52);
        break;
      }

      case snuqasmParser::Identifier: {
        setState(493);
        match(snuqasmParser::Identifier);
        break;
      }

      case snuqasmParser::T__58:
      case snuqasmParser::T__59:
      case snuqasmParser::T__60:
      case snuqasmParser::T__61:
      case snuqasmParser::T__62:
      case snuqasmParser::T__63: {
        setState(494);
        unaryop();
        setState(495);
        match(snuqasmParser::T__10);
        setState(496);
        exp(0);
        setState(497);
        match(snuqasmParser::T__11);
        break;
      }

      case snuqasmParser::T__10: {
        setState(499);
        match(snuqasmParser::T__10);
        setState(500);
        exp(0);
        setState(501);
        match(snuqasmParser::T__11);
        break;
      }

      case snuqasmParser::T__53: {
        setState(503);
        match(snuqasmParser::T__53);
        setState(504);
        exp(1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(513);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExp);
        setState(507);

        if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
        setState(508);
        binop();
        setState(509);
        exp(5); 
      }
      setState(515);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- BinopContext ------------------------------------------------------------------

snuqasmParser::BinopContext::BinopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t snuqasmParser::BinopContext::getRuleIndex() const {
  return snuqasmParser::RuleBinop;
}

void snuqasmParser::BinopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinop(this);
}

void snuqasmParser::BinopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinop(this);
}

snuqasmParser::BinopContext* snuqasmParser::binop() {
  BinopContext *_localctx = _tracker.createInstance<BinopContext>(_ctx, getState());
  enterRule(_localctx, 54, snuqasmParser::RuleBinop);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(516);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 558446353793941504) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryopContext ------------------------------------------------------------------

snuqasmParser::UnaryopContext::UnaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t snuqasmParser::UnaryopContext::getRuleIndex() const {
  return snuqasmParser::RuleUnaryop;
}

void snuqasmParser::UnaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryop(this);
}

void snuqasmParser::UnaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryop(this);
}

snuqasmParser::UnaryopContext* snuqasmParser::unaryop() {
  UnaryopContext *_localctx = _tracker.createInstance<UnaryopContext>(_ctx, getState());
  enterRule(_localctx, 56, snuqasmParser::RuleUnaryop);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(518);
    _la = _input->LA(1);
    if (!(((((_la - 59) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 59)) & 63) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool snuqasmParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 2: return programSempred(antlrcpp::downCast<ProgramContext *>(context), predicateIndex);
    case 26: return expSempred(antlrcpp::downCast<ExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool snuqasmParser::programSempred(ProgramContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 2);
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool snuqasmParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 4);

  default:
    break;
  }
  return true;
}

void snuqasmParser::initialize() {
  ::antlr4::internal::call_once(snuqasmParserOnceFlag, snuqasmParserInitialize);
}
