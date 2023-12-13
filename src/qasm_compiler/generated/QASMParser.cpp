
// Generated from QASM.g4 by ANTLR 4.13.1


#include "QASMListener.h"

#include "QASMParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct QASMParserStaticData final {
  QASMParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  QASMParserStaticData(const QASMParserStaticData&) = delete;
  QASMParserStaticData(QASMParserStaticData&&) = delete;
  QASMParserStaticData& operator=(const QASMParserStaticData&) = delete;
  QASMParserStaticData& operator=(QASMParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag qasmParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
QASMParserStaticData *qasmParserStaticData = nullptr;

void qasmParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (qasmParserStaticData != nullptr) {
    return;
  }
#else
  assert(qasmParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<QASMParserStaticData>(
    std::vector<std::string>{
      "mainprogram", "version", "program", "statement", "declStatement", 
      "regDeclStatement", "qregDeclStatement", "cregDeclStatement", "gateDeclStatement", 
      "opaqueStatement", "gateStatement", "goplist", "gop", "gopUGate", 
      "gopCXGate", "gopBarrier", "gopCustomGate", "gopReset", "idlist", 
      "paramlist", "qopStatement", "qopUGate", "qopCXGate", "qopMeasure", 
      "qopReset", "qopCustomGate", "ifStatement", "barrierStatement", "arglist", 
      "qarg", "carg", "explist", "exp", "complex", "addsub", "binop", "negop", 
      "unaryop"
    },
    std::vector<std::string>{
      "", "'OPENQASM'", "';'", "'qreg'", "'['", "']'", "'creg'", "'opaque'", 
      "'('", "')'", "'gate'", "'{'", "'}'", "'U'", "'CX'", "','", "'barrier'", 
      "'reset'", "'measure'", "'->'", "'if'", "'=='", "'pi'", "'j'", "'+'", 
      "'-'", "'*'", "'/'", "'sin'", "'cos'", "'tan'", "'exp'", "'ln'", "'sqrt'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
      "ID", "NNINTEGER", "REAL", "STRING", "Whitespace", "Newline", "LineComment", 
      "BlockComment"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,41,423,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,2,14,7,
  	14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,7,20,2,21,7,
  	21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,2,27,7,27,2,28,7,
  	28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,7,33,2,34,7,34,2,35,7,
  	35,2,36,7,36,2,37,7,37,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,2,1,2,1,2,1,2,1,
  	2,5,2,89,8,2,10,2,12,2,92,9,2,1,3,1,3,1,3,1,3,3,3,98,8,3,1,4,1,4,3,4,
  	102,8,4,1,5,1,5,3,5,106,8,5,1,6,1,6,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,1,
  	7,1,7,1,7,1,7,1,8,1,8,3,8,124,8,8,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,
  	1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,1,9,3,9,146,8,9,1,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,
  	1,10,1,10,1,10,1,10,3,10,197,8,10,1,11,1,11,1,11,5,11,202,8,11,10,11,
  	12,11,205,9,11,3,11,207,8,11,1,12,1,12,1,12,1,12,1,12,3,12,214,8,12,1,
  	13,1,13,1,13,1,13,1,13,1,13,1,13,1,14,1,14,1,14,1,14,1,14,1,14,1,15,1,
  	15,1,15,1,15,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,
  	16,1,16,1,16,1,16,1,16,1,16,3,16,250,8,16,1,17,1,17,1,17,1,17,1,18,1,
  	18,1,18,1,18,5,18,260,8,18,10,18,12,18,263,9,18,3,18,265,8,18,1,19,1,
  	19,1,19,1,19,5,19,271,8,19,10,19,12,19,274,9,19,3,19,276,8,19,1,20,1,
  	20,1,20,1,20,1,20,3,20,283,8,20,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,
  	22,1,22,1,22,1,22,1,22,1,22,1,23,1,23,1,23,1,23,1,23,1,23,1,24,1,24,1,
  	24,1,24,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,
  	25,1,25,1,25,1,25,1,25,3,25,325,8,25,1,26,1,26,1,26,1,26,1,26,1,26,1,
  	26,1,26,1,27,1,27,1,27,1,27,1,28,1,28,1,28,1,28,5,28,343,8,28,10,28,12,
  	28,346,9,28,3,28,348,8,28,1,29,1,29,1,29,1,29,1,29,3,29,355,8,29,1,30,
  	1,30,1,30,1,30,1,30,3,30,362,8,30,1,31,1,31,1,31,1,31,5,31,368,8,31,10,
  	31,12,31,371,9,31,3,31,373,8,31,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,
  	32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,1,32,3,32,393,8,32,1,
  	32,1,32,1,32,1,32,5,32,399,8,32,10,32,12,32,402,9,32,1,33,1,33,1,33,1,
  	33,1,33,1,33,1,33,1,33,1,33,3,33,413,8,33,1,34,1,34,1,35,1,35,1,36,1,
  	36,1,37,1,37,1,37,0,2,4,64,38,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,
  	30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,
  	0,3,1,0,24,25,1,0,24,27,1,0,28,33,431,0,76,1,0,0,0,2,79,1,0,0,0,4,83,
  	1,0,0,0,6,97,1,0,0,0,8,101,1,0,0,0,10,105,1,0,0,0,12,107,1,0,0,0,14,114,
  	1,0,0,0,16,123,1,0,0,0,18,145,1,0,0,0,20,196,1,0,0,0,22,206,1,0,0,0,24,
  	213,1,0,0,0,26,215,1,0,0,0,28,222,1,0,0,0,30,228,1,0,0,0,32,249,1,0,0,
  	0,34,251,1,0,0,0,36,264,1,0,0,0,38,275,1,0,0,0,40,282,1,0,0,0,42,284,
  	1,0,0,0,44,291,1,0,0,0,46,297,1,0,0,0,48,303,1,0,0,0,50,324,1,0,0,0,52,
  	326,1,0,0,0,54,334,1,0,0,0,56,347,1,0,0,0,58,354,1,0,0,0,60,361,1,0,0,
  	0,62,372,1,0,0,0,64,392,1,0,0,0,66,412,1,0,0,0,68,414,1,0,0,0,70,416,
  	1,0,0,0,72,418,1,0,0,0,74,420,1,0,0,0,76,77,3,2,1,0,77,78,3,4,2,0,78,
  	1,1,0,0,0,79,80,5,1,0,0,80,81,5,36,0,0,81,82,5,2,0,0,82,3,1,0,0,0,83,
  	84,6,2,-1,0,84,85,3,6,3,0,85,90,1,0,0,0,86,87,10,1,0,0,87,89,3,6,3,0,
  	88,86,1,0,0,0,89,92,1,0,0,0,90,88,1,0,0,0,90,91,1,0,0,0,91,5,1,0,0,0,
  	92,90,1,0,0,0,93,98,3,8,4,0,94,98,3,40,20,0,95,98,3,52,26,0,96,98,3,54,
  	27,0,97,93,1,0,0,0,97,94,1,0,0,0,97,95,1,0,0,0,97,96,1,0,0,0,98,7,1,0,
  	0,0,99,102,3,10,5,0,100,102,3,16,8,0,101,99,1,0,0,0,101,100,1,0,0,0,102,
  	9,1,0,0,0,103,106,3,12,6,0,104,106,3,14,7,0,105,103,1,0,0,0,105,104,1,
  	0,0,0,106,11,1,0,0,0,107,108,5,3,0,0,108,109,5,34,0,0,109,110,5,4,0,0,
  	110,111,5,35,0,0,111,112,5,5,0,0,112,113,5,2,0,0,113,13,1,0,0,0,114,115,
  	5,6,0,0,115,116,5,34,0,0,116,117,5,4,0,0,117,118,5,35,0,0,118,119,5,5,
  	0,0,119,120,5,2,0,0,120,15,1,0,0,0,121,124,3,18,9,0,122,124,3,20,10,0,
  	123,121,1,0,0,0,123,122,1,0,0,0,124,17,1,0,0,0,125,126,5,7,0,0,126,127,
  	5,34,0,0,127,128,3,36,18,0,128,129,5,2,0,0,129,146,1,0,0,0,130,131,5,
  	7,0,0,131,132,5,34,0,0,132,133,5,8,0,0,133,134,5,9,0,0,134,135,3,36,18,
  	0,135,136,5,2,0,0,136,146,1,0,0,0,137,138,5,7,0,0,138,139,5,34,0,0,139,
  	140,5,8,0,0,140,141,3,38,19,0,141,142,5,9,0,0,142,143,3,36,18,0,143,144,
  	5,2,0,0,144,146,1,0,0,0,145,125,1,0,0,0,145,130,1,0,0,0,145,137,1,0,0,
  	0,146,19,1,0,0,0,147,148,5,10,0,0,148,149,5,34,0,0,149,150,3,36,18,0,
  	150,151,5,11,0,0,151,152,5,12,0,0,152,197,1,0,0,0,153,154,5,10,0,0,154,
  	155,5,34,0,0,155,156,5,8,0,0,156,157,5,9,0,0,157,158,3,36,18,0,158,159,
  	5,11,0,0,159,160,5,12,0,0,160,197,1,0,0,0,161,162,5,10,0,0,162,163,5,
  	34,0,0,163,164,5,8,0,0,164,165,3,38,19,0,165,166,5,9,0,0,166,167,3,36,
  	18,0,167,168,5,11,0,0,168,169,5,12,0,0,169,197,1,0,0,0,170,171,5,10,0,
  	0,171,172,5,34,0,0,172,173,3,36,18,0,173,174,5,11,0,0,174,175,3,22,11,
  	0,175,176,5,12,0,0,176,197,1,0,0,0,177,178,5,10,0,0,178,179,5,34,0,0,
  	179,180,5,8,0,0,180,181,5,9,0,0,181,182,3,36,18,0,182,183,5,11,0,0,183,
  	184,3,22,11,0,184,185,5,12,0,0,185,197,1,0,0,0,186,187,5,10,0,0,187,188,
  	5,34,0,0,188,189,5,8,0,0,189,190,3,38,19,0,190,191,5,9,0,0,191,192,3,
  	36,18,0,192,193,5,11,0,0,193,194,3,22,11,0,194,195,5,12,0,0,195,197,1,
  	0,0,0,196,147,1,0,0,0,196,153,1,0,0,0,196,161,1,0,0,0,196,170,1,0,0,0,
  	196,177,1,0,0,0,196,186,1,0,0,0,197,21,1,0,0,0,198,207,3,24,12,0,199,
  	203,3,24,12,0,200,202,3,24,12,0,201,200,1,0,0,0,202,205,1,0,0,0,203,201,
  	1,0,0,0,203,204,1,0,0,0,204,207,1,0,0,0,205,203,1,0,0,0,206,198,1,0,0,
  	0,206,199,1,0,0,0,207,23,1,0,0,0,208,214,3,26,13,0,209,214,3,28,14,0,
  	210,214,3,30,15,0,211,214,3,32,16,0,212,214,3,34,17,0,213,208,1,0,0,0,
  	213,209,1,0,0,0,213,210,1,0,0,0,213,211,1,0,0,0,213,212,1,0,0,0,214,25,
  	1,0,0,0,215,216,5,13,0,0,216,217,5,8,0,0,217,218,3,62,31,0,218,219,5,
  	9,0,0,219,220,5,34,0,0,220,221,5,2,0,0,221,27,1,0,0,0,222,223,5,14,0,
  	0,223,224,5,34,0,0,224,225,5,15,0,0,225,226,5,34,0,0,226,227,5,2,0,0,
  	227,29,1,0,0,0,228,229,5,16,0,0,229,230,3,36,18,0,230,231,5,2,0,0,231,
  	31,1,0,0,0,232,233,5,34,0,0,233,234,3,36,18,0,234,235,5,2,0,0,235,250,
  	1,0,0,0,236,237,5,34,0,0,237,238,5,8,0,0,238,239,5,9,0,0,239,240,3,36,
  	18,0,240,241,5,2,0,0,241,250,1,0,0,0,242,243,5,34,0,0,243,244,5,8,0,0,
  	244,245,3,62,31,0,245,246,5,9,0,0,246,247,3,36,18,0,247,248,5,2,0,0,248,
  	250,1,0,0,0,249,232,1,0,0,0,249,236,1,0,0,0,249,242,1,0,0,0,250,33,1,
  	0,0,0,251,252,5,17,0,0,252,253,5,34,0,0,253,254,5,2,0,0,254,35,1,0,0,
  	0,255,265,5,34,0,0,256,261,5,34,0,0,257,258,5,15,0,0,258,260,5,34,0,0,
  	259,257,1,0,0,0,260,263,1,0,0,0,261,259,1,0,0,0,261,262,1,0,0,0,262,265,
  	1,0,0,0,263,261,1,0,0,0,264,255,1,0,0,0,264,256,1,0,0,0,265,37,1,0,0,
  	0,266,276,5,34,0,0,267,272,5,34,0,0,268,269,5,15,0,0,269,271,5,34,0,0,
  	270,268,1,0,0,0,271,274,1,0,0,0,272,270,1,0,0,0,272,273,1,0,0,0,273,276,
  	1,0,0,0,274,272,1,0,0,0,275,266,1,0,0,0,275,267,1,0,0,0,276,39,1,0,0,
  	0,277,283,3,42,21,0,278,283,3,44,22,0,279,283,3,46,23,0,280,283,3,48,
  	24,0,281,283,3,50,25,0,282,277,1,0,0,0,282,278,1,0,0,0,282,279,1,0,0,
  	0,282,280,1,0,0,0,282,281,1,0,0,0,283,41,1,0,0,0,284,285,5,13,0,0,285,
  	286,5,8,0,0,286,287,3,62,31,0,287,288,5,9,0,0,288,289,3,58,29,0,289,290,
  	5,2,0,0,290,43,1,0,0,0,291,292,5,14,0,0,292,293,3,58,29,0,293,294,5,15,
  	0,0,294,295,3,58,29,0,295,296,5,2,0,0,296,45,1,0,0,0,297,298,5,18,0,0,
  	298,299,3,58,29,0,299,300,5,19,0,0,300,301,3,60,30,0,301,302,5,2,0,0,
  	302,47,1,0,0,0,303,304,5,17,0,0,304,305,3,58,29,0,305,306,5,2,0,0,306,
  	49,1,0,0,0,307,308,5,34,0,0,308,309,3,56,28,0,309,310,5,2,0,0,310,325,
  	1,0,0,0,311,312,5,34,0,0,312,313,5,8,0,0,313,314,5,9,0,0,314,315,3,56,
  	28,0,315,316,5,2,0,0,316,325,1,0,0,0,317,318,5,34,0,0,318,319,5,8,0,0,
  	319,320,3,62,31,0,320,321,5,9,0,0,321,322,3,56,28,0,322,323,5,2,0,0,323,
  	325,1,0,0,0,324,307,1,0,0,0,324,311,1,0,0,0,324,317,1,0,0,0,325,51,1,
  	0,0,0,326,327,5,20,0,0,327,328,5,8,0,0,328,329,5,34,0,0,329,330,5,21,
  	0,0,330,331,5,35,0,0,331,332,5,9,0,0,332,333,3,40,20,0,333,53,1,0,0,0,
  	334,335,5,16,0,0,335,336,3,56,28,0,336,337,5,2,0,0,337,55,1,0,0,0,338,
  	348,3,58,29,0,339,344,3,58,29,0,340,341,5,15,0,0,341,343,3,58,29,0,342,
  	340,1,0,0,0,343,346,1,0,0,0,344,342,1,0,0,0,344,345,1,0,0,0,345,348,1,
  	0,0,0,346,344,1,0,0,0,347,338,1,0,0,0,347,339,1,0,0,0,348,57,1,0,0,0,
  	349,355,5,34,0,0,350,351,5,34,0,0,351,352,5,4,0,0,352,353,5,35,0,0,353,
  	355,5,5,0,0,354,349,1,0,0,0,354,350,1,0,0,0,355,59,1,0,0,0,356,362,5,
  	34,0,0,357,358,5,34,0,0,358,359,5,4,0,0,359,360,5,35,0,0,360,362,5,5,
  	0,0,361,356,1,0,0,0,361,357,1,0,0,0,362,61,1,0,0,0,363,373,3,64,32,0,
  	364,369,3,64,32,0,365,366,5,15,0,0,366,368,3,64,32,0,367,365,1,0,0,0,
  	368,371,1,0,0,0,369,367,1,0,0,0,369,370,1,0,0,0,370,373,1,0,0,0,371,369,
  	1,0,0,0,372,363,1,0,0,0,372,364,1,0,0,0,373,63,1,0,0,0,374,375,6,32,-1,
  	0,375,393,5,36,0,0,376,393,5,35,0,0,377,393,5,34,0,0,378,393,3,66,33,
  	0,379,380,3,72,36,0,380,381,3,64,32,4,381,393,1,0,0,0,382,383,3,74,37,
  	0,383,384,5,8,0,0,384,385,3,64,32,0,385,386,5,9,0,0,386,393,1,0,0,0,387,
  	388,5,8,0,0,388,389,3,64,32,0,389,390,5,9,0,0,390,393,1,0,0,0,391,393,
  	5,22,0,0,392,374,1,0,0,0,392,376,1,0,0,0,392,377,1,0,0,0,392,378,1,0,
  	0,0,392,379,1,0,0,0,392,382,1,0,0,0,392,387,1,0,0,0,392,391,1,0,0,0,393,
  	400,1,0,0,0,394,395,10,5,0,0,395,396,3,70,35,0,396,397,3,64,32,6,397,
  	399,1,0,0,0,398,394,1,0,0,0,399,402,1,0,0,0,400,398,1,0,0,0,400,401,1,
  	0,0,0,401,65,1,0,0,0,402,400,1,0,0,0,403,404,5,36,0,0,404,405,3,68,34,
  	0,405,406,5,36,0,0,406,407,5,23,0,0,407,413,1,0,0,0,408,409,3,68,34,0,
  	409,410,5,36,0,0,410,411,5,23,0,0,411,413,1,0,0,0,412,403,1,0,0,0,412,
  	408,1,0,0,0,413,67,1,0,0,0,414,415,7,0,0,0,415,69,1,0,0,0,416,417,7,1,
  	0,0,417,71,1,0,0,0,418,419,5,25,0,0,419,73,1,0,0,0,420,421,7,2,0,0,421,
  	75,1,0,0,0,26,90,97,101,105,123,145,196,203,206,213,249,261,264,272,275,
  	282,324,344,347,354,361,369,372,392,400,412
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  qasmParserStaticData = staticData.release();
}

}

QASMParser::QASMParser(TokenStream *input) : QASMParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

QASMParser::QASMParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  QASMParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *qasmParserStaticData->atn, qasmParserStaticData->decisionToDFA, qasmParserStaticData->sharedContextCache, options);
}

QASMParser::~QASMParser() {
  delete _interpreter;
}

const atn::ATN& QASMParser::getATN() const {
  return *qasmParserStaticData->atn;
}

std::string QASMParser::getGrammarFileName() const {
  return "QASM.g4";
}

const std::vector<std::string>& QASMParser::getRuleNames() const {
  return qasmParserStaticData->ruleNames;
}

const dfa::Vocabulary& QASMParser::getVocabulary() const {
  return qasmParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView QASMParser::getSerializedATN() const {
  return qasmParserStaticData->serializedATN;
}


//----------------- MainprogramContext ------------------------------------------------------------------

QASMParser::MainprogramContext::MainprogramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::VersionContext* QASMParser::MainprogramContext::version() {
  return getRuleContext<QASMParser::VersionContext>(0);
}

QASMParser::ProgramContext* QASMParser::MainprogramContext::program() {
  return getRuleContext<QASMParser::ProgramContext>(0);
}


size_t QASMParser::MainprogramContext::getRuleIndex() const {
  return QASMParser::RuleMainprogram;
}

void QASMParser::MainprogramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMainprogram(this);
}

void QASMParser::MainprogramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMainprogram(this);
}

QASMParser::MainprogramContext* QASMParser::mainprogram() {
  MainprogramContext *_localctx = _tracker.createInstance<MainprogramContext>(_ctx, getState());
  enterRule(_localctx, 0, QASMParser::RuleMainprogram);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(76);
    version();
    setState(77);
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

QASMParser::VersionContext::VersionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::VersionContext::REAL() {
  return getToken(QASMParser::REAL, 0);
}


size_t QASMParser::VersionContext::getRuleIndex() const {
  return QASMParser::RuleVersion;
}

void QASMParser::VersionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVersion(this);
}

void QASMParser::VersionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVersion(this);
}

QASMParser::VersionContext* QASMParser::version() {
  VersionContext *_localctx = _tracker.createInstance<VersionContext>(_ctx, getState());
  enterRule(_localctx, 2, QASMParser::RuleVersion);

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
    match(QASMParser::T__0);
    setState(80);
    match(QASMParser::REAL);
    setState(81);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProgramContext ------------------------------------------------------------------

QASMParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::StatementContext* QASMParser::ProgramContext::statement() {
  return getRuleContext<QASMParser::StatementContext>(0);
}

QASMParser::ProgramContext* QASMParser::ProgramContext::program() {
  return getRuleContext<QASMParser::ProgramContext>(0);
}


size_t QASMParser::ProgramContext::getRuleIndex() const {
  return QASMParser::RuleProgram;
}

void QASMParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void QASMParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


QASMParser::ProgramContext* QASMParser::program() {
   return program(0);
}

QASMParser::ProgramContext* QASMParser::program(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  QASMParser::ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, parentState);
  QASMParser::ProgramContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 4;
  enterRecursionRule(_localctx, 4, QASMParser::RuleProgram, precedence);

    

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
    setState(84);
    statement();
    _ctx->stop = _input->LT(-1);
    setState(90);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ProgramContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleProgram);
        setState(86);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(87);
        statement(); 
      }
      setState(92);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

QASMParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::DeclStatementContext* QASMParser::StatementContext::declStatement() {
  return getRuleContext<QASMParser::DeclStatementContext>(0);
}

QASMParser::QopStatementContext* QASMParser::StatementContext::qopStatement() {
  return getRuleContext<QASMParser::QopStatementContext>(0);
}

QASMParser::IfStatementContext* QASMParser::StatementContext::ifStatement() {
  return getRuleContext<QASMParser::IfStatementContext>(0);
}

QASMParser::BarrierStatementContext* QASMParser::StatementContext::barrierStatement() {
  return getRuleContext<QASMParser::BarrierStatementContext>(0);
}


size_t QASMParser::StatementContext::getRuleIndex() const {
  return QASMParser::RuleStatement;
}

void QASMParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void QASMParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

QASMParser::StatementContext* QASMParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 6, QASMParser::RuleStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(97);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__2:
      case QASMParser::T__5:
      case QASMParser::T__6:
      case QASMParser::T__9: {
        enterOuterAlt(_localctx, 1);
        setState(93);
        declStatement();
        break;
      }

      case QASMParser::T__12:
      case QASMParser::T__13:
      case QASMParser::T__16:
      case QASMParser::T__17:
      case QASMParser::ID: {
        enterOuterAlt(_localctx, 2);
        setState(94);
        qopStatement();
        break;
      }

      case QASMParser::T__19: {
        enterOuterAlt(_localctx, 3);
        setState(95);
        ifStatement();
        break;
      }

      case QASMParser::T__15: {
        enterOuterAlt(_localctx, 4);
        setState(96);
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

//----------------- DeclStatementContext ------------------------------------------------------------------

QASMParser::DeclStatementContext::DeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::RegDeclStatementContext* QASMParser::DeclStatementContext::regDeclStatement() {
  return getRuleContext<QASMParser::RegDeclStatementContext>(0);
}

QASMParser::GateDeclStatementContext* QASMParser::DeclStatementContext::gateDeclStatement() {
  return getRuleContext<QASMParser::GateDeclStatementContext>(0);
}


size_t QASMParser::DeclStatementContext::getRuleIndex() const {
  return QASMParser::RuleDeclStatement;
}

void QASMParser::DeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclStatement(this);
}

void QASMParser::DeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclStatement(this);
}

QASMParser::DeclStatementContext* QASMParser::declStatement() {
  DeclStatementContext *_localctx = _tracker.createInstance<DeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 8, QASMParser::RuleDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(101);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__2:
      case QASMParser::T__5: {
        enterOuterAlt(_localctx, 1);
        setState(99);
        regDeclStatement();
        break;
      }

      case QASMParser::T__6:
      case QASMParser::T__9: {
        enterOuterAlt(_localctx, 2);
        setState(100);
        gateDeclStatement();
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

//----------------- RegDeclStatementContext ------------------------------------------------------------------

QASMParser::RegDeclStatementContext::RegDeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::QregDeclStatementContext* QASMParser::RegDeclStatementContext::qregDeclStatement() {
  return getRuleContext<QASMParser::QregDeclStatementContext>(0);
}

QASMParser::CregDeclStatementContext* QASMParser::RegDeclStatementContext::cregDeclStatement() {
  return getRuleContext<QASMParser::CregDeclStatementContext>(0);
}


size_t QASMParser::RegDeclStatementContext::getRuleIndex() const {
  return QASMParser::RuleRegDeclStatement;
}

void QASMParser::RegDeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRegDeclStatement(this);
}

void QASMParser::RegDeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRegDeclStatement(this);
}

QASMParser::RegDeclStatementContext* QASMParser::regDeclStatement() {
  RegDeclStatementContext *_localctx = _tracker.createInstance<RegDeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 10, QASMParser::RuleRegDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(105);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__2: {
        enterOuterAlt(_localctx, 1);
        setState(103);
        qregDeclStatement();
        break;
      }

      case QASMParser::T__5: {
        enterOuterAlt(_localctx, 2);
        setState(104);
        cregDeclStatement();
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

//----------------- QregDeclStatementContext ------------------------------------------------------------------

QASMParser::QregDeclStatementContext::QregDeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::QregDeclStatementContext::ID() {
  return getToken(QASMParser::ID, 0);
}

tree::TerminalNode* QASMParser::QregDeclStatementContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}


size_t QASMParser::QregDeclStatementContext::getRuleIndex() const {
  return QASMParser::RuleQregDeclStatement;
}

void QASMParser::QregDeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQregDeclStatement(this);
}

void QASMParser::QregDeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQregDeclStatement(this);
}

QASMParser::QregDeclStatementContext* QASMParser::qregDeclStatement() {
  QregDeclStatementContext *_localctx = _tracker.createInstance<QregDeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 12, QASMParser::RuleQregDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(107);
    match(QASMParser::T__2);
    setState(108);
    match(QASMParser::ID);
    setState(109);
    match(QASMParser::T__3);
    setState(110);
    match(QASMParser::NNINTEGER);
    setState(111);
    match(QASMParser::T__4);
    setState(112);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- CregDeclStatementContext ------------------------------------------------------------------

QASMParser::CregDeclStatementContext::CregDeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::CregDeclStatementContext::ID() {
  return getToken(QASMParser::ID, 0);
}

tree::TerminalNode* QASMParser::CregDeclStatementContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}


size_t QASMParser::CregDeclStatementContext::getRuleIndex() const {
  return QASMParser::RuleCregDeclStatement;
}

void QASMParser::CregDeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCregDeclStatement(this);
}

void QASMParser::CregDeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCregDeclStatement(this);
}

QASMParser::CregDeclStatementContext* QASMParser::cregDeclStatement() {
  CregDeclStatementContext *_localctx = _tracker.createInstance<CregDeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 14, QASMParser::RuleCregDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(114);
    match(QASMParser::T__5);
    setState(115);
    match(QASMParser::ID);
    setState(116);
    match(QASMParser::T__3);
    setState(117);
    match(QASMParser::NNINTEGER);
    setState(118);
    match(QASMParser::T__4);
    setState(119);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GateDeclStatementContext ------------------------------------------------------------------

QASMParser::GateDeclStatementContext::GateDeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::OpaqueStatementContext* QASMParser::GateDeclStatementContext::opaqueStatement() {
  return getRuleContext<QASMParser::OpaqueStatementContext>(0);
}

QASMParser::GateStatementContext* QASMParser::GateDeclStatementContext::gateStatement() {
  return getRuleContext<QASMParser::GateStatementContext>(0);
}


size_t QASMParser::GateDeclStatementContext::getRuleIndex() const {
  return QASMParser::RuleGateDeclStatement;
}

void QASMParser::GateDeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGateDeclStatement(this);
}

void QASMParser::GateDeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGateDeclStatement(this);
}

QASMParser::GateDeclStatementContext* QASMParser::gateDeclStatement() {
  GateDeclStatementContext *_localctx = _tracker.createInstance<GateDeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 16, QASMParser::RuleGateDeclStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(123);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__6: {
        enterOuterAlt(_localctx, 1);
        setState(121);
        opaqueStatement();
        break;
      }

      case QASMParser::T__9: {
        enterOuterAlt(_localctx, 2);
        setState(122);
        gateStatement();
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

//----------------- OpaqueStatementContext ------------------------------------------------------------------

QASMParser::OpaqueStatementContext::OpaqueStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::OpaqueStatementContext::ID() {
  return getToken(QASMParser::ID, 0);
}

QASMParser::IdlistContext* QASMParser::OpaqueStatementContext::idlist() {
  return getRuleContext<QASMParser::IdlistContext>(0);
}

QASMParser::ParamlistContext* QASMParser::OpaqueStatementContext::paramlist() {
  return getRuleContext<QASMParser::ParamlistContext>(0);
}


size_t QASMParser::OpaqueStatementContext::getRuleIndex() const {
  return QASMParser::RuleOpaqueStatement;
}

void QASMParser::OpaqueStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOpaqueStatement(this);
}

void QASMParser::OpaqueStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOpaqueStatement(this);
}

QASMParser::OpaqueStatementContext* QASMParser::opaqueStatement() {
  OpaqueStatementContext *_localctx = _tracker.createInstance<OpaqueStatementContext>(_ctx, getState());
  enterRule(_localctx, 18, QASMParser::RuleOpaqueStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(145);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(125);
      match(QASMParser::T__6);
      setState(126);
      match(QASMParser::ID);
      setState(127);
      idlist();
      setState(128);
      match(QASMParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(130);
      match(QASMParser::T__6);
      setState(131);
      match(QASMParser::ID);
      setState(132);
      match(QASMParser::T__7);
      setState(133);
      match(QASMParser::T__8);
      setState(134);
      idlist();
      setState(135);
      match(QASMParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(137);
      match(QASMParser::T__6);
      setState(138);
      match(QASMParser::ID);
      setState(139);
      match(QASMParser::T__7);
      setState(140);
      paramlist();
      setState(141);
      match(QASMParser::T__8);
      setState(142);
      idlist();
      setState(143);
      match(QASMParser::T__1);
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

//----------------- GateStatementContext ------------------------------------------------------------------

QASMParser::GateStatementContext::GateStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::GateStatementContext::ID() {
  return getToken(QASMParser::ID, 0);
}

QASMParser::IdlistContext* QASMParser::GateStatementContext::idlist() {
  return getRuleContext<QASMParser::IdlistContext>(0);
}

QASMParser::ParamlistContext* QASMParser::GateStatementContext::paramlist() {
  return getRuleContext<QASMParser::ParamlistContext>(0);
}

QASMParser::GoplistContext* QASMParser::GateStatementContext::goplist() {
  return getRuleContext<QASMParser::GoplistContext>(0);
}


size_t QASMParser::GateStatementContext::getRuleIndex() const {
  return QASMParser::RuleGateStatement;
}

void QASMParser::GateStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGateStatement(this);
}

void QASMParser::GateStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGateStatement(this);
}

QASMParser::GateStatementContext* QASMParser::gateStatement() {
  GateStatementContext *_localctx = _tracker.createInstance<GateStatementContext>(_ctx, getState());
  enterRule(_localctx, 20, QASMParser::RuleGateStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(196);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(147);
      match(QASMParser::T__9);
      setState(148);
      match(QASMParser::ID);
      setState(149);
      idlist();
      setState(150);
      match(QASMParser::T__10);
      setState(151);
      match(QASMParser::T__11);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(153);
      match(QASMParser::T__9);
      setState(154);
      match(QASMParser::ID);
      setState(155);
      match(QASMParser::T__7);
      setState(156);
      match(QASMParser::T__8);
      setState(157);
      idlist();
      setState(158);
      match(QASMParser::T__10);
      setState(159);
      match(QASMParser::T__11);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(161);
      match(QASMParser::T__9);
      setState(162);
      match(QASMParser::ID);
      setState(163);
      match(QASMParser::T__7);
      setState(164);
      paramlist();
      setState(165);
      match(QASMParser::T__8);
      setState(166);
      idlist();
      setState(167);
      match(QASMParser::T__10);
      setState(168);
      match(QASMParser::T__11);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(170);
      match(QASMParser::T__9);
      setState(171);
      match(QASMParser::ID);
      setState(172);
      idlist();
      setState(173);
      match(QASMParser::T__10);
      setState(174);
      goplist();
      setState(175);
      match(QASMParser::T__11);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(177);
      match(QASMParser::T__9);
      setState(178);
      match(QASMParser::ID);
      setState(179);
      match(QASMParser::T__7);
      setState(180);
      match(QASMParser::T__8);
      setState(181);
      idlist();
      setState(182);
      match(QASMParser::T__10);
      setState(183);
      goplist();
      setState(184);
      match(QASMParser::T__11);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(186);
      match(QASMParser::T__9);
      setState(187);
      match(QASMParser::ID);
      setState(188);
      match(QASMParser::T__7);
      setState(189);
      paramlist();
      setState(190);
      match(QASMParser::T__8);
      setState(191);
      idlist();
      setState(192);
      match(QASMParser::T__10);
      setState(193);
      goplist();
      setState(194);
      match(QASMParser::T__11);
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

QASMParser::GoplistContext::GoplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<QASMParser::GopContext *> QASMParser::GoplistContext::gop() {
  return getRuleContexts<QASMParser::GopContext>();
}

QASMParser::GopContext* QASMParser::GoplistContext::gop(size_t i) {
  return getRuleContext<QASMParser::GopContext>(i);
}


size_t QASMParser::GoplistContext::getRuleIndex() const {
  return QASMParser::RuleGoplist;
}

void QASMParser::GoplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGoplist(this);
}

void QASMParser::GoplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGoplist(this);
}

QASMParser::GoplistContext* QASMParser::goplist() {
  GoplistContext *_localctx = _tracker.createInstance<GoplistContext>(_ctx, getState());
  enterRule(_localctx, 22, QASMParser::RuleGoplist);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(206);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(198);
      gop();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(199);
      gop();
      setState(203);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & 17180090368) != 0)) {
        setState(200);
        gop();
        setState(205);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

//----------------- GopContext ------------------------------------------------------------------

QASMParser::GopContext::GopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::GopUGateContext* QASMParser::GopContext::gopUGate() {
  return getRuleContext<QASMParser::GopUGateContext>(0);
}

QASMParser::GopCXGateContext* QASMParser::GopContext::gopCXGate() {
  return getRuleContext<QASMParser::GopCXGateContext>(0);
}

QASMParser::GopBarrierContext* QASMParser::GopContext::gopBarrier() {
  return getRuleContext<QASMParser::GopBarrierContext>(0);
}

QASMParser::GopCustomGateContext* QASMParser::GopContext::gopCustomGate() {
  return getRuleContext<QASMParser::GopCustomGateContext>(0);
}

QASMParser::GopResetContext* QASMParser::GopContext::gopReset() {
  return getRuleContext<QASMParser::GopResetContext>(0);
}


size_t QASMParser::GopContext::getRuleIndex() const {
  return QASMParser::RuleGop;
}

void QASMParser::GopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGop(this);
}

void QASMParser::GopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGop(this);
}

QASMParser::GopContext* QASMParser::gop() {
  GopContext *_localctx = _tracker.createInstance<GopContext>(_ctx, getState());
  enterRule(_localctx, 24, QASMParser::RuleGop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(213);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__12: {
        enterOuterAlt(_localctx, 1);
        setState(208);
        gopUGate();
        break;
      }

      case QASMParser::T__13: {
        enterOuterAlt(_localctx, 2);
        setState(209);
        gopCXGate();
        break;
      }

      case QASMParser::T__15: {
        enterOuterAlt(_localctx, 3);
        setState(210);
        gopBarrier();
        break;
      }

      case QASMParser::ID: {
        enterOuterAlt(_localctx, 4);
        setState(211);
        gopCustomGate();
        break;
      }

      case QASMParser::T__16: {
        enterOuterAlt(_localctx, 5);
        setState(212);
        gopReset();
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

//----------------- GopUGateContext ------------------------------------------------------------------

QASMParser::GopUGateContext::GopUGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::ExplistContext* QASMParser::GopUGateContext::explist() {
  return getRuleContext<QASMParser::ExplistContext>(0);
}

tree::TerminalNode* QASMParser::GopUGateContext::ID() {
  return getToken(QASMParser::ID, 0);
}


size_t QASMParser::GopUGateContext::getRuleIndex() const {
  return QASMParser::RuleGopUGate;
}

void QASMParser::GopUGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGopUGate(this);
}

void QASMParser::GopUGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGopUGate(this);
}

QASMParser::GopUGateContext* QASMParser::gopUGate() {
  GopUGateContext *_localctx = _tracker.createInstance<GopUGateContext>(_ctx, getState());
  enterRule(_localctx, 26, QASMParser::RuleGopUGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(215);
    match(QASMParser::T__12);
    setState(216);
    match(QASMParser::T__7);
    setState(217);
    explist();
    setState(218);
    match(QASMParser::T__8);
    setState(219);
    match(QASMParser::ID);
    setState(220);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GopCXGateContext ------------------------------------------------------------------

QASMParser::GopCXGateContext::GopCXGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> QASMParser::GopCXGateContext::ID() {
  return getTokens(QASMParser::ID);
}

tree::TerminalNode* QASMParser::GopCXGateContext::ID(size_t i) {
  return getToken(QASMParser::ID, i);
}


size_t QASMParser::GopCXGateContext::getRuleIndex() const {
  return QASMParser::RuleGopCXGate;
}

void QASMParser::GopCXGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGopCXGate(this);
}

void QASMParser::GopCXGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGopCXGate(this);
}

QASMParser::GopCXGateContext* QASMParser::gopCXGate() {
  GopCXGateContext *_localctx = _tracker.createInstance<GopCXGateContext>(_ctx, getState());
  enterRule(_localctx, 28, QASMParser::RuleGopCXGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(222);
    match(QASMParser::T__13);
    setState(223);
    match(QASMParser::ID);
    setState(224);
    match(QASMParser::T__14);
    setState(225);
    match(QASMParser::ID);
    setState(226);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GopBarrierContext ------------------------------------------------------------------

QASMParser::GopBarrierContext::GopBarrierContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::IdlistContext* QASMParser::GopBarrierContext::idlist() {
  return getRuleContext<QASMParser::IdlistContext>(0);
}


size_t QASMParser::GopBarrierContext::getRuleIndex() const {
  return QASMParser::RuleGopBarrier;
}

void QASMParser::GopBarrierContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGopBarrier(this);
}

void QASMParser::GopBarrierContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGopBarrier(this);
}

QASMParser::GopBarrierContext* QASMParser::gopBarrier() {
  GopBarrierContext *_localctx = _tracker.createInstance<GopBarrierContext>(_ctx, getState());
  enterRule(_localctx, 30, QASMParser::RuleGopBarrier);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(228);
    match(QASMParser::T__15);
    setState(229);
    idlist();
    setState(230);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GopCustomGateContext ------------------------------------------------------------------

QASMParser::GopCustomGateContext::GopCustomGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::GopCustomGateContext::ID() {
  return getToken(QASMParser::ID, 0);
}

QASMParser::IdlistContext* QASMParser::GopCustomGateContext::idlist() {
  return getRuleContext<QASMParser::IdlistContext>(0);
}

QASMParser::ExplistContext* QASMParser::GopCustomGateContext::explist() {
  return getRuleContext<QASMParser::ExplistContext>(0);
}


size_t QASMParser::GopCustomGateContext::getRuleIndex() const {
  return QASMParser::RuleGopCustomGate;
}

void QASMParser::GopCustomGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGopCustomGate(this);
}

void QASMParser::GopCustomGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGopCustomGate(this);
}

QASMParser::GopCustomGateContext* QASMParser::gopCustomGate() {
  GopCustomGateContext *_localctx = _tracker.createInstance<GopCustomGateContext>(_ctx, getState());
  enterRule(_localctx, 32, QASMParser::RuleGopCustomGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(249);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(232);
      match(QASMParser::ID);
      setState(233);
      idlist();
      setState(234);
      match(QASMParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(236);
      match(QASMParser::ID);
      setState(237);
      match(QASMParser::T__7);
      setState(238);
      match(QASMParser::T__8);
      setState(239);
      idlist();
      setState(240);
      match(QASMParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(242);
      match(QASMParser::ID);
      setState(243);
      match(QASMParser::T__7);
      setState(244);
      explist();
      setState(245);
      match(QASMParser::T__8);
      setState(246);
      idlist();
      setState(247);
      match(QASMParser::T__1);
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

//----------------- GopResetContext ------------------------------------------------------------------

QASMParser::GopResetContext::GopResetContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::GopResetContext::ID() {
  return getToken(QASMParser::ID, 0);
}


size_t QASMParser::GopResetContext::getRuleIndex() const {
  return QASMParser::RuleGopReset;
}

void QASMParser::GopResetContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGopReset(this);
}

void QASMParser::GopResetContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGopReset(this);
}

QASMParser::GopResetContext* QASMParser::gopReset() {
  GopResetContext *_localctx = _tracker.createInstance<GopResetContext>(_ctx, getState());
  enterRule(_localctx, 34, QASMParser::RuleGopReset);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(251);
    match(QASMParser::T__16);
    setState(252);
    match(QASMParser::ID);
    setState(253);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdlistContext ------------------------------------------------------------------

QASMParser::IdlistContext::IdlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> QASMParser::IdlistContext::ID() {
  return getTokens(QASMParser::ID);
}

tree::TerminalNode* QASMParser::IdlistContext::ID(size_t i) {
  return getToken(QASMParser::ID, i);
}


size_t QASMParser::IdlistContext::getRuleIndex() const {
  return QASMParser::RuleIdlist;
}

void QASMParser::IdlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdlist(this);
}

void QASMParser::IdlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdlist(this);
}

QASMParser::IdlistContext* QASMParser::idlist() {
  IdlistContext *_localctx = _tracker.createInstance<IdlistContext>(_ctx, getState());
  enterRule(_localctx, 36, QASMParser::RuleIdlist);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(264);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(255);
      match(QASMParser::ID);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(256);
      match(QASMParser::ID);
      setState(261);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == QASMParser::T__14) {
        setState(257);
        match(QASMParser::T__14);
        setState(258);
        match(QASMParser::ID);
        setState(263);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

//----------------- ParamlistContext ------------------------------------------------------------------

QASMParser::ParamlistContext::ParamlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> QASMParser::ParamlistContext::ID() {
  return getTokens(QASMParser::ID);
}

tree::TerminalNode* QASMParser::ParamlistContext::ID(size_t i) {
  return getToken(QASMParser::ID, i);
}


size_t QASMParser::ParamlistContext::getRuleIndex() const {
  return QASMParser::RuleParamlist;
}

void QASMParser::ParamlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParamlist(this);
}

void QASMParser::ParamlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParamlist(this);
}

QASMParser::ParamlistContext* QASMParser::paramlist() {
  ParamlistContext *_localctx = _tracker.createInstance<ParamlistContext>(_ctx, getState());
  enterRule(_localctx, 38, QASMParser::RuleParamlist);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(275);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(266);
      match(QASMParser::ID);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(267);
      match(QASMParser::ID);
      setState(272);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == QASMParser::T__14) {
        setState(268);
        match(QASMParser::T__14);
        setState(269);
        match(QASMParser::ID);
        setState(274);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

QASMParser::QopStatementContext::QopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::QopUGateContext* QASMParser::QopStatementContext::qopUGate() {
  return getRuleContext<QASMParser::QopUGateContext>(0);
}

QASMParser::QopCXGateContext* QASMParser::QopStatementContext::qopCXGate() {
  return getRuleContext<QASMParser::QopCXGateContext>(0);
}

QASMParser::QopMeasureContext* QASMParser::QopStatementContext::qopMeasure() {
  return getRuleContext<QASMParser::QopMeasureContext>(0);
}

QASMParser::QopResetContext* QASMParser::QopStatementContext::qopReset() {
  return getRuleContext<QASMParser::QopResetContext>(0);
}

QASMParser::QopCustomGateContext* QASMParser::QopStatementContext::qopCustomGate() {
  return getRuleContext<QASMParser::QopCustomGateContext>(0);
}


size_t QASMParser::QopStatementContext::getRuleIndex() const {
  return QASMParser::RuleQopStatement;
}

void QASMParser::QopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopStatement(this);
}

void QASMParser::QopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopStatement(this);
}

QASMParser::QopStatementContext* QASMParser::qopStatement() {
  QopStatementContext *_localctx = _tracker.createInstance<QopStatementContext>(_ctx, getState());
  enterRule(_localctx, 40, QASMParser::RuleQopStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(282);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::T__12: {
        enterOuterAlt(_localctx, 1);
        setState(277);
        qopUGate();
        break;
      }

      case QASMParser::T__13: {
        enterOuterAlt(_localctx, 2);
        setState(278);
        qopCXGate();
        break;
      }

      case QASMParser::T__17: {
        enterOuterAlt(_localctx, 3);
        setState(279);
        qopMeasure();
        break;
      }

      case QASMParser::T__16: {
        enterOuterAlt(_localctx, 4);
        setState(280);
        qopReset();
        break;
      }

      case QASMParser::ID: {
        enterOuterAlt(_localctx, 5);
        setState(281);
        qopCustomGate();
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

//----------------- QopUGateContext ------------------------------------------------------------------

QASMParser::QopUGateContext::QopUGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::ExplistContext* QASMParser::QopUGateContext::explist() {
  return getRuleContext<QASMParser::ExplistContext>(0);
}

QASMParser::QargContext* QASMParser::QopUGateContext::qarg() {
  return getRuleContext<QASMParser::QargContext>(0);
}


size_t QASMParser::QopUGateContext::getRuleIndex() const {
  return QASMParser::RuleQopUGate;
}

void QASMParser::QopUGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopUGate(this);
}

void QASMParser::QopUGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopUGate(this);
}

QASMParser::QopUGateContext* QASMParser::qopUGate() {
  QopUGateContext *_localctx = _tracker.createInstance<QopUGateContext>(_ctx, getState());
  enterRule(_localctx, 42, QASMParser::RuleQopUGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(284);
    match(QASMParser::T__12);
    setState(285);
    match(QASMParser::T__7);
    setState(286);
    explist();
    setState(287);
    match(QASMParser::T__8);
    setState(288);
    qarg();
    setState(289);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopCXGateContext ------------------------------------------------------------------

QASMParser::QopCXGateContext::QopCXGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<QASMParser::QargContext *> QASMParser::QopCXGateContext::qarg() {
  return getRuleContexts<QASMParser::QargContext>();
}

QASMParser::QargContext* QASMParser::QopCXGateContext::qarg(size_t i) {
  return getRuleContext<QASMParser::QargContext>(i);
}


size_t QASMParser::QopCXGateContext::getRuleIndex() const {
  return QASMParser::RuleQopCXGate;
}

void QASMParser::QopCXGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopCXGate(this);
}

void QASMParser::QopCXGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopCXGate(this);
}

QASMParser::QopCXGateContext* QASMParser::qopCXGate() {
  QopCXGateContext *_localctx = _tracker.createInstance<QopCXGateContext>(_ctx, getState());
  enterRule(_localctx, 44, QASMParser::RuleQopCXGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(291);
    match(QASMParser::T__13);
    setState(292);
    qarg();
    setState(293);
    match(QASMParser::T__14);
    setState(294);
    qarg();
    setState(295);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopMeasureContext ------------------------------------------------------------------

QASMParser::QopMeasureContext::QopMeasureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::QargContext* QASMParser::QopMeasureContext::qarg() {
  return getRuleContext<QASMParser::QargContext>(0);
}

QASMParser::CargContext* QASMParser::QopMeasureContext::carg() {
  return getRuleContext<QASMParser::CargContext>(0);
}


size_t QASMParser::QopMeasureContext::getRuleIndex() const {
  return QASMParser::RuleQopMeasure;
}

void QASMParser::QopMeasureContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopMeasure(this);
}

void QASMParser::QopMeasureContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopMeasure(this);
}

QASMParser::QopMeasureContext* QASMParser::qopMeasure() {
  QopMeasureContext *_localctx = _tracker.createInstance<QopMeasureContext>(_ctx, getState());
  enterRule(_localctx, 46, QASMParser::RuleQopMeasure);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(297);
    match(QASMParser::T__17);
    setState(298);
    qarg();
    setState(299);
    match(QASMParser::T__18);
    setState(300);
    carg();
    setState(301);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopResetContext ------------------------------------------------------------------

QASMParser::QopResetContext::QopResetContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::QargContext* QASMParser::QopResetContext::qarg() {
  return getRuleContext<QASMParser::QargContext>(0);
}


size_t QASMParser::QopResetContext::getRuleIndex() const {
  return QASMParser::RuleQopReset;
}

void QASMParser::QopResetContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopReset(this);
}

void QASMParser::QopResetContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopReset(this);
}

QASMParser::QopResetContext* QASMParser::qopReset() {
  QopResetContext *_localctx = _tracker.createInstance<QopResetContext>(_ctx, getState());
  enterRule(_localctx, 48, QASMParser::RuleQopReset);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(303);
    match(QASMParser::T__16);
    setState(304);
    qarg();
    setState(305);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopCustomGateContext ------------------------------------------------------------------

QASMParser::QopCustomGateContext::QopCustomGateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::QopCustomGateContext::ID() {
  return getToken(QASMParser::ID, 0);
}

QASMParser::ArglistContext* QASMParser::QopCustomGateContext::arglist() {
  return getRuleContext<QASMParser::ArglistContext>(0);
}

QASMParser::ExplistContext* QASMParser::QopCustomGateContext::explist() {
  return getRuleContext<QASMParser::ExplistContext>(0);
}


size_t QASMParser::QopCustomGateContext::getRuleIndex() const {
  return QASMParser::RuleQopCustomGate;
}

void QASMParser::QopCustomGateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopCustomGate(this);
}

void QASMParser::QopCustomGateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopCustomGate(this);
}

QASMParser::QopCustomGateContext* QASMParser::qopCustomGate() {
  QopCustomGateContext *_localctx = _tracker.createInstance<QopCustomGateContext>(_ctx, getState());
  enterRule(_localctx, 50, QASMParser::RuleQopCustomGate);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(324);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(307);
      match(QASMParser::ID);
      setState(308);
      arglist();
      setState(309);
      match(QASMParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(311);
      match(QASMParser::ID);
      setState(312);
      match(QASMParser::T__7);
      setState(313);
      match(QASMParser::T__8);
      setState(314);
      arglist();
      setState(315);
      match(QASMParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(317);
      match(QASMParser::ID);
      setState(318);
      match(QASMParser::T__7);
      setState(319);
      explist();
      setState(320);
      match(QASMParser::T__8);
      setState(321);
      arglist();
      setState(322);
      match(QASMParser::T__1);
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

//----------------- IfStatementContext ------------------------------------------------------------------

QASMParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::IfStatementContext::ID() {
  return getToken(QASMParser::ID, 0);
}

tree::TerminalNode* QASMParser::IfStatementContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}

QASMParser::QopStatementContext* QASMParser::IfStatementContext::qopStatement() {
  return getRuleContext<QASMParser::QopStatementContext>(0);
}


size_t QASMParser::IfStatementContext::getRuleIndex() const {
  return QASMParser::RuleIfStatement;
}

void QASMParser::IfStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStatement(this);
}

void QASMParser::IfStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStatement(this);
}

QASMParser::IfStatementContext* QASMParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 52, QASMParser::RuleIfStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(326);
    match(QASMParser::T__19);
    setState(327);
    match(QASMParser::T__7);
    setState(328);
    match(QASMParser::ID);
    setState(329);
    match(QASMParser::T__20);
    setState(330);
    match(QASMParser::NNINTEGER);
    setState(331);
    match(QASMParser::T__8);
    setState(332);
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

QASMParser::BarrierStatementContext::BarrierStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

QASMParser::ArglistContext* QASMParser::BarrierStatementContext::arglist() {
  return getRuleContext<QASMParser::ArglistContext>(0);
}


size_t QASMParser::BarrierStatementContext::getRuleIndex() const {
  return QASMParser::RuleBarrierStatement;
}

void QASMParser::BarrierStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrierStatement(this);
}

void QASMParser::BarrierStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrierStatement(this);
}

QASMParser::BarrierStatementContext* QASMParser::barrierStatement() {
  BarrierStatementContext *_localctx = _tracker.createInstance<BarrierStatementContext>(_ctx, getState());
  enterRule(_localctx, 54, QASMParser::RuleBarrierStatement);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(334);
    match(QASMParser::T__15);
    setState(335);
    arglist();
    setState(336);
    match(QASMParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArglistContext ------------------------------------------------------------------

QASMParser::ArglistContext::ArglistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<QASMParser::QargContext *> QASMParser::ArglistContext::qarg() {
  return getRuleContexts<QASMParser::QargContext>();
}

QASMParser::QargContext* QASMParser::ArglistContext::qarg(size_t i) {
  return getRuleContext<QASMParser::QargContext>(i);
}


size_t QASMParser::ArglistContext::getRuleIndex() const {
  return QASMParser::RuleArglist;
}

void QASMParser::ArglistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArglist(this);
}

void QASMParser::ArglistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArglist(this);
}

QASMParser::ArglistContext* QASMParser::arglist() {
  ArglistContext *_localctx = _tracker.createInstance<ArglistContext>(_ctx, getState());
  enterRule(_localctx, 56, QASMParser::RuleArglist);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(347);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(338);
      qarg();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(339);
      qarg();
      setState(344);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == QASMParser::T__14) {
        setState(340);
        match(QASMParser::T__14);
        setState(341);
        qarg();
        setState(346);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

//----------------- QargContext ------------------------------------------------------------------

QASMParser::QargContext::QargContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::QargContext::ID() {
  return getToken(QASMParser::ID, 0);
}

tree::TerminalNode* QASMParser::QargContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}


size_t QASMParser::QargContext::getRuleIndex() const {
  return QASMParser::RuleQarg;
}

void QASMParser::QargContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQarg(this);
}

void QASMParser::QargContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQarg(this);
}

QASMParser::QargContext* QASMParser::qarg() {
  QargContext *_localctx = _tracker.createInstance<QargContext>(_ctx, getState());
  enterRule(_localctx, 58, QASMParser::RuleQarg);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(354);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(349);
      match(QASMParser::ID);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(350);
      match(QASMParser::ID);
      setState(351);
      match(QASMParser::T__3);
      setState(352);
      match(QASMParser::NNINTEGER);
      setState(353);
      match(QASMParser::T__4);
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

//----------------- CargContext ------------------------------------------------------------------

QASMParser::CargContext::CargContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::CargContext::ID() {
  return getToken(QASMParser::ID, 0);
}

tree::TerminalNode* QASMParser::CargContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}


size_t QASMParser::CargContext::getRuleIndex() const {
  return QASMParser::RuleCarg;
}

void QASMParser::CargContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCarg(this);
}

void QASMParser::CargContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCarg(this);
}

QASMParser::CargContext* QASMParser::carg() {
  CargContext *_localctx = _tracker.createInstance<CargContext>(_ctx, getState());
  enterRule(_localctx, 60, QASMParser::RuleCarg);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(361);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(356);
      match(QASMParser::ID);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(357);
      match(QASMParser::ID);
      setState(358);
      match(QASMParser::T__3);
      setState(359);
      match(QASMParser::NNINTEGER);
      setState(360);
      match(QASMParser::T__4);
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

QASMParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<QASMParser::ExpContext *> QASMParser::ExplistContext::exp() {
  return getRuleContexts<QASMParser::ExpContext>();
}

QASMParser::ExpContext* QASMParser::ExplistContext::exp(size_t i) {
  return getRuleContext<QASMParser::ExpContext>(i);
}


size_t QASMParser::ExplistContext::getRuleIndex() const {
  return QASMParser::RuleExplist;
}

void QASMParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void QASMParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}

QASMParser::ExplistContext* QASMParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 62, QASMParser::RuleExplist);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(372);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(363);
      exp(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(364);
      exp(0);
      setState(369);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == QASMParser::T__14) {
        setState(365);
        match(QASMParser::T__14);
        setState(366);
        exp(0);
        setState(371);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

QASMParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* QASMParser::ExpContext::REAL() {
  return getToken(QASMParser::REAL, 0);
}

tree::TerminalNode* QASMParser::ExpContext::NNINTEGER() {
  return getToken(QASMParser::NNINTEGER, 0);
}

tree::TerminalNode* QASMParser::ExpContext::ID() {
  return getToken(QASMParser::ID, 0);
}

QASMParser::ComplexContext* QASMParser::ExpContext::complex() {
  return getRuleContext<QASMParser::ComplexContext>(0);
}

QASMParser::NegopContext* QASMParser::ExpContext::negop() {
  return getRuleContext<QASMParser::NegopContext>(0);
}

std::vector<QASMParser::ExpContext *> QASMParser::ExpContext::exp() {
  return getRuleContexts<QASMParser::ExpContext>();
}

QASMParser::ExpContext* QASMParser::ExpContext::exp(size_t i) {
  return getRuleContext<QASMParser::ExpContext>(i);
}

QASMParser::UnaryopContext* QASMParser::ExpContext::unaryop() {
  return getRuleContext<QASMParser::UnaryopContext>(0);
}

QASMParser::BinopContext* QASMParser::ExpContext::binop() {
  return getRuleContext<QASMParser::BinopContext>(0);
}


size_t QASMParser::ExpContext::getRuleIndex() const {
  return QASMParser::RuleExp;
}

void QASMParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void QASMParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


QASMParser::ExpContext* QASMParser::exp() {
   return exp(0);
}

QASMParser::ExpContext* QASMParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  QASMParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  QASMParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 64;
  enterRecursionRule(_localctx, 64, QASMParser::RuleExp, precedence);

    

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
    setState(392);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx)) {
    case 1: {
      setState(375);
      match(QASMParser::REAL);
      break;
    }

    case 2: {
      setState(376);
      match(QASMParser::NNINTEGER);
      break;
    }

    case 3: {
      setState(377);
      match(QASMParser::ID);
      break;
    }

    case 4: {
      setState(378);
      complex();
      break;
    }

    case 5: {
      setState(379);
      negop();
      setState(380);
      exp(4);
      break;
    }

    case 6: {
      setState(382);
      unaryop();
      setState(383);
      match(QASMParser::T__7);
      setState(384);
      exp(0);
      setState(385);
      match(QASMParser::T__8);
      break;
    }

    case 7: {
      setState(387);
      match(QASMParser::T__7);
      setState(388);
      exp(0);
      setState(389);
      match(QASMParser::T__8);
      break;
    }

    case 8: {
      setState(391);
      match(QASMParser::T__21);
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(400);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExp);
        setState(394);

        if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
        setState(395);
        binop();
        setState(396);
        exp(6); 
      }
      setState(402);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ComplexContext ------------------------------------------------------------------

QASMParser::ComplexContext::ComplexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> QASMParser::ComplexContext::REAL() {
  return getTokens(QASMParser::REAL);
}

tree::TerminalNode* QASMParser::ComplexContext::REAL(size_t i) {
  return getToken(QASMParser::REAL, i);
}

QASMParser::AddsubContext* QASMParser::ComplexContext::addsub() {
  return getRuleContext<QASMParser::AddsubContext>(0);
}


size_t QASMParser::ComplexContext::getRuleIndex() const {
  return QASMParser::RuleComplex;
}

void QASMParser::ComplexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComplex(this);
}

void QASMParser::ComplexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComplex(this);
}

QASMParser::ComplexContext* QASMParser::complex() {
  ComplexContext *_localctx = _tracker.createInstance<ComplexContext>(_ctx, getState());
  enterRule(_localctx, 66, QASMParser::RuleComplex);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(412);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case QASMParser::REAL: {
        enterOuterAlt(_localctx, 1);
        setState(403);
        match(QASMParser::REAL);
        setState(404);
        addsub();
        setState(405);
        match(QASMParser::REAL);
        setState(406);
        match(QASMParser::T__22);
        break;
      }

      case QASMParser::T__23:
      case QASMParser::T__24: {
        enterOuterAlt(_localctx, 2);
        setState(408);
        addsub();
        setState(409);
        match(QASMParser::REAL);
        setState(410);
        match(QASMParser::T__22);
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

//----------------- AddsubContext ------------------------------------------------------------------

QASMParser::AddsubContext::AddsubContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t QASMParser::AddsubContext::getRuleIndex() const {
  return QASMParser::RuleAddsub;
}

void QASMParser::AddsubContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddsub(this);
}

void QASMParser::AddsubContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddsub(this);
}

QASMParser::AddsubContext* QASMParser::addsub() {
  AddsubContext *_localctx = _tracker.createInstance<AddsubContext>(_ctx, getState());
  enterRule(_localctx, 68, QASMParser::RuleAddsub);
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
    setState(414);
    _la = _input->LA(1);
    if (!(_la == QASMParser::T__23

    || _la == QASMParser::T__24)) {
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

//----------------- BinopContext ------------------------------------------------------------------

QASMParser::BinopContext::BinopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t QASMParser::BinopContext::getRuleIndex() const {
  return QASMParser::RuleBinop;
}

void QASMParser::BinopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinop(this);
}

void QASMParser::BinopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinop(this);
}

QASMParser::BinopContext* QASMParser::binop() {
  BinopContext *_localctx = _tracker.createInstance<BinopContext>(_ctx, getState());
  enterRule(_localctx, 70, QASMParser::RuleBinop);
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
    setState(416);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 251658240) != 0))) {
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

//----------------- NegopContext ------------------------------------------------------------------

QASMParser::NegopContext::NegopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t QASMParser::NegopContext::getRuleIndex() const {
  return QASMParser::RuleNegop;
}

void QASMParser::NegopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNegop(this);
}

void QASMParser::NegopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNegop(this);
}

QASMParser::NegopContext* QASMParser::negop() {
  NegopContext *_localctx = _tracker.createInstance<NegopContext>(_ctx, getState());
  enterRule(_localctx, 72, QASMParser::RuleNegop);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(418);
    match(QASMParser::T__24);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnaryopContext ------------------------------------------------------------------

QASMParser::UnaryopContext::UnaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t QASMParser::UnaryopContext::getRuleIndex() const {
  return QASMParser::RuleUnaryop;
}

void QASMParser::UnaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryop(this);
}

void QASMParser::UnaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<QASMListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryop(this);
}

QASMParser::UnaryopContext* QASMParser::unaryop() {
  UnaryopContext *_localctx = _tracker.createInstance<UnaryopContext>(_ctx, getState());
  enterRule(_localctx, 74, QASMParser::RuleUnaryop);
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
    setState(420);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 16911433728) != 0))) {
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

bool QASMParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 2: return programSempred(antlrcpp::downCast<ProgramContext *>(context), predicateIndex);
    case 32: return expSempred(antlrcpp::downCast<ExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool QASMParser::programSempred(ProgramContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool QASMParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 5);

  default:
    break;
  }
  return true;
}

void QASMParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  qasmParserInitialize();
#else
  ::antlr4::internal::call_once(qasmParserOnceFlag, qasmParserInitialize);
#endif
}
