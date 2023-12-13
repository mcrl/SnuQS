
// Generated from QASM.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  QASMParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, ID = 34, NNINTEGER = 35, REAL = 36, STRING = 37, Whitespace = 38, 
    Newline = 39, LineComment = 40, BlockComment = 41
  };

  enum {
    RuleMainprogram = 0, RuleVersion = 1, RuleProgram = 2, RuleStatement = 3, 
    RuleDeclStatement = 4, RuleRegDeclStatement = 5, RuleQregDeclStatement = 6, 
    RuleCregDeclStatement = 7, RuleGateDeclStatement = 8, RuleOpaqueStatement = 9, 
    RuleGateStatement = 10, RuleGoplist = 11, RuleGop = 12, RuleGopUGate = 13, 
    RuleGopCXGate = 14, RuleGopBarrier = 15, RuleGopCustomGate = 16, RuleGopReset = 17, 
    RuleIdlist = 18, RuleParamlist = 19, RuleQopStatement = 20, RuleQopUGate = 21, 
    RuleQopCXGate = 22, RuleQopMeasure = 23, RuleQopReset = 24, RuleQopCustomGate = 25, 
    RuleIfStatement = 26, RuleBarrierStatement = 27, RuleArglist = 28, RuleQarg = 29, 
    RuleCarg = 30, RuleExplist = 31, RuleExp = 32, RuleComplex = 33, RuleAddsub = 34, 
    RuleBinop = 35, RuleNegop = 36, RuleUnaryop = 37
  };

  explicit QASMParser(antlr4::TokenStream *input);

  QASMParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~QASMParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class MainprogramContext;
  class VersionContext;
  class ProgramContext;
  class StatementContext;
  class DeclStatementContext;
  class RegDeclStatementContext;
  class QregDeclStatementContext;
  class CregDeclStatementContext;
  class GateDeclStatementContext;
  class OpaqueStatementContext;
  class GateStatementContext;
  class GoplistContext;
  class GopContext;
  class GopUGateContext;
  class GopCXGateContext;
  class GopBarrierContext;
  class GopCustomGateContext;
  class GopResetContext;
  class IdlistContext;
  class ParamlistContext;
  class QopStatementContext;
  class QopUGateContext;
  class QopCXGateContext;
  class QopMeasureContext;
  class QopResetContext;
  class QopCustomGateContext;
  class IfStatementContext;
  class BarrierStatementContext;
  class ArglistContext;
  class QargContext;
  class CargContext;
  class ExplistContext;
  class ExpContext;
  class ComplexContext;
  class AddsubContext;
  class BinopContext;
  class NegopContext;
  class UnaryopContext; 

  class  MainprogramContext : public antlr4::ParserRuleContext {
  public:
    MainprogramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VersionContext *version();
    ProgramContext *program();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MainprogramContext* mainprogram();

  class  VersionContext : public antlr4::ParserRuleContext {
  public:
    VersionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *REAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  VersionContext* version();

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementContext *statement();
    ProgramContext *program();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ProgramContext* program();
  ProgramContext* program(int precedence);
  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DeclStatementContext *declStatement();
    QopStatementContext *qopStatement();
    IfStatementContext *ifStatement();
    BarrierStatementContext *barrierStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StatementContext* statement();

  class  DeclStatementContext : public antlr4::ParserRuleContext {
  public:
    DeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RegDeclStatementContext *regDeclStatement();
    GateDeclStatementContext *gateDeclStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DeclStatementContext* declStatement();

  class  RegDeclStatementContext : public antlr4::ParserRuleContext {
  public:
    RegDeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QregDeclStatementContext *qregDeclStatement();
    CregDeclStatementContext *cregDeclStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  RegDeclStatementContext* regDeclStatement();

  class  QregDeclStatementContext : public antlr4::ParserRuleContext {
  public:
    QregDeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *NNINTEGER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QregDeclStatementContext* qregDeclStatement();

  class  CregDeclStatementContext : public antlr4::ParserRuleContext {
  public:
    CregDeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *NNINTEGER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CregDeclStatementContext* cregDeclStatement();

  class  GateDeclStatementContext : public antlr4::ParserRuleContext {
  public:
    GateDeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    OpaqueStatementContext *opaqueStatement();
    GateStatementContext *gateStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GateDeclStatementContext* gateDeclStatement();

  class  OpaqueStatementContext : public antlr4::ParserRuleContext {
  public:
    OpaqueStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    IdlistContext *idlist();
    ParamlistContext *paramlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  OpaqueStatementContext* opaqueStatement();

  class  GateStatementContext : public antlr4::ParserRuleContext {
  public:
    GateStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    IdlistContext *idlist();
    ParamlistContext *paramlist();
    GoplistContext *goplist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GateStatementContext* gateStatement();

  class  GoplistContext : public antlr4::ParserRuleContext {
  public:
    GoplistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<GopContext *> gop();
    GopContext* gop(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GoplistContext* goplist();

  class  GopContext : public antlr4::ParserRuleContext {
  public:
    GopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    GopUGateContext *gopUGate();
    GopCXGateContext *gopCXGate();
    GopBarrierContext *gopBarrier();
    GopCustomGateContext *gopCustomGate();
    GopResetContext *gopReset();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopContext* gop();

  class  GopUGateContext : public antlr4::ParserRuleContext {
  public:
    GopUGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExplistContext *explist();
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopUGateContext* gopUGate();

  class  GopCXGateContext : public antlr4::ParserRuleContext {
  public:
    GopCXGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> ID();
    antlr4::tree::TerminalNode* ID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopCXGateContext* gopCXGate();

  class  GopBarrierContext : public antlr4::ParserRuleContext {
  public:
    GopBarrierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdlistContext *idlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopBarrierContext* gopBarrier();

  class  GopCustomGateContext : public antlr4::ParserRuleContext {
  public:
    GopCustomGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    IdlistContext *idlist();
    ExplistContext *explist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopCustomGateContext* gopCustomGate();

  class  GopResetContext : public antlr4::ParserRuleContext {
  public:
    GopResetContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GopResetContext* gopReset();

  class  IdlistContext : public antlr4::ParserRuleContext {
  public:
    IdlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> ID();
    antlr4::tree::TerminalNode* ID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IdlistContext* idlist();

  class  ParamlistContext : public antlr4::ParserRuleContext {
  public:
    ParamlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> ID();
    antlr4::tree::TerminalNode* ID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ParamlistContext* paramlist();

  class  QopStatementContext : public antlr4::ParserRuleContext {
  public:
    QopStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QopUGateContext *qopUGate();
    QopCXGateContext *qopCXGate();
    QopMeasureContext *qopMeasure();
    QopResetContext *qopReset();
    QopCustomGateContext *qopCustomGate();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopStatementContext* qopStatement();

  class  QopUGateContext : public antlr4::ParserRuleContext {
  public:
    QopUGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExplistContext *explist();
    QargContext *qarg();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopUGateContext* qopUGate();

  class  QopCXGateContext : public antlr4::ParserRuleContext {
  public:
    QopCXGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QargContext *> qarg();
    QargContext* qarg(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopCXGateContext* qopCXGate();

  class  QopMeasureContext : public antlr4::ParserRuleContext {
  public:
    QopMeasureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QargContext *qarg();
    CargContext *carg();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopMeasureContext* qopMeasure();

  class  QopResetContext : public antlr4::ParserRuleContext {
  public:
    QopResetContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QargContext *qarg();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopResetContext* qopReset();

  class  QopCustomGateContext : public antlr4::ParserRuleContext {
  public:
    QopCustomGateContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    ArglistContext *arglist();
    ExplistContext *explist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopCustomGateContext* qopCustomGate();

  class  IfStatementContext : public antlr4::ParserRuleContext {
  public:
    IfStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *NNINTEGER();
    QopStatementContext *qopStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IfStatementContext* ifStatement();

  class  BarrierStatementContext : public antlr4::ParserRuleContext {
  public:
    BarrierStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArglistContext *arglist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  BarrierStatementContext* barrierStatement();

  class  ArglistContext : public antlr4::ParserRuleContext {
  public:
    ArglistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<QargContext *> qarg();
    QargContext* qarg(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ArglistContext* arglist();

  class  QargContext : public antlr4::ParserRuleContext {
  public:
    QargContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *NNINTEGER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QargContext* qarg();

  class  CargContext : public antlr4::ParserRuleContext {
  public:
    CargContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();
    antlr4::tree::TerminalNode *NNINTEGER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CargContext* carg();

  class  ExplistContext : public antlr4::ParserRuleContext {
  public:
    ExplistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpContext *> exp();
    ExpContext* exp(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ExplistContext* explist();

  class  ExpContext : public antlr4::ParserRuleContext {
  public:
    ExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *REAL();
    antlr4::tree::TerminalNode *NNINTEGER();
    antlr4::tree::TerminalNode *ID();
    ComplexContext *complex();
    NegopContext *negop();
    std::vector<ExpContext *> exp();
    ExpContext* exp(size_t i);
    UnaryopContext *unaryop();
    BinopContext *binop();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ExpContext* exp();
  ExpContext* exp(int precedence);
  class  ComplexContext : public antlr4::ParserRuleContext {
  public:
    ComplexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> REAL();
    antlr4::tree::TerminalNode* REAL(size_t i);
    AddsubContext *addsub();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ComplexContext* complex();

  class  AddsubContext : public antlr4::ParserRuleContext {
  public:
    AddsubContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AddsubContext* addsub();

  class  BinopContext : public antlr4::ParserRuleContext {
  public:
    BinopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  BinopContext* binop();

  class  NegopContext : public antlr4::ParserRuleContext {
  public:
    NegopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  NegopContext* negop();

  class  UnaryopContext : public antlr4::ParserRuleContext {
  public:
    UnaryopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  UnaryopContext* unaryop();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool programSempred(ProgramContext *_localctx, size_t predicateIndex);
  bool expSempred(ExpContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

