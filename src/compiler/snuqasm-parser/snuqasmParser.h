
// Generated from snuqasm.g4 by ANTLR 4.12.0

#pragma once


#include "antlr4-runtime.h"




class  snuqasmParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, T__5 = 6, T__6 = 7, 
    T__7 = 8, T__8 = 9, T__9 = 10, T__10 = 11, T__11 = 12, T__12 = 13, T__13 = 14, 
    T__14 = 15, T__15 = 16, T__16 = 17, T__17 = 18, T__18 = 19, T__19 = 20, 
    T__20 = 21, T__21 = 22, T__22 = 23, T__23 = 24, T__24 = 25, T__25 = 26, 
    T__26 = 27, T__27 = 28, T__28 = 29, T__29 = 30, T__30 = 31, T__31 = 32, 
    T__32 = 33, T__33 = 34, T__34 = 35, T__35 = 36, T__36 = 37, T__37 = 38, 
    T__38 = 39, T__39 = 40, T__40 = 41, T__41 = 42, T__42 = 43, T__43 = 44, 
    T__44 = 45, T__45 = 46, T__46 = 47, T__47 = 48, T__48 = 49, T__49 = 50, 
    T__50 = 51, T__51 = 52, T__52 = 53, T__53 = 54, T__54 = 55, T__55 = 56, 
    T__56 = 57, T__57 = 58, T__58 = 59, T__59 = 60, T__60 = 61, T__61 = 62, 
    T__62 = 63, T__63 = 64, StringLiteral = 65, Real = 66, Integer = 67, 
    Decimal = 68, SciSuffix = 69, Identifier = 70, Whitespace = 71, Newline = 72, 
    LineComment = 73, BlockComment = 74
  };

  enum {
    RuleMainprogram = 0, RuleVersion = 1, RuleProgram = 2, RuleInclude = 3, 
    RuleStatement = 4, RuleDecl = 5, RuleQuantumDecl = 6, RuleClassicalDecl = 7, 
    RuleGatedeclStatement = 8, RuleGoplist = 9, RuleOpaqueDeclStatement = 10, 
    RuleQopStatement = 11, RuleUopStatement = 12, RuleMeasureQop = 13, RuleResetQop = 14, 
    RuleIfStatement = 15, RuleBarrierStatement = 16, RuleUnitaryOp = 17, 
    RuleCustomOp = 18, RuleAnylist = 19, RuleIdlist = 20, RuleDesignatedIdentifier = 21, 
    RuleMixedlist = 22, RuleArglist = 23, RuleArgument = 24, RuleExplist = 25, 
    RuleExp = 26, RuleBinop = 27, RuleUnaryop = 28
  };

  explicit snuqasmParser(antlr4::TokenStream *input);

  snuqasmParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~snuqasmParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class MainprogramContext;
  class VersionContext;
  class ProgramContext;
  class IncludeContext;
  class StatementContext;
  class DeclContext;
  class QuantumDeclContext;
  class ClassicalDeclContext;
  class GatedeclStatementContext;
  class GoplistContext;
  class OpaqueDeclStatementContext;
  class QopStatementContext;
  class UopStatementContext;
  class MeasureQopContext;
  class ResetQopContext;
  class IfStatementContext;
  class BarrierStatementContext;
  class UnitaryOpContext;
  class CustomOpContext;
  class AnylistContext;
  class IdlistContext;
  class DesignatedIdentifierContext;
  class MixedlistContext;
  class ArglistContext;
  class ArgumentContext;
  class ExplistContext;
  class ExpContext;
  class BinopContext;
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
    antlr4::tree::TerminalNode *Real();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  VersionContext* version();

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StatementContext *statement();
    IncludeContext *include();
    ProgramContext *program();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ProgramContext* program();
  ProgramContext* program(int precedence);
  class  IncludeContext : public antlr4::ParserRuleContext {
  public:
    IncludeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *StringLiteral();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IncludeContext* include();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DeclContext *decl();
    GatedeclStatementContext *gatedeclStatement();
    OpaqueDeclStatementContext *opaqueDeclStatement();
    QopStatementContext *qopStatement();
    IfStatementContext *ifStatement();
    BarrierStatementContext *barrierStatement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  StatementContext* statement();

  class  DeclContext : public antlr4::ParserRuleContext {
  public:
    DeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    QuantumDeclContext *quantumDecl();
    ClassicalDeclContext *classicalDecl();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DeclContext* decl();

  class  QuantumDeclContext : public antlr4::ParserRuleContext {
  public:
    QuantumDeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Integer();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QuantumDeclContext* quantumDecl();

  class  ClassicalDeclContext : public antlr4::ParserRuleContext {
  public:
    ClassicalDeclContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Integer();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ClassicalDeclContext* classicalDecl();

  class  GatedeclStatementContext : public antlr4::ParserRuleContext {
  public:
    GatedeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    std::vector<IdlistContext *> idlist();
    IdlistContext* idlist(size_t i);
    GoplistContext *goplist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GatedeclStatementContext* gatedeclStatement();

  class  GoplistContext : public antlr4::ParserRuleContext {
  public:
    GoplistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UopStatementContext *uopStatement();
    BarrierStatementContext *barrierStatement();
    GoplistContext *goplist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  GoplistContext* goplist();

  class  OpaqueDeclStatementContext : public antlr4::ParserRuleContext {
  public:
    OpaqueDeclStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    std::vector<IdlistContext *> idlist();
    IdlistContext* idlist(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  OpaqueDeclStatementContext* opaqueDeclStatement();

  class  QopStatementContext : public antlr4::ParserRuleContext {
  public:
    QopStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UopStatementContext *uopStatement();
    MeasureQopContext *measureQop();
    ResetQopContext *resetQop();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  QopStatementContext* qopStatement();

  class  UopStatementContext : public antlr4::ParserRuleContext {
  public:
    UopStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    UnitaryOpContext *unitaryOp();
    CustomOpContext *customOp();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  UopStatementContext* uopStatement();

  class  MeasureQopContext : public antlr4::ParserRuleContext {
  public:
    MeasureQopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ArgumentContext *> argument();
    ArgumentContext* argument(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MeasureQopContext* measureQop();

  class  ResetQopContext : public antlr4::ParserRuleContext {
  public:
    ResetQopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArgumentContext *argument();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ResetQopContext* resetQop();

  class  IfStatementContext : public antlr4::ParserRuleContext {
  public:
    IfStatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Integer();
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

  class  UnitaryOpContext : public antlr4::ParserRuleContext {
  public:
    UnitaryOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExplistContext *explist();
    std::vector<ArgumentContext *> argument();
    ArgumentContext* argument(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  UnitaryOpContext* unitaryOp();

  class  CustomOpContext : public antlr4::ParserRuleContext {
  public:
    CustomOpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    AnylistContext *anylist();
    ExplistContext *explist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  CustomOpContext* customOp();

  class  AnylistContext : public antlr4::ParserRuleContext {
  public:
    AnylistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdlistContext *idlist();
    MixedlistContext *mixedlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  AnylistContext* anylist();

  class  IdlistContext : public antlr4::ParserRuleContext {
  public:
    IdlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    IdlistContext *idlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  IdlistContext* idlist();

  class  DesignatedIdentifierContext : public antlr4::ParserRuleContext {
  public:
    DesignatedIdentifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Integer();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  DesignatedIdentifierContext* designatedIdentifier();

  class  MixedlistContext : public antlr4::ParserRuleContext {
  public:
    MixedlistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    DesignatedIdentifierContext *designatedIdentifier();
    MixedlistContext *mixedlist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  MixedlistContext* mixedlist();

  class  ArglistContext : public antlr4::ParserRuleContext {
  public:
    ArglistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArgumentContext *argument();
    ArglistContext *arglist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ArglistContext* arglist();

  class  ArgumentContext : public antlr4::ParserRuleContext {
  public:
    ArgumentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    antlr4::tree::TerminalNode *Integer();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ArgumentContext* argument();

  class  ExplistContext : public antlr4::ParserRuleContext {
  public:
    ExplistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpContext *exp();
    ExplistContext *explist();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ExplistContext* explist();

  class  ExpContext : public antlr4::ParserRuleContext {
  public:
    ExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Real();
    antlr4::tree::TerminalNode *Integer();
    antlr4::tree::TerminalNode *Identifier();
    UnaryopContext *unaryop();
    std::vector<ExpContext *> exp();
    ExpContext* exp(size_t i);
    BinopContext *binop();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  ExpContext* exp();
  ExpContext* exp(int precedence);
  class  BinopContext : public antlr4::ParserRuleContext {
  public:
    BinopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;
   
  };

  BinopContext* binop();

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

