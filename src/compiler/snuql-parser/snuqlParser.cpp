
// Generated from snuql.g4 by ANTLR 4.7.2


#include "snuqlListener.h"

#include "snuqlParser.h"


using namespace antlrcpp;
using namespace antlr4;

snuqlParser::snuqlParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

snuqlParser::~snuqlParser() {
  delete _interpreter;
}

std::string snuqlParser::getGrammarFileName() const {
  return "snuql.g4";
}

const std::vector<std::string>& snuqlParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& snuqlParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- MainprogramContext ------------------------------------------------------------------

snuqlParser::MainprogramContext::MainprogramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::HeaderContext* snuqlParser::MainprogramContext::header() {
  return getRuleContext<snuqlParser::HeaderContext>(0);
}

snuqlParser::ProgramContext* snuqlParser::MainprogramContext::program() {
  return getRuleContext<snuqlParser::ProgramContext>(0);
}


size_t snuqlParser::MainprogramContext::getRuleIndex() const {
  return snuqlParser::RuleMainprogram;
}

void snuqlParser::MainprogramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMainprogram(this);
}

void snuqlParser::MainprogramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMainprogram(this);
}

snuqlParser::MainprogramContext* snuqlParser::mainprogram() {
  MainprogramContext *_localctx = _tracker.createInstance<MainprogramContext>(_ctx, getState());
  enterRule(_localctx, 0, snuqlParser::RuleMainprogram);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(58);
    header();
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

//----------------- HeaderContext ------------------------------------------------------------------

snuqlParser::HeaderContext::HeaderContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::VersionContext* snuqlParser::HeaderContext::version() {
  return getRuleContext<snuqlParser::VersionContext>(0);
}

std::vector<snuqlParser::IncludeContext *> snuqlParser::HeaderContext::include() {
  return getRuleContexts<snuqlParser::IncludeContext>();
}

snuqlParser::IncludeContext* snuqlParser::HeaderContext::include(size_t i) {
  return getRuleContext<snuqlParser::IncludeContext>(i);
}


size_t snuqlParser::HeaderContext::getRuleIndex() const {
  return snuqlParser::RuleHeader;
}

void snuqlParser::HeaderContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHeader(this);
}

void snuqlParser::HeaderContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHeader(this);
}

snuqlParser::HeaderContext* snuqlParser::header() {
  HeaderContext *_localctx = _tracker.createInstance<HeaderContext>(_ctx, getState());
  enterRule(_localctx, 2, snuqlParser::RuleHeader);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(62);
    _errHandler->sync(this);

    _la = _input->LA(1);
    if (_la == snuqlParser::T__0) {
      setState(61);
      version();
    }
    setState(67);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == snuqlParser::T__2) {
      setState(64);
      include();
      setState(69);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VersionContext ------------------------------------------------------------------

snuqlParser::VersionContext::VersionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::VersionContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}

tree::TerminalNode* snuqlParser::VersionContext::Real() {
  return getToken(snuqlParser::Real, 0);
}


size_t snuqlParser::VersionContext::getRuleIndex() const {
  return snuqlParser::RuleVersion;
}

void snuqlParser::VersionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVersion(this);
}

void snuqlParser::VersionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVersion(this);
}

snuqlParser::VersionContext* snuqlParser::version() {
  VersionContext *_localctx = _tracker.createInstance<VersionContext>(_ctx, getState());
  enterRule(_localctx, 4, snuqlParser::RuleVersion);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(70);
    match(snuqlParser::T__0);
    setState(71);
    _la = _input->LA(1);
    if (!(_la == snuqlParser::Real

    || _la == snuqlParser::Integer)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(72);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IncludeContext ------------------------------------------------------------------

snuqlParser::IncludeContext::IncludeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::IncludeContext::StringLiteral() {
  return getToken(snuqlParser::StringLiteral, 0);
}


size_t snuqlParser::IncludeContext::getRuleIndex() const {
  return snuqlParser::RuleInclude;
}

void snuqlParser::IncludeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInclude(this);
}

void snuqlParser::IncludeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInclude(this);
}

snuqlParser::IncludeContext* snuqlParser::include() {
  IncludeContext *_localctx = _tracker.createInstance<IncludeContext>(_ctx, getState());
  enterRule(_localctx, 6, snuqlParser::RuleInclude);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(74);
    match(snuqlParser::T__2);
    setState(75);
    match(snuqlParser::StringLiteral);
    setState(76);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ProgramContext ------------------------------------------------------------------

snuqlParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::StatementContext* snuqlParser::ProgramContext::statement() {
  return getRuleContext<snuqlParser::StatementContext>(0);
}

snuqlParser::ProgramContext* snuqlParser::ProgramContext::program() {
  return getRuleContext<snuqlParser::ProgramContext>(0);
}


size_t snuqlParser::ProgramContext::getRuleIndex() const {
  return snuqlParser::RuleProgram;
}

void snuqlParser::ProgramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProgram(this);
}

void snuqlParser::ProgramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProgram(this);
}


snuqlParser::ProgramContext* snuqlParser::program() {
   return program(0);
}

snuqlParser::ProgramContext* snuqlParser::program(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  snuqlParser::ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, parentState);
  snuqlParser::ProgramContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 8;
  enterRecursionRule(_localctx, 8, snuqlParser::RuleProgram, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(79);
    statement();
    _ctx->stop = _input->LT(-1);
    setState(85);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ProgramContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleProgram);
        setState(81);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(82);
        statement(); 
      }
      setState(87);
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

//----------------- StatementContext ------------------------------------------------------------------

snuqlParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::DeclContext* snuqlParser::StatementContext::decl() {
  return getRuleContext<snuqlParser::DeclContext>(0);
}

snuqlParser::GatedeclStatementContext* snuqlParser::StatementContext::gatedeclStatement() {
  return getRuleContext<snuqlParser::GatedeclStatementContext>(0);
}

snuqlParser::QopStatementContext* snuqlParser::StatementContext::qopStatement() {
  return getRuleContext<snuqlParser::QopStatementContext>(0);
}

snuqlParser::IfStatementContext* snuqlParser::StatementContext::ifStatement() {
  return getRuleContext<snuqlParser::IfStatementContext>(0);
}

snuqlParser::BarrierStatementContext* snuqlParser::StatementContext::barrierStatement() {
  return getRuleContext<snuqlParser::BarrierStatementContext>(0);
}


size_t snuqlParser::StatementContext::getRuleIndex() const {
  return snuqlParser::RuleStatement;
}

void snuqlParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void snuqlParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}

snuqlParser::StatementContext* snuqlParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 10, snuqlParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(93);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::T__3:
      case snuqlParser::T__6: {
        enterOuterAlt(_localctx, 1);
        setState(88);
        decl();
        break;
      }

      case snuqlParser::T__8: {
        enterOuterAlt(_localctx, 2);
        setState(89);
        gatedeclStatement();
        break;
      }

      case snuqlParser::T__12:
      case snuqlParser::T__14:
      case snuqlParser::T__18:
      case snuqlParser::T__19:
      case snuqlParser::T__20:
      case snuqlParser::T__22:
      case snuqlParser::T__23:
      case snuqlParser::T__24:
      case snuqlParser::T__25:
      case snuqlParser::T__26:
      case snuqlParser::T__27:
      case snuqlParser::T__28:
      case snuqlParser::T__29:
      case snuqlParser::T__30:
      case snuqlParser::T__31:
      case snuqlParser::T__32:
      case snuqlParser::T__33:
      case snuqlParser::T__34:
      case snuqlParser::T__35:
      case snuqlParser::T__36:
      case snuqlParser::T__37:
      case snuqlParser::T__38:
      case snuqlParser::T__39:
      case snuqlParser::T__40:
      case snuqlParser::T__41:
      case snuqlParser::T__42:
      case snuqlParser::T__43:
      case snuqlParser::T__44:
      case snuqlParser::T__45:
      case snuqlParser::T__46:
      case snuqlParser::T__47:
      case snuqlParser::T__48:
      case snuqlParser::T__49:
      case snuqlParser::T__50:
      case snuqlParser::Identifier: {
        enterOuterAlt(_localctx, 3);
        setState(90);
        qopStatement();
        break;
      }

      case snuqlParser::T__15: {
        enterOuterAlt(_localctx, 4);
        setState(91);
        ifStatement();
        break;
      }

      case snuqlParser::T__17: {
        enterOuterAlt(_localctx, 5);
        setState(92);
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

snuqlParser::DeclContext::DeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::QuantumDeclContext* snuqlParser::DeclContext::quantumDecl() {
  return getRuleContext<snuqlParser::QuantumDeclContext>(0);
}

snuqlParser::ClassicalDeclContext* snuqlParser::DeclContext::classicalDecl() {
  return getRuleContext<snuqlParser::ClassicalDeclContext>(0);
}


size_t snuqlParser::DeclContext::getRuleIndex() const {
  return snuqlParser::RuleDecl;
}

void snuqlParser::DeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecl(this);
}

void snuqlParser::DeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecl(this);
}

snuqlParser::DeclContext* snuqlParser::decl() {
  DeclContext *_localctx = _tracker.createInstance<DeclContext>(_ctx, getState());
  enterRule(_localctx, 12, snuqlParser::RuleDecl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(97);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::T__3: {
        enterOuterAlt(_localctx, 1);
        setState(95);
        quantumDecl();
        break;
      }

      case snuqlParser::T__6: {
        enterOuterAlt(_localctx, 2);
        setState(96);
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

snuqlParser::QuantumDeclContext::QuantumDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::QuantumDeclContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

tree::TerminalNode* snuqlParser::QuantumDeclContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}


size_t snuqlParser::QuantumDeclContext::getRuleIndex() const {
  return snuqlParser::RuleQuantumDecl;
}

void snuqlParser::QuantumDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantumDecl(this);
}

void snuqlParser::QuantumDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantumDecl(this);
}

snuqlParser::QuantumDeclContext* snuqlParser::quantumDecl() {
  QuantumDeclContext *_localctx = _tracker.createInstance<QuantumDeclContext>(_ctx, getState());
  enterRule(_localctx, 14, snuqlParser::RuleQuantumDecl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(99);
    match(snuqlParser::T__3);
    setState(100);
    match(snuqlParser::Identifier);
    setState(101);
    match(snuqlParser::T__4);
    setState(102);
    match(snuqlParser::Integer);
    setState(103);
    match(snuqlParser::T__5);
    setState(104);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ClassicalDeclContext ------------------------------------------------------------------

snuqlParser::ClassicalDeclContext::ClassicalDeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::ClassicalDeclContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

tree::TerminalNode* snuqlParser::ClassicalDeclContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}


size_t snuqlParser::ClassicalDeclContext::getRuleIndex() const {
  return snuqlParser::RuleClassicalDecl;
}

void snuqlParser::ClassicalDeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterClassicalDecl(this);
}

void snuqlParser::ClassicalDeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitClassicalDecl(this);
}

snuqlParser::ClassicalDeclContext* snuqlParser::classicalDecl() {
  ClassicalDeclContext *_localctx = _tracker.createInstance<ClassicalDeclContext>(_ctx, getState());
  enterRule(_localctx, 16, snuqlParser::RuleClassicalDecl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(106);
    match(snuqlParser::T__6);
    setState(107);
    match(snuqlParser::Identifier);
    setState(108);
    match(snuqlParser::T__4);
    setState(109);
    match(snuqlParser::Integer);
    setState(110);
    match(snuqlParser::T__5);
    setState(111);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GatedeclStatementContext ------------------------------------------------------------------

snuqlParser::GatedeclStatementContext::GatedeclStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::GatedeclContext* snuqlParser::GatedeclStatementContext::gatedecl() {
  return getRuleContext<snuqlParser::GatedeclContext>(0);
}

snuqlParser::GoplistContext* snuqlParser::GatedeclStatementContext::goplist() {
  return getRuleContext<snuqlParser::GoplistContext>(0);
}


size_t snuqlParser::GatedeclStatementContext::getRuleIndex() const {
  return snuqlParser::RuleGatedeclStatement;
}

void snuqlParser::GatedeclStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGatedeclStatement(this);
}

void snuqlParser::GatedeclStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGatedeclStatement(this);
}

snuqlParser::GatedeclStatementContext* snuqlParser::gatedeclStatement() {
  GatedeclStatementContext *_localctx = _tracker.createInstance<GatedeclStatementContext>(_ctx, getState());
  enterRule(_localctx, 18, snuqlParser::RuleGatedeclStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(120);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(113);
      gatedecl();
      setState(114);
      goplist();
      setState(115);
      match(snuqlParser::T__7);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(117);
      gatedecl();
      setState(118);
      match(snuqlParser::T__7);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GatedeclContext ------------------------------------------------------------------

snuqlParser::GatedeclContext::GatedeclContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::GatedeclContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

std::vector<snuqlParser::IdlistContext *> snuqlParser::GatedeclContext::idlist() {
  return getRuleContexts<snuqlParser::IdlistContext>();
}

snuqlParser::IdlistContext* snuqlParser::GatedeclContext::idlist(size_t i) {
  return getRuleContext<snuqlParser::IdlistContext>(i);
}


size_t snuqlParser::GatedeclContext::getRuleIndex() const {
  return snuqlParser::RuleGatedecl;
}

void snuqlParser::GatedeclContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGatedecl(this);
}

void snuqlParser::GatedeclContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGatedecl(this);
}

snuqlParser::GatedeclContext* snuqlParser::gatedecl() {
  GatedeclContext *_localctx = _tracker.createInstance<GatedeclContext>(_ctx, getState());
  enterRule(_localctx, 20, snuqlParser::RuleGatedecl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(142);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(122);
      match(snuqlParser::T__8);
      setState(123);
      match(snuqlParser::Identifier);
      setState(124);
      idlist();
      setState(125);
      match(snuqlParser::T__9);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(127);
      match(snuqlParser::T__8);
      setState(128);
      match(snuqlParser::Identifier);
      setState(129);
      match(snuqlParser::T__10);
      setState(130);
      match(snuqlParser::T__11);
      setState(131);
      idlist();
      setState(132);
      match(snuqlParser::T__9);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(134);
      match(snuqlParser::T__8);
      setState(135);
      match(snuqlParser::Identifier);
      setState(136);
      match(snuqlParser::T__10);
      setState(137);
      idlist();
      setState(138);
      match(snuqlParser::T__11);
      setState(139);
      idlist();
      setState(140);
      match(snuqlParser::T__9);
      break;
    }

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

snuqlParser::GoplistContext::GoplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::UopStatementContext* snuqlParser::GoplistContext::uopStatement() {
  return getRuleContext<snuqlParser::UopStatementContext>(0);
}

snuqlParser::BarrierStatementContext* snuqlParser::GoplistContext::barrierStatement() {
  return getRuleContext<snuqlParser::BarrierStatementContext>(0);
}

snuqlParser::GoplistContext* snuqlParser::GoplistContext::goplist() {
  return getRuleContext<snuqlParser::GoplistContext>(0);
}


size_t snuqlParser::GoplistContext::getRuleIndex() const {
  return snuqlParser::RuleGoplist;
}

void snuqlParser::GoplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGoplist(this);
}

void snuqlParser::GoplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGoplist(this);
}

snuqlParser::GoplistContext* snuqlParser::goplist() {
  GoplistContext *_localctx = _tracker.createInstance<GoplistContext>(_ctx, getState());
  enterRule(_localctx, 22, snuqlParser::RuleGoplist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(152);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(144);
      uopStatement();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(145);
      barrierStatement();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(146);
      uopStatement();
      setState(147);
      goplist();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(149);
      barrierStatement();
      setState(150);
      goplist();
      break;
    }

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

snuqlParser::QopStatementContext::QopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::UopStatementContext* snuqlParser::QopStatementContext::uopStatement() {
  return getRuleContext<snuqlParser::UopStatementContext>(0);
}

snuqlParser::MeasureQopContext* snuqlParser::QopStatementContext::measureQop() {
  return getRuleContext<snuqlParser::MeasureQopContext>(0);
}

snuqlParser::ResetQopContext* snuqlParser::QopStatementContext::resetQop() {
  return getRuleContext<snuqlParser::ResetQopContext>(0);
}


size_t snuqlParser::QopStatementContext::getRuleIndex() const {
  return snuqlParser::RuleQopStatement;
}

void snuqlParser::QopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQopStatement(this);
}

void snuqlParser::QopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQopStatement(this);
}

snuqlParser::QopStatementContext* snuqlParser::qopStatement() {
  QopStatementContext *_localctx = _tracker.createInstance<QopStatementContext>(_ctx, getState());
  enterRule(_localctx, 24, snuqlParser::RuleQopStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(157);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::T__18:
      case snuqlParser::T__19:
      case snuqlParser::T__20:
      case snuqlParser::T__22:
      case snuqlParser::T__23:
      case snuqlParser::T__24:
      case snuqlParser::T__25:
      case snuqlParser::T__26:
      case snuqlParser::T__27:
      case snuqlParser::T__28:
      case snuqlParser::T__29:
      case snuqlParser::T__30:
      case snuqlParser::T__31:
      case snuqlParser::T__32:
      case snuqlParser::T__33:
      case snuqlParser::T__34:
      case snuqlParser::T__35:
      case snuqlParser::T__36:
      case snuqlParser::T__37:
      case snuqlParser::T__38:
      case snuqlParser::T__39:
      case snuqlParser::T__40:
      case snuqlParser::T__41:
      case snuqlParser::T__42:
      case snuqlParser::T__43:
      case snuqlParser::T__44:
      case snuqlParser::T__45:
      case snuqlParser::T__46:
      case snuqlParser::T__47:
      case snuqlParser::T__48:
      case snuqlParser::T__49:
      case snuqlParser::T__50:
      case snuqlParser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(154);
        uopStatement();
        break;
      }

      case snuqlParser::T__12: {
        enterOuterAlt(_localctx, 2);
        setState(155);
        measureQop();
        break;
      }

      case snuqlParser::T__14: {
        enterOuterAlt(_localctx, 3);
        setState(156);
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

snuqlParser::UopStatementContext::UopStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::UnitaryOpContext* snuqlParser::UopStatementContext::unitaryOp() {
  return getRuleContext<snuqlParser::UnitaryOpContext>(0);
}

snuqlParser::CustomOpContext* snuqlParser::UopStatementContext::customOp() {
  return getRuleContext<snuqlParser::CustomOpContext>(0);
}


size_t snuqlParser::UopStatementContext::getRuleIndex() const {
  return snuqlParser::RuleUopStatement;
}

void snuqlParser::UopStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUopStatement(this);
}

void snuqlParser::UopStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUopStatement(this);
}

snuqlParser::UopStatementContext* snuqlParser::uopStatement() {
  UopStatementContext *_localctx = _tracker.createInstance<UopStatementContext>(_ctx, getState());
  enterRule(_localctx, 26, snuqlParser::RuleUopStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(161);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::T__18:
      case snuqlParser::T__19:
      case snuqlParser::T__20:
      case snuqlParser::T__22:
      case snuqlParser::T__23:
      case snuqlParser::T__24:
      case snuqlParser::T__25:
      case snuqlParser::T__26:
      case snuqlParser::T__27:
      case snuqlParser::T__28:
      case snuqlParser::T__29:
      case snuqlParser::T__30:
      case snuqlParser::T__31:
      case snuqlParser::T__32:
      case snuqlParser::T__33:
      case snuqlParser::T__34:
      case snuqlParser::T__35:
      case snuqlParser::T__36:
      case snuqlParser::T__37:
      case snuqlParser::T__38:
      case snuqlParser::T__39:
      case snuqlParser::T__40:
      case snuqlParser::T__41:
      case snuqlParser::T__42:
      case snuqlParser::T__43:
      case snuqlParser::T__44:
      case snuqlParser::T__45:
      case snuqlParser::T__46:
      case snuqlParser::T__47:
      case snuqlParser::T__48:
      case snuqlParser::T__49:
      case snuqlParser::T__50: {
        enterOuterAlt(_localctx, 1);
        setState(159);
        unitaryOp();
        break;
      }

      case snuqlParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(160);
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

snuqlParser::MeasureQopContext::MeasureQopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<snuqlParser::ArgumentContext *> snuqlParser::MeasureQopContext::argument() {
  return getRuleContexts<snuqlParser::ArgumentContext>();
}

snuqlParser::ArgumentContext* snuqlParser::MeasureQopContext::argument(size_t i) {
  return getRuleContext<snuqlParser::ArgumentContext>(i);
}


size_t snuqlParser::MeasureQopContext::getRuleIndex() const {
  return snuqlParser::RuleMeasureQop;
}

void snuqlParser::MeasureQopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMeasureQop(this);
}

void snuqlParser::MeasureQopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMeasureQop(this);
}

snuqlParser::MeasureQopContext* snuqlParser::measureQop() {
  MeasureQopContext *_localctx = _tracker.createInstance<MeasureQopContext>(_ctx, getState());
  enterRule(_localctx, 28, snuqlParser::RuleMeasureQop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(163);
    match(snuqlParser::T__12);
    setState(164);
    argument();
    setState(165);
    match(snuqlParser::T__13);
    setState(166);
    argument();
    setState(167);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ResetQopContext ------------------------------------------------------------------

snuqlParser::ResetQopContext::ResetQopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::ArgumentContext* snuqlParser::ResetQopContext::argument() {
  return getRuleContext<snuqlParser::ArgumentContext>(0);
}


size_t snuqlParser::ResetQopContext::getRuleIndex() const {
  return snuqlParser::RuleResetQop;
}

void snuqlParser::ResetQopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterResetQop(this);
}

void snuqlParser::ResetQopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitResetQop(this);
}

snuqlParser::ResetQopContext* snuqlParser::resetQop() {
  ResetQopContext *_localctx = _tracker.createInstance<ResetQopContext>(_ctx, getState());
  enterRule(_localctx, 30, snuqlParser::RuleResetQop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(169);
    match(snuqlParser::T__14);
    setState(170);
    argument();
    setState(171);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IfStatementContext ------------------------------------------------------------------

snuqlParser::IfStatementContext::IfStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::IfStatementContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

tree::TerminalNode* snuqlParser::IfStatementContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}

snuqlParser::QopStatementContext* snuqlParser::IfStatementContext::qopStatement() {
  return getRuleContext<snuqlParser::QopStatementContext>(0);
}


size_t snuqlParser::IfStatementContext::getRuleIndex() const {
  return snuqlParser::RuleIfStatement;
}

void snuqlParser::IfStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIfStatement(this);
}

void snuqlParser::IfStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIfStatement(this);
}

snuqlParser::IfStatementContext* snuqlParser::ifStatement() {
  IfStatementContext *_localctx = _tracker.createInstance<IfStatementContext>(_ctx, getState());
  enterRule(_localctx, 32, snuqlParser::RuleIfStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(173);
    match(snuqlParser::T__15);
    setState(174);
    match(snuqlParser::T__10);
    setState(175);
    match(snuqlParser::Identifier);
    setState(176);
    match(snuqlParser::T__16);
    setState(177);
    match(snuqlParser::Integer);
    setState(178);
    match(snuqlParser::T__11);
    setState(179);
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

snuqlParser::BarrierStatementContext::BarrierStatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::AnylistContext* snuqlParser::BarrierStatementContext::anylist() {
  return getRuleContext<snuqlParser::AnylistContext>(0);
}


size_t snuqlParser::BarrierStatementContext::getRuleIndex() const {
  return snuqlParser::RuleBarrierStatement;
}

void snuqlParser::BarrierStatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrierStatement(this);
}

void snuqlParser::BarrierStatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrierStatement(this);
}

snuqlParser::BarrierStatementContext* snuqlParser::barrierStatement() {
  BarrierStatementContext *_localctx = _tracker.createInstance<BarrierStatementContext>(_ctx, getState());
  enterRule(_localctx, 34, snuqlParser::RuleBarrierStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(181);
    match(snuqlParser::T__17);
    setState(182);
    anylist();
    setState(183);
    match(snuqlParser::T__1);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- UnitaryOpContext ------------------------------------------------------------------

snuqlParser::UnitaryOpContext::UnitaryOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::ExplistContext* snuqlParser::UnitaryOpContext::explist() {
  return getRuleContext<snuqlParser::ExplistContext>(0);
}

std::vector<snuqlParser::ArgumentContext *> snuqlParser::UnitaryOpContext::argument() {
  return getRuleContexts<snuqlParser::ArgumentContext>();
}

snuqlParser::ArgumentContext* snuqlParser::UnitaryOpContext::argument(size_t i) {
  return getRuleContext<snuqlParser::ArgumentContext>(i);
}

snuqlParser::ArglistContext* snuqlParser::UnitaryOpContext::arglist() {
  return getRuleContext<snuqlParser::ArglistContext>(0);
}


size_t snuqlParser::UnitaryOpContext::getRuleIndex() const {
  return snuqlParser::RuleUnitaryOp;
}

void snuqlParser::UnitaryOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnitaryOp(this);
}

void snuqlParser::UnitaryOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnitaryOp(this);
}

snuqlParser::UnitaryOpContext* snuqlParser::unitaryOp() {
  UnitaryOpContext *_localctx = _tracker.createInstance<UnitaryOpContext>(_ctx, getState());
  enterRule(_localctx, 36, snuqlParser::RuleUnitaryOp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(379);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::T__18: {
        enterOuterAlt(_localctx, 1);
        setState(185);
        match(snuqlParser::T__18);
        setState(186);
        match(snuqlParser::T__10);
        setState(187);
        explist();
        setState(188);
        match(snuqlParser::T__11);
        setState(189);
        argument();
        setState(190);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__19: {
        enterOuterAlt(_localctx, 2);
        setState(192);
        match(snuqlParser::T__19);
        setState(193);
        match(snuqlParser::T__10);
        setState(194);
        explist();
        setState(195);
        match(snuqlParser::T__11);
        setState(196);
        argument();
        setState(197);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__20: {
        enterOuterAlt(_localctx, 3);
        setState(199);
        match(snuqlParser::T__20);
        setState(200);
        argument();
        setState(201);
        match(snuqlParser::T__21);
        setState(202);
        argument();
        setState(203);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__22: {
        enterOuterAlt(_localctx, 4);
        setState(205);
        match(snuqlParser::T__22);
        setState(206);
        argument();
        setState(207);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__23: {
        enterOuterAlt(_localctx, 5);
        setState(209);
        match(snuqlParser::T__23);
        setState(210);
        argument();
        setState(211);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__24: {
        enterOuterAlt(_localctx, 6);
        setState(213);
        match(snuqlParser::T__24);
        setState(214);
        argument();
        setState(215);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__25: {
        enterOuterAlt(_localctx, 7);
        setState(217);
        match(snuqlParser::T__25);
        setState(218);
        argument();
        setState(219);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__26: {
        enterOuterAlt(_localctx, 8);
        setState(221);
        match(snuqlParser::T__26);
        setState(222);
        argument();
        setState(223);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__27: {
        enterOuterAlt(_localctx, 9);
        setState(225);
        match(snuqlParser::T__27);
        setState(226);
        argument();
        setState(227);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__28: {
        enterOuterAlt(_localctx, 10);
        setState(229);
        match(snuqlParser::T__28);
        setState(230);
        argument();
        setState(231);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__29: {
        enterOuterAlt(_localctx, 11);
        setState(233);
        match(snuqlParser::T__29);
        setState(234);
        argument();
        setState(235);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__30: {
        enterOuterAlt(_localctx, 12);
        setState(237);
        match(snuqlParser::T__30);
        setState(238);
        argument();
        setState(239);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__31: {
        enterOuterAlt(_localctx, 13);
        setState(241);
        match(snuqlParser::T__31);
        setState(242);
        argument();
        setState(243);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__32: {
        enterOuterAlt(_localctx, 14);
        setState(245);
        match(snuqlParser::T__32);
        setState(246);
        argument();
        setState(247);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__33: {
        enterOuterAlt(_localctx, 15);
        setState(249);
        match(snuqlParser::T__33);
        setState(250);
        match(snuqlParser::T__10);
        setState(251);
        explist();
        setState(252);
        match(snuqlParser::T__11);
        setState(253);
        argument();
        setState(254);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__34: {
        enterOuterAlt(_localctx, 16);
        setState(256);
        match(snuqlParser::T__34);
        setState(257);
        match(snuqlParser::T__10);
        setState(258);
        explist();
        setState(259);
        match(snuqlParser::T__11);
        setState(260);
        argument();
        setState(261);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__35: {
        enterOuterAlt(_localctx, 17);
        setState(263);
        match(snuqlParser::T__35);
        setState(264);
        match(snuqlParser::T__10);
        setState(265);
        explist();
        setState(266);
        match(snuqlParser::T__11);
        setState(267);
        argument();
        setState(268);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__36: {
        enterOuterAlt(_localctx, 18);
        setState(270);
        match(snuqlParser::T__36);
        setState(271);
        match(snuqlParser::T__10);
        setState(272);
        explist();
        setState(273);
        match(snuqlParser::T__11);
        setState(274);
        argument();
        setState(275);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__37: {
        enterOuterAlt(_localctx, 19);
        setState(277);
        match(snuqlParser::T__37);
        setState(278);
        match(snuqlParser::T__10);
        setState(279);
        explist();
        setState(280);
        match(snuqlParser::T__11);
        setState(281);
        argument();
        setState(282);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__38: {
        enterOuterAlt(_localctx, 20);
        setState(284);
        match(snuqlParser::T__38);
        setState(285);
        match(snuqlParser::T__10);
        setState(286);
        explist();
        setState(287);
        match(snuqlParser::T__11);
        setState(288);
        argument();
        setState(289);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__39: {
        enterOuterAlt(_localctx, 21);
        setState(291);
        match(snuqlParser::T__39);
        setState(292);
        argument();
        setState(293);
        match(snuqlParser::T__21);
        setState(294);
        argument();
        setState(295);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__40: {
        enterOuterAlt(_localctx, 22);
        setState(297);
        match(snuqlParser::T__40);
        setState(298);
        argument();
        setState(299);
        match(snuqlParser::T__21);
        setState(300);
        argument();
        setState(301);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__41: {
        enterOuterAlt(_localctx, 23);
        setState(303);
        match(snuqlParser::T__41);
        setState(304);
        argument();
        setState(305);
        match(snuqlParser::T__21);
        setState(306);
        argument();
        setState(307);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__42: {
        enterOuterAlt(_localctx, 24);
        setState(309);
        match(snuqlParser::T__42);
        setState(310);
        argument();
        setState(311);
        match(snuqlParser::T__21);
        setState(312);
        argument();
        setState(313);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__43: {
        enterOuterAlt(_localctx, 25);
        setState(315);
        match(snuqlParser::T__43);
        setState(316);
        argument();
        setState(317);
        match(snuqlParser::T__21);
        setState(318);
        argument();
        setState(319);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__44: {
        enterOuterAlt(_localctx, 26);
        setState(321);
        match(snuqlParser::T__44);
        setState(322);
        match(snuqlParser::T__10);
        setState(323);
        explist();
        setState(324);
        match(snuqlParser::T__11);
        setState(325);
        argument();
        setState(326);
        match(snuqlParser::T__21);
        setState(327);
        argument();
        setState(328);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__45: {
        enterOuterAlt(_localctx, 27);
        setState(330);
        match(snuqlParser::T__45);
        setState(331);
        match(snuqlParser::T__10);
        setState(332);
        explist();
        setState(333);
        match(snuqlParser::T__11);
        setState(334);
        argument();
        setState(335);
        match(snuqlParser::T__21);
        setState(336);
        argument();
        setState(337);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__46: {
        enterOuterAlt(_localctx, 28);
        setState(339);
        match(snuqlParser::T__46);
        setState(340);
        match(snuqlParser::T__10);
        setState(341);
        explist();
        setState(342);
        match(snuqlParser::T__11);
        setState(343);
        argument();
        setState(344);
        match(snuqlParser::T__21);
        setState(345);
        argument();
        setState(346);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__47: {
        enterOuterAlt(_localctx, 29);
        setState(348);
        match(snuqlParser::T__47);
        setState(349);
        match(snuqlParser::T__10);
        setState(350);
        explist();
        setState(351);
        match(snuqlParser::T__11);
        setState(352);
        argument();
        setState(353);
        match(snuqlParser::T__21);
        setState(354);
        argument();
        setState(355);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__48: {
        enterOuterAlt(_localctx, 30);
        setState(357);
        match(snuqlParser::T__48);
        setState(358);
        match(snuqlParser::T__10);
        setState(359);
        explist();
        setState(360);
        match(snuqlParser::T__11);
        setState(361);
        argument();
        setState(362);
        match(snuqlParser::T__21);
        setState(363);
        argument();
        setState(364);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__49: {
        enterOuterAlt(_localctx, 31);
        setState(366);
        match(snuqlParser::T__49);
        setState(367);
        match(snuqlParser::T__10);
        setState(368);
        explist();
        setState(369);
        match(snuqlParser::T__11);
        setState(370);
        argument();
        setState(371);
        match(snuqlParser::T__21);
        setState(372);
        argument();
        setState(373);
        match(snuqlParser::T__1);
        break;
      }

      case snuqlParser::T__50: {
        enterOuterAlt(_localctx, 32);
        setState(375);
        match(snuqlParser::T__50);
        setState(376);
        arglist();
        setState(377);
        match(snuqlParser::T__1);
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

snuqlParser::CustomOpContext::CustomOpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::CustomOpContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

snuqlParser::AnylistContext* snuqlParser::CustomOpContext::anylist() {
  return getRuleContext<snuqlParser::AnylistContext>(0);
}

snuqlParser::ExplistContext* snuqlParser::CustomOpContext::explist() {
  return getRuleContext<snuqlParser::ExplistContext>(0);
}


size_t snuqlParser::CustomOpContext::getRuleIndex() const {
  return snuqlParser::RuleCustomOp;
}

void snuqlParser::CustomOpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCustomOp(this);
}

void snuqlParser::CustomOpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCustomOp(this);
}

snuqlParser::CustomOpContext* snuqlParser::customOp() {
  CustomOpContext *_localctx = _tracker.createInstance<CustomOpContext>(_ctx, getState());
  enterRule(_localctx, 38, snuqlParser::RuleCustomOp);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(398);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(381);
      match(snuqlParser::Identifier);
      setState(382);
      anylist();
      setState(383);
      match(snuqlParser::T__1);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(385);
      match(snuqlParser::Identifier);
      setState(386);
      match(snuqlParser::T__10);
      setState(387);
      match(snuqlParser::T__11);
      setState(388);
      anylist();
      setState(389);
      match(snuqlParser::T__1);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(391);
      match(snuqlParser::Identifier);
      setState(392);
      match(snuqlParser::T__10);
      setState(393);
      explist();
      setState(394);
      match(snuqlParser::T__11);
      setState(395);
      anylist();
      setState(396);
      match(snuqlParser::T__1);
      break;
    }

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

snuqlParser::AnylistContext::AnylistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::IdlistContext* snuqlParser::AnylistContext::idlist() {
  return getRuleContext<snuqlParser::IdlistContext>(0);
}

snuqlParser::MixedlistContext* snuqlParser::AnylistContext::mixedlist() {
  return getRuleContext<snuqlParser::MixedlistContext>(0);
}


size_t snuqlParser::AnylistContext::getRuleIndex() const {
  return snuqlParser::RuleAnylist;
}

void snuqlParser::AnylistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnylist(this);
}

void snuqlParser::AnylistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnylist(this);
}

snuqlParser::AnylistContext* snuqlParser::anylist() {
  AnylistContext *_localctx = _tracker.createInstance<AnylistContext>(_ctx, getState());
  enterRule(_localctx, 40, snuqlParser::RuleAnylist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(402);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(400);
      idlist();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(401);
      mixedlist();
      break;
    }

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

snuqlParser::IdlistContext::IdlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::IdlistContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

snuqlParser::IdlistContext* snuqlParser::IdlistContext::idlist() {
  return getRuleContext<snuqlParser::IdlistContext>(0);
}


size_t snuqlParser::IdlistContext::getRuleIndex() const {
  return snuqlParser::RuleIdlist;
}

void snuqlParser::IdlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdlist(this);
}

void snuqlParser::IdlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdlist(this);
}

snuqlParser::IdlistContext* snuqlParser::idlist() {
  IdlistContext *_localctx = _tracker.createInstance<IdlistContext>(_ctx, getState());
  enterRule(_localctx, 42, snuqlParser::RuleIdlist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(408);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(404);
      match(snuqlParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(405);
      match(snuqlParser::Identifier);
      setState(406);
      match(snuqlParser::T__21);
      setState(407);
      idlist();
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MixedlistContext ------------------------------------------------------------------

snuqlParser::MixedlistContext::MixedlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::MixedlistContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

tree::TerminalNode* snuqlParser::MixedlistContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}

snuqlParser::MixedlistContext* snuqlParser::MixedlistContext::mixedlist() {
  return getRuleContext<snuqlParser::MixedlistContext>(0);
}

snuqlParser::IdlistContext* snuqlParser::MixedlistContext::idlist() {
  return getRuleContext<snuqlParser::IdlistContext>(0);
}


size_t snuqlParser::MixedlistContext::getRuleIndex() const {
  return snuqlParser::RuleMixedlist;
}

void snuqlParser::MixedlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMixedlist(this);
}

void snuqlParser::MixedlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMixedlist(this);
}

snuqlParser::MixedlistContext* snuqlParser::mixedlist() {
  MixedlistContext *_localctx = _tracker.createInstance<MixedlistContext>(_ctx, getState());
  enterRule(_localctx, 44, snuqlParser::RuleMixedlist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(429);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(410);
      match(snuqlParser::Identifier);
      setState(411);
      match(snuqlParser::T__4);
      setState(412);
      match(snuqlParser::Integer);
      setState(413);
      match(snuqlParser::T__5);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(414);
      match(snuqlParser::Identifier);
      setState(415);
      match(snuqlParser::T__21);
      setState(416);
      mixedlist();
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(417);
      match(snuqlParser::Identifier);
      setState(418);
      match(snuqlParser::T__4);
      setState(419);
      match(snuqlParser::Integer);
      setState(420);
      match(snuqlParser::T__5);
      setState(421);
      match(snuqlParser::T__21);
      setState(422);
      mixedlist();
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(423);
      match(snuqlParser::Identifier);
      setState(424);
      match(snuqlParser::T__4);
      setState(425);
      match(snuqlParser::Integer);
      setState(426);
      match(snuqlParser::T__5);
      setState(427);
      match(snuqlParser::T__21);
      setState(428);
      idlist();
      break;
    }

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

snuqlParser::ArglistContext::ArglistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::ArgumentContext* snuqlParser::ArglistContext::argument() {
  return getRuleContext<snuqlParser::ArgumentContext>(0);
}

snuqlParser::ArglistContext* snuqlParser::ArglistContext::arglist() {
  return getRuleContext<snuqlParser::ArglistContext>(0);
}


size_t snuqlParser::ArglistContext::getRuleIndex() const {
  return snuqlParser::RuleArglist;
}

void snuqlParser::ArglistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArglist(this);
}

void snuqlParser::ArglistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArglist(this);
}

snuqlParser::ArglistContext* snuqlParser::arglist() {
  ArglistContext *_localctx = _tracker.createInstance<ArglistContext>(_ctx, getState());
  enterRule(_localctx, 46, snuqlParser::RuleArglist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(436);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(431);
      argument();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(432);
      argument();
      setState(433);
      match(snuqlParser::T__21);
      setState(434);
      arglist();
      break;
    }

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

snuqlParser::ArgumentContext::ArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::ArgumentContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

tree::TerminalNode* snuqlParser::ArgumentContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}


size_t snuqlParser::ArgumentContext::getRuleIndex() const {
  return snuqlParser::RuleArgument;
}

void snuqlParser::ArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArgument(this);
}

void snuqlParser::ArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArgument(this);
}

snuqlParser::ArgumentContext* snuqlParser::argument() {
  ArgumentContext *_localctx = _tracker.createInstance<ArgumentContext>(_ctx, getState());
  enterRule(_localctx, 48, snuqlParser::RuleArgument);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(443);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(438);
      match(snuqlParser::Identifier);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(439);
      match(snuqlParser::Identifier);
      setState(440);
      match(snuqlParser::T__4);
      setState(441);
      match(snuqlParser::Integer);
      setState(442);
      match(snuqlParser::T__5);
      break;
    }

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

snuqlParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

snuqlParser::ExpContext* snuqlParser::ExplistContext::exp() {
  return getRuleContext<snuqlParser::ExpContext>(0);
}

snuqlParser::ExplistContext* snuqlParser::ExplistContext::explist() {
  return getRuleContext<snuqlParser::ExplistContext>(0);
}


size_t snuqlParser::ExplistContext::getRuleIndex() const {
  return snuqlParser::RuleExplist;
}

void snuqlParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void snuqlParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}

snuqlParser::ExplistContext* snuqlParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 50, snuqlParser::RuleExplist);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(450);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(445);
      exp(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(446);
      exp(0);
      setState(447);
      match(snuqlParser::T__21);
      setState(448);
      explist();
      break;
    }

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

snuqlParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* snuqlParser::ExpContext::Real() {
  return getToken(snuqlParser::Real, 0);
}

tree::TerminalNode* snuqlParser::ExpContext::Integer() {
  return getToken(snuqlParser::Integer, 0);
}

tree::TerminalNode* snuqlParser::ExpContext::Identifier() {
  return getToken(snuqlParser::Identifier, 0);
}

snuqlParser::UnaryopContext* snuqlParser::ExpContext::unaryop() {
  return getRuleContext<snuqlParser::UnaryopContext>(0);
}

std::vector<snuqlParser::ExpContext *> snuqlParser::ExpContext::exp() {
  return getRuleContexts<snuqlParser::ExpContext>();
}

snuqlParser::ExpContext* snuqlParser::ExpContext::exp(size_t i) {
  return getRuleContext<snuqlParser::ExpContext>(i);
}

snuqlParser::BinopContext* snuqlParser::ExpContext::binop() {
  return getRuleContext<snuqlParser::BinopContext>(0);
}


size_t snuqlParser::ExpContext::getRuleIndex() const {
  return snuqlParser::RuleExp;
}

void snuqlParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void snuqlParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


snuqlParser::ExpContext* snuqlParser::exp() {
   return exp(0);
}

snuqlParser::ExpContext* snuqlParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  snuqlParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  snuqlParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 52;
  enterRecursionRule(_localctx, 52, snuqlParser::RuleExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(468);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case snuqlParser::Real: {
        setState(453);
        match(snuqlParser::Real);
        break;
      }

      case snuqlParser::Integer: {
        setState(454);
        match(snuqlParser::Integer);
        break;
      }

      case snuqlParser::T__51: {
        setState(455);
        match(snuqlParser::T__51);
        break;
      }

      case snuqlParser::Identifier: {
        setState(456);
        match(snuqlParser::Identifier);
        break;
      }

      case snuqlParser::T__57:
      case snuqlParser::T__58:
      case snuqlParser::T__59:
      case snuqlParser::T__60:
      case snuqlParser::T__61:
      case snuqlParser::T__62: {
        setState(457);
        unaryop();
        setState(458);
        match(snuqlParser::T__10);
        setState(459);
        exp(0);
        setState(460);
        match(snuqlParser::T__11);
        break;
      }

      case snuqlParser::T__10: {
        setState(462);
        match(snuqlParser::T__10);
        setState(463);
        exp(0);
        setState(464);
        match(snuqlParser::T__11);
        break;
      }

      case snuqlParser::T__52: {
        setState(466);
        match(snuqlParser::T__52);
        setState(467);
        exp(1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(476);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleExp);
        setState(470);

        if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
        setState(471);
        binop();
        setState(472);
        exp(5); 
      }
      setState(478);
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

snuqlParser::BinopContext::BinopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t snuqlParser::BinopContext::getRuleIndex() const {
  return snuqlParser::RuleBinop;
}

void snuqlParser::BinopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBinop(this);
}

void snuqlParser::BinopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBinop(this);
}

snuqlParser::BinopContext* snuqlParser::binop() {
  BinopContext *_localctx = _tracker.createInstance<BinopContext>(_ctx, getState());
  enterRule(_localctx, 54, snuqlParser::RuleBinop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(479);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << snuqlParser::T__52)
      | (1ULL << snuqlParser::T__53)
      | (1ULL << snuqlParser::T__54)
      | (1ULL << snuqlParser::T__55)
      | (1ULL << snuqlParser::T__56))) != 0))) {
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

snuqlParser::UnaryopContext::UnaryopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t snuqlParser::UnaryopContext::getRuleIndex() const {
  return snuqlParser::RuleUnaryop;
}

void snuqlParser::UnaryopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnaryop(this);
}

void snuqlParser::UnaryopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<snuqlListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnaryop(this);
}

snuqlParser::UnaryopContext* snuqlParser::unaryop() {
  UnaryopContext *_localctx = _tracker.createInstance<UnaryopContext>(_ctx, getState());
  enterRule(_localctx, 56, snuqlParser::RuleUnaryop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(481);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << snuqlParser::T__57)
      | (1ULL << snuqlParser::T__58)
      | (1ULL << snuqlParser::T__59)
      | (1ULL << snuqlParser::T__60)
      | (1ULL << snuqlParser::T__61)
      | (1ULL << snuqlParser::T__62))) != 0))) {
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

bool snuqlParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 4: return programSempred(dynamic_cast<ProgramContext *>(context), predicateIndex);
    case 26: return expSempred(dynamic_cast<ExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool snuqlParser::programSempred(ProgramContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool snuqlParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 4);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> snuqlParser::_decisionToDFA;
atn::PredictionContextCache snuqlParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN snuqlParser::_atn;
std::vector<uint16_t> snuqlParser::_serializedATN;

std::vector<std::string> snuqlParser::_ruleNames = {
  "mainprogram", "header", "version", "include", "program", "statement", 
  "decl", "quantumDecl", "classicalDecl", "gatedeclStatement", "gatedecl", 
  "goplist", "qopStatement", "uopStatement", "measureQop", "resetQop", "ifStatement", 
  "barrierStatement", "unitaryOp", "customOp", "anylist", "idlist", "mixedlist", 
  "arglist", "argument", "explist", "exp", "binop", "unaryop"
};

std::vector<std::string> snuqlParser::_literalNames = {
  "", "'SOQSL'", "';'", "'include'", "'qreg'", "'['", "']'", "'creg'", "'}'", 
  "'gate'", "'{'", "'('", "')'", "'measure'", "'->'", "'reset'", "'if'", 
  "'=='", "'barrier'", "'U'", "'u'", "'CX'", "','", "'id'", "'h'", "'x'", 
  "'y'", "'z'", "'sx'", "'sy'", "'s'", "'sdg'", "'t'", "'tdg'", "'rx'", 
  "'ry'", "'rz'", "'u1'", "'u2'", "'u3'", "'swap'", "'cx'", "'cy'", "'cz'", 
  "'ch'", "'crx'", "'cry'", "'crz'", "'cu1'", "'cu2'", "'cu3'", "'nswap'", 
  "'pi'", "'-'", "'+'", "'*'", "'/'", "'^'", "'sin'", "'cos'", "'tan'", 
  "'exp'", "'ln'", "'sqrt'"
};

std::vector<std::string> snuqlParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "", "", "", "", "", "", "", "", "", "", "StringLiteral", "Real", "Integer", 
  "Decimal", "SciSuffix", "Identifier", "Whitespace", "Newline", "LineComment", 
  "BlockComment"
};

dfa::Vocabulary snuqlParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> snuqlParser::_tokenNames;

snuqlParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x4b, 0x1e6, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x3, 
    0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x5, 0x3, 0x41, 0xa, 0x3, 0x3, 0x3, 
    0x7, 0x3, 0x44, 0xa, 0x3, 0xc, 0x3, 0xe, 0x3, 0x47, 0xb, 0x3, 0x3, 0x4, 
    0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x7, 0x6, 0x56, 0xa, 
    0x6, 0xc, 0x6, 0xe, 0x6, 0x59, 0xb, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x60, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x5, 
    0x8, 0x64, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0x7b, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 0xc, 0x91, 0xa, 0xc, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x5, 0xd, 0x9b, 0xa, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0xa0, 
    0xa, 0xe, 0x3, 0xf, 0x3, 0xf, 0x5, 0xf, 0xa4, 0xa, 0xf, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 
    0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 
    0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 
    0x5, 0x14, 0x17e, 0xa, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 
    0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 
    0x5, 0x15, 0x191, 0xa, 0x15, 0x3, 0x16, 0x3, 0x16, 0x5, 0x16, 0x195, 
    0xa, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x3, 0x17, 0x5, 0x17, 0x19b, 
    0xa, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 
    0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x18, 0x5, 0x18, 0x1b0, 0xa, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0x1b7, 0xa, 0x19, 0x3, 0x1a, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x1be, 0xa, 0x1a, 0x3, 0x1b, 
    0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x5, 0x1b, 0x1c5, 0xa, 0x1b, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 
    0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x5, 0x1c, 0x1d7, 0xa, 0x1c, 0x3, 0x1c, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x7, 0x1c, 0x1dd, 0xa, 0x1c, 0xc, 0x1c, 
    0xe, 0x1c, 0x1e0, 0xb, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x2, 0x4, 0xa, 0x36, 0x1f, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 
    0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 
    0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x2, 
    0x5, 0x3, 0x2, 0x43, 0x44, 0x3, 0x2, 0x37, 0x3b, 0x3, 0x2, 0x3c, 0x41, 
    0x2, 0x209, 0x2, 0x3c, 0x3, 0x2, 0x2, 0x2, 0x4, 0x40, 0x3, 0x2, 0x2, 
    0x2, 0x6, 0x48, 0x3, 0x2, 0x2, 0x2, 0x8, 0x4c, 0x3, 0x2, 0x2, 0x2, 0xa, 
    0x50, 0x3, 0x2, 0x2, 0x2, 0xc, 0x5f, 0x3, 0x2, 0x2, 0x2, 0xe, 0x63, 
    0x3, 0x2, 0x2, 0x2, 0x10, 0x65, 0x3, 0x2, 0x2, 0x2, 0x12, 0x6c, 0x3, 
    0x2, 0x2, 0x2, 0x14, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x16, 0x90, 0x3, 0x2, 
    0x2, 0x2, 0x18, 0x9a, 0x3, 0x2, 0x2, 0x2, 0x1a, 0x9f, 0x3, 0x2, 0x2, 
    0x2, 0x1c, 0xa3, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xa5, 0x3, 0x2, 0x2, 0x2, 
    0x20, 0xab, 0x3, 0x2, 0x2, 0x2, 0x22, 0xaf, 0x3, 0x2, 0x2, 0x2, 0x24, 
    0xb7, 0x3, 0x2, 0x2, 0x2, 0x26, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x28, 0x190, 
    0x3, 0x2, 0x2, 0x2, 0x2a, 0x194, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x19a, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x30, 0x1b6, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x34, 0x1c4, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0x1d6, 0x3, 0x2, 0x2, 0x2, 0x38, 0x1e1, 0x3, 0x2, 0x2, 0x2, 
    0x3a, 0x1e3, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x3d, 0x5, 0x4, 0x3, 0x2, 0x3d, 
    0x3e, 0x5, 0xa, 0x6, 0x2, 0x3e, 0x3, 0x3, 0x2, 0x2, 0x2, 0x3f, 0x41, 
    0x5, 0x6, 0x4, 0x2, 0x40, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x40, 0x41, 0x3, 
    0x2, 0x2, 0x2, 0x41, 0x45, 0x3, 0x2, 0x2, 0x2, 0x42, 0x44, 0x5, 0x8, 
    0x5, 0x2, 0x43, 0x42, 0x3, 0x2, 0x2, 0x2, 0x44, 0x47, 0x3, 0x2, 0x2, 
    0x2, 0x45, 0x43, 0x3, 0x2, 0x2, 0x2, 0x45, 0x46, 0x3, 0x2, 0x2, 0x2, 
    0x46, 0x5, 0x3, 0x2, 0x2, 0x2, 0x47, 0x45, 0x3, 0x2, 0x2, 0x2, 0x48, 
    0x49, 0x7, 0x3, 0x2, 0x2, 0x49, 0x4a, 0x9, 0x2, 0x2, 0x2, 0x4a, 0x4b, 
    0x7, 0x4, 0x2, 0x2, 0x4b, 0x7, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x4d, 0x7, 
    0x5, 0x2, 0x2, 0x4d, 0x4e, 0x7, 0x42, 0x2, 0x2, 0x4e, 0x4f, 0x7, 0x4, 
    0x2, 0x2, 0x4f, 0x9, 0x3, 0x2, 0x2, 0x2, 0x50, 0x51, 0x8, 0x6, 0x1, 
    0x2, 0x51, 0x52, 0x5, 0xc, 0x7, 0x2, 0x52, 0x57, 0x3, 0x2, 0x2, 0x2, 
    0x53, 0x54, 0xc, 0x3, 0x2, 0x2, 0x54, 0x56, 0x5, 0xc, 0x7, 0x2, 0x55, 
    0x53, 0x3, 0x2, 0x2, 0x2, 0x56, 0x59, 0x3, 0x2, 0x2, 0x2, 0x57, 0x55, 
    0x3, 0x2, 0x2, 0x2, 0x57, 0x58, 0x3, 0x2, 0x2, 0x2, 0x58, 0xb, 0x3, 
    0x2, 0x2, 0x2, 0x59, 0x57, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x60, 0x5, 0xe, 
    0x8, 0x2, 0x5b, 0x60, 0x5, 0x14, 0xb, 0x2, 0x5c, 0x60, 0x5, 0x1a, 0xe, 
    0x2, 0x5d, 0x60, 0x5, 0x22, 0x12, 0x2, 0x5e, 0x60, 0x5, 0x24, 0x13, 
    0x2, 0x5f, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5b, 0x3, 0x2, 0x2, 0x2, 
    0x5f, 0x5c, 0x3, 0x2, 0x2, 0x2, 0x5f, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x5f, 
    0x5e, 0x3, 0x2, 0x2, 0x2, 0x60, 0xd, 0x3, 0x2, 0x2, 0x2, 0x61, 0x64, 
    0x5, 0x10, 0x9, 0x2, 0x62, 0x64, 0x5, 0x12, 0xa, 0x2, 0x63, 0x61, 0x3, 
    0x2, 0x2, 0x2, 0x63, 0x62, 0x3, 0x2, 0x2, 0x2, 0x64, 0xf, 0x3, 0x2, 
    0x2, 0x2, 0x65, 0x66, 0x7, 0x6, 0x2, 0x2, 0x66, 0x67, 0x7, 0x47, 0x2, 
    0x2, 0x67, 0x68, 0x7, 0x7, 0x2, 0x2, 0x68, 0x69, 0x7, 0x44, 0x2, 0x2, 
    0x69, 0x6a, 0x7, 0x8, 0x2, 0x2, 0x6a, 0x6b, 0x7, 0x4, 0x2, 0x2, 0x6b, 
    0x11, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x6d, 0x7, 0x9, 0x2, 0x2, 0x6d, 0x6e, 
    0x7, 0x47, 0x2, 0x2, 0x6e, 0x6f, 0x7, 0x7, 0x2, 0x2, 0x6f, 0x70, 0x7, 
    0x44, 0x2, 0x2, 0x70, 0x71, 0x7, 0x8, 0x2, 0x2, 0x71, 0x72, 0x7, 0x4, 
    0x2, 0x2, 0x72, 0x13, 0x3, 0x2, 0x2, 0x2, 0x73, 0x74, 0x5, 0x16, 0xc, 
    0x2, 0x74, 0x75, 0x5, 0x18, 0xd, 0x2, 0x75, 0x76, 0x7, 0xa, 0x2, 0x2, 
    0x76, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x77, 0x78, 0x5, 0x16, 0xc, 0x2, 0x78, 
    0x79, 0x7, 0xa, 0x2, 0x2, 0x79, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x73, 
    0x3, 0x2, 0x2, 0x2, 0x7a, 0x77, 0x3, 0x2, 0x2, 0x2, 0x7b, 0x15, 0x3, 
    0x2, 0x2, 0x2, 0x7c, 0x7d, 0x7, 0xb, 0x2, 0x2, 0x7d, 0x7e, 0x7, 0x47, 
    0x2, 0x2, 0x7e, 0x7f, 0x5, 0x2c, 0x17, 0x2, 0x7f, 0x80, 0x7, 0xc, 0x2, 
    0x2, 0x80, 0x91, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 0x7, 0xb, 0x2, 0x2, 
    0x82, 0x83, 0x7, 0x47, 0x2, 0x2, 0x83, 0x84, 0x7, 0xd, 0x2, 0x2, 0x84, 
    0x85, 0x7, 0xe, 0x2, 0x2, 0x85, 0x86, 0x5, 0x2c, 0x17, 0x2, 0x86, 0x87, 
    0x7, 0xc, 0x2, 0x2, 0x87, 0x91, 0x3, 0x2, 0x2, 0x2, 0x88, 0x89, 0x7, 
    0xb, 0x2, 0x2, 0x89, 0x8a, 0x7, 0x47, 0x2, 0x2, 0x8a, 0x8b, 0x7, 0xd, 
    0x2, 0x2, 0x8b, 0x8c, 0x5, 0x2c, 0x17, 0x2, 0x8c, 0x8d, 0x7, 0xe, 0x2, 
    0x2, 0x8d, 0x8e, 0x5, 0x2c, 0x17, 0x2, 0x8e, 0x8f, 0x7, 0xc, 0x2, 0x2, 
    0x8f, 0x91, 0x3, 0x2, 0x2, 0x2, 0x90, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x90, 
    0x81, 0x3, 0x2, 0x2, 0x2, 0x90, 0x88, 0x3, 0x2, 0x2, 0x2, 0x91, 0x17, 
    0x3, 0x2, 0x2, 0x2, 0x92, 0x9b, 0x5, 0x1c, 0xf, 0x2, 0x93, 0x9b, 0x5, 
    0x24, 0x13, 0x2, 0x94, 0x95, 0x5, 0x1c, 0xf, 0x2, 0x95, 0x96, 0x5, 0x18, 
    0xd, 0x2, 0x96, 0x9b, 0x3, 0x2, 0x2, 0x2, 0x97, 0x98, 0x5, 0x24, 0x13, 
    0x2, 0x98, 0x99, 0x5, 0x18, 0xd, 0x2, 0x99, 0x9b, 0x3, 0x2, 0x2, 0x2, 
    0x9a, 0x92, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x93, 0x3, 0x2, 0x2, 0x2, 0x9a, 
    0x94, 0x3, 0x2, 0x2, 0x2, 0x9a, 0x97, 0x3, 0x2, 0x2, 0x2, 0x9b, 0x19, 
    0x3, 0x2, 0x2, 0x2, 0x9c, 0xa0, 0x5, 0x1c, 0xf, 0x2, 0x9d, 0xa0, 0x5, 
    0x1e, 0x10, 0x2, 0x9e, 0xa0, 0x5, 0x20, 0x11, 0x2, 0x9f, 0x9c, 0x3, 
    0x2, 0x2, 0x2, 0x9f, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x9f, 0x9e, 0x3, 0x2, 
    0x2, 0x2, 0xa0, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xa1, 0xa4, 0x5, 0x26, 0x14, 
    0x2, 0xa2, 0xa4, 0x5, 0x28, 0x15, 0x2, 0xa3, 0xa1, 0x3, 0x2, 0x2, 0x2, 
    0xa3, 0xa2, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xa5, 
    0xa6, 0x7, 0xf, 0x2, 0x2, 0xa6, 0xa7, 0x5, 0x32, 0x1a, 0x2, 0xa7, 0xa8, 
    0x7, 0x10, 0x2, 0x2, 0xa8, 0xa9, 0x5, 0x32, 0x1a, 0x2, 0xa9, 0xaa, 0x7, 
    0x4, 0x2, 0x2, 0xaa, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xab, 0xac, 0x7, 0x11, 
    0x2, 0x2, 0xac, 0xad, 0x5, 0x32, 0x1a, 0x2, 0xad, 0xae, 0x7, 0x4, 0x2, 
    0x2, 0xae, 0x21, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xb0, 0x7, 0x12, 0x2, 0x2, 
    0xb0, 0xb1, 0x7, 0xd, 0x2, 0x2, 0xb1, 0xb2, 0x7, 0x47, 0x2, 0x2, 0xb2, 
    0xb3, 0x7, 0x13, 0x2, 0x2, 0xb3, 0xb4, 0x7, 0x44, 0x2, 0x2, 0xb4, 0xb5, 
    0x7, 0xe, 0x2, 0x2, 0xb5, 0xb6, 0x5, 0x1a, 0xe, 0x2, 0xb6, 0x23, 0x3, 
    0x2, 0x2, 0x2, 0xb7, 0xb8, 0x7, 0x14, 0x2, 0x2, 0xb8, 0xb9, 0x5, 0x2a, 
    0x16, 0x2, 0xb9, 0xba, 0x7, 0x4, 0x2, 0x2, 0xba, 0x25, 0x3, 0x2, 0x2, 
    0x2, 0xbb, 0xbc, 0x7, 0x15, 0x2, 0x2, 0xbc, 0xbd, 0x7, 0xd, 0x2, 0x2, 
    0xbd, 0xbe, 0x5, 0x34, 0x1b, 0x2, 0xbe, 0xbf, 0x7, 0xe, 0x2, 0x2, 0xbf, 
    0xc0, 0x5, 0x32, 0x1a, 0x2, 0xc0, 0xc1, 0x7, 0x4, 0x2, 0x2, 0xc1, 0x17e, 
    0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 0x7, 0x16, 0x2, 0x2, 0xc3, 0xc4, 0x7, 
    0xd, 0x2, 0x2, 0xc4, 0xc5, 0x5, 0x34, 0x1b, 0x2, 0xc5, 0xc6, 0x7, 0xe, 
    0x2, 0x2, 0xc6, 0xc7, 0x5, 0x32, 0x1a, 0x2, 0xc7, 0xc8, 0x7, 0x4, 0x2, 
    0x2, 0xc8, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xca, 0x7, 0x17, 0x2, 0x2, 
    0xca, 0xcb, 0x5, 0x32, 0x1a, 0x2, 0xcb, 0xcc, 0x7, 0x18, 0x2, 0x2, 0xcc, 
    0xcd, 0x5, 0x32, 0x1a, 0x2, 0xcd, 0xce, 0x7, 0x4, 0x2, 0x2, 0xce, 0x17e, 
    0x3, 0x2, 0x2, 0x2, 0xcf, 0xd0, 0x7, 0x19, 0x2, 0x2, 0xd0, 0xd1, 0x5, 
    0x32, 0x1a, 0x2, 0xd1, 0xd2, 0x7, 0x4, 0x2, 0x2, 0xd2, 0x17e, 0x3, 0x2, 
    0x2, 0x2, 0xd3, 0xd4, 0x7, 0x1a, 0x2, 0x2, 0xd4, 0xd5, 0x5, 0x32, 0x1a, 
    0x2, 0xd5, 0xd6, 0x7, 0x4, 0x2, 0x2, 0xd6, 0x17e, 0x3, 0x2, 0x2, 0x2, 
    0xd7, 0xd8, 0x7, 0x1b, 0x2, 0x2, 0xd8, 0xd9, 0x5, 0x32, 0x1a, 0x2, 0xd9, 
    0xda, 0x7, 0x4, 0x2, 0x2, 0xda, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xdb, 0xdc, 
    0x7, 0x1c, 0x2, 0x2, 0xdc, 0xdd, 0x5, 0x32, 0x1a, 0x2, 0xdd, 0xde, 0x7, 
    0x4, 0x2, 0x2, 0xde, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xdf, 0xe0, 0x7, 0x1d, 
    0x2, 0x2, 0xe0, 0xe1, 0x5, 0x32, 0x1a, 0x2, 0xe1, 0xe2, 0x7, 0x4, 0x2, 
    0x2, 0xe2, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xe3, 0xe4, 0x7, 0x1e, 0x2, 0x2, 
    0xe4, 0xe5, 0x5, 0x32, 0x1a, 0x2, 0xe5, 0xe6, 0x7, 0x4, 0x2, 0x2, 0xe6, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe8, 0x7, 0x1f, 0x2, 0x2, 0xe8, 0xe9, 
    0x5, 0x32, 0x1a, 0x2, 0xe9, 0xea, 0x7, 0x4, 0x2, 0x2, 0xea, 0x17e, 0x3, 
    0x2, 0x2, 0x2, 0xeb, 0xec, 0x7, 0x20, 0x2, 0x2, 0xec, 0xed, 0x5, 0x32, 
    0x1a, 0x2, 0xed, 0xee, 0x7, 0x4, 0x2, 0x2, 0xee, 0x17e, 0x3, 0x2, 0x2, 
    0x2, 0xef, 0xf0, 0x7, 0x21, 0x2, 0x2, 0xf0, 0xf1, 0x5, 0x32, 0x1a, 0x2, 
    0xf1, 0xf2, 0x7, 0x4, 0x2, 0x2, 0xf2, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xf3, 
    0xf4, 0x7, 0x22, 0x2, 0x2, 0xf4, 0xf5, 0x5, 0x32, 0x1a, 0x2, 0xf5, 0xf6, 
    0x7, 0x4, 0x2, 0x2, 0xf6, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xf7, 0xf8, 0x7, 
    0x23, 0x2, 0x2, 0xf8, 0xf9, 0x5, 0x32, 0x1a, 0x2, 0xf9, 0xfa, 0x7, 0x4, 
    0x2, 0x2, 0xfa, 0x17e, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfc, 0x7, 0x24, 0x2, 
    0x2, 0xfc, 0xfd, 0x7, 0xd, 0x2, 0x2, 0xfd, 0xfe, 0x5, 0x34, 0x1b, 0x2, 
    0xfe, 0xff, 0x7, 0xe, 0x2, 0x2, 0xff, 0x100, 0x5, 0x32, 0x1a, 0x2, 0x100, 
    0x101, 0x7, 0x4, 0x2, 0x2, 0x101, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x102, 
    0x103, 0x7, 0x25, 0x2, 0x2, 0x103, 0x104, 0x7, 0xd, 0x2, 0x2, 0x104, 
    0x105, 0x5, 0x34, 0x1b, 0x2, 0x105, 0x106, 0x7, 0xe, 0x2, 0x2, 0x106, 
    0x107, 0x5, 0x32, 0x1a, 0x2, 0x107, 0x108, 0x7, 0x4, 0x2, 0x2, 0x108, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x109, 0x10a, 0x7, 0x26, 0x2, 0x2, 0x10a, 
    0x10b, 0x7, 0xd, 0x2, 0x2, 0x10b, 0x10c, 0x5, 0x34, 0x1b, 0x2, 0x10c, 
    0x10d, 0x7, 0xe, 0x2, 0x2, 0x10d, 0x10e, 0x5, 0x32, 0x1a, 0x2, 0x10e, 
    0x10f, 0x7, 0x4, 0x2, 0x2, 0x10f, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x110, 
    0x111, 0x7, 0x27, 0x2, 0x2, 0x111, 0x112, 0x7, 0xd, 0x2, 0x2, 0x112, 
    0x113, 0x5, 0x34, 0x1b, 0x2, 0x113, 0x114, 0x7, 0xe, 0x2, 0x2, 0x114, 
    0x115, 0x5, 0x32, 0x1a, 0x2, 0x115, 0x116, 0x7, 0x4, 0x2, 0x2, 0x116, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x117, 0x118, 0x7, 0x28, 0x2, 0x2, 0x118, 
    0x119, 0x7, 0xd, 0x2, 0x2, 0x119, 0x11a, 0x5, 0x34, 0x1b, 0x2, 0x11a, 
    0x11b, 0x7, 0xe, 0x2, 0x2, 0x11b, 0x11c, 0x5, 0x32, 0x1a, 0x2, 0x11c, 
    0x11d, 0x7, 0x4, 0x2, 0x2, 0x11d, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x11e, 
    0x11f, 0x7, 0x29, 0x2, 0x2, 0x11f, 0x120, 0x7, 0xd, 0x2, 0x2, 0x120, 
    0x121, 0x5, 0x34, 0x1b, 0x2, 0x121, 0x122, 0x7, 0xe, 0x2, 0x2, 0x122, 
    0x123, 0x5, 0x32, 0x1a, 0x2, 0x123, 0x124, 0x7, 0x4, 0x2, 0x2, 0x124, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x125, 0x126, 0x7, 0x2a, 0x2, 0x2, 0x126, 
    0x127, 0x5, 0x32, 0x1a, 0x2, 0x127, 0x128, 0x7, 0x18, 0x2, 0x2, 0x128, 
    0x129, 0x5, 0x32, 0x1a, 0x2, 0x129, 0x12a, 0x7, 0x4, 0x2, 0x2, 0x12a, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x12c, 0x7, 0x2b, 0x2, 0x2, 0x12c, 
    0x12d, 0x5, 0x32, 0x1a, 0x2, 0x12d, 0x12e, 0x7, 0x18, 0x2, 0x2, 0x12e, 
    0x12f, 0x5, 0x32, 0x1a, 0x2, 0x12f, 0x130, 0x7, 0x4, 0x2, 0x2, 0x130, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x131, 0x132, 0x7, 0x2c, 0x2, 0x2, 0x132, 
    0x133, 0x5, 0x32, 0x1a, 0x2, 0x133, 0x134, 0x7, 0x18, 0x2, 0x2, 0x134, 
    0x135, 0x5, 0x32, 0x1a, 0x2, 0x135, 0x136, 0x7, 0x4, 0x2, 0x2, 0x136, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x137, 0x138, 0x7, 0x2d, 0x2, 0x2, 0x138, 
    0x139, 0x5, 0x32, 0x1a, 0x2, 0x139, 0x13a, 0x7, 0x18, 0x2, 0x2, 0x13a, 
    0x13b, 0x5, 0x32, 0x1a, 0x2, 0x13b, 0x13c, 0x7, 0x4, 0x2, 0x2, 0x13c, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x13e, 0x7, 0x2e, 0x2, 0x2, 0x13e, 
    0x13f, 0x5, 0x32, 0x1a, 0x2, 0x13f, 0x140, 0x7, 0x18, 0x2, 0x2, 0x140, 
    0x141, 0x5, 0x32, 0x1a, 0x2, 0x141, 0x142, 0x7, 0x4, 0x2, 0x2, 0x142, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x143, 0x144, 0x7, 0x2f, 0x2, 0x2, 0x144, 
    0x145, 0x7, 0xd, 0x2, 0x2, 0x145, 0x146, 0x5, 0x34, 0x1b, 0x2, 0x146, 
    0x147, 0x7, 0xe, 0x2, 0x2, 0x147, 0x148, 0x5, 0x32, 0x1a, 0x2, 0x148, 
    0x149, 0x7, 0x18, 0x2, 0x2, 0x149, 0x14a, 0x5, 0x32, 0x1a, 0x2, 0x14a, 
    0x14b, 0x7, 0x4, 0x2, 0x2, 0x14b, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x14c, 
    0x14d, 0x7, 0x30, 0x2, 0x2, 0x14d, 0x14e, 0x7, 0xd, 0x2, 0x2, 0x14e, 
    0x14f, 0x5, 0x34, 0x1b, 0x2, 0x14f, 0x150, 0x7, 0xe, 0x2, 0x2, 0x150, 
    0x151, 0x5, 0x32, 0x1a, 0x2, 0x151, 0x152, 0x7, 0x18, 0x2, 0x2, 0x152, 
    0x153, 0x5, 0x32, 0x1a, 0x2, 0x153, 0x154, 0x7, 0x4, 0x2, 0x2, 0x154, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x155, 0x156, 0x7, 0x31, 0x2, 0x2, 0x156, 
    0x157, 0x7, 0xd, 0x2, 0x2, 0x157, 0x158, 0x5, 0x34, 0x1b, 0x2, 0x158, 
    0x159, 0x7, 0xe, 0x2, 0x2, 0x159, 0x15a, 0x5, 0x32, 0x1a, 0x2, 0x15a, 
    0x15b, 0x7, 0x18, 0x2, 0x2, 0x15b, 0x15c, 0x5, 0x32, 0x1a, 0x2, 0x15c, 
    0x15d, 0x7, 0x4, 0x2, 0x2, 0x15d, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x15e, 
    0x15f, 0x7, 0x32, 0x2, 0x2, 0x15f, 0x160, 0x7, 0xd, 0x2, 0x2, 0x160, 
    0x161, 0x5, 0x34, 0x1b, 0x2, 0x161, 0x162, 0x7, 0xe, 0x2, 0x2, 0x162, 
    0x163, 0x5, 0x32, 0x1a, 0x2, 0x163, 0x164, 0x7, 0x18, 0x2, 0x2, 0x164, 
    0x165, 0x5, 0x32, 0x1a, 0x2, 0x165, 0x166, 0x7, 0x4, 0x2, 0x2, 0x166, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x167, 0x168, 0x7, 0x33, 0x2, 0x2, 0x168, 
    0x169, 0x7, 0xd, 0x2, 0x2, 0x169, 0x16a, 0x5, 0x34, 0x1b, 0x2, 0x16a, 
    0x16b, 0x7, 0xe, 0x2, 0x2, 0x16b, 0x16c, 0x5, 0x32, 0x1a, 0x2, 0x16c, 
    0x16d, 0x7, 0x18, 0x2, 0x2, 0x16d, 0x16e, 0x5, 0x32, 0x1a, 0x2, 0x16e, 
    0x16f, 0x7, 0x4, 0x2, 0x2, 0x16f, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x170, 
    0x171, 0x7, 0x34, 0x2, 0x2, 0x171, 0x172, 0x7, 0xd, 0x2, 0x2, 0x172, 
    0x173, 0x5, 0x34, 0x1b, 0x2, 0x173, 0x174, 0x7, 0xe, 0x2, 0x2, 0x174, 
    0x175, 0x5, 0x32, 0x1a, 0x2, 0x175, 0x176, 0x7, 0x18, 0x2, 0x2, 0x176, 
    0x177, 0x5, 0x32, 0x1a, 0x2, 0x177, 0x178, 0x7, 0x4, 0x2, 0x2, 0x178, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17a, 0x7, 0x35, 0x2, 0x2, 0x17a, 
    0x17b, 0x5, 0x30, 0x19, 0x2, 0x17b, 0x17c, 0x7, 0x4, 0x2, 0x2, 0x17c, 
    0x17e, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xbb, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xc2, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0xc9, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xcf, 0x3, 
    0x2, 0x2, 0x2, 0x17d, 0xd3, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xd7, 0x3, 0x2, 
    0x2, 0x2, 0x17d, 0xdb, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xdf, 0x3, 0x2, 0x2, 
    0x2, 0x17d, 0xe3, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xe7, 0x3, 0x2, 0x2, 0x2, 
    0x17d, 0xeb, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xef, 0x3, 0x2, 0x2, 0x2, 0x17d, 
    0xf3, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xf7, 0x3, 0x2, 0x2, 0x2, 0x17d, 0xfb, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x102, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x109, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x110, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x117, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x125, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x131, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x137, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x13d, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x143, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x14c, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x155, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x15e, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x167, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x170, 
    0x3, 0x2, 0x2, 0x2, 0x17d, 0x179, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x27, 0x3, 
    0x2, 0x2, 0x2, 0x17f, 0x180, 0x7, 0x47, 0x2, 0x2, 0x180, 0x181, 0x5, 
    0x2a, 0x16, 0x2, 0x181, 0x182, 0x7, 0x4, 0x2, 0x2, 0x182, 0x191, 0x3, 
    0x2, 0x2, 0x2, 0x183, 0x184, 0x7, 0x47, 0x2, 0x2, 0x184, 0x185, 0x7, 
    0xd, 0x2, 0x2, 0x185, 0x186, 0x7, 0xe, 0x2, 0x2, 0x186, 0x187, 0x5, 
    0x2a, 0x16, 0x2, 0x187, 0x188, 0x7, 0x4, 0x2, 0x2, 0x188, 0x191, 0x3, 
    0x2, 0x2, 0x2, 0x189, 0x18a, 0x7, 0x47, 0x2, 0x2, 0x18a, 0x18b, 0x7, 
    0xd, 0x2, 0x2, 0x18b, 0x18c, 0x5, 0x34, 0x1b, 0x2, 0x18c, 0x18d, 0x7, 
    0xe, 0x2, 0x2, 0x18d, 0x18e, 0x5, 0x2a, 0x16, 0x2, 0x18e, 0x18f, 0x7, 
    0x4, 0x2, 0x2, 0x18f, 0x191, 0x3, 0x2, 0x2, 0x2, 0x190, 0x17f, 0x3, 
    0x2, 0x2, 0x2, 0x190, 0x183, 0x3, 0x2, 0x2, 0x2, 0x190, 0x189, 0x3, 
    0x2, 0x2, 0x2, 0x191, 0x29, 0x3, 0x2, 0x2, 0x2, 0x192, 0x195, 0x5, 0x2c, 
    0x17, 0x2, 0x193, 0x195, 0x5, 0x2e, 0x18, 0x2, 0x194, 0x192, 0x3, 0x2, 
    0x2, 0x2, 0x194, 0x193, 0x3, 0x2, 0x2, 0x2, 0x195, 0x2b, 0x3, 0x2, 0x2, 
    0x2, 0x196, 0x19b, 0x7, 0x47, 0x2, 0x2, 0x197, 0x198, 0x7, 0x47, 0x2, 
    0x2, 0x198, 0x199, 0x7, 0x18, 0x2, 0x2, 0x199, 0x19b, 0x5, 0x2c, 0x17, 
    0x2, 0x19a, 0x196, 0x3, 0x2, 0x2, 0x2, 0x19a, 0x197, 0x3, 0x2, 0x2, 
    0x2, 0x19b, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x19c, 0x19d, 0x7, 0x47, 0x2, 
    0x2, 0x19d, 0x19e, 0x7, 0x7, 0x2, 0x2, 0x19e, 0x19f, 0x7, 0x44, 0x2, 
    0x2, 0x19f, 0x1b0, 0x7, 0x8, 0x2, 0x2, 0x1a0, 0x1a1, 0x7, 0x47, 0x2, 
    0x2, 0x1a1, 0x1a2, 0x7, 0x18, 0x2, 0x2, 0x1a2, 0x1b0, 0x5, 0x2e, 0x18, 
    0x2, 0x1a3, 0x1a4, 0x7, 0x47, 0x2, 0x2, 0x1a4, 0x1a5, 0x7, 0x7, 0x2, 
    0x2, 0x1a5, 0x1a6, 0x7, 0x44, 0x2, 0x2, 0x1a6, 0x1a7, 0x7, 0x8, 0x2, 
    0x2, 0x1a7, 0x1a8, 0x7, 0x18, 0x2, 0x2, 0x1a8, 0x1b0, 0x5, 0x2e, 0x18, 
    0x2, 0x1a9, 0x1aa, 0x7, 0x47, 0x2, 0x2, 0x1aa, 0x1ab, 0x7, 0x7, 0x2, 
    0x2, 0x1ab, 0x1ac, 0x7, 0x44, 0x2, 0x2, 0x1ac, 0x1ad, 0x7, 0x8, 0x2, 
    0x2, 0x1ad, 0x1ae, 0x7, 0x18, 0x2, 0x2, 0x1ae, 0x1b0, 0x5, 0x2c, 0x17, 
    0x2, 0x1af, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x1af, 0x1a0, 0x3, 0x2, 0x2, 
    0x2, 0x1af, 0x1a3, 0x3, 0x2, 0x2, 0x2, 0x1af, 0x1a9, 0x3, 0x2, 0x2, 
    0x2, 0x1b0, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1b7, 0x5, 0x32, 0x1a, 
    0x2, 0x1b2, 0x1b3, 0x5, 0x32, 0x1a, 0x2, 0x1b3, 0x1b4, 0x7, 0x18, 0x2, 
    0x2, 0x1b4, 0x1b5, 0x5, 0x30, 0x19, 0x2, 0x1b5, 0x1b7, 0x3, 0x2, 0x2, 
    0x2, 0x1b6, 0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b2, 0x3, 0x2, 0x2, 
    0x2, 0x1b7, 0x31, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1be, 0x7, 0x47, 0x2, 
    0x2, 0x1b9, 0x1ba, 0x7, 0x47, 0x2, 0x2, 0x1ba, 0x1bb, 0x7, 0x7, 0x2, 
    0x2, 0x1bb, 0x1bc, 0x7, 0x44, 0x2, 0x2, 0x1bc, 0x1be, 0x7, 0x8, 0x2, 
    0x2, 0x1bd, 0x1b8, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1b9, 0x3, 0x2, 0x2, 
    0x2, 0x1be, 0x33, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1c5, 0x5, 0x36, 0x1c, 
    0x2, 0x1c0, 0x1c1, 0x5, 0x36, 0x1c, 0x2, 0x1c1, 0x1c2, 0x7, 0x18, 0x2, 
    0x2, 0x1c2, 0x1c3, 0x5, 0x34, 0x1b, 0x2, 0x1c3, 0x1c5, 0x3, 0x2, 0x2, 
    0x2, 0x1c4, 0x1bf, 0x3, 0x2, 0x2, 0x2, 0x1c4, 0x1c0, 0x3, 0x2, 0x2, 
    0x2, 0x1c5, 0x35, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c7, 0x8, 0x1c, 0x1, 
    0x2, 0x1c7, 0x1d7, 0x7, 0x43, 0x2, 0x2, 0x1c8, 0x1d7, 0x7, 0x44, 0x2, 
    0x2, 0x1c9, 0x1d7, 0x7, 0x36, 0x2, 0x2, 0x1ca, 0x1d7, 0x7, 0x47, 0x2, 
    0x2, 0x1cb, 0x1cc, 0x5, 0x3a, 0x1e, 0x2, 0x1cc, 0x1cd, 0x7, 0xd, 0x2, 
    0x2, 0x1cd, 0x1ce, 0x5, 0x36, 0x1c, 0x2, 0x1ce, 0x1cf, 0x7, 0xe, 0x2, 
    0x2, 0x1cf, 0x1d7, 0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1d1, 0x7, 0xd, 0x2, 
    0x2, 0x1d1, 0x1d2, 0x5, 0x36, 0x1c, 0x2, 0x1d2, 0x1d3, 0x7, 0xe, 0x2, 
    0x2, 0x1d3, 0x1d7, 0x3, 0x2, 0x2, 0x2, 0x1d4, 0x1d5, 0x7, 0x37, 0x2, 
    0x2, 0x1d5, 0x1d7, 0x5, 0x36, 0x1c, 0x3, 0x1d6, 0x1c6, 0x3, 0x2, 0x2, 
    0x2, 0x1d6, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1d6, 0x1c9, 0x3, 0x2, 0x2, 
    0x2, 0x1d6, 0x1ca, 0x3, 0x2, 0x2, 0x2, 0x1d6, 0x1cb, 0x3, 0x2, 0x2, 
    0x2, 0x1d6, 0x1d0, 0x3, 0x2, 0x2, 0x2, 0x1d6, 0x1d4, 0x3, 0x2, 0x2, 
    0x2, 0x1d7, 0x1de, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1d9, 0xc, 0x6, 0x2, 
    0x2, 0x1d9, 0x1da, 0x5, 0x38, 0x1d, 0x2, 0x1da, 0x1db, 0x5, 0x36, 0x1c, 
    0x7, 0x1db, 0x1dd, 0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1d8, 0x3, 0x2, 0x2, 
    0x2, 0x1dd, 0x1e0, 0x3, 0x2, 0x2, 0x2, 0x1de, 0x1dc, 0x3, 0x2, 0x2, 
    0x2, 0x1de, 0x1df, 0x3, 0x2, 0x2, 0x2, 0x1df, 0x37, 0x3, 0x2, 0x2, 0x2, 
    0x1e0, 0x1de, 0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1e2, 0x9, 0x3, 0x2, 0x2, 
    0x1e2, 0x39, 0x3, 0x2, 0x2, 0x2, 0x1e3, 0x1e4, 0x9, 0x4, 0x2, 0x2, 0x1e4, 
    0x3b, 0x3, 0x2, 0x2, 0x2, 0x16, 0x40, 0x45, 0x57, 0x5f, 0x63, 0x7a, 
    0x90, 0x9a, 0x9f, 0xa3, 0x17d, 0x190, 0x194, 0x19a, 0x1af, 0x1b6, 0x1bd, 
    0x1c4, 0x1d6, 0x1de, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

snuqlParser::Initializer snuqlParser::_init;
