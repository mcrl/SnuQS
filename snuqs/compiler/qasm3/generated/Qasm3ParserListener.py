# Generated from Qasm3Parser.g4 by ANTLR 4.13.1
from antlr4 import *
if "." in __name__:
    from .Qasm3Parser import Qasm3Parser
else:
    from Qasm3Parser import Qasm3Parser

# This class defines a complete listener for a parse tree produced by Qasm3Parser.
class Qasm3ParserListener(ParseTreeListener):

    # Enter a parse tree produced by Qasm3Parser#program.
    def enterProgram(self, ctx:Qasm3Parser.ProgramContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#program.
    def exitProgram(self, ctx:Qasm3Parser.ProgramContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#version.
    def enterVersion(self, ctx:Qasm3Parser.VersionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#version.
    def exitVersion(self, ctx:Qasm3Parser.VersionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#statement.
    def enterStatement(self, ctx:Qasm3Parser.StatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#statement.
    def exitStatement(self, ctx:Qasm3Parser.StatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#annotation.
    def enterAnnotation(self, ctx:Qasm3Parser.AnnotationContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#annotation.
    def exitAnnotation(self, ctx:Qasm3Parser.AnnotationContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#scope.
    def enterScope(self, ctx:Qasm3Parser.ScopeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#scope.
    def exitScope(self, ctx:Qasm3Parser.ScopeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#pragma.
    def enterPragma(self, ctx:Qasm3Parser.PragmaContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#pragma.
    def exitPragma(self, ctx:Qasm3Parser.PragmaContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#statementOrScope.
    def enterStatementOrScope(self, ctx:Qasm3Parser.StatementOrScopeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#statementOrScope.
    def exitStatementOrScope(self, ctx:Qasm3Parser.StatementOrScopeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#calibrationGrammarStatement.
    def enterCalibrationGrammarStatement(self, ctx:Qasm3Parser.CalibrationGrammarStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#calibrationGrammarStatement.
    def exitCalibrationGrammarStatement(self, ctx:Qasm3Parser.CalibrationGrammarStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#includeStatement.
    def enterIncludeStatement(self, ctx:Qasm3Parser.IncludeStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#includeStatement.
    def exitIncludeStatement(self, ctx:Qasm3Parser.IncludeStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#breakStatement.
    def enterBreakStatement(self, ctx:Qasm3Parser.BreakStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#breakStatement.
    def exitBreakStatement(self, ctx:Qasm3Parser.BreakStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#continueStatement.
    def enterContinueStatement(self, ctx:Qasm3Parser.ContinueStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#continueStatement.
    def exitContinueStatement(self, ctx:Qasm3Parser.ContinueStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#endStatement.
    def enterEndStatement(self, ctx:Qasm3Parser.EndStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#endStatement.
    def exitEndStatement(self, ctx:Qasm3Parser.EndStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#forStatement.
    def enterForStatement(self, ctx:Qasm3Parser.ForStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#forStatement.
    def exitForStatement(self, ctx:Qasm3Parser.ForStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#ifStatement.
    def enterIfStatement(self, ctx:Qasm3Parser.IfStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#ifStatement.
    def exitIfStatement(self, ctx:Qasm3Parser.IfStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#returnStatement.
    def enterReturnStatement(self, ctx:Qasm3Parser.ReturnStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#returnStatement.
    def exitReturnStatement(self, ctx:Qasm3Parser.ReturnStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#whileStatement.
    def enterWhileStatement(self, ctx:Qasm3Parser.WhileStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#whileStatement.
    def exitWhileStatement(self, ctx:Qasm3Parser.WhileStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#switchStatement.
    def enterSwitchStatement(self, ctx:Qasm3Parser.SwitchStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#switchStatement.
    def exitSwitchStatement(self, ctx:Qasm3Parser.SwitchStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#switchCaseItem.
    def enterSwitchCaseItem(self, ctx:Qasm3Parser.SwitchCaseItemContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#switchCaseItem.
    def exitSwitchCaseItem(self, ctx:Qasm3Parser.SwitchCaseItemContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#barrierStatement.
    def enterBarrierStatement(self, ctx:Qasm3Parser.BarrierStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#barrierStatement.
    def exitBarrierStatement(self, ctx:Qasm3Parser.BarrierStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#boxStatement.
    def enterBoxStatement(self, ctx:Qasm3Parser.BoxStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#boxStatement.
    def exitBoxStatement(self, ctx:Qasm3Parser.BoxStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#delayStatement.
    def enterDelayStatement(self, ctx:Qasm3Parser.DelayStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#delayStatement.
    def exitDelayStatement(self, ctx:Qasm3Parser.DelayStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#gateCallStatement.
    def enterGateCallStatement(self, ctx:Qasm3Parser.GateCallStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#gateCallStatement.
    def exitGateCallStatement(self, ctx:Qasm3Parser.GateCallStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#measureArrowAssignmentStatement.
    def enterMeasureArrowAssignmentStatement(self, ctx:Qasm3Parser.MeasureArrowAssignmentStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#measureArrowAssignmentStatement.
    def exitMeasureArrowAssignmentStatement(self, ctx:Qasm3Parser.MeasureArrowAssignmentStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#resetStatement.
    def enterResetStatement(self, ctx:Qasm3Parser.ResetStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#resetStatement.
    def exitResetStatement(self, ctx:Qasm3Parser.ResetStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#aliasDeclarationStatement.
    def enterAliasDeclarationStatement(self, ctx:Qasm3Parser.AliasDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#aliasDeclarationStatement.
    def exitAliasDeclarationStatement(self, ctx:Qasm3Parser.AliasDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#classicalDeclarationStatement.
    def enterClassicalDeclarationStatement(self, ctx:Qasm3Parser.ClassicalDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#classicalDeclarationStatement.
    def exitClassicalDeclarationStatement(self, ctx:Qasm3Parser.ClassicalDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#constDeclarationStatement.
    def enterConstDeclarationStatement(self, ctx:Qasm3Parser.ConstDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#constDeclarationStatement.
    def exitConstDeclarationStatement(self, ctx:Qasm3Parser.ConstDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#ioDeclarationStatement.
    def enterIoDeclarationStatement(self, ctx:Qasm3Parser.IoDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#ioDeclarationStatement.
    def exitIoDeclarationStatement(self, ctx:Qasm3Parser.IoDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#oldStyleDeclarationStatement.
    def enterOldStyleDeclarationStatement(self, ctx:Qasm3Parser.OldStyleDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#oldStyleDeclarationStatement.
    def exitOldStyleDeclarationStatement(self, ctx:Qasm3Parser.OldStyleDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#quantumDeclarationStatement.
    def enterQuantumDeclarationStatement(self, ctx:Qasm3Parser.QuantumDeclarationStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#quantumDeclarationStatement.
    def exitQuantumDeclarationStatement(self, ctx:Qasm3Parser.QuantumDeclarationStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defStatement.
    def enterDefStatement(self, ctx:Qasm3Parser.DefStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defStatement.
    def exitDefStatement(self, ctx:Qasm3Parser.DefStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#externStatement.
    def enterExternStatement(self, ctx:Qasm3Parser.ExternStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#externStatement.
    def exitExternStatement(self, ctx:Qasm3Parser.ExternStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#gateStatement.
    def enterGateStatement(self, ctx:Qasm3Parser.GateStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#gateStatement.
    def exitGateStatement(self, ctx:Qasm3Parser.GateStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#assignmentStatement.
    def enterAssignmentStatement(self, ctx:Qasm3Parser.AssignmentStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#assignmentStatement.
    def exitAssignmentStatement(self, ctx:Qasm3Parser.AssignmentStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#expressionStatement.
    def enterExpressionStatement(self, ctx:Qasm3Parser.ExpressionStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#expressionStatement.
    def exitExpressionStatement(self, ctx:Qasm3Parser.ExpressionStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#calStatement.
    def enterCalStatement(self, ctx:Qasm3Parser.CalStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#calStatement.
    def exitCalStatement(self, ctx:Qasm3Parser.CalStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalStatement.
    def enterDefcalStatement(self, ctx:Qasm3Parser.DefcalStatementContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalStatement.
    def exitDefcalStatement(self, ctx:Qasm3Parser.DefcalStatementContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#bitwiseXorExpression.
    def enterBitwiseXorExpression(self, ctx:Qasm3Parser.BitwiseXorExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#bitwiseXorExpression.
    def exitBitwiseXorExpression(self, ctx:Qasm3Parser.BitwiseXorExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#additiveExpression.
    def enterAdditiveExpression(self, ctx:Qasm3Parser.AdditiveExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#additiveExpression.
    def exitAdditiveExpression(self, ctx:Qasm3Parser.AdditiveExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#durationofExpression.
    def enterDurationofExpression(self, ctx:Qasm3Parser.DurationofExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#durationofExpression.
    def exitDurationofExpression(self, ctx:Qasm3Parser.DurationofExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#parenthesisExpression.
    def enterParenthesisExpression(self, ctx:Qasm3Parser.ParenthesisExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#parenthesisExpression.
    def exitParenthesisExpression(self, ctx:Qasm3Parser.ParenthesisExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#comparisonExpression.
    def enterComparisonExpression(self, ctx:Qasm3Parser.ComparisonExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#comparisonExpression.
    def exitComparisonExpression(self, ctx:Qasm3Parser.ComparisonExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#multiplicativeExpression.
    def enterMultiplicativeExpression(self, ctx:Qasm3Parser.MultiplicativeExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#multiplicativeExpression.
    def exitMultiplicativeExpression(self, ctx:Qasm3Parser.MultiplicativeExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#logicalOrExpression.
    def enterLogicalOrExpression(self, ctx:Qasm3Parser.LogicalOrExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#logicalOrExpression.
    def exitLogicalOrExpression(self, ctx:Qasm3Parser.LogicalOrExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#castExpression.
    def enterCastExpression(self, ctx:Qasm3Parser.CastExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#castExpression.
    def exitCastExpression(self, ctx:Qasm3Parser.CastExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#powerExpression.
    def enterPowerExpression(self, ctx:Qasm3Parser.PowerExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#powerExpression.
    def exitPowerExpression(self, ctx:Qasm3Parser.PowerExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#bitwiseOrExpression.
    def enterBitwiseOrExpression(self, ctx:Qasm3Parser.BitwiseOrExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#bitwiseOrExpression.
    def exitBitwiseOrExpression(self, ctx:Qasm3Parser.BitwiseOrExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#callExpression.
    def enterCallExpression(self, ctx:Qasm3Parser.CallExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#callExpression.
    def exitCallExpression(self, ctx:Qasm3Parser.CallExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#bitshiftExpression.
    def enterBitshiftExpression(self, ctx:Qasm3Parser.BitshiftExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#bitshiftExpression.
    def exitBitshiftExpression(self, ctx:Qasm3Parser.BitshiftExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#bitwiseAndExpression.
    def enterBitwiseAndExpression(self, ctx:Qasm3Parser.BitwiseAndExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#bitwiseAndExpression.
    def exitBitwiseAndExpression(self, ctx:Qasm3Parser.BitwiseAndExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#equalityExpression.
    def enterEqualityExpression(self, ctx:Qasm3Parser.EqualityExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#equalityExpression.
    def exitEqualityExpression(self, ctx:Qasm3Parser.EqualityExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#logicalAndExpression.
    def enterLogicalAndExpression(self, ctx:Qasm3Parser.LogicalAndExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#logicalAndExpression.
    def exitLogicalAndExpression(self, ctx:Qasm3Parser.LogicalAndExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#indexExpression.
    def enterIndexExpression(self, ctx:Qasm3Parser.IndexExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#indexExpression.
    def exitIndexExpression(self, ctx:Qasm3Parser.IndexExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#unaryExpression.
    def enterUnaryExpression(self, ctx:Qasm3Parser.UnaryExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#unaryExpression.
    def exitUnaryExpression(self, ctx:Qasm3Parser.UnaryExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#literalExpression.
    def enterLiteralExpression(self, ctx:Qasm3Parser.LiteralExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#literalExpression.
    def exitLiteralExpression(self, ctx:Qasm3Parser.LiteralExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#aliasExpression.
    def enterAliasExpression(self, ctx:Qasm3Parser.AliasExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#aliasExpression.
    def exitAliasExpression(self, ctx:Qasm3Parser.AliasExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#declarationExpression.
    def enterDeclarationExpression(self, ctx:Qasm3Parser.DeclarationExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#declarationExpression.
    def exitDeclarationExpression(self, ctx:Qasm3Parser.DeclarationExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#measureExpression.
    def enterMeasureExpression(self, ctx:Qasm3Parser.MeasureExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#measureExpression.
    def exitMeasureExpression(self, ctx:Qasm3Parser.MeasureExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#rangeExpression.
    def enterRangeExpression(self, ctx:Qasm3Parser.RangeExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#rangeExpression.
    def exitRangeExpression(self, ctx:Qasm3Parser.RangeExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#setExpression.
    def enterSetExpression(self, ctx:Qasm3Parser.SetExpressionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#setExpression.
    def exitSetExpression(self, ctx:Qasm3Parser.SetExpressionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#arrayLiteral.
    def enterArrayLiteral(self, ctx:Qasm3Parser.ArrayLiteralContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#arrayLiteral.
    def exitArrayLiteral(self, ctx:Qasm3Parser.ArrayLiteralContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#indexOperator.
    def enterIndexOperator(self, ctx:Qasm3Parser.IndexOperatorContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#indexOperator.
    def exitIndexOperator(self, ctx:Qasm3Parser.IndexOperatorContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#indexedIdentifier.
    def enterIndexedIdentifier(self, ctx:Qasm3Parser.IndexedIdentifierContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#indexedIdentifier.
    def exitIndexedIdentifier(self, ctx:Qasm3Parser.IndexedIdentifierContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#returnSignature.
    def enterReturnSignature(self, ctx:Qasm3Parser.ReturnSignatureContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#returnSignature.
    def exitReturnSignature(self, ctx:Qasm3Parser.ReturnSignatureContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#gateModifier.
    def enterGateModifier(self, ctx:Qasm3Parser.GateModifierContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#gateModifier.
    def exitGateModifier(self, ctx:Qasm3Parser.GateModifierContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#scalarType.
    def enterScalarType(self, ctx:Qasm3Parser.ScalarTypeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#scalarType.
    def exitScalarType(self, ctx:Qasm3Parser.ScalarTypeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#qubitType.
    def enterQubitType(self, ctx:Qasm3Parser.QubitTypeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#qubitType.
    def exitQubitType(self, ctx:Qasm3Parser.QubitTypeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#arrayType.
    def enterArrayType(self, ctx:Qasm3Parser.ArrayTypeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#arrayType.
    def exitArrayType(self, ctx:Qasm3Parser.ArrayTypeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#arrayReferenceType.
    def enterArrayReferenceType(self, ctx:Qasm3Parser.ArrayReferenceTypeContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#arrayReferenceType.
    def exitArrayReferenceType(self, ctx:Qasm3Parser.ArrayReferenceTypeContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#designator.
    def enterDesignator(self, ctx:Qasm3Parser.DesignatorContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#designator.
    def exitDesignator(self, ctx:Qasm3Parser.DesignatorContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalTarget.
    def enterDefcalTarget(self, ctx:Qasm3Parser.DefcalTargetContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalTarget.
    def exitDefcalTarget(self, ctx:Qasm3Parser.DefcalTargetContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalArgumentDefinition.
    def enterDefcalArgumentDefinition(self, ctx:Qasm3Parser.DefcalArgumentDefinitionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalArgumentDefinition.
    def exitDefcalArgumentDefinition(self, ctx:Qasm3Parser.DefcalArgumentDefinitionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalOperand.
    def enterDefcalOperand(self, ctx:Qasm3Parser.DefcalOperandContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalOperand.
    def exitDefcalOperand(self, ctx:Qasm3Parser.DefcalOperandContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#gateOperand.
    def enterGateOperand(self, ctx:Qasm3Parser.GateOperandContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#gateOperand.
    def exitGateOperand(self, ctx:Qasm3Parser.GateOperandContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#externArgument.
    def enterExternArgument(self, ctx:Qasm3Parser.ExternArgumentContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#externArgument.
    def exitExternArgument(self, ctx:Qasm3Parser.ExternArgumentContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#argumentDefinition.
    def enterArgumentDefinition(self, ctx:Qasm3Parser.ArgumentDefinitionContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#argumentDefinition.
    def exitArgumentDefinition(self, ctx:Qasm3Parser.ArgumentDefinitionContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#argumentDefinitionList.
    def enterArgumentDefinitionList(self, ctx:Qasm3Parser.ArgumentDefinitionListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#argumentDefinitionList.
    def exitArgumentDefinitionList(self, ctx:Qasm3Parser.ArgumentDefinitionListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalArgumentDefinitionList.
    def enterDefcalArgumentDefinitionList(self, ctx:Qasm3Parser.DefcalArgumentDefinitionListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalArgumentDefinitionList.
    def exitDefcalArgumentDefinitionList(self, ctx:Qasm3Parser.DefcalArgumentDefinitionListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#defcalOperandList.
    def enterDefcalOperandList(self, ctx:Qasm3Parser.DefcalOperandListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#defcalOperandList.
    def exitDefcalOperandList(self, ctx:Qasm3Parser.DefcalOperandListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#expressionList.
    def enterExpressionList(self, ctx:Qasm3Parser.ExpressionListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#expressionList.
    def exitExpressionList(self, ctx:Qasm3Parser.ExpressionListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#identifierList.
    def enterIdentifierList(self, ctx:Qasm3Parser.IdentifierListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#identifierList.
    def exitIdentifierList(self, ctx:Qasm3Parser.IdentifierListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#gateOperandList.
    def enterGateOperandList(self, ctx:Qasm3Parser.GateOperandListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#gateOperandList.
    def exitGateOperandList(self, ctx:Qasm3Parser.GateOperandListContext):
        pass


    # Enter a parse tree produced by Qasm3Parser#externArgumentList.
    def enterExternArgumentList(self, ctx:Qasm3Parser.ExternArgumentListContext):
        pass

    # Exit a parse tree produced by Qasm3Parser#externArgumentList.
    def exitExternArgumentList(self, ctx:Qasm3Parser.ExternArgumentListContext):
        pass



del Qasm3Parser