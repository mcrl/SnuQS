grammar QASM;

mainprogram 
: version program
| program
| EOF
;

version
: 'OPENQASM' REAL ';'
;

program
: statement 
| program statement
;

statement
: includeStatement
| declStatement
| qopStatement
| ifStatement
| barrierStatement
;

includeStatement
: 'include' STRING ';'
;

declStatement
: regDeclStatement
| gateDeclStatement
;

regDeclStatement
: qregDeclStatement
| cregDeclStatement
;

qregDeclStatement
: 'qreg' ID '[' NNINTEGER ']' ';'
;

cregDeclStatement
: 'creg' ID '[' NNINTEGER ']' ';'
;

gateDeclStatement
: opaqueStatement
| gateStatement
;

opaqueStatement
: 'opaque' ID idlist ';'
| 'opaque' ID '(' ')' idlist ';'
| 'opaque' ID '(' paramlist ')' idlist ';'
;

gateStatement
: 'gate' ID idlist '{' '}'
| 'gate' ID '(' ')' idlist '{' '}'
| 'gate' ID '(' paramlist ')' idlist '{' '}'
| 'gate' ID idlist '{' goplist '}'
| 'gate' ID '(' ')' idlist '{' goplist '}'
| 'gate' ID '(' paramlist ')' idlist '{' goplist '}'
;

goplist
: gop
| gop gop*
;

gop 
: gopUGate
| gopCXGate
| gopBarrier
| gopCustomGate
| gopReset
;

gopUGate
: 'U' '(' explist ')' ID ';'
;

gopCXGate
: 'CX' ID ',' ID ';'
;

gopBarrier
: 'barrier' idlist ';'
;

gopCustomGate
: ID idlist ';'
| ID '(' ')' idlist ';'
| ID '(' explist ')' idlist ';'
;

gopReset
: 'reset' ID ';'
;

idlist
: ID
| ID (',' ID)*
;

paramlist
: ID
| ID (',' ID)*
;

qopStatement
: qopUGate
| qopCXGate
| qopMeasure
| qopReset
| qopCustomGate
;

qopUGate
: 'U' '(' explist ')' qarg ';'
;

qopCXGate
: 'CX' qarg ',' qarg ';'
;

qopMeasure
: 'measure' qarg '->' carg ';'
;

qopReset
: 'reset' qarg ';'
;

qopCustomGate
: ID arglist ';'
| ID '(' ')' arglist ';'
| ID '(' explist ')' arglist ';'
;


ifStatement
: 'if' '(' ID '==' NNINTEGER ')' qopStatement
;

barrierStatement
: 'barrier' arglist ';'
;

arglist
: qarg
| qarg (',' qarg)*
;

qarg
: ID
| ID '[' NNINTEGER ']'
;

carg
: ID
| ID '[' NNINTEGER ']'
;


explist
: exp
| exp (',' exp)*
;

exp
: REAL
| NNINTEGER
| ID
| complex
| exp binop exp
| negop exp
| unaryop '(' exp ')'
| '(' exp ')'
| 'pi'
;

complex
: REAL addsub REAL 'j'
| addsub REAL 'j'
;

addsub
: '+'
| '-'
;

binop
: '+'
| '-'
| '*'
| '/'
; 

negop
: '-'
;

unaryop
: 'sin'
| 'cos'
| 'tan'
| 'exp'
| 'ln'
| 'sqrt'
;

ID:
[a-z][A-Za-z0-9_]*
;

NNINTEGER:
[1-9]+ [0-9]* 
| '0'
;

REAL:
([0-9]+ '.' [0-9]* | [0-9]* '.' [0-9]+) ([eE][+-]? [0-9]+)?
;

STRING
: '"' ~["\r\t\n]+? '"'
| '\'' ~['\r\t\n]+? '\''
;

Whitespace : [ \t]+ -> skip ;
Newline : [\r\n]+ -> skip ;
LineComment : '//' ~[\r\n]* -> skip;
BlockComment : '/*' .*? '*/' -> skip;
