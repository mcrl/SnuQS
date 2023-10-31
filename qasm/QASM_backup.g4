grammar QASM;

mainprogram 
: version program
;

version
: 'OPENQASM' REAL ';'
;

program
: statement 
| program statement
;

statement
: declStatement
| gatedeclStatement
| opaqueStatement
| qopStatement
| ifStatement
| barrierStatement
;

declStatement
: qregDeclStatement
| cregDeclStatement
;

qregDeclStatement
: 'qreg' ID '[' NNINTEGER ']' ';'
;

cregDeclStatement
: 'creg' ID '[' NNINTEGER ']' ';'
;

gatedeclStatement
: gatedecl goplist '}'
| gatedecl '}'
;

gatedecl
: 'gate' ID idlist '{'
| 'gate' ID '(' ')' idlist '{'
| 'gate' ID '(' paramlist ')' idlist '{'
;

goplist
: gop
| gop goplist
;

gop 
: gopUGate
| gopCXGate
| gopCustomGate
| gopBarrier
;

gopUGate
: 'U' '(' explist ')' ID ';'
;

gopCXGate
: 'CX' ID ',' ID ';'
;

gopCustomGate
: ID idlist ';'
| ID '(' ')' idlist ';'
| ID '(' explist ')' idlist ';'
;

gopBarrier
: 'barrier' idlist ';'
;

paramlist
: idlist
;

idlist
: ID
| ID ',' idlist
;

opaqueStatement
: 'opaque' ID idlist ';'
| 'opaque' ID '(' ')' idlist ';'
| 'opaque' ID '(' idlist ')' idlist ';'
;

qopStatement
: qopUGate
| qopCXGate
| qopCustomGate
| qopMeasure
| qopReset
;

qopUGate
: 'U' '(' explist ')' argument ';'
;

qopCXGate
: 'CX' argument ',' argument ';'
;

qopCustomGate
: ID arglist ';'
| ID '(' ')' arglist ';'
| ID '(' explist ')' arglist ';'
;

qopMeasure
: 'measure' argument '->' argument ';'
;

qopReset
: 'reset' argument ';'
;

ifStatement
: 'if' '(' ID '==' NNINTEGER ')' qopStatement
;

barrierStatement
: 'barrier' arglist ';'
;

arglist
: argument
| argument ',' arglist
;

argument
: ID
| ID '[' NNINTEGER ']'
;

explist
: exp
| exp ',' explist
;

exp
: REAL
| NNINTEGER
| ID
| exp binop exp
| negop exp
| unaryop '(' exp ')'
| '(' exp ')'
| 'pi'
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
