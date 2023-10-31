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
| qopPredefinedGate
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

qopPredefinedGate
: 'u3' '(' exp ',' exp ',' exp ')' argument ';'
| 'u2' '(' exp ',' exp ')' argument ';'
| 'u1' '(' exp ')' argument ';'
| 'cx' argument ',' argument ';'
| 'id' argument ';'
| 'u0' '(' exp ')' argument ';'
| 'u' '(' exp ',' exp ',' exp ')' argument ';'
| 'p' '(' exp ')' argument ';'
| 'x' argument ';'
| 'y' argument ';'
| 'z' argument ';'
| 'h' argument ';'
| 's' argument ';'
| 'sdg' argument ';'
| 't' argument ';'
| 'tdg' argument ';'
| 'rx' '(' exp ')' argument ';'
| 'ry' '(' exp ')' argument ';'
| 'rz' '(' exp ')' argument ';'
| 'sx' argument ';'
| 'sxdg' argument ';'
| 'cz' argument ',' argument ';'
| 'cy' argument ',' argument ';'
| 'swap' argument ',' argument ';'
| 'ch' argument ',' argument ';'
| 'ccx' argument ',' argument ',' argument ';'
| 'cswap' argument ',' argument ',' argument ';'
| 'crx' '(' exp ')' argument ',' argument ';'
| 'cry' '(' exp ')' argument ',' argument ';'
| 'crz' '(' exp ')' argument ',' argument ';'
| 'cu1' '(' exp ')' argument ',' argument ';'
| 'cp' '(' exp ')' argument ',' argument ';'
| 'cu3' '(' exp ',' exp ',' exp ')' argument ',' argument ';'
| 'csx' argument ',' argument ';'
| 'cu' '(' exp ',' exp ',' exp ',' exp ')' argument ',' argument ';'
| 'rxx' '(' exp ')' argument ',' argument ';'
| 'rzz' '(' exp ')' argument ',' argument ';'
| 'rccx' argument ',' argument ',' argument ';'
| 'rc3x' argument ',' argument ',' argument ',' argument ';'
| 'c3x' argument ',' argument ',' argument ',' argument ';'
| 'c3sqrtx' argument ',' argument ',' argument ',' argument ';'
| 'c4x' argument ',' argument ',' argument ',' argument ',' argument ';'
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
