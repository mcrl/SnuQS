grammar snuqasm;

mainprogram 
: version program
;

version
: 'SNUQASM' Real ';'
;

program
: statement 
| include
| program statement
| program include
;

include
: 'include' StringLiteral ';'
;

statement
: decl
| gatedeclStatement
| opaqueDeclStatement
| qopStatement
| ifStatement
| barrierStatement
;

decl
: quantumDecl
| classicalDecl
;

quantumDecl
: 'qreg' Identifier '[' Integer ']' ';'  
;

classicalDecl
: 'creg' Identifier '[' Integer ']' ';'
;

gatedeclStatement
: 'gate' Identifier idlist '{' '}'
| 'gate' Identifier idlist '{' goplist '}'
| 'gate' Identifier '(' ')' idlist '{' '}'
| 'gate' Identifier '(' ')' idlist '{' goplist '}'
| 'gate' Identifier '(' idlist ')' idlist '{' '}'
| 'gate' Identifier '(' idlist ')' idlist '{' goplist '}'
;

goplist
: uopStatement
| barrierStatement
| uopStatement goplist
| barrierStatement goplist
;

opaqueDeclStatement
: 'opaque' Identifier idlist ';'
| 'opaque' Identifier '(' ')' idlist ';'
| 'opaque' Identifier '(' idlist ')' idlist ';'
;

qopStatement
: uopStatement
| measureQop
| resetQop
;

uopStatement
: unitaryOp
| customOp
;

measureQop
: 'measure' argument '->' argument ';'
;

resetQop
: 'reset' argument ';'
;

ifStatement
: 'if' '(' Identifier '==' Integer ')' qopStatement
;

barrierStatement
: 'barrier' arglist ';'
;

unitaryOp
: 'U' '(' explist ')' argument ';'
| 'u' '(' explist ')' argument ';'
| 'CX' argument ',' argument ';'
| 'id' argument ';'
| 'h' argument ';'
| 'x' argument ';'
| 'y' argument ';'
| 'z' argument ';'
| 'sx' argument ';'
| 'sy' argument ';'
| 's' argument ';'
| 'sdg' argument ';'
| 't' argument ';'
| 'tdg' argument ';'
| 'rx' '(' explist ')' argument ';'
| 'ry' '(' explist ')' argument ';'
| 'rz' '(' explist ')' argument ';'
| 'u1' '(' explist ')' argument ';'
| 'u2' '(' explist ')' argument ';'
| 'u3' '(' explist ')' argument ';'
| 'swap' argument ',' argument ';'
| 'cx' argument ',' argument ';'
| 'cy' argument ',' argument ';'
| 'cz' argument ',' argument ';'
| 'ch' argument ',' argument ';'
| 'crx' '(' explist ')' argument ',' argument ';'
| 'cry' '(' explist ')' argument ',' argument ';'
| 'crz' '(' explist ')' argument ',' argument ';'
| 'cu1' '(' explist ')' argument ',' argument ';'
| 'cu2' '(' explist ')' argument ',' argument ';'
| 'cu3' '(' explist ')' argument ',' argument ';'
| 'ccx' argument ',' argument ',' argument ';'
;

customOp
: Identifier anylist ';'
| Identifier '(' ')' anylist ';'
| Identifier '(' explist ')' anylist ';'
;

anylist
: idlist
| mixedlist
;

idlist
: Identifier
| Identifier ',' idlist
;

designatedIdentifier
: Identifier '[' Integer ']' 
;

mixedlist
: Identifier 
| designatedIdentifier
| Identifier ',' mixedlist
| designatedIdentifier ',' mixedlist
;

arglist
: argument
| argument ',' arglist
;

argument
: Identifier
| Identifier '[' Integer ']'
;

explist
: exp
| exp ',' explist
;

exp
: Real
| Integer
| 'pi'
| Identifier
| exp binop exp
| unaryop '(' exp ')'
| '(' exp ')'
| '-' exp
;

binop
: '+'
| '-'
| '*'
| '/'
| '^'
;

unaryop
: 'sin'
| 'cos'
| 'tan'
| 'exp'
| 'ln'
| 'sqrt'
;

StringLiteral
: '"' ~["\r\t\n]+? '"'
| '\'' ~['\r\t\n]+? '\''
;

fragment PlusMinus: [-+];
fragment SciNotation: [eE];
fragment Digit : [0-9];
fragment StartLetter : [a-z];
fragment AnyLetter: [A-Za-z0-9_];
fragment DigitNonZero: [1-9];

Real: Decimal (SciSuffix)? ;
Integer: Digit+;
Decimal: Integer '.' Digit* | Digit* '.' Integer;
SciSuffix: SciNotation PlusMinus? Integer;
Identifier: StartLetter AnyLetter*;



Whitespace : [ \t]+ -> skip ;
Newline : [\r\n]+ -> skip ;

LineComment : '//' ~[\r\n]* -> skip;
BlockComment : '/*' .*? '*/' -> skip;
