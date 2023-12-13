opaque id a;
opaque x a;
opaque y a;
opaque z a;
opaque h a;
opaque s a;
opaque sdg a;
opaque t a;
opaque tdg a;
opaque sx a;
opaque sxdg a;
opaque p(lambda) q;
opaque rx(theta) a;
opaque ry(theta) a;
opaque rz(phi) a;
opaque u0(gamma) q;
opaque u1(lambda) q;
opaque u2(phi,lambda) q;
opaque u3(theta,phi,lambda) q;
opaque u(theta,phi,lambda) q;
opaque cx c,t;
opaque cy a,b;
opaque cz a,b;
opaque swap a,b;
opaque ch a,b;
opaque csx a,b;
opaque crx(lambda) a,b;
opaque cry(lambda) a,b;
opaque crz(lambda) a,b;
opaque cp(lambda) a,b;
opaque cu1(lambda) a,b;
opaque rxx(theta) a,b;
opaque rzz(theta) a,b;
opaque cu3(theta,phi,lambda) c, t;
opaque cu(theta,phi,lambda,gamma) c, t;
opaque ccx a,b,c;
opaque cswap a,b,c;

// relative-phase CCX
gate rccx a,b,c
{
  u2(0,pi) c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  cx a, c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  u2(0,pi) c;
}
// relative-phase 3-controlled X gate
gate rc3x a,b,c,d
{
  u2(0,pi) d;
  u1(pi/4) d;
  cx c,d;
  u1(-pi/4) d;
  u2(0,pi) d;
  cx a,d;
  u1(pi/4) d;
  cx b,d;
  u1(-pi/4) d;
  cx a,d;
  u1(pi/4) d;
  cx b,d;
  u1(-pi/4) d;
  u2(0,pi) d;
  u1(pi/4) d;
  cx c,d;
  u1(-pi/4) d;
  u2(0,pi) d;
}
// 3-controlled X gate
gate c3x a,b,c,d
{
    h d;
    p(pi/8) a;
    p(pi/8) b;
    p(pi/8) c;
    p(pi/8) d;
    cx a, b;
    p(-pi/8) b;
    cx a, b;
    cx b, c;
    p(-pi/8) c;
    cx a, c;
    p(pi/8) c;
    cx b, c;
    p(-pi/8) c;
    cx a, c;
    cx c, d;
    p(-pi/8) d;
    cx b, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx a, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx b, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx a, d;
    h d;
}
// 3-controlled sqrt(X) gate, this equals the C3X gate where the CU1 rotations are -pi/8 not -pi/4
gate c3sqrtx a,b,c,d
{
    h d; cu1(pi/8) a,d; h d;
    cx a,b;
    h d; cu1(-pi/8) b,d; h d;
    cx a,b;
    h d; cu1(pi/8) b,d; h d;
    cx b,c;
    h d; cu1(-pi/8) c,d; h d;
    cx a,c;
    h d; cu1(pi/8) c,d; h d;
    cx b,c;
    h d; cu1(-pi/8) c,d; h d;
    cx a,c;
    h d; cu1(pi/8) c,d; h d;
}
// 4-controlled X gate
gate c4x a,b,c,d,e
{
    h e; cu1(pi/2) d,e; h e;
    c3x a,b,c,d;
    h e; cu1(-pi/2) d,e; h e;
    c3x a,b,c,d;
    c3sqrtx a,b,c,e;
}
