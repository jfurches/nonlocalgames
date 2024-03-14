OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(-1.5372767379533908e-05,-0.3778588660030972,-0.841954817865448) q[0];
u3(-3.9970148397743865e-06,-1.4581118105043307,0.8116943515196942) q[1];
cx q[0],q[1];
ry(1.0202077539491738) q[0];
ry(-2.498712949967879e-06) q[1];
u3(1.7216907179335784e-06,0.34972591103121453,-0.39037509873992104) q[2];
u3(-2.8772446991511777e-07,-1.0353596316939808,-1.7482019885804003) q[3];
cx q[2],q[3];
ry(-0.7996743990599193) q[2];
ry(-3.141593596974464) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
