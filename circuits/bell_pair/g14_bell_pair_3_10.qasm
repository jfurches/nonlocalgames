OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[2];
h q[3];
cx q[2],q[0];
cx q[3],q[1];
u(0.7842574255831707,1.5714613854472508,0.5329789832874051) q[0];
u(3.141620567652899,-4.73415506899868,3.5476032185476494) q[1];
cx q[0],q[1];
ry(-0.7848884470569356) q[0];
ry(2.35663266819999) q[1];
u(-0.00012498815669907324,-0.847683328032983,4.099738463039702) q[2];
u(2.5056644711831205e-05,-2.5913146292814275,-2.2267515750949833) q[3];
cx q[2],q[3];
ry(-1.4894540575945474e-05) q[2];
ry(-2.3013912138405264) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
