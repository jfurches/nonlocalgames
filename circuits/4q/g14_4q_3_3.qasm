OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1,q2,q3 { sdg q0; h q0; cx q2,q0; rz(-1.5707962992125342) q0; cx q2,q0; h q0; s q0; }
gate gate_PauliEvolution_280440190016608(param0) q0,q1,q2,q3 { sdg q0; h q0; sdg q2; h q2; sdg q3; h q3; cx q3,q2; cx q2,q1; cx q1,q0; rz(1.5707922276356872) q0; cx q1,q0; cx q2,q1; cx q3,q2; h q3; s q3; h q2; s q2; h q0; s q0; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
gate_PauliEvolution(-0.7853981496062671) q[0],q[1],q[2],q[3];
gate_PauliEvolution_280440190016608(0.7853961138178436) q[0],q[1],q[2],q[3];
u(0.7842574255831707,1.5714613854472508,0.5329789832874051) q[0];
u(3.141620567652899,-4.73415506899868,3.5476032185476494) q[1];
cx q[0],q[1];
ry(-0.7848884470569356) q[0];
ry(2.35663266819999) q[1];
u(0.7842574255831707,-1.5714613854472508,-0.5329789832874051) q[2];
u(3.141620567652899,4.73415506899868,-3.5476032185476494) q[3];
cx q[2],q[3];
ry(-0.7848884470569356) q[2];
ry(2.35663266819999) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
