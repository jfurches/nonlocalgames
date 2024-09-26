OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1,q2,q3 { sdg q0; h q0; cx q2,q0; rz(-1.5707962992125342) q0; cx q2,q0; h q0; s q0; }
gate gate_PauliEvolution_280440190093056(param0) q0,q1,q2,q3 { sdg q0; h q0; sdg q2; h q2; sdg q3; h q3; cx q3,q2; cx q2,q1; cx q1,q0; rz(1.5707922276356872) q0; cx q1,q0; cx q2,q1; cx q3,q2; h q3; s q3; h q2; s q2; h q0; s q0; }
qreg q[4];
creg c[4];
h q[0];
h q[1];
h q[2];
h q[3];
gate_PauliEvolution(-0.7853981496062671) q[0],q[1],q[2],q[3];
gate_PauliEvolution_280440190093056(0.7853961138178436) q[0],q[1],q[2],q[3];
u(-1.5714385202396843,-1.5706516934063488,0.5334542053837978) q[0];
u(-1.49445276540767e-05,-1.5262517792988692,-0.4019657924634214) q[1];
cx q[0],q[1];
ry(-1.571842147615642) q[0];
ry(-0.0002990457037967593) q[1];
u(0.7867518486856295,1.5710230837164263,2.607972906675082) q[2];
u(3.3848204497847447e-05,-0.26586934548523516,2.979796007739685) q[3];
cx q[2],q[3];
ry(-0.7855306281948701) q[2];
ry(-2.35622243724962) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
