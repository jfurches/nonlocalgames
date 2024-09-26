OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[2];
h q[3];
cx q[2],q[0];
cx q[3],q[1];
u(-1.5714385202396843,-1.5706516934063488,0.5334542053837978) q[0];
u(-1.49445276540767e-05,-1.5262517792988692,-0.4019657924634214) q[1];
cx q[0],q[1];
ry(-1.571842147615642) q[0];
ry(-0.0002990457037967593) q[1];
u(-1.5714385202396843,1.5706516934063488,-0.5334542053837978) q[2];
u(-1.49445276540767e-05,1.5262517792988692,0.4019657924634214) q[3];
cx q[2],q[3];
ry(-1.571842147615642) q[2];
ry(-0.0002990457037967593) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
