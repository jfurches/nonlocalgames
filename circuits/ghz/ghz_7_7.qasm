OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(2.2569476235732002e-05,-0.47794624410588193,0.4105440038579222) q[0];
u3(2.074385746823362e-05,-0.8891489006097917,-1.0466768315071389) q[1];
cx q[0],q[1];
ry(1.8309304079208624) q[0];
ry(2.367760829051558e-05) q[1];
u3(2.2569476235732002e-05,0.47794624410588193,-0.4105440038579222) q[2];
u3(2.074385746823362e-05,0.8891489006097917,1.0466768315071389) q[3];
cx q[2],q[3];
ry(1.8309304079208624) q[2];
ry(2.367760829051558e-05) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
