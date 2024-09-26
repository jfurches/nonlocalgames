OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(4.050308145821962e-06,-0.09180385745651336,-1.0011894705277598) q[0];
u3(-7.661514706203816e-06,-0.2993235968530852,-0.6883587574898277) q[1];
cx q[0],q[1];
ry(-0.8624797150106468) q[0];
ry(5.082118900170752e-06) q[1];
u3(6.6383324939302095e-06,-0.8744356877465147,0.9895034588578784) q[2];
u3(6.384109470305647e-07,0.750645316140673,-0.5482802500176633) q[3];
cx q[2],q[3];
ry(0.8141705434562424) q[2];
ry(-3.1415945334717112) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
