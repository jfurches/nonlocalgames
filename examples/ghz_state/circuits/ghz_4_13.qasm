OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(-0.00013465287500803234,-0.2313883909126745,0.5587395053358405) q[0];
u3(0.0005475739629601477,0.016269512071650612,0.9290441997370964) q[1];
cx q[0],q[1];
ry(-0.9745219407437349) q[0];
ry(-3.970550123225175e-05) q[1];
u3(-3.1415926535898895,1.6187506559076186,-0.5774918828968508) q[2];
u3(-2.644201987834434e-13,-0.8925110554452297,-2.876862323621751) q[3];
cx q[2],q[3];
ry(0.8056217309240511) q[2];
ry(-8.647721805788261e-13) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
