OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(1.0079456191747612e-06,0.33832328595400757,1.5390190286525844) q[0];
u3(-1.9996479897974786e-07,-0.1654911346859796,1.1119496799078075) q[1];
cx q[0],q[1];
ry(-0.8086574108141531) q[0];
ry(-3.14159087065652) q[1];
u3(-0.00013465287500803234,0.2313883909126745,-0.5587395053358405) q[2];
u3(0.0005475739629601477,-0.016269512071650612,-0.9290441997370964) q[3];
cx q[2],q[3];
ry(-0.9745219407437349) q[2];
ry(-3.970550123225175e-05) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
