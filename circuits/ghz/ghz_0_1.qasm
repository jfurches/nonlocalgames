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
u3(-3.5807737316082863e-06,-0.04571259212289503,0.8056605276690753) q[2];
u3(-4.4120589750104625e-05,0.27602660707724325,1.2403382440165929) q[3];
cx q[2],q[3];
ry(2.727547356277457) q[2];
ry(-2.3779280344569615e-05) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
