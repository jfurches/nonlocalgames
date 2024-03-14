OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[3];
u3(-3.5807737316082863e-06,0.04571259212289503,-0.8056605276690753) q[0];
u3(-4.4120589750104625e-05,-0.27602660707724325,-1.2403382440165929) q[1];
cx q[0],q[1];
ry(2.727547356277457) q[0];
ry(-2.3779280344569615e-05) q[1];
u3(-0.0002939238824159688,0.1900813308157749,0.7720005606465199) q[2];
u3(-4.900484010220438e-05,2.491925567645971,0.5658680090245584) q[3];
cx q[2],q[3];
ry(-0.14871022273379333) q[2];
ry(0.0002657668729211251) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
