OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[2];
h q[3];
cx q[2],q[0];
cx q[3],q[1];
u(-0.00012498815669907324,0.847683328032983,-4.099738463039702) q[0];
u(2.5056644711831205e-05,2.5913146292814275,2.2267515750949833) q[1];
cx q[0],q[1];
ry(-1.4894540575945474e-05) q[0];
ry(-2.3013912138405264) q[1];
u(3.14117601359083,-0.8755915519526213,0.06338714010500826) q[2];
u(-3.141721232434335,-1.9354035528401161,-1.5780891701491015) q[3];
cx q[2],q[3];
ry(0.0001899902965140577) q[2];
ry(-1.5712084735733371) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
