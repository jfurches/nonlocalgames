OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[2];
h q[3];
cx q[2],q[0];
cx q[3],q[1];
u(0.7867518486856295,-1.5710230837164263,-2.607972906675082) q[0];
u(3.3848204497847447e-05,0.26586934548523516,-2.979796007739685) q[1];
cx q[0],q[1];
ry(-0.7855306281948701) q[0];
ry(-2.35622243724962) q[1];
u(3.1416137223707694,1.3135489090892158,-3.470179957083861) q[2];
u(6.448041357433954e-05,-3.038715016564204,0.25431838962000747) q[3];
cx q[2],q[3];
ry(0.00033735513349849515) q[2];
ry(-4.712194826194063) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];