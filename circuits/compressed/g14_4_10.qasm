OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[2];
creg ca[2];
creg cb[2];
h a[0];
h a[1];
cx a[0], b[0];
cx a[1], b[1];
h a[0];
h a[1];
h b[0];
h b[1];
u(0.7867518486856295,-1.5710230837164263,-2.607972906675082) a[0];
u(3.3848204497847447e-05,0.26586934548523516,-2.979796007739685) a[1];
cx a[0],a[1];
ry(-0.7855306281948701) a[0];
ry(-2.35622243724962) a[1];
u(-0.00012498815669907324,-0.847683328032983,4.099738463039702) b[0];
u(2.5056644711831205e-05,-2.5913146292814275,-2.2267515750949833) b[1];
cx b[0],b[1];
ry(-1.4894540575945474e-05) b[0];
ry(-2.3013912138405264) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
