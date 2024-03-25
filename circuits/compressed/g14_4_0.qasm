OPENQASM 2.0;
include "qelib1.inc";
qreg a[2];
qreg b[2];
creg ca[2];
creg cb[2];
h b[0];
h b[1];
cx b[0], a[0];
cx b[1], a[1];
u(0.7867518486856295,-1.5710230837164263,-2.607972906675082) a[0];
u(3.3848204497847447e-05,0.26586934548523516,-2.979796007739685) a[1];
cx a[0],a[1];
ry(-0.7855306281948701) a[0];
ry(-2.35622243724962) a[1];
u(-1.5714385202396843,1.5706516934063488,-0.5334542053837978) b[0];
u(-1.49445276540767e-05,1.5262517792988692,0.4019657924634214) b[1];
cx b[0],b[1];
ry(-1.571842147615642) b[0];
ry(-0.0002990457037967593) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
