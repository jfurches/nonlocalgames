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
u(0.7842574255831707,1.5714613854472508,0.5329789832874051) a[0];
u(3.141620567652899,-4.73415506899868,3.5476032185476494) a[1];
cx a[0],a[1];
ry(-0.7848884470569356) a[0];
ry(2.35663266819999) a[1];
u(-1.1085546941920765,-1.5704174853685302,-0.5334704187725228) b[0];
u(1.8863754811500066e-05,3.1821203204261024,-2.8251003209272216) b[1];
cx b[0],b[1];
ry(-2.411793992407277) b[0];
ry(1.5709475766662808) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
