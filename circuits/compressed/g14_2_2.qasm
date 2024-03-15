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
u(1.5710938316063185,1.5708641245916248,0.5334360855486775) a[0];
u(-2.0690776461007513e-05,0.5139368065616461,-2.4416204213583885) a[1];
cx a[0],a[1];
ry(-1.5708162560597627) a[0];
ry(0.0005879538864963184) a[1];
u(1.5710938316063185,-1.5708641245916248,-0.5334360855486775) b[0];
u(-2.0690776461007513e-05,-0.5139368065616461,2.4416204213583885) b[1];
cx b[0],b[1];
ry(-1.5708162560597627) b[0];
ry(0.0005879538864963184) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
