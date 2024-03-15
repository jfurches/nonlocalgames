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
u(3.14117601359083,0.8755915519526213,-0.06338714010500826) a[0];
u(-3.141721232434335,1.9354035528401161,1.5780891701491015) a[1];
cx a[0],a[1];
ry(0.0001899902965140577) a[0];
ry(-1.5712084735733371) a[1];
u(3.14117601359083,-0.8755915519526213,0.06338714010500826) b[0];
u(-3.141721232434335,-1.9354035528401161,-1.5780891701491015) b[1];
cx b[0],b[1];
ry(0.0001899902965140577) b[0];
ry(-1.5712084735733371) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
