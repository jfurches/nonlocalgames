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
u(-0.0005720702104312092,-1.2669863388094338,0.09203137320444613) a[0];
u(-5.038343345043057e-05,-0.10172220972632845,-0.2553663468492025) a[1];
cx a[0],a[1];
ry(-0.0002399448491199051) a[0];
ry(1.569868584817867) a[1];
u(-0.0005720702104312092,1.2669863388094338,-0.09203137320444613) b[0];
u(-5.038343345043057e-05,0.10172220972632845,0.2553663468492025) b[1];
cx b[0],b[1];
ry(-0.0002399448491199051) b[0];
ry(1.569868584817867) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
