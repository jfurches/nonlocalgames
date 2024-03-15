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
u(-0.7861306294378217,-1.5713080556529921,0.5339035498705428) a[0];
u(-0.00015717094124940295,1.3566012919611639,0.6420660448219934) a[1];
cx a[0],a[1];
ry(0.7857061341745188) a[0];
ry(-0.7842572456467424) a[1];
u(-4.247045562496134,1.5715248519226,2.6083151407672016) b[0];
u(-0.00015815122097189262,0.7554565712293971,-0.3981992584005897) b[1];
cx b[0],b[1];
ry(0.7287993823775173) b[0];
ry(1.5705584394432401) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
