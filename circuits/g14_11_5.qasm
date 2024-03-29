OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1,q2,q3 { sdg q0; h q0; cx q2,q0; rz(-1.5707962992125342) q0; cx q2,q0; h q0; s q0; }
gate gate_PauliEvolution_139787688470384(param0) q0,q1,q2,q3 { sdg q0; h q0; sdg q2; h q2; sdg q3; h q3; cx q3,q2; cx q2,q1; cx q1,q0; rz(1.5707922276356872) q0; cx q1,q0; cx q2,q1; cx q3,q2; h q3; s q3; h q2; s q2; h q0; s q0; }
qreg a[2];
qreg b[2];
creg ca[2];
creg cb[2];
reset a[0];
h a[0];
reset a[1];
h a[1];
reset b[0];
h b[0];
reset b[1];
h b[1];
gate_PauliEvolution(-0.7853981496062671) a[0],a[1],b[0],b[1];
gate_PauliEvolution_139787688470384(0.7853961138178436) a[0],a[1],b[0],b[1];
u(-4.247045562496134,-1.5715248519226,-2.6083151407672016) a[0];
u(-0.00015815122097189262,-0.7554565712293971,0.3981992584005897) a[1];
cx a[0],a[1];
ry(0.7287993823775173) a[0];
ry(1.5705584394432401) a[1];
u(3.141515735137349,-2.3736992983461955,-0.4469313681876871) b[0];
u(3.1413371648370116,-1.1419373769172654,-0.7849746854774216) b[1];
cx b[0],b[1];
ry(-3.141307987997077) b[0];
ry(1.5713436896084025) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
