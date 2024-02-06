OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1,q2,q3 { sdg q0; h q0; cx q2,q0; rz(-1.5707962992125342) q0; cx q2,q0; h q0; s q0; }
gate gate_PauliEvolution_139787688492144(param0) q0,q1,q2,q3 { sdg q0; h q0; sdg q2; h q2; sdg q3; h q3; cx q3,q2; cx q2,q1; cx q1,q0; rz(1.5707922276356872) q0; cx q1,q0; cx q2,q1; cx q3,q2; h q3; s q3; h q2; s q2; h q0; s q0; }
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
gate_PauliEvolution_139787688492144(0.7853961138178436) a[0],a[1],b[0],b[1];
u(-1.1085546941920765,1.5704174853685302,0.5334704187725228) a[0];
u(1.8863754811500066e-05,-3.1821203204261024,2.8251003209272216) a[1];
cx a[0],a[1];
ry(-2.411793992407277) a[0];
ry(1.5709475766662808) a[1];
u(3.14117601359083,-0.8755915519526213,0.06338714010500826) b[0];
u(-3.141721232434335,-1.9354035528401161,-1.5780891701491015) b[1];
cx b[0],b[1];
ry(0.0001899902965140577) b[0];
ry(-1.5712084735733371) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
