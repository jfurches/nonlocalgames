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
u(3.1416137223707694,-1.3135489090892158,3.470179957083861) a[0];
u(6.448041357433954e-05,3.038715016564204,-0.25431838962000747) a[1];
cx a[0],a[1];
ry(0.00033735513349849515) a[0];
ry(-4.712194826194063) a[1];
u(3.141422711956604,0.12530395692214297,-0.768667466115249) b[0];
u(2.9927171386368435e-05,4.194446138039751,1.339362272838698) b[1];
cx b[0],b[1];
ry(3.1415472836507616) b[0];
ry(-2.300516097140166) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
