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
u(0.7844784747012941,1.5701991621988325,-0.533047339010624) b[0];
u(-3.1415843222635886,1.683316351552248,1.2549781485596494) b[1];
cx b[0],b[1];
ry(-0.7844295606850257) b[0];
ry(-0.7857312615397688) b[1];
measure a[0] -> ca[0];
measure a[1] -> ca[1];
measure b[0] -> cb[0];
measure b[1] -> cb[1];
