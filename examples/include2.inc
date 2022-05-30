gate cyz(theta, phi, lambda) a,b,c
{
	ccx((phi+lambda)/2, phi, theta) c, b, a;
}

