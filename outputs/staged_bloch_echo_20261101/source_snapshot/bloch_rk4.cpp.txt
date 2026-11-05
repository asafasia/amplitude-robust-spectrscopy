// Two-level Lindblad dynamics. Coordinates: 2 Re(rho01), 2 Im(rho01), p0-p1.
#include <cmath>

struct Vec { double x, y, z; };
static inline Vec rhs(Vec s, double delta, double a, double b,
                      double g1, double g2) {
    return {delta*s.y-b*s.z-g2*s.x,
            a*s.z-delta*s.x-g2*s.y,
            b*s.x-a*s.y+g1*(1.0-s.z)};
}
static inline Vec add(Vec s, Vec k, double h) {
    return {s.x+h*k.x, s.y+h*k.y, s.z+h*k.z};
}
extern "C" void bloch_rk4(double* state, int points, int steps, double dt,
                         const double* det, const double* rabi,
                         const double* re, const double* im,
                         double g1, double g2) {
    for (int n=0; n<steps; ++n) {
        for (int j=0; j<points; ++j) {
            Vec s{state[j],state[points+j],state[2*points+j]};
            const double a=rabi[j], d=det[j];
            Vec k1=rhs(s,d,a*re[2*n],a*im[2*n],g1,g2);
            Vec k2=rhs(add(s,k1,.5*dt),d,a*re[2*n+1],a*im[2*n+1],g1,g2);
            Vec k3=rhs(add(s,k2,.5*dt),d,a*re[2*n+1],a*im[2*n+1],g1,g2);
            Vec k4=rhs(add(s,k3,dt),d,a*re[2*n+2],a*im[2*n+2],g1,g2);
            state[j]=s.x+dt/6*(k1.x+2*k2.x+2*k3.x+k4.x);
            state[points+j]=s.y+dt/6*(k1.y+2*k2.y+2*k3.y+k4.y);
            state[2*points+j]=s.z+dt/6*(k1.z+2*k2.z+2*k3.z+k4.z);
        }
    }
}
