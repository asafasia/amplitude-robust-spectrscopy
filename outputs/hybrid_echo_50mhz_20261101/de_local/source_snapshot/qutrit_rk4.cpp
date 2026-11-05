// Fused real-coordinate implementation of qutrit.py's Lindblad RK4 step.
// Layout: p0,p1,p2,Re01,Im01,Re02,Im02,Re12,Im12, each a grid-sized block.
#include <cmath>

static inline void rhs(const double* s, double* d, double det, double dx,
                       double dy, double alpha, double g1, double gp) {
    constexpr double root2 = 1.4142135623730950488;
    constexpr double invroot2 = 0.7071067811865475244;
    const double e1 = -det, e2 = -2.0 * det + alpha;
    const double j01 = dx * s[4] - dy * s[3];
    const double j12 = root2 * (dx * s[8] - dy * s[7]);
    d[0] = -j01 + g1*s[1];
    d[1] = j01-j12-g1*s[1]+2.0*g1*s[2];
    d[2] = j12-2.0*g1*s[2];
    d[3] = .5*dy*(s[1]-s[0])-e1*s[4]
           -(dx*s[6]-dy*s[5])*invroot2+root2*g1*s[7]-(.5*g1+gp)*s[3];
    d[4] = -.5*dx*(s[1]-s[0])+e1*s[3]
           +(dx*s[5]+dy*s[6])*invroot2+root2*g1*s[8]-(.5*g1+gp)*s[4];
    d[5] = .5*(dx*s[8]+dy*s[7])-(dx*s[4]+dy*s[3])*invroot2
           -e2*s[6]-(g1+4.0*gp)*s[5];
    d[6] = -.5*(dx*s[7]-dy*s[8])+(dx*s[3]-dy*s[4])*invroot2
           +e2*s[5]-(g1+4.0*gp)*s[6];
    d[7] = .5*(dx*s[6]-dy*s[5])+dy*(s[2]-s[1])*invroot2
           -(e2-e1)*s[8]-(1.5*g1+gp)*s[7];
    d[8] = -.5*(dx*s[5]+dy*s[6])-dx*(s[2]-s[1])*invroot2
           +(e2-e1)*s[7]-(1.5*g1+gp)*s[8];
}

extern "C" void qutrit_rk4(double* state, int points, int steps, double dt,
                           const double* det, const double* rabi,
                           const double* real_drive, const double* imag_drive,
                           double alpha, double g1, double gp) {
    for (int step=0; step<steps; ++step) {
        const double a=real_drive[2*step], b=imag_drive[2*step];
        const double c=real_drive[2*step+1], d=imag_drive[2*step+1];
        const double e=real_drive[2*step+2], f=imag_drive[2*step+2];
        for (int j=0; j<points; ++j) {
            double s[9], temp[9], k1[9], k2[9], k3[9], k4[9];
            for (int k=0; k<9; ++k) s[k]=state[k*points+j];
            const double r=rabi[j], delta=det[j];
            rhs(s,k1,delta,r*a,r*b,alpha,g1,gp);
            for (int k=0; k<9; ++k) temp[k]=s[k]+.5*dt*k1[k];
            rhs(temp,k2,delta,r*c,r*d,alpha,g1,gp);
            for (int k=0; k<9; ++k) temp[k]=s[k]+.5*dt*k2[k];
            rhs(temp,k3,delta,r*c,r*d,alpha,g1,gp);
            for (int k=0; k<9; ++k) temp[k]=s[k]+dt*k3[k];
            rhs(temp,k4,delta,r*e,r*f,alpha,g1,gp);
            for (int k=0; k<9; ++k)
                state[k*points+j]=s[k]+dt/6.0*(k1[k]+2.0*k2[k]+2.0*k3[k]+k4[k]);
        }
    }
}
