#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CLS_EQ 0
#define CLS_DIV 1
#define CLS_TARGET 2
#define CLS_OTHER 3
#define CLS_UNKNOWN 4
#define CLS_NONE (-1)

static const char *CLASS_NAMES[] = {"EQ", "DIV", "TARGET", "OTHER", "UNKNOWN"};

typedef struct { double alpha_chua, beta, gamma_chua, m0, m1, a1, a2, rho; int model; } ChuaParams;
typedef struct {
    double alpha_frac, g1, g2, g3, w1, w2, w3, a21, a31, a32, inv_mem_factor;
} EFORKCoeffs;
typedef struct { double h, Lm, TMAX_REF, TMAX_TEST, TBURN_REF, TBURN_TEST; } IntegrationCfg;
typedef struct {
    double R_DIV, EPS_EQ, SEC_TOL, HIT_FRAC_REQ;
    int CAP_WIN, MIN_SEC_MATCH, TEST_MAX_SEC;
} ThresholdCfg;
typedef struct { int n_radii, nsamples; double *radii; uint64_t random_seed; char eq_filter[64]; } SamplingCfg;
typedef struct { char csv_out[512], ref_out[512], summary_csv_out[512]; } FileCfg;
typedef struct { int n_eq; char names[3][3]; double val[3][3]; } EqSet;
typedef struct { double x0, y0, z0; int cls, sec_total, sec_hits; double hit_frac; } SampleRow;
typedef struct { int ref_points, total_target_hits; } RunStats;

static void die(const char *msg){ fprintf(stderr, "%s\n", msg); exit(EXIT_FAILURE);} 
static void die_errno(const char *msg){ fprintf(stderr, "%s: %s\n", msg, strerror(errno)); exit(EXIT_FAILURE);} 

static const char *find_arg(int argc, char **argv, const char *key){ for(int i=1;i<argc-1;++i) if(strcmp(argv[i],key)==0) return argv[i+1]; return NULL; }
static int has_arg(int argc, char **argv, const char *key){ for(int i=1;i<argc;++i) if(strcmp(argv[i],key)==0) return 1; return 0; }
static double parse_double_arg(int argc, char **argv, const char *key){ const char *s=find_arg(argc,argv,key); if(!s){fprintf(stderr,"Falta argumento %s\n",key); exit(EXIT_FAILURE);} char *end=NULL; errno=0; double v=strtod(s,&end); if(errno||end==s||*end!='\0'){fprintf(stderr,"Valor inválido para %s: %s\n",key,s); exit(EXIT_FAILURE);} return v; }
static int parse_int_arg(int argc, char **argv, const char *key){ const char *s=find_arg(argc,argv,key); if(!s){fprintf(stderr,"Falta argumento %s\n",key); exit(EXIT_FAILURE);} char *end=NULL; errno=0; long v=strtol(s,&end,10); if(errno||end==s||*end!='\0'){fprintf(stderr,"Valor inválido para %s: %s\n",key,s); exit(EXIT_FAILURE);} return (int)v; }
static uint64_t parse_u64_arg(int argc, char **argv, const char *key){ const char *s=find_arg(argc,argv,key); if(!s){fprintf(stderr,"Falta argumento %s\n",key); exit(EXIT_FAILURE);} char *end=NULL; errno=0; unsigned long long v=strtoull(s,&end,10); if(errno||end==s||*end!='\0'){fprintf(stderr,"Valor inválido para %s: %s\n",key,s); exit(EXIT_FAILURE);} return (uint64_t)v; }
static void parse_vec3_arg(int argc, char **argv, const char *key, double out[3]){ const char *s=find_arg(argc,argv,key); if(!s){fprintf(stderr,"Falta argumento %s\n",key); exit(EXIT_FAILURE);} char *buf=strdup(s); if(!buf) die_errno("No se pudo reservar memoria para vec3"); char *save=NULL; char *tok=strtok_r(buf,",",&save); for(int i=0;i<3;++i){ if(!tok){free(buf); fprintf(stderr,"Se esperaban 3 componentes en %s\n",key); exit(EXIT_FAILURE);} out[i]=strtod(tok,NULL); tok=strtok_r(NULL,",",&save);} free(buf);} 
static void parse_radii_arg(int argc, char **argv, const char *key, SamplingCfg *s){ const char *txt=find_arg(argc,argv,key); if(!txt) die("Falta argumento --radii"); char *buf=strdup(txt); if(!buf) die_errno("No se pudo reservar memoria para radii"); int count=1; for(const char *p=txt; *p; ++p) if(*p==',') count++; s->radii=(double*)malloc((size_t)count*sizeof(double)); if(!s->radii) die_errno("No se pudo reservar memoria para radii[]"); int idx=0; char *save=NULL; char *tok=strtok_r(buf,",",&save); while(tok){ s->radii[idx++]=strtod(tok,NULL); tok=strtok_r(NULL,",",&save);} s->n_radii=idx; free(buf);} 

static inline int parse_model_name(const char *s){ if(!s) return 0; return (strcmp(s,"arctan")==0 || strcmp(s,"atan")==0 || strcmp(s,"smooth")==0) ? 1 : 0; }
static inline double optional_double_arg(int argc,char **argv,const char *key,double default_value){ const char *s=find_arg(argc,argv,key); if(!s) return default_value; char *end=NULL; errno=0; double v=strtod(s,&end); if(errno||end==s||*end!='\0'){fprintf(stderr,"Valor invÃ¡lido para %s: %s\n",key,s); exit(EXIT_FAILURE);} return v; }
/* piecewise: f(x)=m1*x+psi(x), psi(x)=(m0-m1)*sat(x). */
static inline double f_chua_value(double x, const ChuaParams *p){ if(p->model==1) return p->a1*x + p->a2*atan(p->rho*x); return p->m1*x + 0.5*(p->m0-p->m1)*(fabs(x+1.0)-fabs(x-1.0)); }
static inline void chua_rhs_xyz(double x,double y,double z,const ChuaParams *p,double *dx,double *dy,double *dz){ *dx=p->alpha_chua*(y-x-f_chua_value(x,p)); *dy=x-y+z; *dz=-p->beta*y-p->gamma_chua*z; }
static inline double xdot_func(double x,double y,const ChuaParams *p){ return p->alpha_chua*(y-x-f_chua_value(x,p)); }

static void chua_equilibria(const ChuaParams *p, EqSet *eqs){
    eqs->n_eq=1;
    strcpy(eqs->names[0],"E0");
    eqs->val[0][0]=eqs->val[0][1]=eqs->val[0][2]=0.0;
    if(p->model==1){
        double coeff=1.0+p->a1-p->gamma_chua/(p->beta+p->gamma_chua);
        double prevx=1e-8, prev=coeff*prevx + p->a2*atan(p->rho*prevx);
        double xp=NAN;
        for(int i=1;i<=20000;++i){
            double x=100.0*(double)i/20000.0;
            double cur=coeff*x + p->a2*atan(p->rho*x);
            if(prev*cur<0.0){
                double lo=prevx, hi=x, flo=prev;
                for(int it=0;it<80;++it){
                    double mid=0.5*(lo+hi);
                    double fm=coeff*mid + p->a2*atan(p->rho*mid);
                    if(fabs(fm)<1e-14){ lo=hi=mid; break; }
                    if(flo*fm<=0.0) hi=mid; else { lo=mid; flo=fm; }
                }
                xp=0.5*(lo+hi);
                break;
            }
            prevx=x; prev=cur;
        }
        if(isfinite(xp)){
            double yp=p->gamma_chua/(p->beta+p->gamma_chua)*xp;
            double zp=-p->beta/(p->beta+p->gamma_chua)*xp;
            strcpy(eqs->names[1],"E+");
            strcpy(eqs->names[2],"E-");
            eqs->val[1][0]= xp; eqs->val[1][1]= yp; eqs->val[1][2]= zp;
            eqs->val[2][0]=-xp; eqs->val[2][1]=-yp; eqs->val[2][2]=-zp;
            eqs->n_eq=3;
        }
        return;
    }
    double A=p->m0-p->m1;
    double den=(p->beta+p->gamma_chua)*p->m1 + p->beta;
    if(fabs(den)<1e-14) return;
    double xp=-(p->beta+p->gamma_chua)*A/den;
    if(fabs(xp)>1.0){
        double fp=p->m1*xp + A;
        strcpy(eqs->names[1],"E+");
        strcpy(eqs->names[2],"E-");
        eqs->val[1][0]= xp;  eqs->val[1][1]= xp+fp;  eqs->val[1][2]= fp;
        eqs->val[2][0]=-xp;  eqs->val[2][1]=-(xp+fp); eqs->val[2][2]=-fp;
        eqs->n_eq=3;
    }
}

static EFORKCoeffs efork_coeffs(double alpha_frac, double h){ EFORKCoeffs c; c.alpha_frac=alpha_frac; c.g1=tgamma(1.0+alpha_frac); c.g2=tgamma(1.0+2.0*alpha_frac); c.g3=tgamma(1.0+3.0*alpha_frac); c.a21=1.0/(2.0*c.g1*c.g1); c.a31=((c.g1*c.g1)*c.g2 + 2.0*(c.g2*c.g2) - c.g3)/(4.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2)-c.g3)); c.a32=-c.g2/(4.0*(2.0*(c.g2*c.g2)-c.g3)); c.w1=(8.0*(c.g1*c.g1*c.g1)*(c.g2*c.g2)-6.0*(c.g1*c.g1*c.g1)*c.g3 + c.g2*c.g3)/(c.g1*c.g2*c.g3); c.w2=2.0*(c.g1*c.g1)*(4.0*(c.g2*c.g2)-c.g3)/(c.g2*c.g3); c.w3=-8.0*(c.g1*c.g1)*(2.0*(c.g2*c.g2)-c.g3)/(c.g2*c.g3); c.inv_mem_factor=1.0/(h*tgamma(2.0-alpha_frac)); return c; }
static inline double memory_fractional_scalar(int k,double t,const double *arr,const double *vtn,int nu,const EFORKCoeffs *coef){ int start=k-nu; if(start<0) start=0; double s=0.0, expo=1.0-coef->alpha_frac; for(int j=start;j<k;++j){ double v1=pow(t-vtn[j],expo); double v2=pow(t-vtn[j+1],expo); s += (arr[j+1]-arr[j])*(v1-v2); } return s*coef->inv_mem_factor; }

static int efork_chua_caputo(const ChuaParams *p,double x0[3],double alpha_frac,double h,double t_final,double Lm,double **t_out,double **X_out,int *N_out){ EFORKCoeffs coef=efork_coeffs(alpha_frac,h); int N1=(int)ceil(t_final/h); int nu=(int)(Lm/h); if(nu<1) nu=1; double *t=(double*)calloc((size_t)(N1+1),sizeof(double)); double *x=(double*)calloc((size_t)(N1+1),sizeof(double)); double *y=(double*)calloc((size_t)(N1+1),sizeof(double)); double *z=(double*)calloc((size_t)(N1+1),sizeof(double)); if(!t||!x||!y||!z){ free(t); free(x); free(y); free(z); return 0;} double ha=pow(h,alpha_frac); double xn=x0[0], yn=x0[1], zn=x0[2]; t[0]=0.0; x[0]=xn; y[0]=yn; z[0]=zn; double dx,dy,dz; chua_rhs_xyz(xn,yn,zn,p,&dx,&dy,&dz); double K1x=ha*dx, K1y=ha*dy, K1z=ha*dz; chua_rhs_xyz(xn+coef.a21*K1x, yn+coef.a21*K1y, zn+coef.a21*K1z, p,&dx,&dy,&dz); double K2x=ha*dx, K2y=ha*dy, K2z=ha*dz; chua_rhs_xyz(xn+coef.a31*K2x+coef.a32*K1x, yn+coef.a31*K2y+coef.a32*K1y, zn+coef.a31*K2z+coef.a32*K1z,p,&dx,&dy,&dz); double K3x=ha*dx, K3y=ha*dy, K3z=ha*dz; double xn1=xn+coef.w1*K1x+coef.w2*K2x+coef.w3*K3x; double yn1=yn+coef.w1*K1y+coef.w2*K2y+coef.w3*K3y; double zn1=zn+coef.w1*K1z+coef.w2*K2z+coef.w3*K3z; if(N1>=1){ t[1]=h; x[1]=xn1; y[1]=yn1; z[1]=zn1;} xn=xn1; yn=yn1; zn=zn1; for(int n=1;n<N1;++n){ double tn=n*h; double mem_x=memory_fractional_scalar(n,tn,x,t,nu,&coef); double mem_y=memory_fractional_scalar(n,tn,y,t,nu,&coef); double mem_z=memory_fractional_scalar(n,tn,z,t,nu,&coef); chua_rhs_xyz(xn,yn,zn,p,&dx,&dy,&dz); double f1x=dx-mem_x, f1y=dy-mem_y, f1z=dz-mem_z; K1x=ha*f1x; K1y=ha*f1y; K1z=ha*f1z; chua_rhs_xyz(xn+coef.a21*K1x, yn+coef.a21*K1y, zn+coef.a21*K1z, p,&dx,&dy,&dz); K2x=ha*dx; K2y=ha*dy; K2z=ha*dz; chua_rhs_xyz(xn+coef.a31*K2x+coef.a32*K1x, yn+coef.a31*K2y+coef.a32*K1y, zn+coef.a31*K2z+coef.a32*K1z,p,&dx,&dy,&dz); K3x=ha*dx; K3y=ha*dy; K3z=ha*dz; xn1=xn+coef.w1*K1x+coef.w2*K2x+coef.w3*K3x; yn1=yn+coef.w1*K1y+coef.w2*K2y+coef.w3*K3y; zn1=zn+coef.w1*K1z+coef.w2*K2z+coef.w3*K3z; t[n+1]=(n+1)*h; x[n+1]=xn1; y[n+1]=yn1; z[n+1]=zn1; xn=xn1; yn=yn1; zn=zn1; } double *X=(double*)malloc((size_t)(N1+1)*3u*sizeof(double)); if(!X){ free(t); free(x); free(y); free(z); return 0;} for(int i=0;i<=N1;++i){ X[3*i+0]=x[i]; X[3*i+1]=y[i]; X[3*i+2]=z[i]; } free(x); free(y); free(z); *t_out=t; *X_out=X; *N_out=N1+1; return 1; }

static int classify_equilibrium_or_divergence(const double *X,int N,const EqSet *eqs,const ThresholdCfg *thr){ double R2=thr->R_DIV*thr->R_DIV, eps2=thr->EPS_EQ*thr->EPS_EQ; int hits[3]={0,0,0}; for(int i=0;i<N;++i){ double x=X[3*i+0], y=X[3*i+1], z=X[3*i+2]; double r2=x*x+y*y+z*z; if(r2>R2) return CLS_DIV; for(int e=0;e<eqs->n_eq;++e){ double dx=x-eqs->val[e][0], dy=y-eqs->val[e][1], dz=z-eqs->val[e][2]; double d2=dx*dx+dy*dy+dz*dz; if(d2<=eps2){ hits[e]++; if(hits[e]>=thr->CAP_WIN) return CLS_EQ; } else hits[e]=0; }} return CLS_NONE; }
static int section_points(const double *t,const double *X,int N,const ChuaParams *p,double t_burn,int max_pts,double *out_yz){ int start_idx=1; while(start_idx<N && t[start_idx]<t_burn) start_idx++; int count=0; for(int k=start_idx;k<N;++k){ double xp=X[3*(k-1)+0], x=X[3*k+0]; if(xp<0.0 && x>=0.0){ double fcur=xdot_func(X[3*k+0],X[3*k+1],p); if(fcur>0.0){ double lam=(0.0-xp)/((x-xp)+1e-30); double ys=X[3*(k-1)+1] + lam*(X[3*k+1]-X[3*(k-1)+1]); double zs=X[3*(k-1)+2] + lam*(X[3*k+2]-X[3*(k-1)+2]); out_yz[2*count+0]=ys; out_yz[2*count+1]=zs; count++; if(count>=max_pts) break; }}} return count; }
static inline double min_dist_to_ref(const double *ref,int nref,double y,double z){ if(nref<=0) return INFINITY; double best=INFINITY; for(int i=0;i<nref;++i){ double dy=ref[2*i+0]-y, dz=ref[2*i+1]-z; double d2=dy*dy+dz*dz; if(d2<best) best=d2; } return sqrt(best); }

static uint64_t splitmix64_next(uint64_t *x){ uint64_t z=(*x += UINT64_C(0x9E3779B97F4A7C15)); z=(z^(z>>30))*UINT64_C(0xBF58476D1CE4E5B9); z=(z^(z>>27))*UINT64_C(0x94D049BB133111EB); return z^(z>>31);} 
static double rng_uniform01(uint64_t *state){ uint64_t u=splitmix64_next(state); return (u>>11)*(1.0/9007199254740992.0);} 
static void sample_in_ball(const double center[3],double radius,uint64_t *state,double out[3]){ for(;;){ double ux=2.0*rng_uniform01(state)-1.0, uy=2.0*rng_uniform01(state)-1.0, uz=2.0*rng_uniform01(state)-1.0; double r2=ux*ux+uy*uy+uz*uz; if(r2<=1.0){ out[0]=center[0]+radius*ux; out[1]=center[1]+radius*uy; out[2]=center[2]+radius*uz; return; } }}
static uint64_t make_sample_seed(uint64_t base,int eq_idx,int radius_idx,int sample_idx){ uint64_t s=base; s ^= UINT64_C(0xA24BAED4963EE407)*(uint64_t)(eq_idx+1); s ^= UINT64_C(0x9FB21C651E98DF25)*(uint64_t)(radius_idx+1); s ^= UINT64_C(0xD6E8FEB86659FD93)*(uint64_t)(sample_idx+1); return s; }

static int eq_filter_allows(const SamplingCfg *s,const char *name){
    const char *filter=s->eq_filter[0] ? s->eq_filter : getenv("HIDDEN_VERIFY_EQ_FILTER");
    if(!filter || !filter[0] || strcmp(filter,"all")==0 || strcmp(filter,"ALL")==0) return 1;
    char buf[64];
    snprintf(buf,sizeof(buf),"%s",filter);
    char *save=NULL;
    char *tok=strtok_r(buf,",",&save);
    while(tok){
        while(*tok==' '||*tok=='\t') tok++;
        char *end=tok+strlen(tok);
        while(end>tok && (end[-1]==' '||end[-1]=='\t'||end[-1]=='\r'||end[-1]=='\n')) *--end='\0';
        if(strcmp(tok,name)==0) return 1;
        tok=strtok_r(NULL,",",&save);
    }
    return 0;
}

static int build_reference_section(const ChuaParams *p,const EqSet *eqs,const IntegrationCfg *integ,const ThresholdCfg *thr,double frac_order,const double target_seed[3],double **ref_out,int *nref_out){ double *t=NULL,*X=NULL; int N=0; double seed_local[3]={target_seed[0],target_seed[1],target_seed[2]}; if(!efork_chua_caputo(p,seed_local,frac_order,integ->h,integ->TMAX_REF,integ->Lm,&t,&X,&N)) die("No se pudo integrar la semilla de referencia"); int cls=classify_equilibrium_or_divergence(X,N,eqs,thr); if(cls==CLS_EQ||cls==CLS_DIV){ free(t); free(X); return cls; } int max_pts=thr->TEST_MAX_SEC*4; double *ref=(double*)malloc((size_t)max_pts*2u*sizeof(double)); if(!ref) die_errno("No se pudo reservar memoria para la referencia seccional"); int nref=section_points(t,X,N,p,integ->TBURN_REF,max_pts,ref); free(t); free(X); if(nref<thr->MIN_SEC_MATCH){ free(ref); return CLS_UNKNOWN; } *ref_out=ref; *nref_out=nref; return CLS_TARGET; }

static int classify_to_target(const ChuaParams *p,const EqSet *eqs,const IntegrationCfg *integ,const ThresholdCfg *thr,double frac_order,const double *ref,int nref,const double x0[3],int *sec_total,int *sec_hits,double *hit_frac){ double *t=NULL,*X=NULL; int N=0; double seed_local[3]={x0[0],x0[1],x0[2]}; if(!efork_chua_caputo(p,seed_local,frac_order,integ->h,integ->TMAX_TEST,integ->Lm,&t,&X,&N)) die("No se pudo integrar una trayectoria de prueba"); int cls0=classify_equilibrium_or_divergence(X,N,eqs,thr); if(cls0!=CLS_NONE){ *sec_total=0; *sec_hits=0; *hit_frac=0.0; free(t); free(X); return cls0; } int max_pts=thr->TEST_MAX_SEC; double *sec=(double*)malloc((size_t)max_pts*2u*sizeof(double)); if(!sec) die_errno("No se pudo reservar memoria para la sección de prueba"); int nsec=section_points(t,X,N,p,integ->TBURN_TEST,max_pts,sec); free(t); free(X); if(nsec<thr->MIN_SEC_MATCH){ *sec_total=nsec; *sec_hits=0; *hit_frac=0.0; free(sec); return CLS_UNKNOWN; } int hits=0; for(int i=0;i<nsec;++i){ double dmin=min_dist_to_ref(ref,nref,sec[2*i+0],sec[2*i+1]); if(dmin<=thr->SEC_TOL) hits++; } *sec_total=nsec; *sec_hits=hits; *hit_frac=(double)hits/(double)((nsec>0)?nsec:1); free(sec); return (*hit_frac>=thr->HIT_FRAC_REQ)?CLS_TARGET:CLS_OTHER; }

static void write_reference_csv(const char *path,const double *ref,int nref){ FILE *f=fopen(path,"w"); if(!f) die_errno("No se pudo abrir reference_section.csv"); fprintf(f,"y,z\n"); for(int i=0;i<nref;++i) fprintf(f,"%.17g,%.17g\n",ref[2*i+0],ref[2*i+1]); fclose(f); }

static void run_backend(const ChuaParams *p,double frac_order,const double target_seed[3],const IntegrationCfg *integ,const ThresholdCfg *thr,const SamplingCfg *sampling,const FileCfg *files,RunStats *stats){ EqSet eqs; chua_equilibria(p,&eqs); printf("Equilibrios:\n"); for(int i=0;i<eqs.n_eq;++i) printf("%s = (%.10f, %.10f, %.10f)\n",eqs.names[i],eqs.val[i][0],eqs.val[i][1],eqs.val[i][2]); printf("\nSemilla objetivo = (%.10f, %.10f, %.10f)\n\n",target_seed[0],target_seed[1],target_seed[2]); const char *active_filter=sampling->eq_filter[0] ? sampling->eq_filter : getenv("HIDDEN_VERIFY_EQ_FILTER"); if(active_filter&&active_filter[0]) printf("Filtro de equilibrios: %s\n\n",active_filter); double *ref=NULL; int nref=0; int ref_cls=build_reference_section(p,&eqs,integ,thr,frac_order,target_seed,&ref,&nref); if(ref_cls!=CLS_TARGET) die("No se pudo construir una referencia robusta del atractor objetivo. Ajusta tiempos, h, Lm o la semilla."); stats->ref_points=nref; write_reference_csv(files->ref_out,ref,nref); printf("Referencia construida con %d puntos de sección.\n\n",nref); FILE *fcsv=fopen(files->csv_out,"w"); if(!fcsv) die_errno("No se pudo abrir csv_out"); fprintf(fcsv,"equilibrium,radius,sample_id,x0,y0,z0,class,sec_total,sec_hits,hit_frac\n"); FILE *fsum=fopen(files->summary_csv_out,"w"); if(!fsum) die_errno("No se pudo abrir summary_csv_out"); fprintf(fsum,"equilibrium,radius,EQ,DIV,TARGET,OTHER,UNKNOWN\n"); int total_target_hits=0; for(int e=0;e<eqs.n_eq;++e){ if(!eq_filter_allows(sampling,eqs.names[e])) continue; printf("=== Muestreo alrededor de %s ===\n",eqs.names[e]); for(int ir=0;ir<sampling->n_radii;++ir){ double radius=sampling->radii[ir]; int ns=sampling->nsamples; SampleRow *rows=(SampleRow*)calloc((size_t)ns,sizeof(SampleRow)); if(!rows) die_errno("No se pudo reservar memoria para rows"); 
#ifdef _OPENMP
#pragma omp parallel for if(ns > 1)
#endif
for(int i=0;i<ns;++i){ uint64_t state=make_sample_seed(sampling->random_seed,e,ir,i); double x0[3]; sample_in_ball(eqs.val[e],radius,&state,x0); rows[i].x0=x0[0]; rows[i].y0=x0[1]; rows[i].z0=x0[2]; rows[i].cls=classify_to_target(p,&eqs,integ,thr,frac_order,ref,nref,x0,&rows[i].sec_total,&rows[i].sec_hits,&rows[i].hit_frac);} int counts[5]={0,0,0,0,0}; for(int i=0;i<ns;++i){ int cls=rows[i].cls; if(cls<0||cls>4) cls=CLS_UNKNOWN; counts[cls]++; fprintf(fcsv,"%s,%.17g,%d,%.17g,%.17g,%.17g,%s,%d,%d,%.17g\n",eqs.names[e],radius,i,rows[i].x0,rows[i].y0,rows[i].z0,CLASS_NAMES[cls],rows[i].sec_total,rows[i].sec_hits,rows[i].hit_frac); } fprintf(fsum,"%s,%.17g,%d,%d,%d,%d,%d\n",eqs.names[e],radius,counts[CLS_EQ],counts[CLS_DIV],counts[CLS_TARGET],counts[CLS_OTHER],counts[CLS_UNKNOWN]); total_target_hits += counts[CLS_TARGET]; printf("r=%-8.1e  EQ=%3d  DIV=%3d  TARGET=%3d  OTHER=%3d  UNKNOWN=%3d\n",radius,counts[CLS_EQ],counts[CLS_DIV],counts[CLS_TARGET],counts[CLS_OTHER],counts[CLS_UNKNOWN]); fflush(fcsv); fflush(fsum); free(rows);} printf("\n"); } fclose(fcsv); fclose(fsum); free(ref); stats->total_target_hits=total_target_hits; }

static void usage(const char *prog){ fprintf(stderr,"Uso: %s \\\n  --alpha_chua A --beta B --gamma_chua G --m0 M0 --m1 M1 \\\n  --frac_order q --target_seed x,y,z \\\n  --h h --Lm Lm --TMAX_REF tr --TMAX_TEST tt --TBURN_REF br --TBURN_TEST bt \\\n  --R_DIV rdiv --EPS_EQ epseq --CAP_WIN cap --SEC_TOL stol --MIN_SEC_MATCH msec --TEST_MAX_SEC tsec --HIT_FRAC_REQ hreq \\\n  --radii r1,r2,... --nsamples N --random_seed S \\\n  --csv_out out.csv --ref_out ref.csv --summary_csv_out summary.csv\n",prog); }

int main(int argc,char **argv){ setvbuf(stdout,NULL,_IONBF,0); if(argc==1||has_arg(argc,argv,"--help")){ usage(argv[0]); return (argc==1)?EXIT_FAILURE:EXIT_SUCCESS; } ChuaParams p; IntegrationCfg integ; ThresholdCfg thr; SamplingCfg sampling={0}; FileCfg files; double frac_order, target_seed[3]; p.alpha_chua=parse_double_arg(argc,argv,"--alpha_chua"); p.beta=parse_double_arg(argc,argv,"--beta"); p.gamma_chua=parse_double_arg(argc,argv,"--gamma_chua"); p.m0=parse_double_arg(argc,argv,"--m0"); p.m1=parse_double_arg(argc,argv,"--m1"); p.model=parse_model_name(find_arg(argc,argv,"--model")); p.a1=optional_double_arg(argc,argv,"--a1",0.4); p.a2=optional_double_arg(argc,argv,"--a2",-1.5585); p.rho=optional_double_arg(argc,argv,"--rho",1.0); if(!(p.rho>0.0)) die("--rho debe ser positivo"); frac_order=parse_double_arg(argc,argv,"--frac_order"); if(!(frac_order>0.0 && frac_order<=1.0)) die("El orden fraccionario q debe cumplir 0 < q <= 1."); parse_vec3_arg(argc,argv,"--target_seed",target_seed); integ.h=parse_double_arg(argc,argv,"--h"); integ.Lm=parse_double_arg(argc,argv,"--Lm"); integ.TMAX_REF=parse_double_arg(argc,argv,"--TMAX_REF"); integ.TMAX_TEST=parse_double_arg(argc,argv,"--TMAX_TEST"); integ.TBURN_REF=parse_double_arg(argc,argv,"--TBURN_REF"); integ.TBURN_TEST=parse_double_arg(argc,argv,"--TBURN_TEST"); thr.R_DIV=parse_double_arg(argc,argv,"--R_DIV"); thr.EPS_EQ=parse_double_arg(argc,argv,"--EPS_EQ"); thr.CAP_WIN=parse_int_arg(argc,argv,"--CAP_WIN"); thr.SEC_TOL=parse_double_arg(argc,argv,"--SEC_TOL"); thr.MIN_SEC_MATCH=parse_int_arg(argc,argv,"--MIN_SEC_MATCH"); thr.TEST_MAX_SEC=parse_int_arg(argc,argv,"--TEST_MAX_SEC"); thr.HIT_FRAC_REQ=parse_double_arg(argc,argv,"--HIT_FRAC_REQ"); parse_radii_arg(argc,argv,"--radii",&sampling); sampling.nsamples=parse_int_arg(argc,argv,"--nsamples"); sampling.random_seed=parse_u64_arg(argc,argv,"--random_seed"); const char *csv_out=find_arg(argc,argv,"--csv_out"), *ref_out=find_arg(argc,argv,"--ref_out"), *sum_out=find_arg(argc,argv,"--summary_csv_out"); if(!csv_out||!ref_out||!sum_out){ free(sampling.radii); die("Faltan rutas de salida: --csv_out, --ref_out o --summary_csv_out"); } snprintf(files.csv_out,sizeof(files.csv_out),"%s",csv_out); snprintf(files.ref_out,sizeof(files.ref_out),"%s",ref_out); snprintf(files.summary_csv_out,sizeof(files.summary_csv_out),"%s",sum_out); RunStats stats={0,0}; run_backend(&p,frac_order,target_seed,&integ,&thr,&sampling,&files,&stats); printf("Resumen final:\n"); if(stats.total_target_hits==0) printf("No se detectaron trayectorias que cayeran en el atractor objetivo desde las vecindades muestreadas de los equilibrios.\n"); else printf("Se detectaron %d trayectorias clasificadas como TARGET.\n",stats.total_target_hits); printf("\nArchivos numéricos generados:\n- %s\n- %s\n- %s\n",files.csv_out,files.ref_out,files.summary_csv_out); free(sampling.radii); return EXIT_SUCCESS; }
