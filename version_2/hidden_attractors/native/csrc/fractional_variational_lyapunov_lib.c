#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #define API_EXPORT __declspec(dllexport)
#else
  #define API_EXPORT __attribute__((visibility("default")))
#endif

#define ABI_VERSION 1
#define STATE_DIM 3
#define EXT_DIM 12
#define SYSTEM_RF 1
#define SYSTEM_LORENZ 2
#define CONTRACT_DK2018_BLOCK_RESTART_ABM_GS 1
#define CONTRACT_FIXED_LOWER_LIMIT_FULL_HISTORY_QR 2
#define CONVOLUTION_DIRECT 1
#define CONVOLUTION_FFT_BLOCK 2
#define STATUS_OK 0
#define STATUS_INVALID_REQUEST -1
#define STATUS_ALLOCATION_FAILED -2
#define STATUS_NONFINITE -3
#define STATUS_DIVERGED -4
#define STATUS_OUTPUT_TOO_SMALL -5

typedef struct {
    int abi_version;
    int system_id;
    int execution_contract;
    int convolution_mode;
    int fft_block_size;
    double q;
    double h;
    double t_final;
    double t_burn;
    double reorthonormalization_time;
    double divergence_norm;
    double x0[4];
    double parameters[8];
} FractionalLyapunovRequest;

typedef struct {
    int abi_version;
    int status_code;
    int steps_completed;
    int convergence_rows;
    double exponents[4];
    double final_state[4];
} FractionalLyapunovResult;

typedef struct {
    int mode;
    int dim;
    int nsteps;
    int block_size;
    int active_start;
    int active_count;
    double *history;
    double *active;
    double *future_pred;
    double *future_corr;
    const double *pred_kernel;
    const double *corr_kernel;
} ConvStream;

static int next_pow2(int value) {
    int n = 1;
    while (n < value) n <<= 1;
    return n;
}

static void fft(double complex *values, int n, int inverse) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            const double complex tmp = values[i];
            values[i] = values[j];
            values[j] = tmp;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * acos(-1.0) / (double)len;
        const double complex wlen = cos(angle) + I * sin(angle);
        for (int i = 0; i < n; i += len) {
            double complex w = 1.0;
            for (int j = 0; j < len / 2; ++j) {
                const double complex u = values[i + j];
                const double complex v = values[i + j + len / 2] * w;
                values[i + j] = u + v;
                values[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (int i = 0; i < n; ++i) values[i] /= (double)n;
    }
}

static int add_fft_block(
    double *future,
    const double *source,
    int source_start,
    int source_count,
    int dim,
    const double *kernel,
    int max_target,
    int skip_global_zero
) {
    const int kernel_len = max_target - source_start + 1;
    const int nfft = next_pow2(source_count + kernel_len - 1);
    double complex *kernel_fft = (double complex*)calloc((size_t)nfft, sizeof(double complex));
    if (!kernel_fft) return STATUS_ALLOCATION_FAILED;
    for (int i = 0; i < kernel_len; ++i) kernel_fft[i] = kernel[i];
    fft(kernel_fft, nfft, 0);

    for (int d = 0; d < dim; ++d) {
        double complex *work = (double complex*)calloc((size_t)nfft, sizeof(double complex));
        if (!work) {
            free(kernel_fft);
            return STATUS_ALLOCATION_FAILED;
        }
        for (int i = 0; i < source_count; ++i) {
            const int global_index = source_start + i;
            work[i] = (skip_global_zero && global_index == 0) ? 0.0 : source[(size_t)i * dim + d];
        }
        fft(work, nfft, 0);
        for (int i = 0; i < nfft; ++i) work[i] *= kernel_fft[i];
        fft(work, nfft, 1);
        /*
         * Keep the block-end convolution too: after f[end] is appended the
         * next ABM step still evaluates its history sum at index end.
         */
        for (int target = source_start + source_count - 1; target <= max_target; ++target) {
            future[(size_t)target * dim + d] += creal(work[target - source_start]);
        }
        free(work);
    }
    free(kernel_fft);
    return STATUS_OK;
}

static int stream_init(
    ConvStream *stream,
    int mode,
    int dim,
    int nsteps,
    int block_size,
    const double *pred_kernel,
    const double *corr_kernel
) {
    memset(stream, 0, sizeof(*stream));
    stream->mode = mode;
    stream->dim = dim;
    stream->nsteps = nsteps;
    stream->block_size = block_size > 0 ? block_size : 256;
    stream->pred_kernel = pred_kernel;
    stream->corr_kernel = corr_kernel;
    if (mode == CONVOLUTION_DIRECT) {
        stream->history = (double*)calloc((size_t)(nsteps + 1) * dim, sizeof(double));
        return stream->history ? STATUS_OK : STATUS_ALLOCATION_FAILED;
    }
    if (mode != CONVOLUTION_FFT_BLOCK) return STATUS_INVALID_REQUEST;
    stream->active = (double*)calloc((size_t)stream->block_size * dim, sizeof(double));
    stream->future_pred = (double*)calloc((size_t)(nsteps + 1) * dim, sizeof(double));
    stream->future_corr = (double*)calloc((size_t)(nsteps + 1) * dim, sizeof(double));
    return (stream->active && stream->future_pred && stream->future_corr)
        ? STATUS_OK : STATUS_ALLOCATION_FAILED;
}

static void stream_free(ConvStream *stream) {
    free(stream->history);
    free(stream->active);
    free(stream->future_pred);
    free(stream->future_corr);
    memset(stream, 0, sizeof(*stream));
}

static int stream_flush(ConvStream *stream) {
    if (stream->mode != CONVOLUTION_FFT_BLOCK || stream->active_count == 0) return STATUS_OK;
    int rc = add_fft_block(
        stream->future_pred, stream->active, stream->active_start, stream->active_count,
        stream->dim, stream->pred_kernel, stream->nsteps, 0
    );
    if (rc != STATUS_OK) return rc;
    rc = add_fft_block(
        stream->future_corr, stream->active, stream->active_start, stream->active_count,
        stream->dim, stream->corr_kernel, stream->nsteps, 1
    );
    if (rc != STATUS_OK) return rc;
    stream->active_start += stream->active_count;
    stream->active_count = 0;
    memset(stream->active, 0, (size_t)stream->block_size * stream->dim * sizeof(double));
    return STATUS_OK;
}

static int stream_append(ConvStream *stream, int index, const double *value) {
    if (stream->mode == CONVOLUTION_DIRECT) {
        memcpy(&stream->history[(size_t)index * stream->dim], value, (size_t)stream->dim * sizeof(double));
        return STATUS_OK;
    }
    if (index != stream->active_start + stream->active_count) return STATUS_INVALID_REQUEST;
    memcpy(&stream->active[(size_t)stream->active_count * stream->dim], value, (size_t)stream->dim * sizeof(double));
    stream->active_count += 1;
    return stream->active_count == stream->block_size ? stream_flush(stream) : STATUS_OK;
}

static void stream_sums(const ConvStream *stream, int index, double *pred, double *corr) {
    memset(pred, 0, (size_t)stream->dim * sizeof(double));
    memset(corr, 0, (size_t)stream->dim * sizeof(double));
    if (stream->mode == CONVOLUTION_DIRECT) {
        for (int j = 0; j <= index; ++j) {
            const double wp = stream->pred_kernel[index - j];
            const double wc = stream->corr_kernel[index - j];
            for (int d = 0; d < stream->dim; ++d) {
                const double f = stream->history[(size_t)j * stream->dim + d];
                pred[d] += wp * f;
                if (j > 0) corr[d] += wc * f;
            }
        }
        return;
    }
    memcpy(pred, &stream->future_pred[(size_t)index * stream->dim], (size_t)stream->dim * sizeof(double));
    memcpy(corr, &stream->future_corr[(size_t)index * stream->dim], (size_t)stream->dim * sizeof(double));
    for (int offset = 0; offset < stream->active_count; ++offset) {
        const int j = stream->active_start + offset;
        if (j > index) break;
        const double wp = stream->pred_kernel[index - j];
        const double wc = stream->corr_kernel[index - j];
        for (int d = 0; d < stream->dim; ++d) {
            const double f = stream->active[(size_t)offset * stream->dim + d];
            pred[d] += wp * f;
            if (j > 0) corr[d] += wc * f;
        }
    }
}

static void transform_variational(double *value, const double *right) {
    double tmp[9];
    for (int row = 0; row < STATE_DIM; ++row) {
        for (int col = 0; col < STATE_DIM; ++col) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += value[STATE_DIM + row * STATE_DIM + k] * right[k * STATE_DIM + col];
            }
            tmp[row * STATE_DIM + col] = sum;
        }
    }
    memcpy(&value[STATE_DIM], tmp, sizeof(tmp));
}

static void stream_transform(ConvStream *stream, int current_index, const double *right) {
    if (stream->mode == CONVOLUTION_DIRECT) {
        for (int j = 0; j <= current_index; ++j) {
            transform_variational(&stream->history[(size_t)j * stream->dim], right);
        }
        return;
    }
    for (int j = 0; j < stream->active_count; ++j) {
        transform_variational(&stream->active[(size_t)j * stream->dim], right);
    }
    for (int j = current_index; j <= stream->nsteps; ++j) {
        transform_variational(&stream->future_pred[(size_t)j * stream->dim], right);
        transform_variational(&stream->future_corr[(size_t)j * stream->dim], right);
    }
}

static void rhs_jacobian(
    int system_id,
    const double *params,
    const double *x,
    double *rhs,
    double *jac
) {
    const double x1 = x[0], x2 = x[1], x3 = x[2];
    if (system_id == SYSTEM_RF) {
        const double a = params[0], b = params[1];
        rhs[0] = x2 * (x3 - 1.0 + x1 * x1) + a * x1;
        rhs[1] = x1 * (3.0 * x3 + 1.0 - x1 * x1) + a * x2;
        rhs[2] = -2.0 * x3 * (b + x1 * x2);
        jac[0] = 2.0 * x1 * x2 + a; jac[1] = x1 * x1 + x3 - 1.0; jac[2] = x2;
        jac[3] = -3.0 * x1 * x1 + 3.0 * x3 + 1.0; jac[4] = a; jac[5] = 3.0 * x1;
        jac[6] = -2.0 * x2 * x3; jac[7] = -2.0 * x1 * x3; jac[8] = -2.0 * (x1 * x2 + b);
        return;
    }
    const double sigma = params[0], beta = params[1], rho = params[2];
    rhs[0] = sigma * (x2 - x1);
    rhs[1] = x1 * (rho - x3) - x2;
    rhs[2] = x1 * x2 - beta * x3;
    jac[0] = -sigma; jac[1] = sigma; jac[2] = 0.0;
    jac[3] = rho - x3; jac[4] = -1.0; jac[5] = -x1;
    jac[6] = x2; jac[7] = x1; jac[8] = -beta;
}

static void extended_rhs(const FractionalLyapunovRequest *req, const double *state, double *out) {
    double jac[9];
    rhs_jacobian(req->system_id, req->parameters, state, out, jac);
    for (int row = 0; row < STATE_DIM; ++row) {
        for (int col = 0; col < STATE_DIM; ++col) {
            double sum = 0.0;
            for (int k = 0; k < STATE_DIM; ++k) {
                sum += jac[row * STATE_DIM + k] * state[STATE_DIM + k * STATE_DIM + col];
            }
            out[STATE_DIM + row * STATE_DIM + col] = sum;
        }
    }
}

static int invert3(const double *matrix, double *inverse) {
    double a[3][6];
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) a[row][col] = matrix[row * 3 + col];
        for (int col = 0; col < 3; ++col) a[row][3 + col] = row == col ? 1.0 : 0.0;
    }
    for (int col = 0; col < 3; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 3; ++row) {
            if (fabs(a[row][col]) > fabs(a[pivot][col])) pivot = row;
        }
        if (fabs(a[pivot][col]) < 1e-300) return STATUS_NONFINITE;
        if (pivot != col) {
            for (int k = 0; k < 6; ++k) {
                const double tmp = a[col][k]; a[col][k] = a[pivot][k]; a[pivot][k] = tmp;
            }
        }
        const double scale = a[col][col];
        for (int k = 0; k < 6; ++k) a[col][k] /= scale;
        for (int row = 0; row < 3; ++row) {
            if (row == col) continue;
            const double factor = a[row][col];
            for (int k = 0; k < 6; ++k) a[row][k] -= factor * a[col][k];
        }
    }
    for (int row = 0; row < 3; ++row) for (int col = 0; col < 3; ++col) inverse[row * 3 + col] = a[row][3 + col];
    return STATUS_OK;
}

static int orthonormalize(double *state, double *log_diag, double *right_inverse) {
    double q[9] = {0.0}, r[9] = {0.0};
    const double *phi = &state[STATE_DIM];
    for (int col = 0; col < 3; ++col) {
        double v[3] = {phi[col], phi[3 + col], phi[6 + col]};
        for (int prev = 0; prev < col; ++prev) {
            double dot = 0.0;
            for (int row = 0; row < 3; ++row) dot += q[row * 3 + prev] * v[row];
            r[prev * 3 + col] = dot;
            for (int row = 0; row < 3; ++row) v[row] -= dot * q[row * 3 + prev];
        }
        double norm = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (!isfinite(norm) || norm < 1e-300) return STATUS_NONFINITE;
        r[col * 3 + col] = norm;
        log_diag[col] = log(norm);
        for (int row = 0; row < 3; ++row) q[row * 3 + col] = v[row] / norm;
    }
    if (invert3(r, right_inverse) != STATUS_OK) return STATUS_NONFINITE;
    memcpy(&state[STATE_DIM], q, sizeof(q));
    return STATUS_OK;
}

static int make_kernels(int nsteps, double q, double **pred_out, double **corr_out) {
    double *pred = (double*)calloc((size_t)nsteps + 1, sizeof(double));
    double *corr = (double*)calloc((size_t)nsteps + 1, sizeof(double));
    if (!pred || !corr) {
        free(pred); free(corr);
        return STATUS_ALLOCATION_FAILED;
    }
    for (int lag = 0; lag <= nsteps; ++lag) {
        pred[lag] = pow((double)(lag + 1), q) - pow((double)lag, q);
        corr[lag] = pow((double)(lag + 2), q + 1.0)
            + pow((double)lag, q + 1.0)
            - 2.0 * pow((double)(lag + 1), q + 1.0);
    }
    *pred_out = pred;
    *corr_out = corr;
    return STATUS_OK;
}

static int integrate_segment(
    const FractionalLyapunovRequest *req,
    double *state,
    int nsteps,
    int convolution_mode,
    int fft_block_size
) {
    double *pred_kernel = NULL, *corr_kernel = NULL;
    if (make_kernels(nsteps, req->q, &pred_kernel, &corr_kernel) != STATUS_OK) return STATUS_ALLOCATION_FAILED;
    ConvStream stream;
    int rc = stream_init(&stream, convolution_mode, EXT_DIM, nsteps, fft_block_size, pred_kernel, corr_kernel);
    if (rc != STATUS_OK) {
        free(pred_kernel); free(corr_kernel);
        return rc;
    }
    double anchor[EXT_DIM], f0[EXT_DIM], fp[EXT_DIM], fnew[EXT_DIM], pred_sum[EXT_DIM], corr_sum[EXT_DIM], predictor[EXT_DIM], corrected[EXT_DIM];
    memcpy(anchor, state, sizeof(anchor));
    extended_rhs(req, state, f0);
    rc = stream_append(&stream, 0, f0);
    const double pred_scale = pow(req->h, req->q) / tgamma(req->q + 1.0);
    const double corr_scale = pow(req->h, req->q) / tgamma(req->q + 2.0);
    for (int n = 0; rc == STATUS_OK && n < nsteps; ++n) {
        stream_sums(&stream, n, pred_sum, corr_sum);
        for (int d = 0; d < EXT_DIM; ++d) predictor[d] = anchor[d] + pred_scale * pred_sum[d];
        extended_rhs(req, predictor, fp);
        const double a0 = pow((double)n, req->q + 1.0) - ((double)n - req->q) * pow((double)(n + 1), req->q);
        for (int d = 0; d < EXT_DIM; ++d) corrected[d] = anchor[d] + corr_scale * (a0 * f0[d] + corr_sum[d] + fp[d]);
        for (int d = 0; d < EXT_DIM; ++d) if (!isfinite(corrected[d])) rc = STATUS_NONFINITE;
        if (req->divergence_norm > 0.0) {
            const double norm = sqrt(corrected[0] * corrected[0] + corrected[1] * corrected[1] + corrected[2] * corrected[2]);
            if (norm >= req->divergence_norm) rc = STATUS_DIVERGED;
        }
        if (rc != STATUS_OK) break;
        memcpy(state, corrected, sizeof(corrected));
        extended_rhs(req, state, fnew);
        rc = stream_append(&stream, n + 1, fnew);
    }
    stream_free(&stream);
    free(pred_kernel); free(corr_kernel);
    return rc;
}

static int run_dk2018(
    const FractionalLyapunovRequest *req,
    FractionalLyapunovResult *result,
    double *times,
    double *convergence,
    int max_rows
) {
    const int total_steps = (int)llround(req->t_final / req->h);
    const int interval = (int)llround(req->reorthonormalization_time / req->h);
    if (interval < 1) return STATUS_INVALID_REQUEST;
    double state[EXT_DIM] = {0.0}, sums[3] = {0.0};
    memcpy(state, req->x0, 3 * sizeof(double));
    state[3] = state[7] = state[11] = 1.0;
    int completed = 0, rows = 0;
    while (completed < total_steps) {
        const int segment_steps = (total_steps - completed < interval) ? total_steps - completed : interval;
        int rc = integrate_segment(req, state, segment_steps, req->convolution_mode, req->fft_block_size);
        if (rc != STATUS_OK) return rc;
        completed += segment_steps;
        double log_diag[3], right_inverse[9];
        rc = orthonormalize(state, log_diag, right_inverse);
        if (rc != STATUS_OK) return rc;
        const double t = completed * req->h;
        if (t > req->t_burn) {
            for (int d = 0; d < 3; ++d) sums[d] += log_diag[d];
            if (rows >= max_rows) return STATUS_OUTPUT_TOO_SMALL;
            times[rows] = t;
            for (int d = 0; d < 3; ++d) convergence[(size_t)rows * 3 + d] = sums[d] / (t - req->t_burn);
            rows += 1;
        }
    }
    result->steps_completed = completed;
    result->convergence_rows = rows;
    for (int d = 0; d < 3; ++d) {
        result->exponents[d] = rows ? convergence[(size_t)(rows - 1) * 3 + d] : NAN;
        result->final_state[d] = state[d];
    }
    return STATUS_OK;
}

static int run_fixed_lower_limit(
    const FractionalLyapunovRequest *req,
    FractionalLyapunovResult *result,
    double *times,
    double *convergence,
    int max_rows
) {
    const int total_steps = (int)llround((req->t_burn + req->t_final) / req->h);
    const int burn_steps = (int)llround(req->t_burn / req->h);
    const int interval = (int)llround(req->reorthonormalization_time / req->h);
    if (interval < 1) return STATUS_INVALID_REQUEST;
    double *pred_kernel = NULL, *corr_kernel = NULL;
    if (make_kernels(total_steps, req->q, &pred_kernel, &corr_kernel) != STATUS_OK) return STATUS_ALLOCATION_FAILED;
    ConvStream stream;
    int rc = stream_init(&stream, req->convolution_mode, EXT_DIM, total_steps, req->fft_block_size, pred_kernel, corr_kernel);
    if (rc != STATUS_OK) {
        free(pred_kernel); free(corr_kernel);
        return rc;
    }
    double state[EXT_DIM] = {0.0}, anchor[EXT_DIM] = {0.0}, f0[EXT_DIM], fp[EXT_DIM], fnew[EXT_DIM];
    double pred_sum[EXT_DIM], corr_sum[EXT_DIM], predictor[EXT_DIM], corrected[EXT_DIM], sums[3] = {0.0};
    memcpy(state, req->x0, 3 * sizeof(double));
    state[3] = state[7] = state[11] = 1.0;
    memcpy(anchor, state, sizeof(anchor));
    extended_rhs(req, state, f0);
    rc = stream_append(&stream, 0, f0);
    const double pred_scale = pow(req->h, req->q) / tgamma(req->q + 1.0);
    const double corr_scale = pow(req->h, req->q) / tgamma(req->q + 2.0);
    int rows = 0, completed = 0;
    for (int n = 0; rc == STATUS_OK && n < total_steps; ++n) {
        stream_sums(&stream, n, pred_sum, corr_sum);
        for (int d = 0; d < EXT_DIM; ++d) predictor[d] = anchor[d] + pred_scale * pred_sum[d];
        extended_rhs(req, predictor, fp);
        const double a0 = pow((double)n, req->q + 1.0) - ((double)n - req->q) * pow((double)(n + 1), req->q);
        for (int d = 0; d < EXT_DIM; ++d) corrected[d] = anchor[d] + corr_scale * (a0 * f0[d] + corr_sum[d] + fp[d]);
        for (int d = 0; d < EXT_DIM; ++d) if (!isfinite(corrected[d])) rc = STATUS_NONFINITE;
        if (req->divergence_norm > 0.0) {
            const double norm = sqrt(corrected[0] * corrected[0] + corrected[1] * corrected[1] + corrected[2] * corrected[2]);
            if (norm >= req->divergence_norm) rc = STATUS_DIVERGED;
        }
        if (rc != STATUS_OK) break;
        memcpy(state, corrected, sizeof(corrected));
        extended_rhs(req, state, fnew);
        rc = stream_append(&stream, n + 1, fnew);
        completed = n + 1;
        const int burn_end = burn_steps > 0 && completed == burn_steps;
        if (rc == STATUS_OK && (completed % interval == 0 || burn_end)) {
            double log_diag[3], right_inverse[9];
            rc = orthonormalize(state, log_diag, right_inverse);
            if (rc != STATUS_OK) break;
            transform_variational(anchor, right_inverse);
            transform_variational(f0, right_inverse);
            stream_transform(&stream, completed, right_inverse);
            if (completed > burn_steps && !burn_end) {
                const double elapsed = (completed - burn_steps) * req->h;
                for (int d = 0; d < 3; ++d) sums[d] += log_diag[d];
                if (rows >= max_rows) { rc = STATUS_OUTPUT_TOO_SMALL; break; }
                times[rows] = elapsed;
                for (int d = 0; d < 3; ++d) convergence[(size_t)rows * 3 + d] = sums[d] / elapsed;
                rows += 1;
            }
        }
    }
    result->steps_completed = completed;
    result->convergence_rows = rows;
    for (int d = 0; d < 3; ++d) {
        result->exponents[d] = rows ? convergence[(size_t)(rows - 1) * 3 + d] : NAN;
        result->final_state[d] = state[d];
    }
    stream_free(&stream);
    free(pred_kernel); free(corr_kernel);
    return rc;
}

API_EXPORT int fractional_lyapunov_abi_version(void) {
    return ABI_VERSION;
}

API_EXPORT int fractional_lyapunov_rhs_jacobian(
    int system_id,
    const double *parameters,
    const double *state,
    double *rhs,
    double *jacobian
) {
    if (!parameters || !state || !rhs || !jacobian || (system_id != SYSTEM_RF && system_id != SYSTEM_LORENZ)) {
        return STATUS_INVALID_REQUEST;
    }
    rhs_jacobian(system_id, parameters, state, rhs, jacobian);
    return STATUS_OK;
}

API_EXPORT int fractional_lyapunov_run(
    const FractionalLyapunovRequest *request,
    FractionalLyapunovResult *result,
    double *times,
    double *convergence,
    int max_rows
) {
    if (!request || !result || !times || !convergence || max_rows < 1) return STATUS_INVALID_REQUEST;
    memset(result, 0, sizeof(*result));
    result->abi_version = ABI_VERSION;
    if (
        request->abi_version != ABI_VERSION
        || (request->system_id != SYSTEM_RF && request->system_id != SYSTEM_LORENZ)
        || (request->execution_contract != CONTRACT_DK2018_BLOCK_RESTART_ABM_GS
            && request->execution_contract != CONTRACT_FIXED_LOWER_LIMIT_FULL_HISTORY_QR)
        || (request->convolution_mode != CONVOLUTION_DIRECT && request->convolution_mode != CONVOLUTION_FFT_BLOCK)
        || !(request->q > 0.0 && request->q < 1.0)
        || !(request->h > 0.0)
        || !(request->t_final > 0.0)
        || !(request->reorthonormalization_time > 0.0)
    ) {
        result->status_code = STATUS_INVALID_REQUEST;
        return STATUS_INVALID_REQUEST;
    }
    int rc = request->execution_contract == CONTRACT_DK2018_BLOCK_RESTART_ABM_GS
        ? run_dk2018(request, result, times, convergence, max_rows)
        : run_fixed_lower_limit(request, result, times, convergence, max_rows);
    result->status_code = rc;
    return rc;
}
