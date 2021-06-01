import cupy as cp

remove_nonvisible_kernel = cp.RawKernel('''
extern "C" __global__
void remove_nonvisible(float *y, int size, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % size3;
		if (y[id] >= x) {
			y[id] = 0;
		}
	}
}''','remove_nonvisible')

remove_occluded_kernel = cp.RawKernel('''
extern "C" __global__
void remove_occluded(float *y, int size, int size3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % size3;
		for (int i = 1; x + i < size3; i++) {
			if (i - y[id + i] < -y[id]) {
				y[id] = 0;
				break;
			}
		}
	}
}''','remove_occluded')

remove_white_kernel = cp.RawKernel('''
extern "C"
__global__ void remove_white(float *x, float *y, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (x[id] == 255) {
			y[id] = 0;
		}
	}
}
''', 'remove_white')

grayscale_kernel = cp.RawKernel('''
extern "C" 
__global__ void grayscale(float* in, float* out, int w, int h)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col < w && row < h)
	{
		int grayOff = row * w + col;
		int rgbOff = grayOff * 3;
		float r = in[rgbOff];
		float g = in[rgbOff+1];
		float b = in[rgbOff+2];
		out[grayOff] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}
''','grayscale')


sj_kernel = cp.RawKernel('''
extern "C" 
__global__ void StereoJoin_(float *input_L, float *input_R, float *output_L, float *output_R, int size1_input, int size1, int size3, int size23)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size23) {
		int dim3 = id % size3;
		assert(size1_input <= 128);
		float L_cache[128];
		for (int i = 0; i < size1_input; i++) {
			L_cache[i] = input_L[i * size23 + id];
		}

		for (int d = 0; d < size1; d++) {
			if (dim3 - d >= 0) {
				float sum = 0;
				for (int i = 0; i < size1_input; i++) {
					sum -= L_cache[i] * input_R[i * size23 + id - d];
				}
				output_L[d * size23 + id] = sum;
				output_R[d * size23 + id - d] = sum;
			}
		}
	}
}
''','StereoJoin_')

sgm1_kernel = cp.RawKernel("""
#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))
#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)
extern "C" 
__global__ void sgm(float *x0, float *x1, float *vol, float *tmp, float *out, int dim1, int dim2, int dim3, float pi1, float pi2, float tau_so, float alpha1, float sgm_q1, float sgm_q2, int sgm_direction, int direction)
{
	int x, y, dx, dy;

	dx = dy = 0;
	if (sgm_direction <= 1) {
		y = blockIdx.x * blockDim.x + threadIdx.x;
		if (y >= dim2) {
			return;
		}
		if (sgm_direction == 0) {
			x = 0;
			dx = 1;
		} else if (sgm_direction == 1) {
			x = dim3 - 1;
			dx = -1;
		}
	} else if (sgm_direction <= 3) {
		x = blockIdx.x * blockDim.x + threadIdx.x;
		if (x >= dim3) {
			return;
		}
		if (sgm_direction == 2) {
			y = 0;
			dy = 1;
		} else if (sgm_direction == 3) {
			y = dim2 - 1;
			dy = -1;
		}
	}

	assert(dim1 <= 400);
	float tmp_curr_[400];
	float tmp_prev_[400];
	float *tmp_curr = tmp_curr_;
	float *tmp_prev = tmp_prev_;

	float min_prev = CUDART_INF;
	for (; 0 <= y && y < dim2 && 0 <= x && x < dim3; x += dx, y += dy) {
		float min_curr = CUDART_INF;
		for (int d = 0; d < dim1; d++) {
			int ind = (d * dim2 + y) * dim3 + x;

			if (x + d * direction < 0 ||
				x + d * direction >= dim3 || 
				y - dy < 0 || 
				y - dy >= dim2 || 
				x + d * direction - dx < 0 || 
				x + d * direction - dx >= dim3 ||
				x - dx < 0 ||
				x - dx >= dim3) {

				out[ind] += vol[ind];
				tmp_curr[d] = vol[ind];
			} else {
				int ind2 = y * dim3 + x;

				float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * dim3 - dx);
				float D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * dim3 - dx);
				float P1, P2;
				if (D1 < tau_so && D2 < tau_so) { 
					P1 = pi1; 
					P2 = (pi1 * pi2); 
				} else if (D1 > tau_so && D2 > tau_so) { 
					P1 = pi1 / (sgm_q1 * sgm_q2);
					P2 = (pi1 * pi2) / (sgm_q1 * sgm_q2);
				} else {
					P1 = pi1 / sgm_q1;
					P2 = (pi1 * pi2) / sgm_q1;
				}

				assert(min_prev != CUDART_INF);
				float cost = min(tmp_prev[d], min_prev + P2);
				if (d > 0) {
					cost = min(cost, tmp_prev[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
				}
				if (d < dim1 - 1) {
					cost = min(cost, tmp_prev[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
				}
				float val = vol[ind] + cost - min_prev;
				out[ind] += val;
				tmp_curr[d] = val;
			}
			if (tmp_curr[d] < min_curr) {
				min_curr = tmp_curr[d];
			}
		}
		min_prev = min_curr;

		float *swap = tmp_curr;
		tmp_curr = tmp_prev;
		tmp_prev = swap;
	}
}
""", "sgm")


sgm2_kernel = cp.RawKernel('''
#define INDEX(dim0, dim1, dim2, dim3) \
	assert((dim1) >= 0 && (dim1) < size1 && (dim2) >= 0 && (dim2) < size2 && (dim3) >= 0 && (dim3) < size3), \
	((((dim0) * size1 + (dim1)) * size2 + (dim2)) * size3 + dim3)

#define COLOR_DIFF(x, i, j) (abs(x[i] - x[j]))

extern "C" 
__global__ void sgm2(float *x0, float *x1, float *input, float *output, 
float *tmp, float pi1, float pi2, float tau_so, 
                    float alpha1, float sgm_q1, float sgm_q2, int direction, 
                    int size1, int size2, int size3, int step, int sgm_direction)
{
	int x, y, dx, dy;
	int d = threadIdx.x;

	if (sgm_direction == 0) {
		/* right */
		x = step;
		y = blockIdx.x;
		dx = 1;
		dy = 0;
	} else if (sgm_direction == 1) {
		/* left */
		x = size2 - 1 - step;
		y = blockIdx.x;
		dx = -1;
		dy = 0;
	} else if (sgm_direction == 2) {
		/* down */
		x = blockIdx.x;
		y = step;
		dx = 0;
		dy = 1;
	} else if (sgm_direction == 3) {
		/* up */
		x = blockIdx.x;
		y = size1 - 1 - step;
		dx = 0;
		dy = -1;
	}

	if (y - dy < 0 || y - dy >= size1 || x - dx < 0 || x - dx >= size2) {
		float val = input[INDEX(0, y, x, d)];
		output[INDEX(0, y, x, d)] += val;
		tmp[d * size2 + blockIdx.x] = val;
		return;
	}

	__shared__ float output_s[400], output_min[400];

	output_s[d] = output_min[d] = tmp[d * size2 + blockIdx.x];
	__syncthreads();

	for (int i = 256; i > 0; i /= 2) {
		if (d < i && d + i < size3 && output_min[d + i] < output_min[d]) {
			output_min[d] = output_min[d + i];
		}
		__syncthreads();
	}

	int ind2 = y * size2 + x;
	float D1 = COLOR_DIFF(x0, ind2, ind2 - dy * size2 - dx);
	float D2;
	int xx = x + d * direction;
	if (xx < 0 || xx >= size2 || xx - dx < 0 || xx - dx >= size2) {
		D2 = 10;
	} else {
		D2 = COLOR_DIFF(x1, ind2 + d * direction, ind2 + d * direction - dy * size2 - dx);
	}
	float P1, P2;
	if (D1 < tau_so && D2 < tau_so) {
		P1 = pi1;
		P2 = pi2;
	} else if (D1 > tau_so && D2 > tau_so) {
		P1 = pi1 / (sgm_q1 * sgm_q2);
		P2 = pi2 / (sgm_q1 * sgm_q2);
	} else {
		P1 = pi1 / sgm_q1;
		P2 = pi2 / sgm_q1;
	}

	float cost = min(output_s[d], output_min[0] + P2);
	if (d - 1 >= 0) {
		cost = min(cost, output_s[d - 1] + (sgm_direction == 2 ? P1 / alpha1 : P1));
	}
	if (d + 1 < size3) {
		cost = min(cost, output_s[d + 1] + (sgm_direction == 3 ? P1 / alpha1 : P1));
	}

	float val = input[INDEX(0, y, x, d)] + cost - output_min[0];
	output[INDEX(0, y, x, d)] += val;
	tmp[d * size2 + blockIdx.x] = val;
}
''', 'sgm2')


outlier_detection_kernel = cp.RawKernel('''
extern "C"
__global__ void outlier_detection(float *d0, float *d1, float *outlier,
                        int size, int dim3, int disp_max)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int d0i = d0[id];
		if (x - d0i < 0) {
			//assert(0);
			outlier[id] = 1;
		} else if (abs(d0[id] - d1[id - d0i]) < 1.1) {
			outlier[id] = 0; /* match */
		} else {
			outlier[id] = 1; /* occlusion */
			for (int d = 0; d < disp_max; d++) {
				if (x - d >= 0 && abs(d - d1[id - d]) < 1.1) {
					outlier[id] = 2; /* mismatch */
					break;
				}
			}
		}
	}
}
''','outlier_detection')


interpolate_occlusion_kernel = cp.RawKernel('''
extern "C"
__global__ void interpolate_occlusion(float *d0, float *outlier, float *out, int size, int dim3)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 1) {
			out[id] = d0[id];
			return;
		}
		int x = id % dim3;

		int dx = 0;
		while (x + dx >= 0 && outlier[id + dx] != 0) {
			dx--;
		}
		if (x + dx < 0) {
			dx = 0;
			while (x + dx < dim3 && outlier[id + dx] != 0) {
				dx++;
			}
		}
		if (x + dx < dim3) {
			out[id] = d0[id + dx];
		} else {
			out[id] = d0[id];
		}
	}
}
''','interpolate_occlusion')

interpolate_mismatch_kernel = cp.RawKernel('''
__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}

extern "C"
__global__ void interpolate_mismatch(float *d0, float *outlier, float *out, int size, int dim2, int dim3)
{
	const float dir[] = {
		0	,  1,
		-0.5,  1,
		-1	,  1,
		-1	,  0.5,
		-1	,  0,
		-1	, -0.5,
		-1	, -1,
		-0.5, -1,
		0	, -1,
		0.5 , -1,
		1	, -1,
		1	, -0.5,
		1	,  0,
		1	,  0.5,
		1	,  1,
		0.5 ,  1
	};

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		if (outlier[id] != 2) {
			out[id] = d0[id];
			return;
		}

		float vals[16];
		int vals_size = 0;

		int x = id % dim3;
		int y = id / dim3;
		for (int d = 0; d < 16; d++) {
			float dx = dir[2 * d];
			float dy = dir[2 * d + 1];
			float xx = x;
			float yy = y;
			int xx_i = round(xx);
			int yy_i = round(yy);
			while (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3 && outlier[yy_i * dim3 + xx_i] == 2) {
				xx += dx;
				yy += dy;
				xx_i = round(xx);
				yy_i = round(yy);
			}

			int ind = yy_i * dim3 + xx_i;
			if (0 <= yy_i && yy_i < dim2 && 0 <= xx_i && xx_i < dim3) {
				assert(outlier[ind] != 2);
				vals[vals_size++] = d0[ind];
			}
		}
		assert(vals_size > 0);
		sort(vals, vals_size);
		out[id] = vals[vals_size / 2];
	}
}
''','interpolate_mismatch')

subpixel_enchancement_kernel = cp.RawKernel('''
extern "C"
__global__ void subpixel_enchancement(float *d0, float *c2, float *out, int size, int dim23, int disp_max) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int d = d0[id];
		out[id] = d;
		if (1 <= d && d < disp_max - 1) {
			float cn = c2[(d - 1) * dim23 + id];
			float cz = c2[d * dim23 + id];
			float cp = c2[(d + 1) * dim23 + id];
			float denom = 2 * (cp + cn - 2 * cz);
			if (denom > 1e-5) {
				out[id] = d - min(1.0, max(-1.0, (cp - cn) / denom));
			}
		}
	}
}
''','subpixel_enchancement')

median_filter_kernel = cp.RawKernel('''

__device__ void sort(float *x, int n)
{
	for (int i = 0; i < n - 1; i++) {
		int min = i;
		for (int j = i + 1; j < n; j++) {
			if (x[j] < x[min]) {
				min = j;
			}
		}
		float tmp = x[min];
		x[min] = x[i];
		x[i] = tmp;
	}
}

extern "C"
__global__ void median2d(float *img, float *out, int size, int dim2, int dim3, int kernel_radius)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float xs[11 * 11];
		int xs_size = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2) {
					xs[xs_size++] = img[yy * dim3 + xx];
				}
			}
		}
		sort(xs, xs_size);
		out[id] = xs[xs_size / 2];
	}
}
''','median2d')

bilateral_filter_kernel = cp.RawKernel('''
extern "C"
__global__ void mean2d(float *img, float *kernel, float *out, int size, int kernel_radius, int dim2, int dim3, float alpha2)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < size) {
		int x = id % dim3;
		int y = id / dim3;

		float sum = 0;
		float cnt = 0;
		int i = 0;
		for (int xx = x - kernel_radius; xx <= x + kernel_radius; xx++) {
			for (int yy = y - kernel_radius; yy <= y + kernel_radius; yy++, i++) {
				if (0 <= xx && xx < dim3 && 0 <= yy && yy < dim2 && abs(img[yy * dim3 + xx] - img[y * dim3 + x]) < alpha2) {
					sum += img[yy * dim3 + xx] * kernel[i];
					cnt += kernel[i];
				}
			}
		}
		out[id] = sum / cnt;
	}
}
''','mean2d')