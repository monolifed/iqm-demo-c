
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static bool isle(void)
{
	union
	{
		int i;
		uint8_t b[sizeof(int)];
	} conv;
	conv.i = 1;
	return conv.b[0] != 0;
}

static void leswap16(uint8_t *buffer, size_t len)
{
	if (isle())
		return;
	uint8_t temp;
	for (unsigned i = 1; i < len; i += 2)
	{
		temp = buffer[i - 1];
		buffer[i - 1] = buffer[i];
		buffer[i] = temp;
	}
}

static void leswap32(uint8_t *buffer, size_t len)
{
	if (isle())
		return;
	uint8_t temp;
	for (unsigned i = 3; i < len; i += 4)
	{
		temp = buffer[i - 3];
		buffer[i - 3] = buffer[i];
		buffer[i] = temp;
		temp = buffer[i - 2];
		buffer[i - 2] = buffer[i - 1];
		buffer[i - 1] = temp;
	}
}

//static float clampf(float val, float min, float max)
//{
//	return val < min ? min : (val > max ? max : val);
//}

typedef float vec3[3];
typedef float vec4[4];
typedef vec4 quat;
typedef vec4 mat3x4[3]; //row major

//https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
static void mat3x4_from_rst(mat3x4 m, const quat r, const vec3 t, const vec3 s)
{
	float sx = 2 * s[0], sy = 2 * s[1], sz = 2 * s[2];
	float x = r[0], y = r[1], z = r[2], w = r[3];
	float n = sqrtf(x * x + y * y + z * z + w * w);
	x /= n; y /= n; z /= n; w /= n;
	m[0][0] = sx * (0.5 - (y * y + z * z));
	m[0][1] = sy * (x * y - w * z);
	m[0][2] = sz * (x * z + w * y);
	m[0][3] = t[0];
	m[1][0] = sx * (x * y + w * z);
	m[1][1] = sy * (0.5 - (x * x + z * z));
	m[1][2] = sz * (y * z - w * x);
	m[1][3] = t[1];
	m[2][0] = sx * (x * z - w * y);
	m[2][1] = sy * (y * z + w * x);
	m[2][2] = sz * (0.5 - (x * x + y * y));
	m[2][3] = t[2];
}

static void vec4_matmul(vec4 a, mat3x4 m, vec4 b)
{
	a[0] = m[0][0] * b[0] + m[1][0] * b[1] + m[2][0] * b[2];
	a[1] = m[0][1] * b[0] + m[1][1] * b[1] + m[2][1] * b[2];
	a[2] = m[0][2] * b[0] + m[1][2] * b[1] + m[2][2] * b[2];
	a[3] = m[0][3] * b[0] + m[1][3] * b[1] + m[2][3] * b[2] + b[3];
}

static void mat3x4_mul(mat3x4 ret, mat3x4 f, mat3x4 g)
{
	mat3x4 m;
	vec4_matmul(m[0], g, f[0]);
	vec4_matmul(m[1], g, f[1]);
	vec4_matmul(m[2], g, f[2]);
/*
	m[0][0] = g[0][0] * f[0][0] + g[1][0] * f[0][1] + g[2][0] * f[0][2];
	m[0][1] = g[0][1] * f[0][0] + g[1][1] * f[0][1] + g[2][1] * f[0][2];
	m[0][2] = g[0][2] * f[0][0] + g[1][2] * f[0][1] + g[2][2] * f[0][2];
	m[0][3] = g[0][3] * f[0][0] + g[1][3] * f[0][1] + g[2][3] * f[0][2] + f[0][3];
	m[1][0] = g[0][0] * f[1][0] + g[1][0] * f[1][1] + g[2][0] * f[1][2];
	m[1][1] = g[0][1] * f[1][0] + g[1][1] * f[1][1] + g[2][1] * f[1][2];
	m[1][2] = g[0][2] * f[1][0] + g[1][2] * f[1][1] + g[2][2] * f[1][2];
	m[1][3] = g[0][3] * f[1][0] + g[1][3] * f[1][1] + g[2][3] * f[1][2] + f[1][3];
	m[2][0] = g[0][0] * f[2][0] + g[1][0] * f[2][1] + g[2][0] * f[2][2];
	m[2][1] = g[0][1] * f[2][0] + g[1][1] * f[2][1] + g[2][1] * f[2][2];
	m[2][2] = g[0][2] * f[2][0] + g[1][2] * f[2][1] + g[2][2] * f[2][2];
	m[2][3] = g[0][3] * f[2][0] + g[1][3] * f[2][1] + g[2][3] * f[2][2] + f[2][3];
*/
	memcpy(ret, m, sizeof(mat3x4));
}

static void mat3x4_invert(mat3x4 i, mat3x4 m)
{
	float n, x, y, z;
	
	x = m[0][0]; y = m[1][0]; z = m[2][0];
	n = sqrtf(x * x + y * y + z * z);
	i[0][0] = x / n; i[0][1] = y / n; i[0][2] = z / n;
	
	x = m[0][1]; y = m[1][1]; z = m[2][1];
	n = sqrtf(x * x + y * y + z * z);
	i[1][0] = x / n; i[1][1] = y / n; i[1][2] = z / n;
	
	x = m[0][2]; y = m[1][2]; z = m[2][2];
	n = sqrtf(x * x + y * y + z * z);
	i[2][0] = x / n; i[2][1] = y / n; i[2][2] = z / n;
	
	x = m[0][3]; y = m[1][3]; z = m[2][3];
	
	i[0][3] = -(i[0][0] * x + i[0][1] * y + i[0][2] * z);
	i[1][3] = -(i[1][0] * x + i[1][1] * y + i[1][2] * z);
	i[2][3] = -(i[2][0] * x + i[2][1] * y + i[2][2] * z);
}

// These are not used by GPU skinning demo
void mat3x4_scale(mat3x4 m, float s, mat3x4 n)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			m[i][j] = s * n[i][j];
}

void mat3x4_scaleadd(mat3x4 m, float s, mat3x4 n, mat3x4 a)
{
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 4; j++)
			m[i][j] = s * n[i][j] + a[i][j];
}

void vec3_matmul(vec3 a, mat3x4 m, const vec3 b)
{
	a[0] = m[0][0] * b[0] + m[0][1] * b[1] + m[0][2] * b[2] + m[0][3];
	a[1] = m[1][0] * b[0] + m[1][1] * b[1] + m[1][2] * b[2] + m[1][3];
	a[2] = m[2][0] * b[0] + m[2][1] * b[1] + m[2][2] * b[2] + m[2][3];
}

void vec3_cross(vec3 a, vec3 b, vec3 c)
{
	a[0] = b[1] * c[2] - b[2] * c[1];
	a[1] = b[2] * c[0] - b[0] * c[2];
	a[2] = b[0] * c[1] - b[1] * c[0];
}

void vec3_scale(vec3 a, float s, vec3 b)
{
	a[0] = s * b[0];
	a[1] = s * b[1];
	a[2] = s * b[2];
}