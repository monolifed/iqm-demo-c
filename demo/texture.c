#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#include <GL/gl.h>
#include <GL/glext.h>

static int clampi(int x, int min, int max)
{
	return x < min ? min : (x > max ? max : x);
}

#define FUNCNAME(name) name##1
#define DEFPIXEL uint32_t OP(r, 0);
#define PIXELOP OP(r, 0);
#define BPP 1
#include "scale.h"

#define FUNCNAME(name) name##2
#define DEFPIXEL uint32_t OP(r, 0), OP(g, 1);
#define PIXELOP OP(r, 0); OP(g, 1);
#define BPP 2
#include "scale.h"

#define FUNCNAME(name) name##3
#define DEFPIXEL uint32_t OP(r, 0), OP(g, 1), OP(b, 2);
#define PIXELOP OP(r, 0); OP(g, 1); OP(b, 2);
#define BPP 3
#include "scale.h"

#define FUNCNAME(name) name##4
#define DEFPIXEL uint32_t OP(r, 0), OP(g, 1), OP(b, 2), OP(a, 3);
#define PIXELOP OP(r, 0); OP(g, 1); OP(b, 2); OP(a, 3);
#define BPP 4
#include "scale.h"

static void scaletexture(uint8_t *src, uint32_t sw, uint32_t sh, uint32_t bpp, uint32_t pitch, uint8_t *dst, uint32_t dw, uint32_t dh)
{
	if (sw == dw * 2 && sh == dh * 2)
	{
		switch (bpp)
		{
		case 1: halvetexture1(src, sw, sh, pitch, dst); return;
		case 2: halvetexture2(src, sw, sh, pitch, dst); return;
		case 3: halvetexture3(src, sw, sh, pitch, dst); return;
		case 4: halvetexture4(src, sw, sh, pitch, dst); return;
		}
	}
	else if (sw < dw || sh < dh || sw & (sw - 1) || sh & (sh - 1) || dw & (dw - 1) || dh & (dh - 1))
	{
		switch (bpp)
		{
		case 1: scaletexture1(src, sw, sh, pitch, dst, dw, dh); return;
		case 2: scaletexture2(src, sw, sh, pitch, dst, dw, dh); return;
		case 3: scaletexture3(src, sw, sh, pitch, dst, dw, dh); return;
		case 4: scaletexture4(src, sw, sh, pitch, dst, dw, dh); return;
		}
	}
	else
	{
		switch (bpp)
		{
		case 1: shifttexture1(src, sw, sh, pitch, dst, dw, dh); return;
		case 2: shifttexture2(src, sw, sh, pitch, dst, dw, dh); return;
		case 3: shifttexture3(src, sw, sh, pitch, dst, dw, dh); return;
		case 4: shifttexture4(src, sw, sh, pitch, dst, dw, dh); return;
		}
	}
}

static inline void bgr2rgb(uint8_t *data, int len, int bpp)
{
	uint8_t temp;
	for (uint8_t *end = &data[len]; data < end; data += bpp)
	{
		temp = data[0];
		data[0] = data[2];
		data[2] = temp;
	}
}

typedef struct TGAHeader
{
	uint8_t  identsize;
	uint8_t  cmaptype;
	uint8_t  imagetype;
	uint8_t  cmaporigin[2];
	uint8_t  cmapsize[2];
	uint8_t  cmapentrysize;
	uint8_t  xorigin[2];
	uint8_t  yorigin[2];
	uint8_t  width[2];
	uint8_t  height[2];
	uint8_t  pixelsize;
	uint8_t  descbyte;
} TGAHeader;

static uint8_t *loadtga(const char *fname, int *width, int *height, int *bitsperpixel)
{
	FILE *f = fopen(fname, "rb");
	if (!f)
		return NULL;
		
	uint8_t *data = NULL, *cmap = NULL;
	TGAHeader hdr;
	if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr))
		goto error;
	if (fseek(f, hdr.identsize, SEEK_CUR) < 0)
		goto error;
	if (hdr.pixelsize != 8 && hdr.pixelsize != 24 && hdr.pixelsize != 32)
		goto error;
		
	int bpp = hdr.pixelsize / 8;
	int w = hdr.width[0] + (hdr.width[1] << 8);
	int h = hdr.height[0] + (hdr.height[1] << 8);
	
	if (hdr.imagetype == 1)
	{
		int cmapsize = hdr.cmapsize[0] + (hdr.cmapsize[1] << 8);
		if (hdr.cmapentrysize != 8 || hdr.cmapentrysize != 24 || hdr.cmapentrysize != 32)
			goto error;
		bpp = hdr.cmapentrysize / 8;
		cmap = malloc(sizeof(uint8_t) * bpp * cmapsize);
		if ((int) fread(cmap, 1, bpp * cmapsize, f) != bpp * cmapsize)
			goto error;
		if (bpp >= 3)
			bgr2rgb(cmap, bpp * cmapsize, bpp);
		data = malloc(sizeof(uint8_t) * bpp * w * h);
		uint8_t *idxs = &data[(bpp - 1) * w * h];
		if ((int) fread(idxs, 1, w * h, f) != w * h)
			goto error;
		uint8_t *src = idxs, *dst = &data[bpp * w * h];
		for (int i = 0; i < h; i++)
		{
			dst -= bpp * w;
			uint8_t *row = dst;
			for (int j = 0; j < w; j++)
			{
				memcpy(row, &cmap[*src++ * bpp], bpp);
				row += bpp;
			}
		}
	}
	else if (hdr.imagetype == 2)
	{
		data = malloc(sizeof(uint8_t) * bpp * w * h);
		uint8_t *dst = &data[bpp * w * h];
		for (int i = 0; i < h; i++)
		{
			dst -= bpp * w;
			if ((int) fread(dst, 1, bpp * w, f) != bpp * w)
				goto error;
		}
		if (bpp >= 3)
			bgr2rgb(data, bpp * w * h, bpp);
	}
	else if (hdr.imagetype == 9)
	{
		int cmapsize = hdr.cmapsize[0] + (hdr.cmapsize[1] << 8);
		if (hdr.cmapentrysize != 8 || hdr.cmapentrysize != 24 || hdr.cmapentrysize != 32)
			goto error;
		bpp = hdr.cmapentrysize / 8;
		cmap = malloc(sizeof(uint8_t) * bpp * cmapsize);
		if ((int) fread(cmap, 1, bpp * cmapsize, f) != bpp * cmapsize)
			goto error;
		if (bpp >= 3)
			bgr2rgb(cmap, bpp * cmapsize, bpp);
		data = malloc(sizeof(uint8_t) * bpp * w * h);
		uint8_t buf[128];
		for (uint8_t *end = &data[bpp * w * h], *dst = end - bpp * w; dst >= data;)
		{
			int c = fgetc(f);
			if (c == EOF)
				goto error;
			if (c & 0x80)
			{
				int idx = fgetc(f);
				if (idx == EOF)
					goto error;
				const uint8_t *col = &cmap[idx * bpp];
				c -= 0x7F;
				c *= bpp;
				while (c > 0 && dst >= data)
				{
					int n = (int)(end - dst);
					n = c < n ? c : n;
					for (uint8_t *run = dst + n; dst < run; dst += bpp)
						memcpy(dst, col, bpp);
					c -= n;
					if (dst >= end)
					{
						end -= bpp * w;
						dst = end - bpp * w;
					}
				}
			}
			else
			{
				c += 1;
				while (c > 0 && dst >= data)
				{
					int n = ((int)(end - dst)) / bpp;
					n = c < n ? c : n;
					if ((int) fread(buf, 1, n, f) != n)
						goto error;
					for (uint8_t *src = buf; src < &buf[n]; dst += bpp)
						memcpy(dst, &cmap[*src++ * bpp], bpp);
					c -= n;
					if (dst >= end)
					{
						end -= bpp * w;
						dst = end - bpp * w;
					}
				}
			}
		}
	}
	else if (hdr.imagetype == 10)
	{
		data = malloc(sizeof(uint8_t) * bpp * w * h);
		uint8_t buf[4];
		for (uint8_t *end = &data[bpp * w * h], *dst = end - bpp * w; dst >= data;)
		{
			int c = fgetc(f);
			if (c == EOF)
				goto error;
			if (c & 0x80)
			{
				if ((int) fread(buf, 1, bpp, f) != bpp)
					goto error;
				c -= 0x7F;
				if (bpp >= 3)
				{
					uint8_t temp = buf[0];
					buf[0] = buf[2];
					buf[2] = temp;
				}
				c *= bpp;
				while (c > 0)
				{
					int n = (int)(end - dst);
					n = c < n ? c : n;
					for (uint8_t *run = dst + n; dst < run; dst += bpp)
						memcpy(dst, buf, bpp);
					c -= n;
					if (dst >= end)
					{
						end -= bpp * w;
						dst = end - bpp * w;
						if (dst < data)
							break;
					}
				}
			}
			else
			{
				c += 1;
				c *= bpp;
				while (c > 0)
				{
					int n = (int)(end - dst);
					n = c < n ? c : n;
					if ((int) fread(dst, 1, n, f) != n)
						goto error;
					if (bpp >= 3)
						bgr2rgb(dst, n, bpp);
					dst += n;
					c -= n;
					if (dst >= end)
					{
						end -= bpp * w;
						dst = end - bpp * w;
						if (dst < data)
							break;
					}
				}
			}
		}
	}
	else
		goto error;
		
	if (cmap)
		free(cmap);
	fclose(f);
	*width = w; *height = h; *bitsperpixel = bpp;
	return data;
	
error:
	if (data)
		free(data);
	if (cmap)
		free(cmap);
	fclose(f);
	return NULL;
}

static GLenum texformat(int bpp)
{
	switch (bpp)
	{
	case 8: return GL_LUMINANCE;
	case 16: return GL_LUMINANCE_ALPHA;
	case 24: return GL_RGB;
	case 32: return GL_RGBA;
	default: return 0;
	}
}

int formatsize(GLenum format)
{
	switch (format)
	{
	case GL_LUMINANCE:
	case GL_ALPHA: return 1;
	case GL_LUMINANCE_ALPHA: return 2;
	case GL_RGB: return 3;
	case GL_RGBA: return 4;
	default: return 4;
	}
}

void resizetexture(int w, int h, bool mipmap, GLenum target, int *twidth, int *theight)
{
	(void) target;
	GLint sizelimit = 4096;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &sizelimit);
	w = w < sizelimit ? w : sizelimit;
	h = h < sizelimit ? h : sizelimit;
	int tw = w, th = h;
	if (mipmap || w & (w - 1) || h & (h - 1))
	{
		tw = th = 1;
		while (tw < w)
			tw *= 2;
		while (th < h)
			th *= 2;
		if (w < tw - tw / 4)
			tw /= 2;
		if (h < th - th / 4)
			th /= 2;
	}
	*twidth = tw;
	*theight = th;
}

void uploadtexture(GLenum target, GLenum internal, int tw, int th, GLenum format, GLenum type, void *pixels, int pw, int ph, bool mipmap)
{
	int bpp = formatsize(format);
	uint8_t *buf = NULL;
	if (pw != tw || ph != th)
	{
		buf = malloc(sizeof(uint8_t) * tw * th * bpp);
		scaletexture((uint8_t *) pixels, pw, ph, bpp, pw * bpp, buf, tw, th);
	}
	for (int level = 0;; level++)
	{
		uint8_t *src = buf ? buf : (uint8_t *) pixels;
		if (target == GL_TEXTURE_1D)
			glTexImage1D(target, level, internal, tw, 0, format, type, src);
		else
			glTexImage2D(target, level, internal, tw, th, 0, format, type, src);
		if (!mipmap || (tw <= 1 && th <= 1))
			break;
		int srcw = tw, srch = th;
		if (tw > 1)
			tw /= 2;
		if (th > 1)
			th /= 2;
		if (!buf)
			buf = malloc(sizeof(uint8_t) * tw * th * bpp);
		scaletexture(src, srcw, srch, bpp, srcw * bpp, buf, tw, th);
	}
	if (buf)
		free(buf);
}

void createtexture(int tnum, int w, int h, void *pixels, int clamp, int filter, GLenum component)
{
	GLenum target = GL_TEXTURE_2D;
	int pw = 0, ph = 0;
	glBindTexture(target, tnum);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, clamp & 1 ? GL_CLAMP_TO_EDGE : GL_REPEAT);
	if (target != GL_TEXTURE_1D)
		glTexParameteri(target, GL_TEXTURE_WRAP_T, clamp & 2 ? GL_CLAMP_TO_EDGE : GL_REPEAT);
	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter > 1 ? GL_LINEAR_MIPMAP_LINEAR : (filter ? GL_LINEAR : GL_NEAREST));
	
	GLenum format = component, type = GL_UNSIGNED_BYTE;
	switch (component)
	{
	case GL_RGB5:
	case GL_RGB8:
	case GL_RGB16:
		format = GL_RGB;
		break;
		
	case GL_RGBA8:
	case GL_RGBA16:
		format = GL_RGBA;
		break;
	}
	
	if (!pw)
		pw = w;
	if (!ph)
		ph = h;
	int tw = w, th = h;
	bool mipmap = filter > 1;
	if (pixels)
		resizetexture(w, h, mipmap, target, &tw, &th);
	uploadtexture(target, component, tw, th, format, type, pixels, pw, ph, mipmap && pixels);
}

GLuint loadtexture(const char *name, int clamp)
{
	int w, h, b;
	uint8_t *data = loadtga(name, &w, &h, &b);
	if (!data)
	{
		printf("%s: failed loading\n", name);
		return 0;
	}
	GLenum format = texformat(b * 8);
	if (!format)
	{
		printf("%s: failed loading\n", name);
		free(data);
		return 0;
	}
	
	GLuint tex;
	glGenTextures(1, &tex);
	createtexture(tex, w, h, data, clamp, 2, format);
	
	free(data);
	return tex;
}

