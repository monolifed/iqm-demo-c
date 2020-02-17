static void FUNCNAME(halvetexture)(uint8_t *src, uint32_t sw, uint32_t sh, uint32_t stride, uint8_t *dst)
{
	for (uint8_t *yend = &src[sh * stride]; src < yend;)
	{
		for (uint8_t *xend = &src[sw * BPP], *xsrc = src; xsrc < xend; xsrc += 2 * BPP, dst += BPP)
		{
#define OP(c, n) dst[n] = ((uint32_t) xsrc[n] + (uint32_t) xsrc[n+BPP] + (uint32_t) xsrc[stride+n] + (uint32_t) xsrc[stride+n+BPP])>>2
			PIXELOP
#undef OP
		}
		src += 2 * stride;
	}
}

static void FUNCNAME(shifttexture)(uint8_t *src, uint32_t sw, uint32_t sh, uint32_t stride, uint8_t *dst, uint32_t dw, uint32_t dh)
{
	uint32_t wfrac = sw / dw, hfrac = sh / dh, wshift = 0, hshift = 0;
	while (dw << wshift < sw)
		wshift++;
	while (dh << hshift < sh)
		hshift++;
	uint32_t tshift = wshift + hshift;
	for (uint8_t *yend = &src[sh * stride]; src < yend;)
	{
		for (uint8_t *xend = &src[sw * BPP], *xsrc = src; xsrc < xend; xsrc += wfrac * BPP, dst += BPP)
		{
#define OP(c, n) c##t = 0
			DEFPIXEL
#undef OP
			for (uint8_t *ycur = xsrc, *xend = &ycur[wfrac * BPP], *yend = &src[hfrac * stride];
			    ycur < yend;
			    ycur += stride, xend += stride)
			{
				for (uint8_t *xcur = ycur; xcur < xend; xcur += BPP)
				{
#define OP(c, n) c##t += xcur[n]
					PIXELOP
#undef OP
				}
			}
#define OP(c, n) dst[n] = (c##t)>>tshift
			PIXELOP
#undef OP
		}
		src += hfrac * stride;
	}
}

static void FUNCNAME(scaletexture)(uint8_t *src, uint32_t sw, uint32_t sh, uint32_t stride, uint8_t *dst, uint32_t dw, uint32_t dh)
{
	uint32_t wfrac = (sw << 12) / dw, hfrac = (sh << 12) / dh, darea = dw * dh, sarea = sw * sh;
	int over, under;
	for (over = 0; (darea >> over) > sarea; over++);
	for (under = 0; (darea << under) < sarea; under++);
	uint32_t cscale = clampi(under, over - 12, 12);
	uint32_t ascale = clampi(12 + under - over, 0, 24);
	uint32_t dscale = ascale + 12 - cscale;
	uint32_t area = ((uint64_t)darea << ascale) / sarea;
	dw *= wfrac;
	dh *= hfrac;
	for (uint32_t y = 0; y < dh; y += hfrac)
	{
		const uint32_t yn = y + hfrac - 1, yi = y >> 12, h = (yn >> 12) - yi, ylow = ((yn | (-((int) h) >> 24)) & 0xFFFU) + 1 - (y & 0xFFFU), yhigh = (yn & 0xFFFU) + 1;
		const uint8_t *ysrc = &src[yi * stride];
		for (uint32_t x = 0; x < dw; x += wfrac, dst += BPP)
		{
			const uint32_t xn = x + wfrac - 1, xi = x >> 12, w = (xn >> 12) - xi, xlow = ((w + 0xFFFU) & 0x1000U) - (x & 0xFFFU), xhigh = (xn & 0xFFFU) + 1;
			const uint8_t *xsrc = &ysrc[xi * BPP], *xend = &xsrc[w * BPP];
#define OP(c, n) c##t = 0
			DEFPIXEL
#undef OP
			for (const uint8_t *xcur = &xsrc[BPP]; xcur < xend; xcur += BPP)
			{
#define OP(c, n) c##t += xcur[n]
				PIXELOP
#undef OP
			}
#define OP(c, n) c##t = (ylow*(c##t + ((xsrc[n]*xlow + xend[n]*xhigh)>>12)))>>cscale
			PIXELOP
#undef OP
			if (h)
			{
				xsrc += stride;
				xend += stride;
				for (uint32_t hcur = h; --hcur; xsrc += stride, xend += stride)
				{
#define OP(c, n) c = 0
					DEFPIXEL
#undef OP
					for (const uint8_t *xcur = &xsrc[BPP]; xcur < xend; xcur += BPP)
					{
#define OP(c, n) c += xcur[n]
						PIXELOP
#undef OP
					}
#define OP(c, n) c##t += ((c<<12) + xsrc[n]*xlow + xend[n]*xhigh)>>cscale
					PIXELOP
#undef OP
				}
#define OP(c, n) c = 0
				DEFPIXEL
#undef OP
				for (const uint8_t *xcur = &xsrc[BPP]; xcur < xend; xcur += BPP)
				{
#define OP(c, n) c += xcur[n]
					PIXELOP
#undef OP
				}
#define OP(c, n) c##t += (yhigh*(c + ((xsrc[n]*xlow + xend[n]*xhigh)>>12)))>>cscale
				PIXELOP
#undef OP
			}
#define OP(c, n) dst[n] = (c##t * area)>>dscale
			PIXELOP
#undef OP
		}
	}
}

#undef FUNCNAME
#undef DEFPIXEL
#undef PIXELOP
#undef BPP

