#ifndef __IQM_H__
#define __IQM_H__

#define IQM_MAGIC "INTERQUAKEMODEL"
#define IQM_VERSION 2

typedef struct iqmheader_s
{
	uint8_t magic[16];
	uint32_t version;
	uint32_t filesize;
	uint32_t flags;
	uint32_t num_text, ofs_text;
	uint32_t num_meshes, ofs_meshes;
	uint32_t num_vertexarrays, num_vertexes, ofs_vertexarrays;
	uint32_t num_triangles, ofs_triangles, ofs_adjacency;
	uint32_t num_joints, ofs_joints;
	uint32_t num_poses, ofs_poses;
	uint32_t num_anims, ofs_anims;
	uint32_t num_frames, num_framechannels, ofs_frames, ofs_bounds;
	uint32_t num_comment, ofs_comment;
	uint32_t num_extensions, ofs_extensions;
} iqmheader;

typedef struct iqmmesh_s
{
	uint32_t name;
	uint32_t material;
	uint32_t first_vertex, num_vertexes;
	uint32_t first_triangle, num_triangles;
} iqmmesh;

typedef enum
{
	IQM_POSITION     = 0,
	IQM_TEXCOORD     = 1,
	IQM_NORMAL       = 2,
	IQM_TANGENT      = 3,
	IQM_BLENDINDEXES = 4,
	IQM_BLENDWEIGHTS = 5,
	IQM_COLOR        = 6,
	IQM_CUSTOM       = 0x10
} iqmvatype;

typedef enum
{
	IQM_BYTE   = 0,
	IQM_UBYTE  = 1,
	IQM_SHORT  = 2,
	IQM_USHORT = 3,
	IQM_INT    = 4,
	IQM_UINT   = 5,
	IQM_HALF   = 6,
	IQM_FLOAT  = 7,
	IQM_DOUBLE = 8,
} iqmvaformat;

typedef struct iqmtriangle_s
{
	uint32_t vertex[3];
} iqmtriangle;

typedef struct iqmadjacency_s
{
	uint32_t triangle[3];
} iqmadjacency;

typedef struct iqmjoint_s
{
	uint32_t name;
	int32_t parent;
	float translate[3], rotate[4], scale[3];
} iqmjoint;

typedef struct iqmpose_s
{
	int32_t parent;
	uint32_t mask;
	float channeloffset[10];
	float channelscale[10];
} iqmpose;

typedef struct iqmanim_s
{
	uint32_t name;
	uint32_t first_frame, num_frames;
	float framerate;
	uint32_t flags;
} iqmanim;

typedef enum
{
	IQM_LOOP = 1 << 0
} iqmanimflag;

typedef struct iqmvertexarray_s
{
	uint32_t type;
	uint32_t flags;
	uint32_t format;
	uint32_t size;
	uint32_t offset;
} iqmvertexarray;

typedef struct iqmbounds_s
{
	float bbmin[3], bbmax[3];
	float xyradius, radius;
} iqmbounds;

#endif

