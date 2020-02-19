#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <stdbool.h>

//#include <glad/gl.h>
#define GLFW_INCLUDE_GLEXT
#include <GLFW/glfw3.h>

#include "iqm.h"
#include "common.h"
#include "stb_easy_font.h"

#define VSYNC 0
#define WINDOW_TITLE "IQM GPU Skinning Demo"

extern GLuint loadtexture(const char *name, int clamp);

int FPS = 0;
const char *modelname = "mrfixit.iqm";
char keyinfo[50000];
int keyinfo_numquads = 0;

static void print_keyinfo(float x,  float y)
{
	char textbuf[256];
	snprintf(textbuf, sizeof textbuf,
		"MODEL: %s\n"
		"---------------\n"
		"[A] TURN LEFT\n"
		"[D] TURN RIGHT\n"
		"[S] MOVE AWAY\n"
		"[W] MOVE CLOSER\n"
		"[Q] MOVE DOWN\n"
		"[E] MOVE UP\n",
		modelname);
	keyinfo_numquads = stb_easy_font_print(x, y, textbuf, NULL, keyinfo, sizeof keyinfo);
}

static void glprintf(float x, float y, char *format, ...)
{
	char textbuf[256];
	va_list args;
	va_start(args, format);
	vsnprintf(textbuf, sizeof textbuf, format, args);
	va_end(args);
	
	static char buffer[99999]; // ~500 chars
	int num_quads;

	num_quads = stb_easy_font_print(x, y, textbuf, NULL, buffer, sizeof buffer);
	glVertexPointer(2, GL_FLOAT, 16, buffer);
	glDrawArrays(GL_QUADS, 0, num_quads*4);
}

#define REGISTER_EXTENSIONS \
	REXT(PFNGLUSEPROGRAMPROC, glUseProgram, true) \
	REXT(PFNGLCREATEPROGRAMPROC, glCreateProgram, true) \
	REXT(PFNGLCREATESHADERPROC, glCreateShader, true) \
	REXT(PFNGLDELETEPROGRAMPROC, glDeleteProgram, true) \
	REXT(PFNGLDELETESHADERPROC, glDeleteShader, true) \
	REXT(PFNGLATTACHSHADERPROC, glAttachShader, true) \
	REXT(PFNGLBINDATTRIBLOCATIONPROC, glBindAttribLocation, true) \
	REXT(PFNGLCOMPILESHADERPROC, glCompileShader, true) \
	REXT(PFNGLLINKPROGRAMPROC, glLinkProgram, true) \
	REXT(PFNGLSHADERSOURCEPROC, glShaderSource, true) \
	REXT(PFNGLGETPROGRAMIVPROC, glGetProgramiv, true) \
	REXT(PFNGLGETSHADERIVPROC, glGetShaderiv, true) \
	REXT(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog, true) \
	REXT(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog, true) \
	REXT(PFNGLDISABLEVERTEXATTRIBARRAYPROC, glDisableVertexAttribArray, true) \
	REXT(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray, true) \
	REXT(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer, true) \
	REXT(PFNGLUNIFORMMATRIX3X4FVPROC, glUniformMatrix3x4fv, true) \
	REXT(PFNGLUNIFORM1IPROC, glUniform1i, true) \
	REXT(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation, true) \
	REXT(PFNGLBINDBUFFERPROC, glBindBuffer, true) \
	REXT(PFNGLDELETEBUFFERSPROC, glDeleteBuffers, true) \
	REXT(PFNGLGENBUFFERSPROC, glGenBuffers, true) \
	REXT(PFNGLBUFFERDATAPROC, glBufferData, true) \
	REXT(PFNGLBUFFERSUBDATAPROC, glBufferSubData, true) \
	REXT(PFNGLGETUNIFORMINDICESPROC, glGetUniformIndices, hasUBO) \
	REXT(PFNGLGETACTIVEUNIFORMSIVPROC, glGetActiveUniformsiv, hasUBO) \
	REXT(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex, hasUBO) \
	REXT(PFNGLGETACTIVEUNIFORMBLOCKIVPROC, glGetActiveUniformBlockiv, hasUBO) \
	REXT(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding, hasUBO) \
	REXT(PFNGLBINDBUFFERBASEPROC, glBindBufferBase, hasUBO) \
	REXT(PFNGLBINDBUFFERRANGEPROC, glBindBufferRange, hasUBO)

#define REXT(type, name, required) type name##_ = NULL;
REGISTER_EXTENSIONS
#undef REXT

GLFWglproc requireext(const char *procname)
{
	GLFWglproc fn = glfwGetProcAddress(procname);
	if (!fn)
	{
		fprintf(stderr, "failed getting proc address: %s", procname);
		exit(EXIT_FAILURE);
	}
	return fn;
}

bool hasUBO = false;

typedef struct binding_s
{
	const char *name;
	GLint index;
} binding;

typedef struct shader_s
{
	const char *name, *vsstr, *psstr;
	const binding *attribs, *texs;
	GLuint vs, ps, program, vsobj, psobj;
} shader;

static const binding gpuskinattribs[] = {{"vtangent", 1}, {"vweights", 6}, {"vbones", 7}, {NULL, -1}};
static const binding gpuskintexs[] = {{"tex", 0}, {NULL, -1}};
static shader gpuskin =
{
	.name = "gpu skin",
	
	.vsstr =
	"#version 120\n"
	"#ifdef GL_ARB_uniform_buffer_object\n"
	"  #extension GL_ARB_uniform_buffer_object : enable\n"
	"  layout(std140) uniform animdata\n"
	"  {\n"
	"     uniform mat3x4 bonemats[80];\n"
	"  };\n"
	"#else\n"
	"  uniform mat3x4 bonemats[80];\n"
	"#endif\n"
	"attribute vec4 vweights;\n"
	"attribute vec4 vbones;\n"
	"attribute vec4 vtangent;\n"
	"void main(void)\n"
	"{\n"
	"   mat3x4 m = bonemats[int(vbones.x)] * vweights.x;\n"
	"   m += bonemats[int(vbones.y)] * vweights.y;\n"
	"   m += bonemats[int(vbones.z)] * vweights.z;\n"
	"   m += bonemats[int(vbones.w)] * vweights.w;\n"
	"   vec4 mpos = vec4(gl_Vertex * m, gl_Vertex.w);\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * mpos;\n"
	"   gl_TexCoord[0] = gl_MultiTexCoord0;\n"
	"   mat3 madjtrans = mat3(cross(m[1].xyz, m[2].xyz), cross(m[2].xyz, m[0].xyz), cross(m[0].xyz, m[1].xyz));\n"
	"   vec3 mnormal = gl_Normal * madjtrans;\n"
	"   vec3 mtangent = vtangent.xyz * madjtrans; // tangent not used, just here as an example\n"
	"   vec3 mbitangent = cross(mnormal, mtangent) * vtangent.w; // bitangent not used, just here as an example\n"
	"   gl_FrontColor = gl_Color * (clamp(dot(normalize(gl_NormalMatrix * mnormal), gl_LightSource[0].position.xyz), 0.0, 1.0) * gl_LightSource[0].diffuse + gl_LightSource[0].ambient);\n"
	"}\n",
	
	.psstr =
	"uniform sampler2D tex;\n"
	"void main(void)\n"
	"{\n"
	"   gl_FragColor = gl_Color * texture2D(tex, gl_TexCoord[0].xy);\n"
	"}\n",
	
	.attribs = gpuskinattribs,
	.texs = gpuskintexs,
	.vs = 0, .ps = 0, .program = 0, .vsobj = 0, .psobj = 0
};

static const binding noskinattribs[] = {{"vtangent", 1}, {NULL, -1}};
static const binding noskintexs[] = {{"tex", 0}, {NULL, -1}};
static shader noskin =
{
	.name = "no skin",
	
	.vsstr =
	"attribute vec4 vtangent;\n"
	"void main(void)\n"
	"{\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
	"   gl_TexCoord[0] = gl_MultiTexCoord0;\n"
	"   vec3 vbitangent = cross(gl_Normal, vtangent.xyz) * vtangent.w; // bitangent not used, just here as an example\n"
	"   gl_FrontColor = gl_Color * (clamp(dot(normalize(gl_NormalMatrix * gl_Normal), gl_LightSource[0].position.xyz), 0.0, 1.0) * gl_LightSource[0].diffuse + gl_LightSource[0].ambient);\n"
	"}\n",
	
	.psstr =
	"uniform sampler2D tex;\n"
	"void main(void)\n"
	"{\n"
	"   gl_FragColor = gl_Color * texture2D(tex, gl_TexCoord[0].xy);\n"
	"}\n",
	
	.attribs = noskinattribs,
	.texs = noskintexs,
	.vs = 0, .ps = 0, .program = 0, .vsobj = 0, .psobj = 0
};

static void showinfo(GLuint obj, const char *tname, const char *name)
{
	GLint length = 0;
	if (!strcmp(tname, "PROG"))
		glGetProgramiv_(obj, GL_INFO_LOG_LENGTH, &length);
	else
		glGetShaderiv_(obj, GL_INFO_LOG_LENGTH, &length);
	if (length > 1)
	{
		GLchar *log = malloc(sizeof(GLchar) * length);
		if (!strcmp(tname, "PROG"))
			glGetProgramInfoLog_(obj, length, &length, log);
		else
			glGetShaderInfoLog_(obj, length, &length, log);
		printf("GLSL ERROR (%s:%s)\n", tname, name);
		puts(log);
		free(log);
	}
}

static void compshader(GLenum type, GLuint *obj, const char *def, const char *tname, const char *name)
{
	const GLchar *source = (const GLchar *)(def + strspn(def, " \t\r\n"));
	*obj = glCreateShader_(type);
	glShaderSource_(*obj, 1, &source, NULL);
	glCompileShader_(*obj);
	GLint success;
	glGetShaderiv_(*obj, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		showinfo(*obj, tname, name);
		glDeleteShader_(*obj);
		*obj = 0;
		fprintf(stderr, "error compiling shader\n");
		exit(EXIT_FAILURE);
	}
}

static void shader_link(shader *s)
{
	s->program = s->vsobj && s->psobj ? glCreateProgram_() : 0;
	GLint success = 0;
	if (s->program)
	{
		glAttachShader_(s->program, s->vsobj);
		glAttachShader_(s->program, s->psobj);
		
		if (s->attribs)
			for (const binding *a = s->attribs; a->name; a++)
				glBindAttribLocation_(s->program, a->index, a->name);
				
		glLinkProgram_(s->program);
		glGetProgramiv_(s->program, GL_LINK_STATUS, &success);
	}
	if (!success)
	{
		if (s->program)
		{
			showinfo(s->program, "PROG", s->name);
			glDeleteProgram_(s->program);
			s->program = 0;
		}
		fprintf(stderr, "error linking shader\n");
		exit(EXIT_FAILURE);
	}
}

static void shader_compile(shader *s)
{
	if (!s->vsstr || !s->psstr)
		return;
	compshader(GL_VERTEX_SHADER,   &s->vsobj, s->vsstr, "VS", s->name);
	compshader(GL_FRAGMENT_SHADER, &s->psobj, s->psstr, "PS", s->name);
	shader_link(s);
}

static GLint shader_getparam(shader *s, const char *pname)
{
	return glGetUniformLocation_(s->program, pname);
}

static void shader_set(shader *s)
{
	glUseProgram_(s->program);
	if (!s->texs)
		return;
	GLint loc;
	for (const binding *t = s->texs; t->name; t++)
	{
		loc = shader_getparam(s, t->name);
		if (loc != -1)
			glUniform1i_(loc, t->index);
	}
}

static void loadexts()
{
	hasUBO = glfwExtensionSupported("GL_ARB_uniform_buffer_object") == GLFW_TRUE;
	
#define REXT(type, name, required) \
	if (required) { name##_ = (type) requireext(#name); }
	REGISTER_EXTENSIONS
#undef REXT
}

typedef struct vertex_s
{
	GLfloat position[3];
	GLfloat normal[3];
	GLfloat tangent[4];
	GLfloat texcoord[2];
	GLubyte blendindex[4];
	GLubyte blendweight[4];
} vertex;



// Note that while this demo stores pointer directly into mesh data in a buffer
// of the entire IQM file's data, it is recommended that you copy the data and
// convert it into a more suitable internal representation for whichever 3D
// engine you use.
uint8_t *meshdata = NULL, *animdata = NULL;
int nummeshes = 0, numtris = 0, numverts = 0, numjoints = 0, numframes = 0, numanims = 0;
iqmtriangle *tris = NULL, *adjacency = NULL;
iqmmesh *meshes = NULL;
GLuint *textures = NULL;
iqmjoint *joints = NULL;
iqmpose *poses = NULL;
iqmanim *anims = NULL;
iqmbounds *bounds = NULL;
mat3x4 *baseframe = NULL, *inversebaseframe = NULL, *outframe = NULL, *frames = NULL;

GLuint notexture = 0;
uint8_t *incolor = NULL;

GLuint ebo = 0, vbo = 0, ubo = 0;
GLint ubosize = 0, bonematsoffset = 0;

void cleanupiqm()
{
	if (textures)
	{
		glDeleteTextures(nummeshes, textures);
		free(textures);
	}
	if (notexture)
		glDeleteTextures(1, &notexture);
	free(baseframe);
	free(inversebaseframe);
	free(outframe);
	free(frames);
	
	if (ebo)
		glDeleteBuffers_(1, &ebo);
	if (vbo)
		glDeleteBuffers_(1, &vbo);
	if (ubo)
		glDeleteBuffers_(1, &ubo);
}

bool loadiqmmeshes(const char *filename, const iqmheader *hdr, uint8_t *buf)
{
	if (meshdata)
		return false;
		
	leswap32(buf + hdr->ofs_vertexarrays, hdr->num_vertexarrays * sizeof(iqmvertexarray));
	leswap32(buf + hdr->ofs_triangles, hdr->num_triangles * sizeof(iqmtriangle));
	leswap32(buf + hdr->ofs_meshes, hdr->num_meshes * sizeof(iqmmesh));
	leswap32(buf + hdr->ofs_joints, hdr->num_joints * sizeof(iqmjoint));
	if (hdr->ofs_adjacency)
		leswap32(buf + hdr->ofs_adjacency, hdr->num_triangles * sizeof(iqmtriangle));
		
	meshdata = buf;
	nummeshes = hdr->num_meshes;
	numtris = hdr->num_triangles;
	numverts = hdr->num_vertexes;
	numjoints = hdr->num_joints;
	outframe = malloc(sizeof(mat3x4) * hdr->num_joints);
	textures = malloc(sizeof(GLuint) * nummeshes);
	memset(textures, 0, nummeshes * sizeof(GLuint));
	
	float *inposition = NULL, *innormal = NULL, *intangent = NULL, *intexcoord = NULL;
	uint8_t *inblendindex = NULL, *inblendweight = NULL;
	const char *str = hdr->ofs_text ? (char *) &buf[hdr->ofs_text] : "";
	iqmvertexarray *vas = (iqmvertexarray *) &buf[hdr->ofs_vertexarrays];
	for (int i = 0; i < (int) hdr->num_vertexarrays; i++)
	{
		iqmvertexarray *va = vas + i;
		switch (va->type)
		{
		case IQM_POSITION:
			if (va->format != IQM_FLOAT || va->size != 3)
				return false;
			leswap32(buf + va->offset, 3 * hdr->num_vertexes * sizeof(float));
			inposition = (float *) &buf[va->offset]; break;
		case IQM_NORMAL:
			if (va->format != IQM_FLOAT || va->size != 3)
				return false;
			leswap32(buf + va->offset, 3 * hdr->num_vertexes * sizeof(float));
			innormal = (float *) &buf[va->offset]; break;
		case IQM_TANGENT:
			if (va->format != IQM_FLOAT || va->size != 4)
				return false;
			leswap32(buf + va->offset, 4 * hdr->num_vertexes * sizeof(float));
			intangent = (float *) &buf[va->offset]; break;
		case IQM_TEXCOORD:
			if (va->format != IQM_FLOAT || va->size != 2)
				return false;
			leswap32(buf + va->offset, 2 * hdr->num_vertexes * sizeof(float));
			intexcoord = (float *) &buf[va->offset]; break;
		case IQM_BLENDINDEXES:
			if (va->format != IQM_UBYTE || va->size != 4)
				return false;
			inblendindex = (uint8_t *) &buf[va->offset]; break;
		case IQM_BLENDWEIGHTS:
			if (va->format != IQM_UBYTE || va->size != 4)
				return false;
			inblendweight = (uint8_t *) &buf[va->offset]; break;
		case IQM_COLOR:
			if (va->format != IQM_UBYTE || va->size != 4)
				return false;
			incolor = (uint8_t *) &buf[va->offset]; break;
		}
	}
	tris = (iqmtriangle *) &buf[hdr->ofs_triangles];
	meshes = (iqmmesh *) &buf[hdr->ofs_meshes];
	joints = (iqmjoint *) &buf[hdr->ofs_joints];
	if (hdr->ofs_adjacency)
		adjacency = (iqmtriangle *) &buf[hdr->ofs_adjacency];
		
	baseframe = malloc(sizeof(mat3x4) * hdr->num_joints);
	inversebaseframe = malloc(sizeof(mat3x4) * hdr->num_joints);
	for (int i = 0; i < (int) hdr->num_joints; i++)
	{
		iqmjoint *j = joints + i;
		mat3x4_from_rst(baseframe[i], j->rotate, j->translate, j->scale);
		mat3x4_invert(inversebaseframe[i], baseframe[i]);
		if (j->parent >= 0)
		{
			//tempmat = baseframe[i];
			mat3x4_mul(baseframe[i], baseframe[j->parent], baseframe[i]);
			mat3x4_mul(inversebaseframe[i], inversebaseframe[i], inversebaseframe[j->parent]);
		}
	}
	
	for (int i = 0; i < (int) hdr->num_meshes; i++)
	{
		iqmmesh *m = meshes + i;
		printf("%s: loaded mesh: %s\n", filename, &str[m->name]);
		textures[i] = loadtexture(&str[m->material], 0);
		if (textures[i])
			printf("%s: loaded material: %s\n", filename, &str[m->material]);
	}
	
	if (!ebo)
		glGenBuffers_(1, &ebo);
	glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData_(GL_ELEMENT_ARRAY_BUFFER, hdr->num_triangles * sizeof(iqmtriangle), tris, GL_STATIC_DRAW);
	glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	vertex *verts = malloc(sizeof(vertex) * hdr->num_vertexes);
	memset(verts, 0, hdr->num_vertexes * sizeof(vertex));
	for (int i = 0; i < (int) hdr->num_vertexes; i++)
	{
		vertex *v = verts + i;
		if (inposition)
			memcpy(v->position, &inposition[i * 3], sizeof(v->position));
		if (innormal)
			memcpy(v->normal, &innormal[i * 3], sizeof(v->normal));
		if (intangent)
			memcpy(v->tangent, &intangent[i * 4], sizeof(v->tangent));
		if (intexcoord)
			memcpy(v->texcoord, &intexcoord[i * 2], sizeof(v->texcoord));
		if (inblendindex)
			memcpy(v->blendindex, &inblendindex[i * 4], sizeof(v->blendindex));
		if (inblendweight)
			memcpy(v->blendweight, &inblendweight[i * 4], sizeof(v->blendweight));
	}
	
	if (!vbo)
		glGenBuffers_(1, &vbo);
	glBindBuffer_(GL_ARRAY_BUFFER, vbo);
	glBufferData_(GL_ARRAY_BUFFER, hdr->num_vertexes * sizeof(vertex), verts, GL_STATIC_DRAW);
	glBindBuffer_(GL_ARRAY_BUFFER, 0);
	free(verts);
	
	return true;
}

bool loadiqmanims(const char *filename, const iqmheader *hdr, uint8_t *buf)
{
	if ((int) hdr->num_poses != numjoints)
		return false;
		
	if (animdata)
	{
		if (animdata != meshdata)
			free(animdata);
		free(frames);
		animdata = NULL;
		anims = NULL;
		frames = 0;
		numframes = 0;
		numanims = 0;
	}
	
	leswap32(buf + hdr->ofs_poses, hdr->num_poses * sizeof(iqmpose));
	leswap32(buf + hdr->ofs_anims, hdr->num_anims * sizeof(iqmanim));
	leswap16(buf + hdr->ofs_frames, hdr->num_frames * hdr->num_framechannels * sizeof(uint16_t));
	if (hdr->ofs_bounds)
		leswap32(buf + hdr->ofs_bounds, hdr->num_frames * sizeof(iqmbounds));
		
	animdata = buf;
	numanims = hdr->num_anims;
	numframes = hdr->num_frames;
	
	const char *str = hdr->ofs_text ? (char *) &buf[hdr->ofs_text] : "";
	anims = (iqmanim *) &buf[hdr->ofs_anims];
	poses = (iqmpose *) &buf[hdr->ofs_poses];
	frames = malloc(sizeof(mat3x4) * hdr->num_frames * hdr->num_poses);
	uint16_t *framedata = (uint16_t *) &buf[hdr->ofs_frames];
	if (hdr->ofs_bounds)
		bounds = (iqmbounds *) &buf[hdr->ofs_bounds];
	
	mat3x4 mat;
	float rotate[4], translate[3], scale[3];
	for (int i = 0; i < (int) hdr->num_frames; i++)
	{
		for (int j = 0; j < (int) hdr->num_poses; j++)
		{
			iqmpose *p = poses + j;
			translate[0] = p->channeloffset[0]; if (p->mask &  0x01) translate[0] += *framedata++ * p->channelscale[0];
			translate[1] = p->channeloffset[1]; if (p->mask &  0x02) translate[1] += *framedata++ * p->channelscale[1];
			translate[2] = p->channeloffset[2]; if (p->mask &  0x04) translate[2] += *framedata++ * p->channelscale[2];
			rotate[0]    = p->channeloffset[3]; if (p->mask &  0x08) rotate[0]    += *framedata++ * p->channelscale[3];
			rotate[1]    = p->channeloffset[4]; if (p->mask &  0x10) rotate[1]    += *framedata++ * p->channelscale[4];
			rotate[2]    = p->channeloffset[5]; if (p->mask &  0x20) rotate[2]    += *framedata++ * p->channelscale[5];
			rotate[3]    = p->channeloffset[6]; if (p->mask &  0x40) rotate[3]    += *framedata++ * p->channelscale[6];
			scale[0]     = p->channeloffset[7]; if (p->mask &  0x80) scale[0]     += *framedata++ * p->channelscale[7];
			scale[1]     = p->channeloffset[8]; if (p->mask & 0x100) scale[1]     += *framedata++ * p->channelscale[8];
			scale[2]     = p->channeloffset[9]; if (p->mask & 0x200) scale[2]     += *framedata++ * p->channelscale[9];
			// Concatenate each pose with the inverse base pose to avoid doing this at animation time.
			// If the joint has a parent, then it needs to be pre-concatenated with its parent's base pose.
			// Thus it all negates at animation time like so:
			//   (parentPose * parentInverseBasePose) * (parentBasePose * childPose * childInverseBasePose) =>
			//   parentPose * (parentInverseBasePose * parentBasePose) * childPose * childInverseBasePose =>
			//   parentPose * childPose * childInverseBasePose
			mat3x4_from_rst(mat, rotate, translate, scale);
			if (p->parent >= 0)
			{
				mat3x4_mul(mat, mat, inversebaseframe[j]);
				mat3x4_mul(frames[i * hdr->num_poses + j], baseframe[p->parent], mat);
			}
			else
			{
				mat3x4_mul(frames[i * hdr->num_poses + j], mat, inversebaseframe[j]);
			}
		}
	}
	
	for (int i = 0; i < (int) hdr->num_anims; i++)
	{
		//iqmanim *a = anims + i;
		printf("%s: loaded anim: %s\n", filename, &str[anims[i].name]);
	}
	
	return true;
}

bool loadiqm(const char *filename)
{
	FILE *f = fopen(filename, "rb");
	if (!f)
		return false;
		
	uint8_t *buf = NULL;
	iqmheader hdr;
	if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr) || memcmp(hdr.magic, IQM_MAGIC, sizeof(hdr.magic)))
		goto error;
	leswap32((uint8_t *) &hdr.version, sizeof(hdr) - sizeof(hdr.magic));
	if (hdr.version != IQM_VERSION)
		goto error;
	if (hdr.filesize > (16 << 20))
		goto error; // sanity check... don't load files bigger than 16 MB
	buf = malloc(sizeof(uint8_t) * hdr.filesize);
	if (fread(buf + sizeof(hdr), 1, hdr.filesize - sizeof(hdr), f) != hdr.filesize - sizeof(hdr))
		goto error;
		
	if (hdr.num_meshes > 0 && !loadiqmmeshes(filename, &hdr, buf))
		goto error;
	if (hdr.num_anims > 0 && !loadiqmanims(filename, &hdr, buf))
		goto error;
		
	fclose(f);
	return true;
	
error:
	printf("%s: error while loading\n", filename);
	if (buf != meshdata && buf != animdata)
		free(buf);
	fclose(f);
	return false;
}

// Note that this animates all attributes (position, normal, tangent, bitangent)
// for expository purposes, even though this demo does not use all of them for rendering.
void animateiqm(float curframe)
{
	if (numframes <= 0)
		return;
	
	int frame1 = (int) floor(curframe),
	    frame2 = frame1 + 1;
	float frameoffset = curframe - frame1;
	frame1 %= numframes;
	frame2 %= numframes;
	mat3x4 *mat1 = &frames[frame1 * numjoints],
	        *mat2 = &frames[frame2 * numjoints];
	// Interpolate matrixes between the two closest frames and concatenate with parent matrix if necessary.
	// Concatenate the result with the inverse of the base pose.
	// You would normally do animation blending and inter-frame blending here in a 3D engine.
	mat3x4 mat;
	for (int i = 0; i < numjoints; i++)
	{
		for (int k = 0; k < 3; k++)
			for (int l = 0; l < 4; l++)
				mat[k][l] = mat1[i][k][l] * (1 - frameoffset) + mat2[i][k][l] * frameoffset;
		if (joints[i].parent >= 0)
			mat3x4_mul(outframe[i], outframe[joints[i].parent], mat);
		else
			memcpy(outframe[i], mat, sizeof(mat3x4));
	}
}


float scale = 1, rotate = 0;
vec3 translate = {0, 0, 0};

void renderiqm()
{
	static const GLfloat zero[4] = {0, 0, 0, 0};
	static const GLfloat one[4] = {1, 1, 1, 1};
	static const GLfloat ambientcol[4] = {0.5f, 0.5f, 0.5f, 1};
	static const GLfloat diffusecol[4] = {0.5f, 0.5f, 0.5f, 1};
	       const GLfloat lightdir[4] = {cosf(-M_PI / 3), 0, sinf(-M_PI / 3), 0};
	
	glPushMatrix();
	glTranslatef(translate[0] * scale, translate[1] * scale, translate[2] * scale);
	glRotatef(rotate, 0, 0, -1);
	glScalef(scale, scale, scale);
	
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, zero);
	glMaterialfv(GL_FRONT, GL_SPECULAR, zero);
	glMaterialfv(GL_FRONT, GL_EMISSION, zero);
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, one);
	glLightfv(GL_LIGHT0, GL_SPECULAR, zero);
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientcol);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffusecol);
	glLightfv(GL_LIGHT0, GL_POSITION, lightdir);
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
	
	glColor3f(1, 1, 1);
	
	if (numframes > 0)
	{
		shader_set(&gpuskin);
		
		if (hasUBO)
		{
			glBindBuffer_(GL_UNIFORM_BUFFER, ubo);
			glBufferData_(GL_UNIFORM_BUFFER, ubosize, NULL, GL_STREAM_DRAW);
			glBufferSubData_(GL_UNIFORM_BUFFER, bonematsoffset, numjoints * sizeof(mat3x4), outframe[0][0]);
			glBindBuffer_(GL_UNIFORM_BUFFER, 0);
			
			glBindBufferBase_(GL_UNIFORM_BUFFER, 0, ubo);
		}
		else
			glUniformMatrix3x4fv_(shader_getparam(&gpuskin, "bonemats"), numjoints, GL_FALSE, outframe[0][0]);
	}
	else
		shader_set(&noskin);
		
	glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBindBuffer_(GL_ARRAY_BUFFER, vbo);
	
	vertex *vert = NULL;
	glVertexPointer(3, GL_FLOAT, sizeof(vertex), &vert->position);
	glNormalPointer(GL_FLOAT, sizeof(vertex), &vert->normal);
	glTexCoordPointer(2, GL_FLOAT, sizeof(vertex), &vert->texcoord);
	glVertexAttribPointer_(1, 4, GL_FLOAT, GL_FALSE, sizeof(vertex), &vert->tangent);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	
	if (incolor)
	{
		glColorPointer(4, GL_UNSIGNED_BYTE, 0, incolor);
		
		glEnableClientState(GL_COLOR_ARRAY);
	}

	glEnableVertexAttribArray_(1);
	if (numframes > 0)
	{
		glVertexAttribPointer_(6, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(vertex), &vert->blendweight);
		glVertexAttribPointer_(7, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(vertex), &vert->blendindex);
		glEnableVertexAttribArray_(6);
		glEnableVertexAttribArray_(7);
	}
	
	glEnable(GL_TEXTURE_2D);
	iqmtriangle *tindex = NULL;
	for (int i = 0; i < nummeshes; i++)
	{
		iqmmesh *m = meshes + i;
		glBindTexture(GL_TEXTURE_2D, textures[i] ? textures[i] : notexture);
		glDrawElements(GL_TRIANGLES, 3 * m->num_triangles, GL_UNSIGNED_INT, &tindex[m->first_triangle]);
	}
	
	glDisable(GL_TEXTURE_2D);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableVertexAttribArray_(1);
	if (numframes > 0)
	{
		glDisableVertexAttribArray_(6);
		glDisableVertexAttribArray_(7);
	}
	glBindBuffer_(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer_(GL_ARRAY_BUFFER, 0);
	
	if (incolor)
		glDisableClientState(GL_COLOR_ARRAY);
		
	glDisable(GL_NORMALIZE);
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);
	
	glPopMatrix();
}

void initgl()
{
	glClearColor(0.2, 0.2, 0.2, 0);
	glClearDepth(1);
	glDisable(GL_FOG);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	notexture = loadtexture("notexture.tga", 0);
	
	shader_compile(&gpuskin);
	
	if (hasUBO)
	{
		GLuint blockidx = glGetUniformBlockIndex_(gpuskin.program, "animdata"), bonematsidx;
		const GLchar *bonematsname = "bonemats";
		glGetUniformIndices_(gpuskin.program, 1, &bonematsname, &bonematsidx);
		glGetActiveUniformBlockiv_(gpuskin.program, blockidx, GL_UNIFORM_BLOCK_DATA_SIZE, &ubosize);
		glGetActiveUniformsiv_(gpuskin.program, 1, &bonematsidx, GL_UNIFORM_OFFSET, &bonematsoffset);
		glUniformBlockBinding_(gpuskin.program, blockidx, 0);
		if (!ubo)
			glGenBuffers_(1, &ubo);
	}
	
	shader_compile(&noskin);
	
}

int scrw = 0, scrh = 0;

void reshapefunc(int w, int h)
{
	scrw = w;
	scrh = h;
	glViewport(0, 0, w, h);
}

float camyaw = -90, campitch = 0, camroll = 0;
vec3 campos = {10, 0, 5};

void setupcamera()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	GLdouble aspect = ((double) scrw) / scrh,
	         fov = M_PI / 2,
	         fovy = 2 * atan2(tan(fov / 2), aspect),
	         nearplane = 1e-2f, farplane = 1000,
	         ydist = nearplane * tan(fovy / 2), xdist = ydist * aspect;
	glFrustum(-xdist, xdist, -ydist, ydist, nearplane, farplane);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glRotatef(camroll, 0, 0, 1);
	glRotatef(campitch, -1, 0, 0);
	glRotatef(camyaw, 0, 1, 0);
	glRotatef(-90, 1, 0, 0);
	glScalef(1, -1, 1);
	glTranslatef(-campos[0], -campos[1], -campos[2]);
}

float animate = 0;

void timerfunc(float dt)
{
	animate += dt;
}

void drawui()
{
	shader_set(&noskin);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, scrw, scrh, 0, -100, 100);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glColor4f(1, 1, 1, 1);
	glEnableClientState(GL_VERTEX_ARRAY);
	
	glBindTexture(GL_TEXTURE_2D, notexture);
	glprintf(5, 5, "FPS: %i", FPS);
	glVertexPointer(2, GL_FLOAT, 16, keyinfo);
	glDrawArrays(GL_QUADS, 0, keyinfo_numquads * 4);
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

void displayfunc(GLFWwindow *window)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	setupcamera();
	
	animateiqm(animate);
	renderiqm();
	drawui();

#if VSYNC == 0
	(void) window;
	glFlush();
#else
	glfwSwapBuffers(window);
#endif
}

static void keycallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	(void) scancode; (void) mods;
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
		return;
	}
	if (action == GLFW_REPEAT)
	{
		switch(key)
		{
		case GLFW_KEY_A: rotate -= 5; break;
		case GLFW_KEY_D: rotate += 5; break;
		case GLFW_KEY_S: translate[0] -= .1; break;
		case GLFW_KEY_W: translate[0] += .1; break;
		case GLFW_KEY_Q: translate[2] -= .1; break;
		case GLFW_KEY_E: translate[2] += .1; break;
		}
	}
}


static void errcallback(int errid, const char *errtext)
{
	(void) errid;
	fprintf(stderr, "Error: %s\n", errtext);
}

int main(int argc, char **argv)
{
	(void) argc; (void) argv;
	if (!glfwInit())
		return EXIT_FAILURE;
	glfwSetErrorCallback(errcallback);
	
#if VSYNC == 0
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
#else
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
#endif
	//glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	GLFWwindow *window = glfwCreateWindow(640, 480, WINDOW_TITLE, NULL, NULL);
	
	if (!window)
	{
		glfwTerminate();
		return EXIT_FAILURE;
	}
	
	glfwSetKeyCallback(window, keycallback);
	glfwMakeContextCurrent(window);
	//gladLoadGL(glfwGetProcAddress);
	glfwSwapInterval(1);
	
	loadexts();
	
	atexit(cleanupiqm);
	if (!meshdata && !loadiqm(modelname))
		return EXIT_FAILURE;
	
	print_keyinfo(5, 25);
	initgl();
	
	int framecount = 0;
	double curtime = 0, prevtime = 0, dt = 0;
	while (!glfwWindowShouldClose(window))
	{
		dt = curtime;
		curtime = glfwGetTime();
		dt = curtime - dt;
		framecount++;
		if (curtime - prevtime >= 1.0)
		{
			FPS = framecount;
			framecount = 0;
			prevtime = curtime;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		timerfunc(dt * 24);
		reshapefunc(width, height);
		displayfunc(window);
		glfwPollEvents();
	}
	
	glfwDestroyWindow(window);
	glfwTerminate();
	
	return EXIT_SUCCESS;
}

