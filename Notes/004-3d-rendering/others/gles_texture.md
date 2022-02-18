# GLES Texture

# 概念
GLES的Texture有几个概念：

1. texture unit，应用显卡中的硬件单元，用于对纹理进入采样和写入
2. texture name， 一个整数，代码一个context内的唯一的纹理。用glGenTextures获得。
3. texture object, 与name一一对应的。用glBindTexture获得
4. texture image, 一个texture object中可以有多个image，一个image 代码2D纹理的一个mipmap level，或是Cube Map纹理中一个面的一个mipmap level。

# texture unit

texture unit通过glActiveTexture激活，成为当前上下文的默认texture:


```c
/**
 * Sampler-related subset of a texture unit, like current texture objects.
 */
struct gl_texture_unit
{
   GLfloat LodBias;		/**< for biasing mipmap levels */

   /** Texture targets that have a non-default texture bound */
   GLbitfield _BoundTextures;

   /** Current sampler object (GL_ARB_sampler_objects) */
   struct gl_sampler_object *Sampler;

   /** Current texture object pointers */
   struct gl_texture_object *CurrentTex[NUM_TEXTURE_TARGETS];

   /** Points to highest priority, complete and enabled texture object */
   struct gl_texture_object *_Current;
};

struct gl_texture_unit Unit[MAX_COMBINED_TEXTURE_IMAGE_UNITS];
struct gl_fixedfunc_texture_unit FixedFuncUnit[MAX_TEXTURE_COORD_UNITS];

```

一个上下文中有MAX_COMBINED_TEXTURE_IMAGE_UNITS个texture_unit。这是一个显卡常量。

# texture name

入口在_mesa_GenTextures:

```c
   first = _mesa_HashFindFreeKeyBlock(ctx->Shared->TexObjects, n);

   /* Allocate new, empty texture objects */
   for (i = 0; i < n; i++) {
      struct gl_texture_object *texObj;
      GLuint name = first + i;
      texObj = ctx->Driver.NewTextureObject(ctx, name, target);
      if (!texObj) {
         _mesa_HashUnlockMutex(ctx->Shared->TexObjects);
         _mesa_error(ctx, GL_OUT_OF_MEMORY, "%s", caller);
         return;
      }

      /* insert into hash table */
      _mesa_HashInsertLocked(ctx->Shared->TexObjects, texObj->Name, texObj);

      textures[i] = name;
   }
```

分配一个数字，然后为这个数字分配了一个空的gl_texture_object。这个mesa的实现。
在swiftshader的实现中，仅分配了一个数字。

# texture object

入口在_mesa_BindTexture:

在gl_texture_unit中有几个gl_texture_object指针，在glBindTexture调用中，一次调用，就是为其中一个指针赋值。
绑定的过程中会对旧的texture_object减少引用计数，对新的增加引用计数。旧的如果引用计数为0就会被释放。

```c
   /* If the refcount on the previously bound texture is decremented to
    * zero, it'll be deleted here.
    */
   _mesa_reference_texobj(&texUnit->CurrentTex[targetIndex], texObj);
```

texture_object中有一个二维数组，第一维是faces, 第二维的mipmap levels，存的是gl_texture_image指针。

```cpp
/**
 * Texture object state.  Contains the array of mipmap images, border color,
 * wrap modes, filter modes, and shadow/texcompare state.
 */
struct gl_texture_object
{
   simple_mtx_t Mutex;         /**< for thread safety */
   GLint RefCount;             /**< reference count */
   GLuint Name;                /**< the user-visible texture object ID */
   GLenum16 Target;            /**< GL_TEXTURE_1D, GL_TEXTURE_2D, etc. */
   GLenum16 DepthMode;         /**< GL_ARB_depth_texture */
   GLchar *Label;              /**< GL_KHR_debug */

   struct gl_sampler_object Sampler;

   gl_texture_index TargetIndex; /**< The gl_texture_unit::CurrentTex index.
...
   GLboolean Immutable;        /**< GL_ARB_texture_storage */
...
   /** Actual texture images, indexed by [cube face] and [mipmap level] */
   struct gl_texture_image *Image[MAX_FACES][MAX_TEXTURE_LEVELS];

};

```

# texture image

入口在_mesa_TexImage2D或_mesa_TexStorage2D:

前者分配一个image，放在某target的某个mipmap level上。并且可以填充或不填充数据。
后者分配一组image，直接把某target的所有mipmap level弄完整。后者创建是immutable image。
immutable image的意义不是纹理不可写，而是这个纹理不可以resize和替换backing store。

```c
/**
 * Texture image state.  Drivers will typically create a subclass of this
 * with extra fields for memory buffers, etc.
 */
struct gl_texture_image
{
   GLint InternalFormat;	/**< Internal format as given by the user */
   GLenum16 _BaseFormat;	/**< Either GL_RGB, GL_RGBA, GL_ALPHA,
                                 *   GL_LUMINANCE, GL_LUMINANCE_ALPHA,
                                 *   GL_INTENSITY, GL_DEPTH_COMPONENT or
                                 *   GL_DEPTH_STENCIL_EXT only. Used for
                                 *   choosing TexEnv arithmetic.
                                 */
   mesa_format TexFormat;         /**< The actual texture memory format */

   GLuint Border;		/**< 0 or 1 */
   GLuint Width;		/**< = 2^WidthLog2 + 2*Border */
   GLuint Height;		/**< = 2^HeightLog2 + 2*Border */
   GLuint Depth;		/**< = 2^DepthLog2 + 2*Border */
   GLuint Width2;		/**< = Width - 2*Border */
   GLuint Height2;		/**< = Height - 2*Border */
   GLuint Depth2;		/**< = Depth - 2*Border */
   GLuint WidthLog2;		/**< = log2(Width2) */
   GLuint HeightLog2;		/**< = log2(Height2) */
   GLuint DepthLog2;		/**< = log2(Depth2) */
   GLuint MaxNumLevels;		/**< = maximum possible number of mipmap
                                       levels, computed from the dimensions */

   struct gl_texture_object *TexObject;  /**< Pointer back to parent object */
   GLuint Level;                /**< Which mipmap level am I? */
   /** Cube map face: index into gl_texture_object::Image[] array */
   GLuint Face;

   /** GL_ARB_texture_multisample */
   GLuint NumSamples;            /**< Sample count, or 0 for non-multisample */
   GLboolean FixedSampleLocations; /**< Same sample locations for all pixels? */
};

```

## mutable image

_mesa_TexImage2D分配的是mutable image，比较关键的gl_texture_image与gl_texture_object对应的代码在这里:
```c
/**
 * Like _mesa_select_tex_image() but if the image doesn't exist, allocate
 * it and install it.  Only return NULL if passed a bad parameter or run
 * out of memory.
 */
struct gl_texture_image *
_mesa_get_tex_image(struct gl_context *ctx, struct gl_texture_object *texObj,
                    GLenum target, GLint level)
{
   struct gl_texture_image *texImage;

   if (!texObj)
      return NULL;

   texImage = _mesa_select_tex_image(texObj, target, level);
   if (!texImage) {
      texImage = ctx->Driver.NewTextureImage(ctx);
      if (!texImage) {
         _mesa_error(ctx, GL_OUT_OF_MEMORY, "texture image allocation");
         return NULL;
      }

      set_tex_image(texObj, target, level, texImage);
   }

   return texImage;
}


struct gl_texture_image *
_mesa_select_tex_image(const struct gl_texture_object *texObj,
		                 GLenum target, GLint level)
{
   const GLuint face = _mesa_tex_target_to_face(target);
   return texObj->Image[face][level];
}

static void
set_tex_image(struct gl_texture_object *tObj,
              GLenum target, GLint level,
              struct gl_texture_image *texImage)
{
   const GLuint face = _mesa_tex_target_to_face(target);

   tObj->Image[face][level] = texImage;

   /* Set the 'back' pointer */
   texImage->TexObject = tObj;
   texImage->Level = level;
   texImage->Face = face;
}

```

分配了gl_texture_image，并且关联给gl_texture_object某face某level之后，调用vendor driver的TexImage来分配内存：

```c
static void
st_TexImage(struct gl_context * ctx, GLuint dims,
            struct gl_texture_image *texImage,
            GLenum format, GLenum type, const void *pixels,
            const struct gl_pixelstore_attrib *unpack)
{
   assert(dims == 1 || dims == 2 || dims == 3);

   prep_teximage(ctx, texImage, format, type);

   if (texImage->Width == 0 || texImage->Height == 0 || texImage->Depth == 0)
      return;

   /* allocate storage for texture data */
   if (!ctx->Driver.AllocTextureImageBuffer(ctx, texImage)) {
      _mesa_error(ctx, GL_OUT_OF_MEMORY, "glTexImage%uD", dims);
      return;
   }

   st_TexSubImage(ctx, dims, texImage, 0, 0, 0,
                  texImage->Width, texImage->Height, texImage->Depth,
                  format, type, pixels, unpack);
}

/**
 * Called via ctx->Driver.AllocTextureImageBuffer().
 * If the texture object/buffer already has space for the indicated image,
 * we're done.  Otherwise, allocate memory for the new texture image.
 */
static GLboolean
st_AllocTextureImageBuffer(struct gl_context *ctx,
                           struct gl_texture_image *texImage)
{
   struct st_context *st = st_context(ctx);
   struct st_texture_image *stImage = st_texture_image(texImage);
   struct st_texture_object *stObj = st_texture_object(texImage->TexObject);
... 

   if (stObj->pt &&
       st_texture_match_image(st, stObj->pt, texImage)) {
      /* The image will live in the object's mipmap memory */
      pipe_resource_reference(&stImage->pt, stObj->pt);
      assert(stImage->pt);
      return GL_TRUE;
   }
   else {
...
      stImage->pt = st_texture_create(st,
                                      gl_target_to_pipe(stObj->base.Target),
                                      format,
                                      0, /* lastLevel */
                                      ptWidth,
                                      ptHeight,
                                      ptDepth,
                                      ptLayers, 0,
                                      bindings);
      return stImage->pt != NULL;
   }
}
```

### st_texture_image

st_texture_create返回的是pipe_resource指针，填入了st_texture_image的pt成员里。pipe_resource就是代码buffer或image的backing store。

```c
/**
 * Subclass of gl_texure_image.
 */
struct st_texture_image
{
   struct gl_texture_image base;

   /* If stImage->pt != NULL, image data is stored here.
    * Else there is no image data.
    */
   struct pipe_resource *pt;
   ...

};
```

### pipe_resource

st_texture_create函数主要调用screen->resource_create来创建pipe_resource的一个子类, pipe_resource代表一个buffer或image，它只描述，不具体真实的backing store，真实的backing store在子类里：
```c
/**
 * Allocate a new pipe_resource object
 * width0, height0, depth0 are the dimensions of the level 0 image
 * (the highest resolution).  last_level indicates how many mipmap levels
 * to allocate storage for.  For non-mipmapped textures, this will be zero.
 */
struct pipe_resource *
st_texture_create(struct st_context *st,
                  enum pipe_texture_target target,
                  enum pipe_format format,
                  GLuint last_level,
                  GLuint width0,
                  GLuint height0,
                  GLuint depth0,
                  GLuint layers,
                  GLuint nr_samples,
                  GLuint bind)
{
   struct pipe_resource pt, *newtex;
   struct pipe_screen *screen = st->pipe->screen;
...
   memset(&pt, 0, sizeof(pt));
   pt.target = target;
   pt.format = format;
   pt.last_level = last_level;
   pt.width0 = width0;
   pt.height0 = height0;
   pt.depth0 = depth0;
   pt.array_size = layers;
   pt.usage = PIPE_USAGE_DEFAULT;
   pt.bind = bind;
   /* only set this for OpenGL textures, not renderbuffers */
   pt.flags = PIPE_RESOURCE_FLAG_TEXTURING_MORE_LIKELY;
   pt.nr_samples = nr_samples;
   pt.nr_storage_samples = nr_samples;

   newtex = screen->resource_create(screen, &pt); //返回回子类
   return newtex;
}


```

### fd resource
screen->resource_create在adreno中就是fd_resource_create函数, 这个函数经过一系列调用，最后通过创建了一个pipe_resource的子类对象，fd_resource：

```c
struct fd_resource {
	struct pipe_resource base;
	struct fd_bo *bo;
	enum pipe_format internal_format;
	struct fdl_layout layout;
...
	/* Sequence # incremented each time bo changes: */
	uint16_t seqno;
...
	bool lrz_valid : 1;
	enum fd_lrz_direction lrz_direction : 2;
	uint16_t lrz_width;  // for lrz clear, does this differ from lrz_pitch?
	uint16_t lrz_height;
	uint16_t lrz_pitch;
	struct fd_bo *lrz;
};
```

fd_resource的bo指向的就是一个buffer object，这个bo的创建也不直观，代码里是先设置fd_resource的各种属性，然后调用realloc_bo来分配空间的：

```c
static void
realloc_bo(struct fd_resource *rsc, uint32_t size)
{
	...
    rsc->bo = fd_bo_new(screen->dev, size, flags, "%ux%ux%u@%u:%x",
			prsc->width0, prsc->height0, prsc->depth0, rsc->layout.cpp, prsc->bind);
    ...
	/* Zero out the UBWC area on allocation.  This fixes intermittent failures
	 * with UBWC, which I suspect are due to the HW having a hard time
	 * interpreting arbitrary values populating the flags buffer when the BO
	 * was recycled through the bo cache (instead of fresh allocations from
	 * the kernel, which are zeroed).  sleep(1) in this spot didn't work
	 * around the issue, but any memset value seems to.
	 */
	if (rsc->layout.ubwc) {
		void *buf = fd_bo_map(rsc->bo);
	}
    ...
	rsc->seqno = p_atomic_inc_return(&screen->rsc_seqno);
}
```

### fd_bo
```c
struct fd_bo {
	struct fd_device *dev;
	uint32_t size;
	uint32_t handle;
	uint32_t name;
	int32_t refcnt;
	uint32_t flags; /* flags like FD_RELOC_DUMP to use for relocs to this BO */
	uint64_t iova;
	void *map;
	const struct fd_bo_funcs *funcs;

	enum {
		NO_CACHE = 0,
		BO_CACHE = 1,
		RING_CACHE = 2,
	} bo_reuse;

	struct list_head list;   /* bucket-list entry */
	time_t free_time;        /* time when added to bucket-list */
};
```

bo就是buffer object，它的创建分两步，一是申请显存并得到句柄，二是获取iova填充数据结构, 申请显存调的是drm接口：

```c

/* allocate a buffer handle: */
int msm_bo_new_handle(struct fd_device *dev,
		uint32_t size, uint32_t flags, uint32_t *handle)
{
	struct drm_msm_gem_new req = {
			.size = size,
			.flags = MSM_BO_WC,  // TODO figure out proper flags..
	};
    ...
	ret = drmCommandWriteRead(dev->fd, DRM_MSM_GEM_NEW,
			&req, sizeof(req));
    ...
	*handle = req.handle;

	return 0;
}

/* allocate a new buffer object, call w/ table_lock held */
static struct fd_bo * bo_from_handle(struct fd_device *dev,
		uint32_t size, uint32_t handle)
{
	auto bo = dev->funcs->bo_from_handle(dev, size, handle); // msm_bo_from_handle
	if (!bo) {
		drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &req);
		return NULL;
	}
	bo->size = size;
	bo->handle = handle;
	bo->iova = bo->funcs->iova(bo);
	/* add ourself into the handle table: */
	_mesa_hash_table_insert(dev->handle_table, &bo->handle, bo);
	return bo;
}
```

## immutable image

immutable image的分配是在st_AllocTextureStorage（其他显卡可以重写这个函数），它只分配一个image, 给所有的faces和level用。
分配到的是pipe_resource，存在st_texture_object的pt指针下面, st_texture_object是gl_texture_object的子类：

```c
/**
 * Subclass of gl_texure_object.
 */
struct st_texture_object
{
   struct gl_texture_object base;       /* The "parent" object */
...
   struct pipe_resource *pt;
...
};
void st_texture_storage()
{
    ...
      stObj->pt = st_texture_create(st,
                                    gl_target_to_pipe(texObj->Target),
                                    fmt,
                                    levels - 1,
                                    ptWidth,
                                    ptHeight,
                                    ptDepth,
                                    ptLayers, num_samples,
                                    bindings);
    ...
  
}
```

st_texture_create之后的实现参考上一节，都是一样的。

# 示例代码

```c
glActiveTexture(GL_TEXTURE0); //激活texture unit

unsigned int textureID;
glGenTextures(1, &textureID); // 分配一个名字，就是整数

glBindTexture(GL_TEXTURE_CUBE_MAP, textureID); // 创建texture object

int width, height, nrChannels;
unsigned char *data;  
for(unsigned int i = 0; i < textures_faces.size(); i++)
{
    data = stbi_load(textures_faces[i].c_str(), &width, &height, &nrChannels, 0);
    glTexImage2D(
        GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 
        0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
    );  // 创建mutable image并填充数据
}

glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

```