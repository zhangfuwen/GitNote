# GLES Texture

## 概念

GLES的Texture有几个概念：

1. texture unit，应用显卡中的硬件单元，用于对纹理进入采样和写入
2. texture name， 一个整数，代码一个context内的唯一的纹理。用glGenTextures获得。
3. texture object, 与name一一对应的。用glBindTexture获得
4. texture image, 一个texture object中可以有多个image，一个image 代码2D纹理的一个mipmap level，或是Cube Map纹理中一个面的一个mipmap level。

## texture unit

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

## texture name

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

## texture object

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

## texture image

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

### mutable image

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

#### st_texture_image

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

#### pipe_resource

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

#### fd resource

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

#### fd_bo

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


### immutable image

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

## eglImage vs gl_texture_image

在Android中经常需要渲染到AHardwareBuffer。其中的原理是什么呢？

### 用法

#### 做法一

```c

// Try creating a 32x32 AHardwareBuffer and attaching it to a multiview
// framebuffer, with various formats and depths.
AHardwareBuffer_Desc desc = {};
desc.width = 32;
desc.height = 32;
desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT;
const int layers[] = {2, 4};
const int formats[] = {
    AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM,
    AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM,
    // Do not test AHARDWAREBUFFER_FORMAT_BLOB, it isn't color-renderable.
};
const int samples[] = {1, 2, 4};
for (int nsamples : samples) {
    for (auto nlayers : layers) {
        for (auto format : formats) {
            desc.layers = nlayers;
            desc.format = format;
            testEglImageArray(env, desc, nsamples);
        }
    }
}

static void testEglImageArray(JNIEnv* env, AHardwareBuffer_Desc desc,
        int nsamples) {
 AHardwareBuffer* hwbuffer = nullptr;
    int error = AHardwareBuffer_allocate(&desc, &hwbuffer);

    // Create EGLClientBuffer from the AHardwareBuffer.
    EGLClientBuffer native_buffer = eglGetNativeClientBufferANDROID(hwbuffer);

    // Create EGLImage from EGLClientBuffer.
    EGLint attrs[] = {EGL_NONE};
    EGLImageKHR image =
        eglCreateImageKHR(eglGetCurrentDisplay(), EGL_NO_CONTEXT,
                          EGL_NATIVE_BUFFER_ANDROID, native_buffer, attrs);

    // Create OpenGL texture from the EGLImage.
    GLuint texid;
    glGenTextures(1, &texid);
    glBindTexture(GL_TEXTURE_2D_ARRAY, texid);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_2D_ARRAY, image);

    // Create FBO and add multiview attachment.
    GLuint fboid;
    glGenFramebuffers(1, &fboid);
    glBindFramebuffer(GL_FRAMEBUFFER, fboid);
    const GLint miplevel = 0;
    const GLint base_view = 0;
    const GLint num_views = desc.layers;
    if (nsamples == 1) {
        glFramebufferTextureMultiviewOVR(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                         texid, miplevel, base_view, num_views);
    } else {
        glFramebufferTextureMultisampleMultiviewOVR(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texid, miplevel, nsamples,
            base_view, num_views);
    }

    glCheckFramebufferStatus(GL_FRAMEBUFFER);
    //do some render
  
    glDeleteTextures(1, &texid);
    glDeleteFramebuffers(1, &fboid);
    AHardwareBuffer_release(hwbuffer);
  
```

#### 做法二

也可以通过eglCreateNativeClientBufferANDROID获取EGLClientBuffer.

```c
EGLint attrs[] = {
	EGL_WIDTH, 10,
	EGL_HEIGHT,10,
	EGL_RED_SIZE,8,
	EGL_GREEN_SIZE,8,
	EGL_BLUE_SIZE 8,
	EGL_ALPHA_SIZE,8,
	EGL_NATIVE_BUFFER_USAGE_ANDROID,EGL_NATIVE_BUFFER_USAGE_TEXTURE_BIT_ANDROID,
	EGL_NONE };
  
EGLClientBuffer native_buffer = eglCreateNativeClientBufferANDROID(attrs);
```

### 原理

在前面的章节我们知道texture的一个image，在底层的backing store是一个通过drm获取的一个buffer object。

那么eglImage与gl_texture_image是不是一一对应的关系呢？

#### eglCreateImageKHR的实现

```c
src/egl/drivers/dri2/platform_android.c:1343:   .create_image = droid_create_image_khr,

```

具体到native buffer的实现在`dri2_create_image_android_native_buffer`:

```c
static _EGLImage *
dri2_create_image_android_native_buffer(_EGLDisplay *disp,
                                        _EGLContext *ctx,
                                        struct ANativeWindowBuffer *buf)
{
   if (ctx != NULL) {
      /* From the EGL_ANDROID_image_native_buffer spec:
       *
       *     * If <target> is EGL_NATIVE_BUFFER_ANDROID and <ctx> is not
       *       EGL_NO_CONTEXT, the error EGL_BAD_CONTEXT is generated.
       */
      _eglError(EGL_BAD_CONTEXT, "eglCreateEGLImageKHR: for "
                "EGL_NATIVE_BUFFER_ANDROID, the context must be "
                "EGL_NO_CONTEXT");
      return NULL;
   }

   if (!buf || buf->common.magic != ANDROID_NATIVE_BUFFER_MAGIC ||
       buf->common.version != sizeof(*buf)) {
      _eglError(EGL_BAD_PARAMETER, "eglCreateEGLImageKHR");
      return NULL;
   }

   __DRIimage *dri_image =
      droid_create_image_from_native_buffer(disp, buf, buf);

#ifdef HAVE_DRM_GRALLOC
   if (dri_image == NULL)
      dri_image = droid_create_image_from_name(disp, buf, buf);
#endif

   if (dri_image) {
#if ANDROID_API_LEVEL >= 26
      AHardwareBuffer_acquire(ANativeWindowBuffer_getHardwareBuffer(buf));
#endif
      return dri2_create_image_from_dri(disp, dri_image);
   }

   return NULL;
}
```

从ahardwarebuffer创建dri_image最终调到了这里：

```c
static __DRIimage *
droid_create_image_from_buffer_info(struct dri2_egl_display *dri2_dpy,
    ¦   ¦   ¦   ¦   ¦   ¦   ¦   ¦   struct buffer_info *buf_info,
    ¦   ¦   ¦   ¦   ¦   ¦   ¦   ¦   void *priv)
{
   unsigned error;

   if (dri2_dpy->image->base.version >= 15 &&
    ¦  dri2_dpy->image->createImageFromDmaBufs2 != NULL) {
    ¦ return dri2_dpy->image->createImageFromDmaBufs2(
    ¦   ¦dri2_dpy->dri_screen, buf_info->width, buf_info->height,
    ¦   ¦buf_info->drm_fourcc, buf_info->modifier, buf_info->fds,
    ¦   ¦buf_info->num_planes, buf_info->pitches, buf_info->offsets,
    ¦   ¦buf_info->yuv_color_space, buf_info->sample_range,
    ¦   ¦buf_info->horizontal_siting, buf_info->vertical_siting, &error,
    ¦   ¦priv);
   }

   return dri2_dpy->image->createImageFromDmaBufs(
    ¦ dri2_dpy->dri_screen, buf_info->width, buf_info->height,
    ¦ buf_info->drm_fourcc, buf_info->fds, buf_info->num_planes,
    ¦ buf_info->pitches, buf_info->offsets, buf_info->yuv_color_space,
    ¦ buf_info->sample_range, buf_info->horizontal_siting,
    ¦ buf_info->vertical_siting, &error, priv);                                                                                                 
}

```

两个函数指针的具体实现分别为:`dri2_from_dma_bufs`和 `dri2_from_dma_bufs2`，这两个函数都是调用`dri2_create_image_from_fd`：
这个函数从fd创建了一个DRIImage，fd是放在了一个pipe_resource里面(与gl_texture_image同)。DRIImage的数据结构(alias of DRIImageRec)：

```c
struct __DRIimageRec {
   struct pipe_resource *texture; // 这里是一个链表，对ahardwarebuffer可能带来的多个fd,用pipe_resource->next串起来。 
   unsigned level;
   unsigned layer;
   uint32_t dri_format;
   uint32_t dri_fourcc;
   uint32_t dri_components;
   unsigned use;
   unsigned plane;

   void *loader_private;

   boolean imported_dmabuf;
   /** 
   ¦* Provided by EGL_EXT_image_dma_buf_import.
   ¦*/ 
   enum __DRIYUVColorSpace yuv_color_space;                                                                                                     
   enum __DRISampleRange sample_range;
   enum __DRIChromaSiting horizontal_siting;
   enum __DRIChromaSiting vertical_siting;

   /* DRI loader screen */
   __DRIscreen *sPriv;
};

```

DRIImage的texture是通过对每个fd调用tex = pscreen->resource_from_handle并prepend实现的，resource_from_handle针对高通的实现是
`fd_resource_from_handle`, 实际上创建的是pipe_resource的子类fd_resource。

从这里可以看到eglImage和gl_texture_object或gl_texture_image都不一样，它代表一串，几个pipe_resource。

#### glEGLImageTargetTexture2DOES

它的实现是__mesa_EGLImageTargetTexture2DOES， 该函数主要逻辑：

```c
texImage = _mesa_get_tex_image(ctx, texObj, target, 0); //最后一个参数是level，就是只用texture的level 0
   if (!texImage) {
      _mesa_error(ctx, GL_OUT_OF_MEMORY, "%s", caller);
   } else {
      st_FreeTextureImageBuffer(ctx, texImage); // 释放到gl_texture_image的backing store

      texObj->External = GL_TRUE;

      if (tex_storage) {
         st_egl_image_target_tex_storage(ctx, target, texObj, texImage,
                                         image); 
      } else {
         st_egl_image_target_texture_2d(ctx, target, texObj, texImage,
                                        image);
      }

      _mesa_dirty_texobj(ctx, texObj);
   }

```

上面代码中调用的st_egl_image_target_tex_storage和st_egl_image_target_texture_2d的实现基本是相同的：

```c
void
st_egl_image_target_tex_storage(struct gl_context *ctx, GLenum target,
    ¦   ¦   ¦   ¦   ¦   ¦   ¦   struct gl_texture_object *texObj,
    ¦   ¦   ¦   ¦   ¦   ¦   ¦   struct gl_texture_image *texImage,
    ¦   ¦   ¦   ¦   ¦   ¦   ¦   GLeglImageOES image_handle)
{
   struct st_egl_image stimg;
   bool native_supported;

   if (!st_get_egl_image(ctx, image_handle, PIPE_BIND_SAMPLER_VIEW,
    ¦   ¦   ¦   ¦   ¦   ¦"glEGLImageTargetTexture2D", &stimg,
    ¦   ¦   ¦   ¦   ¦   ¦&native_supported))
    ¦ return;

   st_bind_egl_image(ctx, texObj, texImage, &stimg, true, native_supported);                                                                      
   pipe_resource_reference(&stimg.texture, NULL);
}

```

函数的逻辑就是取到st_egl_image然后后gl_texture_object绑定，即调用st_bind_egl_iamge。
st_egl_image不是DRIImage的子类，但它们的pipe_resource *texture指向相同。

st_bind_egl_image主要做的就是把stimg->texture填到texObj->pt和texImage->pt上。

```c
 pipe_resource_reference(&texObj->pt, stimg->texture); //pipe_resource_reference实际是将第一个参数指向的内容释放，然后再第二个参数指针的内容赋值给第一个参数
   st_texture_release_all_sampler_views(st, texObj);                                                                                              
   pipe_resource_reference(&texImage->pt, texObj->pt);

```

经过这一步之后是这样的：

```plantuml
class gl_texture_object {
  
    pipe_resource * pt;
    gl_texture_image *Image[faces][levels];
}

class gl_texture_image {
}

class st_texture_image {
    pipe_resource * pt;
}
st_texture_image "extends"--|> gl_texture_image

gl_texture_object "Image[0][0]" -> gl_texture_image

class st_egl_image {
    pipe_resource * texture;
}

object fd_resource_linked_list {
    pipe_resource * next;
}

Object texObj<<st_texture_object>> 
texObj "pt"..>  fd_resource_linked_list
Object texImg<<st_texture_image>>
texImg "pt"..>  fd_resource_linked_list
Object stImg<<st_egl_image>>
stImg "texture"..>  fd_resource_linked_list

```

## 示例代码

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

# DRM vs GBM

drm提供了一组ioctl来管理buffer的创建、释放、map, 送显等：
https://gitlab.freedesktop.org/mesa/drm/-/blob/main/include/drm/drm.h

gbm是一个通用的buffer管理接口，api比较易用，它有多种backend,典型的就是drm。

## drm api

https://gitlab.freedesktop.org/mesa/drm/-/blob/main/include/drm/drm.h

## gbm api

```c
/**
 * \file gbmint.h
 * \brief Internal implementation details of gbm
 */

/**
 * The device used for the memory allocation.
 *
 * The members of this structure should be not accessed directly
 */
struct gbm_device {
   /* Hack to make a gbm_device detectable by its first element. */
   struct gbm_device *(*dummy)(int);

   int fd;
   const char *name;
   unsigned int refcount;
   struct stat stat;

   void (*destroy)(struct gbm_device *gbm);
   int (*is_format_supported)(struct gbm_device *gbm,
                              uint32_t format,
                              uint32_t usage);
   int (*get_format_modifier_plane_count)(struct gbm_device *device,
                                          uint32_t format,
                                          uint64_t modifier);

   struct gbm_bo *(*bo_create)(struct gbm_device *gbm,
                               uint32_t width, uint32_t height,
                               uint32_t format,
                               uint32_t usage,
                               const uint64_t *modifiers,
                               const unsigned int count);
   struct gbm_bo *(*bo_import)(struct gbm_device *gbm, uint32_t type,
                               void *buffer, uint32_t usage);
   void *(*bo_map)(struct gbm_bo *bo,
                               uint32_t x, uint32_t y,
                               uint32_t width, uint32_t height,
                               uint32_t flags, uint32_t *stride,
                               void **map_data);
   void (*bo_unmap)(struct gbm_bo *bo, void *map_data);
   int (*bo_write)(struct gbm_bo *bo, const void *buf, size_t data);
   int (*bo_get_fd)(struct gbm_bo *bo);
   int (*bo_get_planes)(struct gbm_bo *bo);
   union gbm_bo_handle (*bo_get_handle)(struct gbm_bo *bo, int plane);
   uint32_t (*bo_get_stride)(struct gbm_bo *bo, int plane);
   uint32_t (*bo_get_offset)(struct gbm_bo *bo, int plane);
   uint64_t (*bo_get_modifier)(struct gbm_bo *bo);
   void (*bo_destroy)(struct gbm_bo *bo);

   struct gbm_surface *(*surface_create)(struct gbm_device *gbm,
                                         uint32_t width, uint32_t height,
                                         uint32_t format, uint32_t flags,
                                         const uint64_t *modifiers,
                                         const unsigned count);
   struct gbm_bo *(*surface_lock_front_buffer)(struct gbm_surface *surface);
   void (*surface_release_buffer)(struct gbm_surface *surface,
                                  struct gbm_bo *bo);
   int (*surface_has_free_buffers)(struct gbm_surface *surface);
   void (*surface_destroy)(struct gbm_surface *surface);
};
```

## 通过GBM离屏渲染


```c
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <assert.h>
#include <fcntl.h>
#include <gbm.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* a dummy compute shader that does nothing */
#define COMPUTE_SHADER_SRC "          \
#version 310 es\n                                                       \
                                                                        \
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;       \
                                                                        \
void main(void) {                                                       \
   /* awesome compute code here */                                      \
}                                                                       \
"

int32_t
main (int32_t argc, char* argv[])
{
   bool res;

   int32_t fd = open ("/dev/dri/renderD128", O_RDWR);
   assert (fd > 0);

   struct gbm_device *gbm = gbm_create_device (fd);
   assert (gbm != NULL);

   /* setup EGL from the GBM device */
   EGLDisplay egl_dpy = eglGetPlatformDisplay (EGL_PLATFORM_GBM_MESA, gbm, NULL);
   assert (egl_dpy != NULL);

   res = eglInitialize (egl_dpy, NULL, NULL);
   assert (res);

   const char *egl_extension_st = eglQueryString (egl_dpy, EGL_EXTENSIONS);
   assert (strstr (egl_extension_st, "EGL_KHR_create_context") != NULL);
   assert (strstr (egl_extension_st, "EGL_KHR_surfaceless_context") != NULL);

   static const EGLint config_attribs[] = {
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
      EGL_NONE
   };
   EGLConfig cfg;
   EGLint count;

   res = eglChooseConfig (egl_dpy, config_attribs, &cfg, 1, &count);
   assert (res);

   res = eglBindAPI (EGL_OPENGL_ES_API);
   assert (res);

   static const EGLint attribs[] = {
      EGL_CONTEXT_CLIENT_VERSION, 3,
      EGL_NONE
   };
   EGLContext core_ctx = eglCreateContext (egl_dpy,
                                           cfg,
                                           EGL_NO_CONTEXT,
                                           attribs);
   assert (core_ctx != EGL_NO_CONTEXT);

   res = eglMakeCurrent (egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, core_ctx);
   assert (res);

   /* print some compute limits (not strictly necessary) */
   GLint work_group_count[3] = {0};
   for (unsigned i = 0; i < 3; i++)
      glGetIntegeri_v (GL_MAX_COMPUTE_WORK_GROUP_COUNT,
                       i,
                       &work_group_count[i]);
   printf ("GL_MAX_COMPUTE_WORK_GROUP_COUNT: %d, %d, %d\n",
           work_group_count[0],
           work_group_count[1],
           work_group_count[2]);

   GLint work_group_size[3] = {0};
   for (unsigned i = 0; i < 3; i++)
      glGetIntegeri_v (GL_MAX_COMPUTE_WORK_GROUP_SIZE, i, &work_group_size[i]);
   printf ("GL_MAX_COMPUTE_WORK_GROUP_SIZE: %d, %d, %d\n",
           work_group_size[0],
           work_group_size[1],
           work_group_size[2]);

   GLint max_invocations;
   glGetIntegerv (GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_invocations);
   printf ("GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS: %d\n", max_invocations);

   GLint mem_size;
   glGetIntegerv (GL_MAX_COMPUTE_SHARED_MEMORY_SIZE, &mem_size);
   printf ("GL_MAX_COMPUTE_SHARED_MEMORY_SIZE: %d\n", mem_size);

   /* setup a compute shader */
   GLuint compute_shader = glCreateShader (GL_COMPUTE_SHADER);

   assert (glGetError () == GL_NO_ERROR);
   const char *shader_source = COMPUTE_SHADER_SRC;

   glShaderSource (compute_shader, 1, &shader_source, NULL);
   assert (glGetError () == GL_NO_ERROR);

   glCompileShader (compute_shader);
   assert (glGetError () == GL_NO_ERROR);

   GLuint shader_program = glCreateProgram ();

   glAttachShader (shader_program, compute_shader);
   assert (glGetError () == GL_NO_ERROR);

   glLinkProgram (shader_program);
   assert (glGetError () == GL_NO_ERROR);

   glDeleteShader (compute_shader);

   glUseProgram (shader_program);
   assert (glGetError () == GL_NO_ERROR);

   /* dispatch computation */
   glDispatchCompute (1, 1, 1);
   assert (glGetError () == GL_NO_ERROR);

   printf ("Compute shader dispatched and finished successfully\n");

   /* free stuff */
   glDeleteProgram (shader_program);
   eglDestroyContext (egl_dpy, core_ctx);
   eglTerminate (egl_dpy);
   gbm_device_destroy (gbm);
   close (fd);

   return 0;
}
```