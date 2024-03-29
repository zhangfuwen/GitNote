# NV_texture_array

Name

    NV_texture_array

Name Strings

    GL_NV_texture_array

Contributors

    Contributors to the OpenGL EXT_texture_array extension
    Mathias Heyer, NVIDIA
    Greg Roth, NVIDIA
    Ian Stewart, NVIDIA
    Nuno Subtil, NVIDIA

Contact

    Ian Stewart, NVIDIA Corporation (istewart 'at' nvidia.com)

Status

    Complete.

Version

    Last Modifed Date: Sep 25, 2012
    NVIDIA Revision: 4

Number

    OpenGL ES Extension #133

Dependencies

    This extension is written against the OpenGL ES 2.0.25
    Specification.

    This extension depends on EXT_unpack_subimage.

    This extension interacts with OES_texture_3D.
    This extension interacts with OES_depth_texture.
    This extension interacts with OES_packed_depth_stencil.
    This extension interacts with OES_texture_npot.
    This extension interacts with NV_pixel_buffer_object.
    This extension interacts with NV_texture_npot_2D_mipmap.
    This extension interacts with NV_framebuffer_blit.

Overview

    This extension introduces the notion of two-dimensional array
    textures.  An array texture is a collection of two-dimensional
    images of identical size and format, arranged in layers.  Array
    textures are specified using TexImage3DNV, where the depth is
    used to indicate the number of layers in the image.

    An array texture is accessed as a single unit in a programmable
    shader, using a single coordinate vector.  A single layer is
    selected, using the "p" texture coordinate, and that layer is then
    accessed as though it were a two-dimensional texture.  The layer
    coordinate is provided as an unnormalized floating-point value in
    the range [0,<n>-1], where <n> is the number of layers in the
    array texture.  Texture lookups do not filter between layers,
    though such filtering can be achieved using programmable shaders.
    When mipmapping is used, each level of an array texture has the
    same number of layers as the base level; the number of layers is
    not reduced as the image size decreases.

    Single layers of array textures can be rendered to by binding them
    to a framebuffer object using the FramebufferTextureLayerNV
    function.

New Procedures and Functions

    void TexImage3DNV(enum target, int level, enum internalFormat,
                      sizei width, sizei height, sizei depth, int border,
                      enum format, enum type, const void *pixels)

    void TexSubImage3DNV(enum target, int level,
                         int xoffset, int yoffset, int zoffset,
                         sizei width, sizei height, sizei depth,
                         enum format, enum type, const void *pixels)

    void CopyTexSubImage3DNV(enum target, int level,
                             int xoffset, int yoffset, int zoffset,
                             int x, int y, sizei width, sizei height)

    void CompressedTexImage3DNV(enum target, int level, enum internalformat,
                                sizei width, sizei height, sizei depth,
                                int border, sizei imageSize, const void *data)

    void CompressedTexSubImage3DNV(enum target, int level,
                                   int xoffset, int yoffset, int zoffset,
                                   sizei width, sizei height, sizei depth,
                                   enum format, sizei imageSize,
                                   const void *data)

    void FramebufferTextureLayerNV(enum target, enum attachment,
                                   uint texture, int level, int layer);

New Tokens

    Accepted by the <target> parameter of TexImage3DNV,
    TexSubImage3DNV, CopyTexSubImage3DNV, CompressedTexImage3DNV,
    CompressedTexSubImage3DNV, TexParameteri, TexParameteriv,
    TexParameterf, TexParameterfv, GenerateMipmap, and BindTexture:

        TEXTURE_2D_ARRAY_NV                             0x8C1A

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv and
    GetFloatv:

        TEXTURE_BINDING_2D_ARRAY_NV                     0x8C1D
        MAX_ARRAY_TEXTURE_LAYERS_NV                     0x88FF

    Accepted by the <pname> parameter of
    GetFramebufferAttachmentParameteriv:

        FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_NV         0x8CD4

    Returned by the <type> parameter of GetActiveUniform:

        SAMPLER_2D_ARRAY_NV                             0x8DC1

    Accepted by the <pname> parameter of PixelStorei, GetBooleanv,
    GetIntegerv and GetFloatv:

        UNPACK_SKIP_IMAGES_NV                           0x806D
        UNPACK_IMAGE_HEIGHT_NV                          0x806E

New GLSL defines

    #define GL_NV_texture_array 1

New GLSL sampler types

    sampler2DArrayNV

GLSL Grammar Changes

    The token SAMPLER2DARRAYNV is added to the list of tokens returned
    from lexical analysis and the type_specifier_no_prec production.

New GLSL functions

    texture2DArrayNV()
    texture2DArrayLodNV()

Additions to Chapter 2 of the OpenGL ES 2.0.25 Specification

    Modify section 2.10.4, "Shader Variables"

    (add the following new return type to the description of
     GetActiveUniform)

        SAMPLER_2D_ARRAY_NV

Additions to Chapter 3 of the OpenGL ES 2.0.25 Specification

    Modify Section 3.6.1, "Pixel Storage Modes"

    Replace the first sentence:

    Pixel storage modes affect the operation of TexImage*D,
    TexSubImage*D, and ReadPixels when one of these commands is issued. 

    Add to table 3.1

    Parameter Name          Type      Initial Value   Valid Range
    ----------------------  -------   -------------   ------------
    UNPACK_SKIP_IMAGES_NV   integer   0               [0,Infinity)
    UNPACK_IMAGE_HEIGHT_NV  integer   0               [0,Infinity)

    Modify Section 3.7.1, "Texture Image Specification"

    Replace the entire section with:

    The command

      void TexImage3DNV(enum target, int level, int internalformat,
                        sizei width, sizei height, sizei depth, int border,
                        enum format, enum type, void *data);

    is used to specify a three-dimensional image for a two-dimensional
    array texture. <target> must be TEXTURE_2D_ARRAY_NV. <format>,
    <type>, and <data> specify the format of the image data, the type of
    those data, and a reference to the image data in the currently bound
    pixel unpack buffer or client memory, as described in section 3.6.2.

    The groups in memory are treated as being arranged in a sequence of
    adjacent rectangles. Each rectangle is a two-dimensional image,
    whose size and organization are specified by the <width> and
    <height> parameters to TexImage3DNV. The values of UNPACK_ROW_-
    LENGTH_EXT and UNPACK_ALIGNMENT control the row-to-row spacing in
    these images as described in section 3.6.2. If the value of the
    integer parameter UNPACK_IMAGE_HEIGHT_NV is zero, then the number of
    rows in each two-dimensional image is height; otherwise the number
    of rows is UNPACK_IMAGE_HEIGHT_NV. Each two-dimensional image
    comprises an integral number of rows, and is exactly adjacent to its
    neighbor images.

    The mechanism for selecting a sub-volume of a three-dimensional
    image relies on the integer parameter UNPACK_SKIP_IMAGES_NV. If
    UNPACK_SKIP_IMAGES_NV is positive, the pointer is advanced by
    UNPACK_SKIP_IMAGES_NV times the number of elements in one two-
    dimensional image before obtaining the first group from memory.
    Then <depth> two-dimensional images are processed, each having a
    subimage extracted as described in section 3.6.2.
    
    The selected groups are transferred to the GL as described in
    section 3.6.2 and then clamped to the representable range of the
    internal format. For color component groups, if the <internalformat>
    of the texture is signed or unsigned normalized fixed-point,
    components are clamped to [-1, 1] or [0, 1], respectively. For depth
    component groups, the depth value is clamped to [0, 1]. Otherwise,
    values are not modified.

    Components are then selected from the resulting R, G, B, A, depth,
    or stencil values to obtain a texture with the base internal format
    specified by <internalformat>. Table 3.8 summarizes the mapping of
    R, G, B, A, depth, or stencil values to texture components, as a
    function of the base internal format of the texture image.
    <internalformat> may be one of the five internal format symbolic
    constants listed in table 3.8. Specifying a value for
    <internalformat> that is not one of the above values generates the
    error INVALID_VALUE. If internalformat does not match format, the
    error INVALID_OPERATION is generated.

    The GL stores the resulting texture with internal component
    resolutions of its own choosing. The allocation of internal
    component resolution may vary based on any TexImage3DNV or
    TexImage2D (see below) parameter (except target), but the allocation
    must not be a function of any other state and cannot be changed once
    established. Allocations must be invariant; the same allocation must
    be chosen each time a texture image is specified with the same
    parameter values.

    The image itself (referred to by data) is a sequence of groups of
    values. The first group is the lower left back corner of the texture
    image. Subsequent groups fill out rows of width width from left to
    right; height rows are stacked from bottom to top forming a single
    two-dimensional image slice; and depth slices are stacked from back
    to front. When the final R, G, B, and A components have been
    computed for a group, they are assigned to components of a texel as
    described by table 3.8. Counting from zero, each resulting Nth texel
    is assigned internal integer coordinates (i, j, k), where

        i = (N mod width)

        j = (floor(N/width) mod height)

        k = (floor(N/(width * height) mod depth)

    Thus the last two-dimensional image slice of the three-dimensional
    image is indexed with the highest value of k.

    If the internal data type of the image array is signed or unsigned
    normalized fixed-point, each color component is converted as
    described in section 2.1.2. If the internal type is floating-point,
    components are clamped to the representable range of the
    corresponding internal component, but are not converted.

    The <level> argument to TexImage3DNV is an integer level-of-detail
    number. Levels of detail are discussed below, under Mipmapping. The
    main texture image has a level of detail number of 0. If a level-of-
    detail less than zero is specified, the error INVALID_VALUE is
    generated. If <level> is greater than zero, and either width or
    height is not a power of two, the error INVALID_VALUE is generated.

    If <border> is not zero, then the error INVALID_VALUE is generated.

    If <width>, <height>, or <depth> are less than zero, then the error
    INVALID_VALUE is generated.

    The maximum allowable width and height of a texel array for a two-
    dimensional texture or two-dimensional array texture is an
    implementation-dependent function of the level-of-detail and
    internal format of the resulting image array. It must be at least
    2^(k-lod) for image arrays of level-of-detail 0 through k, where k
    is the log base 2 of MAX_TEXTURE_SIZE and lod is the level-of-detail
    of the image array. The maximum allowable width and height of a cube
    map texture must be the same, and must be at least 2^(k-lod) for
    image arrays level 0 through k, where k is the log base 2 of
    MAX_CUBE_MAP_TEXTURE_SIZE.  The maximum number of layers for two-
    dimensional array textures (depth) must be at least MAX_ARRAY_-
    TEXTURE_LAYERS_NV for for image arrays level 0 through k. The
    maximum width, height, and depth may be zero for image arrays of any
    level-of-detail greater than k. The error INVALID_VALUE is generated
    if the specified image is too large to be stored under any
    conditions.

    An implementation may allow an image of level zero to be created
    only if that single image can be supported. Additional constraints
    on the creation of images of level one or greater are described in
    more detail in section 3.7.10.

    If a pixel unpack buffer object is bound and storing texture data
    would access memory beyond the end of the pixel unpack buffer, an
    INVALID_OPERATION error results.

    The command

        void TexImage2D(enum target, int level, int internalformat,
                        sizei width, sizei height, int border, enum format,
                        enum type, const void *data);

    is used to specify a two-dimensional texture image. target must be
    one of TEXTURE_2D for a two-dimensional texture, or one of TEXTURE_-
    CUBE_MAP_POSITIVE_X, TEXTURE_CUBE_MAP_NEGATIVE_X, TEXTURE_CUBE_MAP_-
    POSITIVE_Y, TEXTURE_CUBE_MAP_NEGATIVE_Y, TEXTURE_CUBE_MAP_POSITIVE_Z,
    or TEXTURE_CUBE_MAP_NEGATIVE_Z for a cube map texture. The other
    parameters match the corresponding parameters of TexImage3DNV.

    For the purposes of decoding the texture image, TexImage2D is
    equivalent to calling TexImage3DNV with corresponding arguments and
    depth of 1, except that UNPACK_SKIP_IMAGES_NV is ignored.

    A two-dimensional texture consists of a single two-dimensional
    texture image. A cube map texture is a set of six two-dimensional
    texture images. The six cube map texture targets form a single cube
    map texture though each target names a distinct face of the cube
    map. The TEXTURE_CUBE_MAP_* targets listed above update their
    appropriate cube map face 2D texture image. Note that the six cube
    map two-dimensional image tokens such as TEXTURE_CUBE_MAP_POSITIVE_X
    are used when specifying, updating, or querying one of a cube map's
    six two-dimensional images, but when binding to a cube map texture
    object (that is when the cube map is accessed as a whole as opposed
    to a particular two-dimensional image), the TEXTURE_CUBE_MAP target
    is specified.

    When the target parameter to TexImage2D is one of the six cube map
    two-dimensional image targets, the error INVALID_VALUE is generated
    if the width and height parameters are not equal.

    An INVALID_VALUE error is generated if border is non-zero.

    The image indicated to the GL by the image pointer is decoded and
    copied into the GL's internal memory.

    We shall refer to the decoded image as the texel array. A three-
    dimensional texel array has width, height, and depth <wt>, <ht>, and
    <dt>. A two-dimensional texel array has depth <dt> = 1, with height
    <ht> and width <wt> as above.

    An element (i, j, k) of the texel array is called a texel (for a
    non-array two-dimensional texture, k is irrelevant). The texture
    value used in texturing a fragment is determined by sampling the
    texture in a shader, but may not correspond to any actual texel. See
    figure 3.6.

    If the <data> argument of TexImage2D or TexImage3DNV is a NULL
    pointer, and the pixel unpack buffer object is zero, a two- or
    three-dimensional texel array is created with the specified target,
    level, internalformat, border, width, height, and depth, but with
    unspecified image contents. In this case no pixel values are
    accessed in client memory, and no pixel processing is performed.
    Errors are generated, however, exactly as though the data pointer
    were valid. Otherwise if the pixel unpack buffer object is non-zero,
    the data argument is treatedly normally to refer to the beginning of
    the pixel unpack buffer object's data.

    Modify Section 3.7.2, "Alternate Texture Image Specification
    Commands"

    Replace paragraphs 4 through 6 starting with the description of
    additional commands:

    Four additional commands,

        void TexSubImage3DNV(enum target, int level, int xoffset,
                           int yoffset, int zoffset, sizei width,
                           sizei height, sizei depth, enum format,
                           enum type, const void *data);
        void TexSubImage2D(enum target, int level, int xoffset,
                           int yoffset, sizei width, sizei height,
                           enum format, enum type, const void *data);
        void CopyTexSubImage3DNV(enum target, int level, int xoffset,
                               int yoffset, int zoffset, int x, int y,
                               sizei width, sizei height);
        void CopyTexSubImage2D(enum target, int level, int xoffset,
                               int yoffset, int x, int y, sizei width,
                               sizei height);

    respecify only a rectangular subregion of an existing texel array.
    No change is made to the <internalformat>, <width>, <height>,
    <depth>, or <border> parameters of the specified texel array, nor is
    any change made to texel values outside the specified subregion.
    The <target> arguments of TexSubImage2D and CopyTexSubImage2D must
    be one of TEXTURE_2D, TEXTURE_CUBE_MAP_POSITIVE_X, TEXTURE_CUBE_-
    MAP_NEGATIVE_X, TEXTURE_CUBE_MAP_POSITIVE_Y, TEXTURE_CUBE_MAP_-
    NEGATIVE_Y, TEXTURE_CUBE_MAP_POSITIVE_Z, or TEXTURE_CUBE_MAP_-
    NEGATIVE_Z, and the <target> arguments of TexSubImage3DNV and
    CopyTexSubImage3DNV must be TEXTURE_2D_ARRAY_NV. The
    <level> parameter of each command specifies the level of the texel
    array that is modified. If level is less than zero or greater than
    the base 2 logarithm of the maximum texture width, height, or depth,
    the error INVALID_VALUE is generated.

    TexSubImage3DNV arguments <width>, <height>, <depth>, <format>,
    <type>, and <data> match the corresponding arguments to
    TexImage3DNV, meaning that they are specified using the same values,
    and have the same meanings. Likewise, TexSubImage2D arguments
    <width>, <height>, <format>, <type>, and <data> match the
    corresponding arguments to TexImage2D.

    CopyTexSubImage3DNV and CopyTexSubImage2D arguments <x>, <y>,
    <width>, and <height> match the corresponding arguments to
    CopyTexImage2D. Each of the TexSubImage commands interprets and
    processes pixel groups in exactly the manner of its TexImage
    counterpart, except that the assignment of R, G, B, A, depth, and
    stencil pixel group values to the texture components is controlled
    by the <internalformat> of the texel array, not by an argument to
    the command. The same constraints and errors apply to the
    TexSubImage commands' argument <format> and the <internalformat> of
    the texel array being respecified as apply to the <format> and
    <internalformat> arguments of its TexImage counterparts.

    Arguments <xoffset>, <yoffset>, and <zoffset> of TexSubImage3DNV and
    CopyTexSubImage3DNV specify the lower left texel coordinates of a
    <width>-wide by <height>-high by <depth>-deep rectangular subregion
    of the texel array. The <depth> argument associated with
    CopyTexSubImage3DNV is always 1, because framebuffer memory is two-
    dimensional - only a portion of a single <s>, <t> slice of a three-
    dimensional texture is replaced by CopyTexSubImage3DNV.

    Taking <wt>, <ht>, and <dt> to be the specified width, height, and
    depth of the texel array, and taking <x>, <y>, <z>, <w>, <h>, and
    <d> to be the <xoffset>, <yoffset>, <zoffset>, <width>, <height>,
    and <depth> argument values, any of the following relationships
    generates the error INVALID_VALUE:

        x < 0
        x + w > wt
        y < 0
        y + h > ht
        z < 0
        z + d > dt

    Counting from zero, the <n>th pixel group is assigned to the texel
    with internal integer coordinates [i, j, k], where

        i = x + (n mod w)
        j = y + (floor(n/w) mod h)
        k = z + (floor (n/(width*height)) mod d)

    Replace the last two paragraphs of 3.7.2:

    Calling CopyTexSubImage3DNV, CopyTexImage2D, or CopyTexSubImage2D
    will result in an INVALID_FRAMEBUFFER_OPERATION error if the object
    bound to READ_FRAMEBUFFER_BINDING_NV is not framebuffer complete
    (see section 4.4.5).

    "Texture Copying Feedback Loops"

    Calling CopyTexSubImage3DNV, CopyTexImage2D, or CopyTexSubImage2D
    will result in undefined behavior if the destination texture image
    level is also bound to the selected read buffer (see section 4.3.1)
    of the read framebuffer. This situation is discussed in more detail
    in the description of feedback loops in section 4.4.4.

    Modify Section 3.7.3, "Compressed Texture Images"

    Replace the second paragraph:
    
    The commands

        void CompressedTexImage2D(enum target, int level,
                                  enum internalformat, sizei width,
                                  sizei height, int border,
                                  sizei imageSize, const void *data);
        void CompressedTexImage3DNV(enum target, int level,
                                    enum internalformat, sizei width,
                                    sizei height, sizei depth,
                                    int border, sizei imageSize,
                                    const void *data);

    define two- and three-dimensional texture images, respectively, with
    incoming data stored in a compressed image format. The <target>,
    <level>, <internalformat>, <width>, <height>, <depth>, and <border>
    parameters have the same meaning as in TexImage2D and TexImage3DNV.
    <data> refers to compressed image data stored in the compressed
    image format corresponding to <internalformat>. If a pixel unpack
    buffer is bound (as indicated by a non-zero value of PIXEL_UNPACK_-
    BUFFER_BINDING), <data> is an offset into the pixel unpack buffer
    and the compressed data is read from the buffer relative to this
    offset; otherwise, data is a pointer to client memory and the
    compressed data is read from client memory relative to the pointer.


    Replace fifth through the seventh paragraph of 3.7.3:

    Any restrictions imposed by specific compressed internal formats
    will be invariant with respect to image contents, meaning that if
    the GL accepts and stores a texture image in compressed form,
    CompressedTexImage2D or CompressedTexImage3DNV will accept any
    properly encoded compressed texture image of the same width, height,
    compressed image size, and compressed internal format for storage at
    the same texture level.

    Respecifying Subimages of Compressed Textures

    The commands

        void CompressedTexSubImage2D(enum target, int level, int xoffset,
                                    int yoffset, sizei width, sizei height,
                                    enum format, sizei imageSize, void *data);
        void CompressedTexSubImage3DNV(enum target, int level, int xoffset,
                                       int yoffset, int zoffset, sizei width,
                                       sizei height, sizei depth,
                                       enum format, sizei imageSize,
                                       const void *data);

    respecify only a rectangular region of an existing texel array, with
    incoming data stored in a known compressed image format. The
    <target>, <level>, <xoffset>, <yoffset>, <zoffset>, <width>,
    <height>, <depth>, and <format> parameters have the same meaning as
    in TexSubImage2D and TexSubImage3DNV. data points to compressed
    image data stored in the compressed image format corresponding to
    <format>.

    The image pointed to by <data> and the <imageSize> parameter are
    interpreted as though they were provided to CompressedTexImage2D and
    CompressedTexImage3DNV. These commands do not provide for image
    format conversion, so an INVALID_OPERATION error results if <format>
    does not match the internal format of the texture image being
    modified. If the <imageSize> parameter is not consistent with the
    format, dimensions, and contents of the compressed image (too little
    or too much data), an INVALID_VALUE error results.
    

    Replace the last two paragraphs of section 3.7.3:

    Any restrictions imposed by specific compressed internal formats
    will be invariant with respect to image contents, meaning that if GL
    accepts and stores a texture image in compressed form, Compressed-
    TexSubImage2D or CompressedTexSubImage3DNV will accept any properly
    encoded compressed texture image of the same width, height,
    compressed image size, and compressed internal format for storage at
    the same texture level.

    Calling CompressedTexSubImage2D or CompressedTexSubImage3DNV will
    result in an INVALID_OPERATION error if <xoffset>, <yoffset>, or
    <zoffset> are not equal to zero, or if <width>, <height>, and
    <depth> do not match the dimensions of the texture level. These
    restrictions may be relaxed for specific compressed internal formats
    whose images are easily modified.

    Modify Section 3.7.10, "Texture Completeness and Non-Power-Of-Two
    Textures"

    (modify second paragraph)

    A two-dimensional texture or two-dimensional texture array is
    complete if the following conditions all hold true:

    Modify Section 3.7.11, "Mipmap Generation"

    (modify first paragraph)

    Mipmaps can be generated with the command

        void GenerateMipmap(enum target);

    where <target> is the target, which muct be TEXTURE_2D,
    TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP.

    Modify Section 3.7.13, "Texture Objects"

    In addition to the default textures TEXTURE_2D,
    TEXTURE_2D_ARRAY_NV, and TEXTURE_CUBE_MAP, named two-dimensional,
    two-dimensional array, and cube map texture objects can be created
    and operated upon.  The name space for texture objects is the
    unsigned integers, with zero reserved by the GL.

    A texture object is created by binding an unused name to
    TEXTURE_2D, TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP. The binding
    is effected by calling

        void BindTexture(enum target, uint texture);

    with <target> set to the desired texture target and <texture> set
    to the unused name. The resulting texture object is a new state
    vector, comprising all the state values listed in section 3.7.12,
    set to the same initial values. If the new texture object is bound
    to TEXTURE_2D, TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP, it is and
    remains a two-dimensional, two-dimensional array, or cube map
    texture respectively until it is deleted.

    BindTexture may also be used to bind an existing texture object to
    TEXTURE_2D, TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP. The error
    INVALID_OPERATION is generated if an attempt is made to bind a
    texture object of different dimensionality than the specified
    target. If the bind is successful no change is made to the state
    of the bound texture object, and any previous binding to target is
    broken.  While a texture object is bound, GL operations on the
    target to which it is bound affect the bound object, and queries
    of the target to which it is bound return state from the bound
    object.

    In the initial state, TEXTURE_2D, TEXTURE_2D_ARRAY_NV, and
    TEXTURE_CUBE_MAP have two-dimensional, two-dimensional array, and
    cube map texture state vectors respectively associated with them.
    In order that access to these initial textures not be lost, they
    are treated as texture objects all of whose names are 0. The
    initial two-dimensional, two-dimensional array, and cube map
    texture are therefore operated upon, queried, and applied as
    TEXTURE_2D, TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP respectively
    while 0 is bound to the corresponding targets.

    Texture objects are deleted by calling

        void DeleteTextures(sizei n, uint *textures);

    <textures> contains <n> names of texture objects to be deleted.
    After a texture object is deleted, it has no contents or
    dimensionality, and its name is again unused. If a texture that is
    currently bound to one of the targets TEXTURE_2D,
    TEXTURE_2D_ARRAY_NV, or TEXTURE_CUBE_MAP is deleted, it is as
    though BindTexture had been executed with the same target and
    texture zero. Unused names in textures are silently ignored, as is
    the value zero.

Additions to Chapter 4 of the OpenGL ES 2.0.25 Specification

    Modify Section 4.3, "Reading Pixels"

    Pixels may be read from the framebuffer to client memory using the
    ReadPixels commands, as described below. Pixels may also be copied
    from client memory or the framebuffer to texture images in the GL
    using the CopyTexImage2D, CopyTexSubImage2D, and
    CopyTexSubImage3DNV commands, as described in section 3.7.2.

    Modify Section 4.4.3, "Renderbuffer Objects"

    (modify paragraph following "Attaching Texture Images to a
     Framebuffer")

    OpenGL ES supports copying the rendered contents of the
    framebuffer into the images of a texture object through the use of
    the routines CopyTexImage2D, CopyTexSubImage2D, and
    CopyTexSubImage3DNV. Additionally, OpenGL ES supports rendering
    directly into the images of a texture object.

    (add the following to the end of the section)

    The command

      void FramebufferTextureLayerNV(enum target, enum attachment,
                                     uint texture, int level, int layer);

    operates similarly to FramebufferTexture2D, except that it
    attaches a single layer of a two-dimensional array texture level.

    <layer> specifies the layer of the two-dimensional image within
    <texture>. An INVALID_VALUE error is generated if <layer> is
    larger than the value of MAX_ARRAY_TEXTURE_LAYERS_NV minus one.
    The error INVALID_VALUE is generated if <texture> is non-zero and
    <layer> is negative.

    <level> must be greater than or equal to zero and no larger than
    log2 of the value of MAX_TEXTURE_SIZE. Otherwise, an INVALID_VALUE
    error is generated.

    The error INVALID_OPERATION is generated if <texture> is non-zero
    and is not the name of a two-dimensional array texture. Unlike
    FramebufferTexture2D, no <textarget> parameter is accepted.

    If <texture> is non-zero and the command does not result in an
    error, the framebuffer attachment state corresponding to
    <attachment> is updated as in FramebufferTexture2D, except that
    the value of FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_NV is set to
    <layer>.

    Modify Section 4.4.5, "Framebuffer Completeness"

    (modify fourth bullet of actions that may change the framebuffer
     completeness)

      * Changing the width, height, or internal format of a texture
        image that is attached to the framebuffer by calling
        TexImage2D, CopyTexImage2D, CompressedTexImage2D,
        TexImage3DNV, or CompressedTexImage3DNV.

Additions to Chapter 6 of the OpenGL ES 2.0.25 Specification

    Modify Section 6.1.3, "Enumerated Queries"

    (modify first paragraph)

    The command

        void GetTexParameter{if}v(enum target, enum value, T data);

    returns information about <target>, which may be
    one of TEXTURE_2D, TEXTURE_2D_ARRAY, or TEXTURE_CUBE_MAP,
    indicating the currently bound two-dimensional, two-dimensional
    array, or cube map texture object.

    (add the following item after "If the value of
     FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE is TEXTURE, then")

      * If <pname> is FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_NV and the
        texture object named FRAMEBUFFER_ATTACHMENT_OBJECT_NAME is a
        two-dimensional array texture, then <params> will contain the
        number of texture layer attached to the attachment point.
        Otherwise, <params> will contain the value zero.

Additions to Chapter 4 of the OpenGL ES Shading Language Specification

    Add the following to the table of basic types in section 4.1:

        sampler2DArrayNV   a handle for accessing a 2D array texture

Additions to Chapter 8 of the OpenGL ES Shading Language Specification

    The built-in texture lookup functions texture2DArrayNV and
    texture2DArrayLodNV are optional, and must be enabled by

        #extension GL_NV_texture_array : enable

    before being used.

    Syntax:

        vec4 texture2DArrayNV(sampler2DArrayNV sampler, vec3 coord [, float bias])
        vec4 texture2DArrayLodNV(sampler2DArrayNV sampler, vec3 coord, float lod)

    Description:

    Use the first two elements (coord.s, coord.t) of texture
    coordinate coord to do a texture lookup in the layer indicated by
    the third coordinate (coord.p) of the 2D texture array currently
    bound to sampler. The layer to access is computed by layer =
    max (0, min(d - 1, floor (coord.p + 0.5)) where 'd' is the depth
    of the texture array.

Interactions with OES_texture_3D

    FRAMEBUFFER_ATTACHMENT_TEXTURE_LAYER_NV is an alias for the
    FRAMEBUFFER_ATTACHMENT_TEXTURE_3D_ZOFFSET_OES token provided in
    OES_texture_3D. This extension generalizes the notion of
    "<zoffset>" to include layers of an array texture.

    If OES_texture_3D is supported, descriptions of TexImage3DNV,
    TexSubImage3DNV, CopyTexSubImage3DNV, CompressedTexImage3DNV, and
    CompressedTexSubImage3DNV should be merged with the *OES
    counterparts added by that extension and should be considered
    equivalents to the OES counterparts.

Interactions with OES_depth_texture

    If OES_depth_texture is not supported, all references to depth
    texture data should be ignored.

Interactions with OES_packed_depth_stencil

    If OES_packed_depth_stencil is not supported, all references to
    stencil texture data should be ignored.

Interactions with OES_texture_npot and NV_texture_npot_2D_mipmap

    If either OES_texture_npot or NV_texture_npot_2D_mipmap are
    supported, ignore INVALID_VALUE error generation by TexImage3DNV
    when level is greater than zero and width or height are non-power-
    of-two.

Interactions with NV_pixel_buffer_object

    If NV_pixel_buffer_object is not supported, ignore all references to
    non-zero unpack buffer objects and corresponding behavior.

Interactions with NV_framebuffer_blit

    If NV_framebuffer_blit is not supported, replace the reference to
    READ_FRAMEBUFFER_BINDING_NV in section 3.7.2 to FRAMEBUFFER_BINDING.

Errors

    None.

New State

                                                 Initial
    Get Value                   Type Get Command  Value  Description
    --------------------------- ---- -----------  -----  -------------------------------------------
    TEXTURE_BINDING_2D_ARRAY_NV  Z+  GetIntegerv    0    texture object bound to TEXTURE_2D_ARRAY_NV

New Implementation Dependent State


                                                 Minimum
    Get Value                   Type Get Command  Value  Description
    --------------------------- ---- -----------  -----  ------------------------------------------
    MAX_ARRAY_TEXTURE_LAYERS_NV  Z+  GetIntegerv    64   maximum number of layers for texture array

Issues

    None

Revision History

    Rev.    Date        Author       Changes
    ----   --------     ---------    -------------------------------------
     4     25 Sep 2012  groth        Minor typographical fixes. fbo_blit interaction
     3     13 Sep 2012  groth        Added missing suffixes. Removed tex_3d
                                     language. Added more interactions.
                                     Clarified enumerated additions.
     2     05 Sep 2012  groth        Addition of lots of detail in chapter 3
                                     and interactions with other extensions
     1     30 Aug 2012  istewart     First revision, based of the
                                     EXT_texture_array specification.

