# EXT_texture_storage

Name

    EXT_texture_storage

Name Strings

    GL_EXT_texture_storage

Contact

    Bruce Merry (bmerry 'at' gmail.com)
    Ian Romanick, Intel (ian.d.romanick 'at' intel.com)

Contributors

    Jeremy Sandmel, Apple
    Bruce Merry, ARM
    Tom Olson, ARM
    Benji Bowman, Imagination Technologies
    Ian Romanick, Intel
    Jeff Bolz, NVIDIA
    Pat Brown, NVIDIA
    Maurice Ribble, Qualcomm
    Lingjun Chen, Qualcomm
    Daniel Koch, Transgaming Inc
    Benj Lipchak, Apple

Status

    Complete. 

Version

    Last Modified Date: 22 September, 2021
    Author Revision: 29

Number

    OpenGL ES Extension #108
    OpenGL Extension #556

Dependencies

    OpenGL ES 1.0, OpenGL ES 2.0 or OpenGL 1.2 is required.

    OES_texture_npot, OES_texture_cube_map, OES_texture_3D,
    OES_depth_texture, OES_packed_depth_stencil,
    OES_compressed_paletted_texture, OES_texture_float, OES_texture_half_float
    EXT_texture_type_2_10_10_10_REV, EXT_texture_format_BGRA8888,
    EXT_texture3D, EXT_texture_rg, APPLE_texture_2D_limited_npot,
    APPLE_rgb_422, APPLE_texture_format_BGRA8888, 
    ARB_texture_cube_map, ARB_texture_cube_map_array,
    ARB_texture_rectangle, SGIS_generate_mipmap,
    EXT_direct_state_access, OES_EGL_image, WGL_ARB_render_texture,
    GLX_EXT_texture_from_pixmap, and core specifications that
    incorporate these extensions affect the definition of this
    extension.

    This extension is written against the OpenGL 3.2 Core Profile
    specification.

Overview

    The texture image specification commands in OpenGL allow each level
    to be separately specified with different sizes, formats, types and
    so on, and only imposes consistency checks at draw time. This adds
    overhead for implementations.

    This extension provides a mechanism for specifying the entire
    structure of a texture in a single call, allowing certain
    consistency checks and memory allocations to be done up front. Once
    specified, the format and dimensions of the image array become
    immutable, to simplify completeness checks in the implementation.

    When using this extension, it is no longer possible to supply texture
    data using TexImage*. Instead, data can be uploaded using TexSubImage*,
    or produced by other means (such as render-to-texture, mipmap generation,
    or rendering to a sibling EGLImage).

    This extension has complicated interactions with other extensions.
    The goal of most of these interactions is to ensure that a texture
    is always mipmap complete (and cube complete for cubemap textures).

IP Status

    No known IP claims

New Procedures and Functions

    void TexStorage1DEXT(enum target, sizei levels,
                         enum internalformat,
                         sizei width);

    void TexStorage2DEXT(enum target, sizei levels,
                         enum internalformat,
                         sizei width, sizei height);

    void TexStorage3DEXT(enum target, sizei levels,
                         enum internalformat,
                         sizei width, sizei height, sizei depth);

    void TextureStorage1DEXT(uint texture, enum target, sizei levels,
                         enum internalformat,
                         sizei width);

    void TextureStorage2DEXT(uint texture, enum target, sizei levels,
                         enum internalformat,
                         sizei width, sizei height);

    void TextureStorage3DEXT(uint texture, enum target, sizei levels,
                         enum internalformat,
                         sizei width, sizei height, sizei depth);

New Types

    None

New Tokens

    Accepted by the <value> parameter of GetTexParameter{if}v:

        TEXTURE_IMMUTABLE_FORMAT_EXT   0x912F

    Accepted by the <internalformat> parameter of TexStorage* when
    implemented on OpenGL ES:

        ALPHA8_EXT                     0x803C  /* reuse tokens from EXT_texture */
        LUMINANCE8_EXT                 0x8040
        LUMINANCE8_ALPHA8_EXT          0x8045

      (if OES_texture_float is supported)
        RGBA32F_EXT                    0x8814  /* reuse tokens from ARB_texture_float */
        RGB32F_EXT                     0x8815
        ALPHA32F_EXT                   0x8816
        LUMINANCE32F_EXT               0x8818
        LUMINANCE_ALPHA32F_EXT         0x8819

      (if OES_texture_half_float is supported)
        RGBA16F_EXT                    0x881A  /* reuse tokens from ARB_texture_float */
        RGB16F_EXT                     0x881B
        ALPHA16F_EXT                   0x881C
        LUMINANCE16F_EXT               0x881E
        LUMINANCE_ALPHA16F_EXT         0x881F

      (if EXT_texture_type_2_10_10_10_REV is supported)
        RGB10_A2_EXT                   0x8059  /* reuse tokens from EXT_texture */
        RGB10_EXT                      0x8052

      (if EXT_texture_format_BGRA8888 or APPLE_texture_format_BGRA8888 is supported)
        BGRA8_EXT                      0x93A1

      (if EXT_texture_rg is supported)
        R8_EXT                         0x8229  /* reuse tokens from ARB_texture_rg */
        RG8_EXT                        0x822B

      (if EXT_texture_rg and OES_texture_float are supported)
        R32F_EXT                       0x822E  /* reuse tokens from ARB_texture_rg */
        RG32F_EXT                      0x8230

      (if EXT_texture_rg and OES_texture_half_float are supported)
        R16F_EXT                       0x822D  /* reuse tokens from ARB_texture_g */
        RG16F_EXT                      0x822F

      (APPLE_rgb_422 is supported)
        RGB_RAW_422_APPLE              0x8A51


Additions to Chapter 2 of the OpenGL 3.2 Core Profile Specification
(OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL 3.2 Core Profile Specification
(Rasterization)

    After section 3.8.1 (Texture Image Specification) add a new
    subsection called "Immutable-format texture images":

    "An alterative set of commands is provided for specifying the
    properties of all levels of a texture at once. Once a texture is
    specified with such a command, the format and dimensions of all
    levels becomes immutable, unless it is a proxy texture (since
    otherwise it would no longer be possible to use the proxy). The
    contents of the images and the parameters can still be modified.
    Such a texture is referred to as an "immutable-format" texture. The
    immutability status of a texture can be determined by calling
    GetTexParameter with <pname> TEXTURE_IMMUTABLE_FORMAT_EXT.

    Each of the commands below is described by pseudo-code which
    indicates the effect on the dimensions and format of the texture.
    For all of the commands, the following apply in addition to the
    pseudo-code:

    - If the default texture object is bound to <target>, an
      INVALID_OPERATION error is generated.
    - If executing the pseudo-code would lead to an error, the error is
      generated and the command will have no effect.
    - Any existing levels that are not replaced are reset to their
      initial state.
    - If <width>, <height>, <depth> or <levels> is less than 1, the
      error INVALID_VALUE is generated.
    - The pixel unpack buffer should be considered to be zero i.e.,
      the image contents are unspecified.
    - Since no pixel data are provided, the <format> and <type> values
      used in the pseudo-code are irrelevant; they can be considered to
      be any values that are legal to use with <internalformat>.
    - If the command is successful, TEXTURE_IMMUTABLE_FORMAT_EXT becomes
      TRUE.
    - If <internalformat> is a specific compressed texture format, then
      references to TexImage* should be replaced by CompressedTexImage*,
      with <format>, <type> and <data> replaced by any valid <imageSize> and
      <data>. If there is no <imageSize> for which this command would have
      been valid, an INVALID_OPERATION error is generated [fn: This
      condition is not required for OpenGL, but is necessary for OpenGL
      ES which does not support on-the-fly compression.]
    - If <internalformat> is one of the internal formats listed in table
      3.11, an INVALID_ENUM error is generated. [fn: The corresponding table
      in OpenGL ES 2.0 is table 3.8.]

    The command

        void TexStorage1DEXT(enum target, sizei levels,
                             enum internalformat,
                             sizei width);

    specifies all the levels of a one-dimensional texture (or proxy) at
    the same time. It is described by the pseudo-code below:

        for (i = 0; i < levels; i++)
        {
            TexImage1D(target, i, internalformat, width, 0,
                       format, type, NULL);
            width = max(1, floor(width / 2));
        }

    If <target> is not TEXTURE_1D or PROXY_TEXTURE_1D then INVALID_ENUM
    is generated. If <levels> is greater than floor(log_2(width)) + 1
    then INVALID_OPERATION is generated.

    The command

        void TexStorage2DEXT(enum target, sizei levels,
                             enum internalformat,
                             sizei width, sizei height);

    specifies all the levels of a two-dimensional, cube-map,
    one-dimension array or rectangle texture (or proxy) at the same
    time. The pseudo-code depends on the <target>:

    [PROXY_]TEXTURE_2D, [PROXY_]TEXTURE_RECTANGLE or
    PROXY_TEXTURE_CUBE_MAP:

        for (i = 0; i < levels; i++)
        {
            TexImage2D(target, i, internalformat, width, height, 0,
                       format, type, NULL);
            width = max(1, floor(width / 2));
            height = max(1, floor(height / 2));
        }

    TEXTURE_CUBE_MAP:

        for (i = 0; i < levels; i++)
        {
            for face in (+X, -X, +Y, -Y, +Z, -Z)
            {
                TexImage2D(face, i, internalformat, width, height, 0,
                           format, type, NULL);
            }
            width = max(1, floor(width / 2));
            height = max(1, floor(height / 2));
        }

    [PROXY_]TEXTURE_1D_ARRAY:

        for (i = 0; i < levels; i++)
        {
            TexImage2D(target, i, internalformat, width, height, 0,
                       format, type, NULL);
            width = max(1, floor(width / 2));
        }

    If <target> is not one of those listed above, the error INVALID_ENUM
    is generated.

    The error INVALID_OPERATION is generated if any of the following
    conditions hold:
    - <target> is [PROXY_]TEXTURE_1D_ARRAY and <levels> is greater than
      floor(log_2(width)) + 1
    - <target> is not [PROXY_]TEXTURE_1D_ARRAY and <levels> is greater
    than floor(log_2(max(width, height))) + 1

    The command

        void TexStorage3DEXT(enum target, sizei levels, enum internalformat,
                             sizei width, sizei height, sizei depth);

    specifies all the levels of a three-dimensional, two-dimensional
    array texture, or cube-map array texture (or proxy). The pseudo-code
    depends on <target>:

    [PROXY_]TEXTURE_3D:

        for (i = 0; i < levels; i++)
        {
            TexImage3D(target, i, internalformat, width, height, depth, 0,
                       format, type, NULL);
            width = max(1, floor(width / 2));
            height = max(1, floor(height / 2));
            depth = max(1, floor(depth / 2));
        }

    [PROXY_]TEXTURE_2D_ARRAY, [PROXY_]TEXTURE_CUBE_MAP_ARRAY_ARB:

        for (i = 0; i < levels; i++)
        {
            TexImage3D(target, i, internalformat, width, height, depth, 0,
                       format, type, NULL);
            width = max(1, floor(width / 2));
            height = max(1, floor(height / 2));
        }

    If <target> is not one of those listed above, the error INVALID_ENUM
    is generated.

    The error INVALID_OPERATION is generated if any of the following
    conditions hold:
    - <target> is [PROXY_]TEXTURE_3D and <levels> is greater than
      floor(log_2(max(width, height, depth))) + 1
    - <target> is [PROXY_]TEXTURE_2D_ARRAY or
      [PROXY_]TEXTURE_CUBE_MAP_ARRAY_EXT and <levels> is greater than
      floor(log_2(max(width, height))) + 1

    After a successful call to any TexStorage* command with a non-proxy
    target, the value of TEXTURE_IMMUTABLE_FORMAT_EXT for this texture
    object is set to TRUE, and no further changes to the dimensions or
    format of the texture object may be made. Other commands may only
    alter the texel values and texture parameters. Using any of the
    following commands with the same texture will result in the error
    INVALID_OPERATION being generated, even if it does not affect the
    dimensions or format:

        - TexImage*
        - CompressedTexImage*
        - CopyTexImage*
        - TexStorage*

    The TextureStorage* commands operate identically to the
    corresponding command where "Texture" is substituted for "Tex"
    except, rather than updating the current bound texture for the
    texture unit indicated by the current active texture state and the
    target parameter, these "Texture" commands update the texture object
    named by the initial texture parameter. The error INVALID_VALUE
    is generated if <texture> is zero.
    "

    In section 3.8.6 (Texture Parameters), after the sentence

    "In the remainder of section 3.8, denote by lod_min, lod_max,
    level_base, and level_max the values of the texture parameters
    TEXTURE_MIN_LOD, TEXTURE_MAX_LOD, TEXTURE_BASE_LEVEL, and
    TEXTURE_MAX_LEVEL respectively."

    add

    "However, if TEXTURE_IMMUTABLE_FORMAT_EXT is
    TRUE, then level_base is clamped to the range [0, <levels> - 1] and
    level_max is then clamped to the range [level_base, <levels> - 1],
    where <levels> is the parameter passed the call to TexStorage* for
    the texture object.

    In section 3.8.9 (Rendering feedback loops) replace all references
    to TEXTURE_BASE_LEVEL by level_base.

    In section 3.8.9 (Mipmapping), replace the paragraph starting "Each
    array in a mipmap is defined..." by

    "Each array in a mipmap is defined using TexImage3D, TexImage2D,
    CopyTexImage2D, TexImage1D, CopyTexImage1D, or by functions that are
    defined in terms of these functions. Level-of-detail numbers proceed
    from level_base for the original texel array through the maximum
    level p, with each unit increase indicating an array of half the
    dimensions of the previous one (rounded down to the next integer if
    fractional) as already described. For immutable-format textures,
    p is one less than the <levels> parameter passed to TexStorage*;
    otherwise p = floor(log_2(maxsize)) + level_base.  All arrays from
    level_base through q = min(p, level_max) must be defined, as
    discussed in section 3.8.12."

    In section 3.8.12 (Texture Completeness), modify the last sentence
    to avoid refering to level_base and level_max:

    "An implementation may allow a texture image array of level 1 or
    greater to be created only if a mipmap complete set of image arrays
    consistent with the requested array can be supported where the
    values of TEXTURE_BASE_LEVEL and TEXTURE_MAX_LEVEL are 0 and 1000
    respectively."

    Modify section 3.8.13 (Texture State and Proxy State) to add the new
    state:

    "Each set consists of ..., and a boolean flag indicating whether the
    format and dimensions of the texture are immutable."

    Add
    "The initial value of TEXTURE_IMMUTABLE_FORMAT_EXT is FALSE."

Additions to Chapter 4 of the OpenGL 3.2 Core Profile Specification
(Per-Fragment Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL 3.2 Compatibility Profile Specification
(Special Functions)

    In section 5.4.1 (Commands Not Usable in Display Lists), add
    TexStorage* to the list of commands that cannot be used.

Additions to Chapter 6 of the OpenGL 3.2 Core Profile Specification
(State and State Requests)

    Replace the following statement in 6.1.3 (Enumerated Queries):

    "<value> must be one of the symbolic values in table 3.10."

    with

    "<value> must be TEXTURE_IMMUTABLE_FORMAT_EXT or one of the symbolic
    values in table 3.22."

Additions to the AGL/EGL/GLX/WGL Specifications

    None

Additions to OES_compressed_ETC1_RGB8_texture

    Add the following to the additions to Chapter 3:

    "Since ETC1 images are easily edited along 4x4 texel boundaries, the
    limitations on CompressedTexSubImage2D are relaxed.
    CompressedTexSubImage2D will result in an INVALID_OPERATION error
    only if one of the following conditions occurs:

        * <width> is not a multiple of four, and <width> plus <xoffset> is not
          equal to the texture width;

        * <height> is not a multiple of four, and <height> plus <yoffset> is
          not equal to the texture height; or

        * <xoffset> or <yoffset> is not a multiple of four.

    Remove CompressedTexSubImage2D from this error:

    "INVALID_OPERATION is generated by CompressedTexSubImage2D,
    TexSubImage2D, or CopyTexSubImage2D if the texture image <level>
    bound to <target> has internal format ETC1_RGB8_OES."

    Add the following error:

    "INVALID_OPERATION is generated by CompressedTexSubImage2D
    if the region to be modified is not aligned to block boundaries
    (refer to the extension text for details)."

Additions to AMD_compressed_ATC_texture and AMD_compressed_3DC_texture:

    Apply the same changes as for OES_compressed_ETC1_RGB8_texture
    above, substituting the appropriate internal format tokens from
    these extensions.

Dependencies on EXT_direct_state_access

    If EXT_direct_state_access is not present, references to
    TextureStorage* should be ignored.

Dependencies on OpenGL ES

    On OpenGL ES without extensions introducing TEXTURE_MAX_LEVEL,
    mipmapped textures specified with TexStorage are required to have a
    full set of mipmaps. If TEXTURE_MAX_LEVEL is not supported, this
    extension is modified as follows:

    - Where an upper bound is placed on <levels> in this extension (i.e.
      the maximum number of mipmap levels for a texture of the given
      target and dimensions), an INVALID_OPERATION error is generated if
      <levels> is neither 1 nor this upper bound.
    - q (the effective maximum number of levels) is redefined to clamp
      to the number of levels present in immutable-format textures.

    OpenGL ES does not accept sized internal formats (e.g., RGBA8) and
    instead derives an internal format from the <format> and <type>
    parameters of TexImage2D. Since TexStorage* does not specify texel
    data, the API doesn't include <format> and <type> parameters.
    On an OpenGL ES implementation, the values in the <internalformat>
    column in the tables below are accepted as <internalformat>
    parameters, and base internal formats are not accepted. The
    TexImage* calls in the TexStorage* pseudocode are modified so that
    the <internalformat>, <format> and <type> parameters are
    taken from the <format>, <format> and <type> columns (respectively)
    in the tables below, according to the <internalformat>
    specified in the TexStorage* command.

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        RGB565                 RGB                UNSIGNED_SHORT_5_6_5
        RGBA4                  RGBA               UNSIGNED_SHORT_4_4_4_4
        RGB5_A1                RGBA               UNSIGNED_SHORT_5_5_5_1
        RGB8_OES               RGB                UNSIGNED_BYTE
        RGBA8_OES              RGBA               UNSIGNED_BYTE
        LUMINANCE8_ALPHA8_EXT  LUMINANCE_ALPHA    UNSIGNED_BYTE
        LUMINANCE8_EXT         LUMINANCE          UNSIGNED_BYTE
        ALPHA8_EXT             ALPHA              UNSIGNED_BYTE

    If OES_depth_texture is supported:

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        DEPTH_COMPONENT16_OES  DEPTH_COMPONENT    UNSIGNED_SHORT
        DEPTH_COMPONENT32_OES  DEPTH_COMPONENT    UNSIGNED_INT

    If OES_packed_depth_stencil is supported:

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        DEPTH24_STENCIL8_OES   DEPTH_STENCIL_OES  UNSIGNED_INT

    If OES_texture_float is supported:

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        RGBA32F_EXT            RGBA               FLOAT
        RGB32F_EXT             RGB                FLOAT
        LUMINANCE_ALPHA32F_EXT LUMINANCE_ALPHA    FLOAT
        LUMINANCE32F_EXT       LUMINANCE          FLOAT 
        ALPHA32F_EXT           ALPHA              FLOAT

    If OES_texture_half_float is supported:

        <internalformat>       <format>           <type>
        ----------------       --------           ------
        RGBA16F_EXT            RGBA               HALF_FLOAT_OES
        RGB16F_EXT             RGB                HALF_FLOAT_OES
        LUMINANCE_ALPHA16F_EXT LUMINANCE_ALPHA    HALF_FLOAT_OES
        LUMINANCE16F_EXT       LUMINANCE          HALF_FLOAT_OES 
        ALPHA16F_EXT           ALPHA              HALF_FLOAT_OES

    If EXT_texture_type_2_10_10_10_REV is supported:

        <internalformat>    <format>   <type>
        ----------------    --------   ------
        RGB10_A2_EXT        RGBA       UNSIGNED_INT_2_10_10_10_REV_EXT
        RGB10_EXT           RGB        UNSIGNED_INT_2_10_10_10_REV_EXT

    If EXT_texture_format_BGRA8888 or APPLE_texture_format_BGRA8888 is supported:

        <internalformat>    <format>   <type>
        ----------------    --------   ------
        BGRA8_EXT           BGRA_EXT   UNSIGNED_BYTE

    If EXT_texture_rg is supported:

        <internalformat>    <format>   <type>
        ----------------    --------   ------
        R8_EXT              RED_EXT    UNSIGNED_BYTE
        RG8_EXT             RG_EXT     UNSIGNED_BYTE

    If EXT_texture_rg and OES_texture_float are supported:

        <internalformat>    <format>   <type>
        ----------------    --------   ------
        R32F_EXT            RED_EXT    FLOAT
        RG32F_EXT           RG_EXT     FLOAT

    If EXT_texture_rg and OES_texture_half_float are supported:

        <internalformat>    <format>   <type>
        ----------------    --------   ------
        R16F_EXT            RED_EXT    HALF_FLOAT_OES
        RG16F_EXT           RG_EXT     HALF_FLOAT_OES

    If APPLE_rgb_422 is supported:

        <internalformat>    <format>       <type>
        ----------------    --------       ------
        RGB_RAW_422_APPLE   RGB_422_APPLE  UNSIGNED_SHORT_8_8_APPLE


Dependencies on texture targets

    If a particular texture target is not supported by the
    implementation, passing it as a <target> to TexStorage* will
    generate an INVALID_ENUM error. If as a result, any of the commands
    defined in this extension would no longer have any valid <target>,
    all references to the command should be ignored.

    In particular, note that OpenGL ES 1.x/2.0 do not have proxy textures,
    1D textures, or 3D textures, and thus only the TexStorage2DEXT
    entry point is required. If OES_texture_3D is supported, the
    TexStorage3DEXT entry point is also required.

Dependencies on OES_texture_npot

    If OpenGL ES 2.0 or APPLE_texture_2D_limited_npot is present but
    OES_texture_npot is not present, then INVALID_OPERATION is
    generated by TexStorage* and TexStorage3DEXT if <levels> is
    not one and <width>, <height> or <depth> is not a power of
    two.

Dependencies on WGL_ARB_render_texture, GLX_EXT_texture_from_pixmap, EGL
1.4 and GL_OES_EGL_image

    The commands eglBindTexImage, wglBindTexImageARB, glXBindTexImageEXT or
    EGLImageTargetTexture2DOES are not permitted on an immutable-format
    texture.
    They will generate the following errors:
      - EGLImageTargetTexture2DOES: INVALID_OPERATION
      - eglBindTexImage: EGL_BAD_MATCH
      - wglBindTexImage: ERROR_INVALID_OPERATION
      - glXBindTexImageEXT: BadMatch

Dependencies on OES_compressed_paletted_texture

    The compressed texture formats exposed by
    OES_compressed_paletted_texture are not supported by TexStorage*.
    Passing one of these tokens to TexStorage* will generate an
    INVALID_ENUM error.

Dependencies on APPLE_rgb_422

    UNSIGNED_SHORT_8_8_APPLE is implied as the <type> when TexStorage2DEXT
    is called with <internalformat> RGB_RAW_422_APPLE.  Subsequently supplying 
    UNSIGNED_SHORT_8_8_REV_APPLE as the <type> to a TexSubImage2D updating
    such an immutable texture will generate an INVALID_OPERATION error.

Errors

    Note that dependencies above modify the errors.

    If TexStorage* is called with a <width>, <height>, <depth> or
    <levels> parameter that is less than one, then the error
    INVALID_VALUE is generated.

    If the <target> parameter to TexStorage1DEXT is not
    [PROXY_]TEXTURE_1D, then the error INVALID_ENUM is generated.

    If the <target> parameter to TexStorage2DEXT is not
    [PROXY_]TEXTURE_2D, [PROXY_]TEXTURE_CUBE_MAP,
    [PROXY_]TEXTURE_RECTANGLE or [PROXY_]TEXTURE_1D_ARRAY, then the
    error INVALID_ENUM is generated.

    If the <target> parameter to TexStorage3DEXT is not
    [PROXY_]TEXTURE_3D, [PROXY_]TEXTURE_2D_ARRAY or
    [PROXY_]TEXTURE_CUBE_MAP_ARRAY then the error INVALID_ENUM is
    generated.

    If the <levels> parameter to TexStorage* is greater than the
    <target>-specific value listed below then the error
    INVALID_OPERATION is generated:
        [PROXY_]TEXTURE_{1D,1D_ARRAY}:
            floor(log_2(width)) + 1
        [PROXY_]TEXTURE_{2D,2D_ARRAY,CUBE_MAP,CUBE_MAP_ARRAY}:
            floor(log_2(max(width, height))) + 1
        [PROXY_]TEXTURE_3D:
            floor(log_2(max(width, height, depth))) + 1
        [PROXY_]TEXTURE_RECTANGLE:
            1

    If the default texture object is bound to the <target> passed to
    TexStorage*, then the error INVALID_OPERATION is generated.

    If the <target> parameter to TextureStorage* does not match the
    dimensionality of <texture>, then the error INVALID_OPERATION is
    generated.

    If the <texture> parameter to TextureStorage* is zero, then the
    INVALID_VALUE is generated.

    If any pseudo-code listed in this extension would generate an error,
    then that error is generated.

    Calling any of the following functions on a texture for which
    TEXTURE_IMMUTABLE_FORMAT_EXT is TRUE will generate an
    INVALID_OPERATION error:
        - TexImage*
        - CompressedTexImage*
        - CopyTexImage*

New State

    Additions to Table 6.8 Textures (state per texture object)

                                                               Initial
        Get Value                      Type   Get Command      Value    Description                Sec.
        ---------                      ----   -----------      -------  -----------                ----
        TEXTURE_IMMUTABLE_FORMAT_EXT   B      GetTexParameter  FALSE    Size and format immutable  2.6

New Implementation Dependent State

    None

Issues

    1. What should this extension be called?

    RESOLVED: EXT_texture_storage is chosen for consistency with the
    glRenderbufferStorage entry point.

    2. Should TexStorage* accept a border parameter?

    RESOLVED: no.

    DISCUSSION: Currently it does not, since borders are a deprecated
    feature which is not supported by all hardware. Users of the
    compatibility profile can continue to use the existing texture
    specification functions, but there is an argument that users of
    compatibility profile may also want to use this extension.

    3. What is the correct error when <levels> specifies a partial
    mipmap pyramid for OpenGL ES?

    RESOLVED: INVALID_OPERATION, since it is an interaction between
    parameters rather than a single value being invalid. It also makes
    sense to relax this condition for desktop GL where it makes sense to
    use a truncated pyramid with TEXTURE_MAX_LEVEL.

    4. Should use of these entry-points make the metadata (format and
    dimensions) immutable?

    RESOLVED: Yes.

    DISCUSSION: The benefits of knowing metadata can't change will
    probably outweigh the extra cost of checking the
    TEXTURE_IMMUTABLE_FORMAT_EXT flag on each texture specification
    call.

    5. Should it be legal to completely replace the texture using a new call
    to TexStorage*?

    RESOLVED. It will not be allowed.

    DISCUSSION: This is useful to invalidate all levels of a texture.
    Allowing the metadata to be changed here seems easier than trying to
    define a portable definition of what it means to change the metadata
    (e.g. what if you used an unsized internal format the first time and
    the corresponding sized internal format the second time, or vice
    versa)?

    However, while this is largely similar to deleting the old texture
    object and replacing it with a new one, it does lose some of the
    advantages of immutability. Specifically, because doing so does not
    reset bindings, it doesn't allow a migration path to an API that
    validates the texture format at bind time.

    6. Should it be legal to use TexImage* after TexStorage* if it doesn't
    affect the metadata?

    RESOLVED: No.

    DISCUSSION: A potential use case is to allow a single level of a
    texture to be invalidated using a NULL pointer. However, as noted
    above it is non-trivial to determine what constitutes a change.

    7. How does this extension interact with APPLE_texture_2D_limited_npot?

    RESOLVED. APPLE_texture_2D_limited_npot is equivalent to the NPOT
    support in OpenGL ES 2.0.

    8. Should this extension be written to work with desktop OpenGL?

    RESOLVED: Yes.

    DISCUSSION: There has been been interest and it will future-proof it
    against further additions to OpenGL ES.

    9. Which texture targets should be supported?

    RESOLVED. All targets except multisample and buffer textures are
    supported.

    Initially all targets except TEXTURE_BUFFER were supported. It was
    noted that the entrypoints for multisample targets added no useful
    functionality, since multisample textures have no completeness
    checks beyond being non-empty.

    Rectangle textures have completeness checks to prevent filtering of
    integer textures. However, since we decided to only force mipmap
    completeness, this becomes less useful.

    10. Should this extension support proxy textures?

    RESOLVED: Yes.

    DISCUSSION: It should be orthogonal.

    11. Are the <format> and <type> parameters necessary?

    RESOLVED. No, they will be removed.

    DISCUSSION: For OpenGL ES the type parameter was necessary to
    determine the precision of the texture, but this can be solved by
    having these functions accept sized internal formats (which are
    already accepted by renderbuffers).

    12. Should it be legal to make the default texture (id 0)
    immutable-format?

    RESOLVED: No.

    DISCUSSION: This would make it impossible to restore the context to
    it's default state, which is deemed undesirable. There is no good
    reason not to use named texture objects.

    13. Should we try to guarantee that textures made through this path
    will always be complete?

    RESOLVED: It should be guaranteed that the texture will be mipmap
    complete.

    DISCUSSION: Future separation between images and samplers will still
    allow users to create combinations that are invalid, but
    constraining the simple cases will make these APIs easier to use for
    beginners.

    14. Should these functions use a EXT_direct_state_access approach to
    specifying the texture objects?

    UNRESOLVED.

    DISCUSSION: as a standalone extension, no DSA-like functions will be
    added. However, interactions with EXT_direct_state_access and
    ARB_direct_state_access need to be resolved.

    15. Should these functions accept generic compressed formats?

    RESOLVED: Yes. Note that the spec language will need to be modified
    to allow this for ES, since the pseudocode is written in terms of
    TexImage2D, which does not allow compressed texture formats in ES.
    See also issues 23 and 27.

    16. How should completeness be forced when TEXTURE_MAX_LEVEL is not
    present?

    RESOLVED. The maximum level q will be redefined to clamp to the
    highest level available.

    DISCUSSION: A single-level texture can be made complete either by
    making it mipmap complete (by setting TEXTURE_MAX_LEVEL to 0) or by
    turning off mipmapping (by choose an appropriate minification
    filter).

    Some options:

    A: Specify that TexStorage* changes the default minification filter
    for OpenGL ES. This makes it awkward to add TEXTURE_MAX_LEVEL
    support to OpenGL ES later, since switching to match GL would break
    compatibility. The two mechanisms also do not give identical
    results, since the magnification threshold depends on the
    minification filter.

    B: Specify that the texture behaves as though TEXTURE_MAX_LEVEL were
    zero. To specify this properly probably requires fairly intrusive
    changes to the OpenGL ES full specification to add back all the
    language relating to the max level. It also does not solve the
    similar problem of what to do with NPOT textures; and it may have
    hardware impacts due to the change in the min/mag crossover.

    C: Specify that TexStorage* changes the default minification filter
    for all implementations when a single-level texture is specified.
    This may be slightly counter-intuitive to desktop GL users, but will
    give consistent behaviour across variants of GL and avoids changing
    the functional behaviour of this extension based on the presence or
    absence of some other feature.

    Currently B is specified. This has potential hardware implications
    for OpenGL ES because of the effect of the minification filter on
    the min/mag crossover. However, C has potential hardware implications
    for OpenGL due to the separation of texture and sampler state.

    17. How should completeness be forced when only ES2-style NPOT is
    available?

    RESOLVED. It is not worth trying to do this, in light of issue 13.

    Previous revisions of this extension overrode the minification
    filter and wrap modes, but that is no longer the case. Since
    OES_texture_npot removes the caveats on NPOT textures anyway, it
    might not be worth trying to "fix" this.

    18. For OpenGL ES, how do the new sized internal formats interact
    with OES_required_internal_format?

    RESOLVED.

    If OES_required_internal_format is not present, then the
    <internalformat> parameter is intended merely to indicate what the
    corresponding <format> and <type> would have been, had TexImage*
    been used instead. If OES_required_internal_format is present, then
    it is intended that the <internalformat> will be interpreted as if
    it had been passed directly to TexImage*.

    19. Should there be some hinting mechanism to indicate whether data
    is coming immediately or later?

    RESOLVED. No parameter is needed. An extension can be added to provide
    a TexParameter value which is latched at TexStorage time.

    DISCUSSION: Some members felt that this would be useful so that they
    could defer allocation when suitable, particularly if higher-
    resolution images will be streamed in later; or to choose a memory
    type or layout appropriate to the usage. However, implementation
    experience with BufferData is that developers frequently provide
    wrong values and implementations have to guess anyway.

    One option suggested was the <usage> parameter currently passed to
    BufferData. Another option was to set it with TexParameter.

    20. How should this extension interact with
    EGLImageTargetTexture2DOES, eglBindTexImage, glXBindTexImage and
    wglBindTexImage?

    RESOLVED. These functions will not be permitted after glTexStorage*.

    Several options are possible:

    A) Disallow these functions.
    B) Allow them, but have them reset the TEXTURE_IMMUTABLE_FORMAT_EXT
       flag.
    C) Allow them unconditionally.

    C would violate the design principle that the dimensions and format
    of the mipmap array are immutable. B does not so much modify the
    dimension and formats as replace them with an entirely different
    set.

    21. Should there be a single function for specifying 1D, 2D and 3D
    targets?

    RESOLVED. No, we will stick with existing precedent.

    22. Is it possible to use GenerateMipmap with an incomplete mipmap
    pyramid?

    RESOLVED. Yes, because the effective max level is limited to the
    levels that were specified, and so GenerateMipmap does not generate
    any new levels.

    However, to make automatic mipmap generation work, it is necessary
    to redefine p rather than q, since automatic mipmap generation
    ignores the max level.

    23. How should this extension interact with
    OES_compressed_paletted_texture?

    RESOLVED. Paletted textures will not be permitted, and will
    generate INVALID_ENUM.

    DISCUSSION: OES_compressed_paletted_texture supplies all the mipmaps
    in a single function call, with the palette specified once. That's
    incompatible with the upload model in this extension.

    24. How can ETC1 textures be used with this extension?

    RESOLVED. Add language in this extension to allow subregion uploads
    for ETC1.

    DISCUSSION: GL_OES_compressed_ETC1_RGB8_texture doesn't allow
    CompressedTexSubImage*, so it would be impossible to use this
    extension with ETC1. This is seen as an oversight in the ETC1
    extension. While it cannot be fixed in that extension (since it is
    already shipping), this extension can add that capability.

    25. Should any other compressed formats be similarly modified?

    RESOLVED. Yes, AMD_compressed_ATC_texture and
    AMD_compressed_3DC_texture can be modified similarly to ETC1
    (Maurice Ribble indicated that both formats use 4x4 blocks). Desktop
    OpenGL requires that whole-image replacement is supported for any
    compressed texture format, and the OpenGL ES extensions
    EXT_texture_compression_dxt1 and IMG_texture_compression_pvrtc
    already allow whole-image replacement, so it is not necessary to
    modify them to be used with this extension.

    26. Should these commands be permitted in display lists?

    RESOLVED. No.

    DISCUSSION: Display lists are most useful for repeating commands,
    and TexStorage* commands cannot be repeated because the first call
    makes the format immutable.

    27. Should these commands accept unsized internal formats?

    RESOLVED: No, for both OpenGL and OpenGL ES.

    DISCUSSION: normally the <type> parameter to TexImage* can serve as
    a hint to select a sized format (and in OpenGL ES, this is the only
    mechanism available); since TexStorage* does not have a <type>
    parameter, the implementation has no information on which to base a
    decision.

Revision History

    Revision 29, 2021/09/22 (Adam Jackson)
      - Assign OpenGL extension number

    Revision 28, 2013/09/18 (Benj Lipchak)
      - Add interaction with APPLE_texture_format_BGRA8888.
      - Fix interaction with APPLE_rgb_422.

    Revision 27, 2012/07/24 (Benj Lipchak)
      - Add interaction with APPLE_rgb_422.

    Revision 26, 2012/02/29 (Benj Lipchak)
      - Add interaction with EXT_texture_rg.

    Revision 25, 2012/01/19 (bmerry)
      - Clarify that the pixel unpack buffer has no effect.

    Revision 24, 2011/11/11 (dgkoch)
      - Mark complete. Clarify ES clarifications.

    Revision 23, 2011/11/10 (dgkoch)
      - Add GLES clarifcations and interactions with more GLES extensions

    Revision 22, 2011/11/10 (bmerry)
      - Update my contact details

    Revision 21, 2011/07/25 (bmerry)
      - Remove dangling references to MultiTexStorage in Errors section

    Revision 20, 2011/07/21 (bmerry)
      - Remove dangling reference to <samples> in Errors section

    Revision 19, 2011/05/02 (Jon Leech)
      - Assign enum value

    Revision 18, 2011/01/24 (bmerry)
      - Disallow unsized internal formats (oversight in revision 17).

    Revision 17, 2011/01/24 (bmerry)
      - Added and resolved issue 26.
      - Split issue 27 out from issue 15.
      - Disallow TexStorage* in display lists.
      - Use the term "immutable-format" consistently (bug 7281).

    Revision 16, 2010/11/23 (bmerry)
      - Disallowed TexStorage on an immutable-format texture
        (resolves issue 5).
      - Deleted MultiTexStorage* commands (other DSA functions still
        unresolved).
      - Some minor wording changes suggested by Pat Brown (bug 7002).

    Revision 15, 2010/11/09 (bmerry)
      - Reopened issue 5.
      - Reopened issue 14, pending stabilisation of
        ARB_direct_state_access.
      - Marked issue 9 resolved, pending any objections.
      - Fix references to no object being bound (was meant to refer to
        the default object).
      - Adding missing pseudocode for TEXTURE_1D_ARRAY.
      - Corrected TEXTURE_2D_ARRAY -> TEXTURE_1D_ARRAY in error checks.
      - Changed "levels... are removed" to "levels... are reset to their
        init state", since desktop GL has per-level state apart from the
        texels.
      - Miscellaneous wording fixes.

    Revision 14, 2010/09/25 (bmerry)
      - Add issues 24-25 and alterations to
        OES_compressed_ETC1_RGB8_texture, AMD_compressed_ATC_texture and
        AMD_compressed_3DC_texture.

    Revision 13, 2010/09/19 (bmerry)
      - Two typo fixes from Daniel Koch

    Revision 12, 2010/09/18 (bmerry)
      - Changed resolution to issue 20
      - Added and resolved issue 23
      - Added explanation of how to upload data (in overview)
      - Added spec language to implement resolution to issue 15

    Revision 11, 2010/07/21 (bmerry)
      - Resolved issue 16
      - Reopen issue 20
      - Fix some typos

    Revision 10, 2010/07/15 (bmerry)
      - Update some issues to match core text
      - Resolved issue 17

    Revision 9, 2010/05/24 (bmerry)
      - Marked issue 2 as resolved
      - Resolved issue 19 (as no change)
      - Resolved issue 20
      - Add issues 21-22
      - Add in spec language to forbid use on default textures
      - Redefine level_base, level_max to be clamped forms of
        TEXTURE_BASE_LEVEL/TEXTURE_MAX_LEVEL when using immutable
        textures
      - Redefine p to also be clamped to the provided levels for
        immutable textures, to support automatic mipmap generation
      - Removed multisample functions
      - Removed language stating that texture parameters were reset to
        defaults

    Revision 8, 2010/05/18 (bmerry)
      - Added issue about EGLimage
      - Marked issue 14 as resolved

    Revision 7, 2010/05/04 (bmerry)
      - Removed some lingering <format>, <type> parameters to the new
        functions that should have been removed in revision 4
      - Trivial typo fixes

    Revision 6, 2010/02/18 (bmerry)
      - Resolved issues 5, 6 and 18
      - Added MultiTexStorage* functions for DSA interaction
      - Added error for texture-target mismatch in DSA
      - Allowed TexStorage* to be called again

    Revision 5, 2010/01/25 (bmerry)
      - Added to contributors list
      - Require OpenGL 1.2, to simplify interactions with
        TEXTURE_BASE_LEVEL/TEXTURE_MAX_LEVEL and CLAMP_TO_EDGE
      - Change default wrap modes to always be CLAMP_TO_EDGE
      - Change default filters to always be NEAREST
      - Moved language about generating new levels into an interaction,
        since it can only happen on OpenGL ES
      - Added interaction with EXT_direct_state_access
      - Added extra <internalformats> for GL ES when OES_depth_texture,
        OES_packed_depth_stencil and EXT_texture_type_2_10_10_10_REV are
        present.
      - Minor non-functional wording fixes and typos
      - Resolved issue 16
      - Added issues 17-19

    Revision 4, 2010/01/13 (bmerry)
      - Changed suffix from ARM to EXT
      - Added list of contributors
      - Added language to force the texture to always be complete
      - Removed <format> and <type> arguments
      - Added issues 14-16
      - Reopened issue 2
      - Reformatted issues to separate resolution and discussion
      - Resolved issues 1, 9 and 11-13
      - Fixed the max number of levels in a cube map array

    Revision 3, 2009/12/17 (bmerry)
      - Added missing vendor suffix to TEXTURE_IMMUTABLE_FORMAT_ARM
      - Rewritten to against desktop OpenGL
      - Added prototypes for 1D and multisample storage functions
      - Added issues 8-13

    Revision 2, 2009/08/20 (bmerry)
      - Resolved issue 2 (no border parameter)
      - Resolved issue 4 (metadata becomes immutable)
      - Added interaction with OES_texture_cube_map
      - Added error if width != height in a cube map
      - Added issues 5-7

    Revision 1, 2009/05/06 (bmerry)
      - First draft
