# EXT_EGL_image_storage_compression

Name

    EXT_EGL_image_storage_compression

Name Strings

    GL_EXT_EGL_image_storage_compression

Contact

    Jan-Harald Fredriksen (jan-harald.fredriksen 'at' arm.com)

Contributors

    Jan-Harald Fredriksen, Arm
    Lisa Wu, Arm
    Anton Berko, Arm
    Laurie Hedge, Imagination Technologies

Status

    Complete

Version

    Version 1 - November 15, 2021

Number

    OpenGL ES Extension #337

Dependencies

    Requires OpenGL ES 3.0.

    Requires GL_EXT_EGL_image_storage.
    This extension is written based on the wording of the OpenGL ES 3.2
    Specification.

Overview

    Applications may wish to take advantage of framebuffer compression. Some
    platforms may support framebuffer compression at fixed bitrates. Such
    compression algorithms generally produce results that are visually lossless,
    but the results are typically not bit-exact when compared to a non-compressed
    result.

    This extension enables applications to specify a texture by taking all
    properties from a fixed-rate compressed EGLImage.

Glossary

    Please see the EGL_KHR_image specification for a list of terms
    used by this specification.

New Types

    None.

New Procedures and Functions

    None.

New Tokens

    New attributes accepted by the <attrib_list> argument of
    EGLImageTargetTexStorageEXT:
        SURFACE_COMPRESSION_EXT                     0x96C0

    Accepted as attribute values for SURFACE_COMPRESSION_EXT by
    EGLImageTargetTexStorageEXT:
        SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT     0x96C1
        SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT  0x96C2

Additions to Chapter 8 of the OpenGL ES 3.2 Specification (Textures and
Samplers)

    - (8.18, p. 210) Modify the final paragraph of the description of the
      EGLImageTargetTexStorageEXT command:

    <attrib_list> specifies a list of attributes for the texture.

    All attribute names in <attrib_list> are immediately followed by the
    corresponding value. The list is terminated with GL_NONE. If an
    attribute is not specified in <attrib_list>, then the default value
    is used.

    <attrib_list> may be NULL or empty (first attribute is GL_NONE), in which
    case all attributes assume their default value as described below.

    Attributes that can be specified in <attrib_list> include
    SURFACE_COMPRESSION_EXT.

    SURFACE_COMPRESSION_EXT specifies whether fixed-rate compression
    is enabled for <image>.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT, then <image>
    must not use fixed-rate compression.
    In this case, if <image> refers to a compressed image, the error
    INVALID_OPERATION is generated.

    If its value is SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT, then
    <image> may use fixed-rate compression.

    The default value of SURFACE_COMPRESSION_EXT is
    SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT.

    If <attrib_list> is neither NULL nor a value described above,
    the error INVALID_VALUE is generated.

    Fixed-rate compression is done in an implementation-defined manner and may
    be applied at block granularity. In that case, a write to an individual
    texel may modify the value of other texels in the same block.

    Modify section 8.23, "Texture Image Loads and Stores":

    Add to the list of errors for BindImageTexture:

    "An INVALID_VALUE error is generated if the value of
    SURFACE_COMPRESSION_EXT for <texture> is not
    SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT."

    Add to the bullet list of conditions for when image access is considered invalid:

    " * the value of SURFACE_COMPRESSION_EXT for the texture is not
        SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT;"

Issues

    1. Can fixed-rate compressed images be imported using EGLImageTargetTexture2DOES?

    RESOLVED. No. We only allow fixed-rate compression for immutable textures.

    2. Do we need to specify compression rates in this extension?

    RESOLVED. No. The source EGLImage object must have been allocated elsewhere,
    and the compression ratio must have been decided at that time.

Revision History
    Version 1, 2021/11/15
      - Internal revisions
