# EXT_texture_storage_compression

Name

    EXT_texture_storage_compression

Name Strings

    GL_EXT_texture_storage_compression


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

    OpenGL ES Extension #336

Dependencies

    Requires OpenGL ES 3.0.

    This extension is written based on the wording of the OpenGL ES 3.2
    Specification.

Overview

    Applications may wish to take advantage of framebuffer compression. Some
    platforms may support framebuffer compression at fixed bitrates. Such
    compression algorithms generally produce results that are visually lossless,
    but the results are typically not bit-exact when compared to a non-compressed
    result.

    This extension enables applications to opt-in to compression for
    immutable textures.

New Types

    None.

New Procedures and Functions

   void TexStorageAttribs2DEXT(enum target, sizei levels, enum internalformat,
                               sizei width, sizei height, const int *attrib_list);

   void TexStorageAttribs3DEXT(enum target, sizei levels, enum internalformat,
                               sizei width, sizei height, sizei depth, const int *attrib_list);


New Tokens

    New attributes accepted by the <attrib_list> argument of
    TexStorageAttribs2DEXT and TexStorageAttribs3DEXT, and as the <pname>
    argument to GetTexParameter*:
        SURFACE_COMPRESSION_EXT                     0x96C0

    New attributes accepted by the <pname> argument of
    GetInternalformativ:
        NUM_SURFACE_COMPRESSION_FIXED_RATES_EXT     0x8F6E

    Accepted as attribute values for SURFACE_COMPRESSION_EXT by TexStorageAttribs2DEXT
    and TexStorageAttribs3DEXT:
        SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT     0x96C1
        SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT  0x96C2

        SURFACE_COMPRESSION_FIXED_RATE_1BPC_EXT     0x96C4
        SURFACE_COMPRESSION_FIXED_RATE_2BPC_EXT     0x96C5
        SURFACE_COMPRESSION_FIXED_RATE_3BPC_EXT     0x96C6
        SURFACE_COMPRESSION_FIXED_RATE_4BPC_EXT     0x96C7
        SURFACE_COMPRESSION_FIXED_RATE_5BPC_EXT     0x96C8
        SURFACE_COMPRESSION_FIXED_RATE_6BPC_EXT     0x96C9
        SURFACE_COMPRESSION_FIXED_RATE_7BPC_EXT     0x96CA
        SURFACE_COMPRESSION_FIXED_RATE_8BPC_EXT     0x96CB
        SURFACE_COMPRESSION_FIXED_RATE_9BPC_EXT     0x96CC
        SURFACE_COMPRESSION_FIXED_RATE_10BPC_EXT    0x96CD
        SURFACE_COMPRESSION_FIXED_RATE_11BPC_EXT    0x96CE
        SURFACE_COMPRESSION_FIXED_RATE_12BPC_EXT    0x96CF

Additions to Chapter 8 of the OpenGL ES 3.2 Specification (Textures and
Samplers)

    Add to 8.11.2, "Texture Parameter Queries"

    Add SURFACE_COMPRESSION_EXT to the values accepted by <pname>.

    "Querying <pname> SURFACE_COMPRESSION_EXT returns the fixed-rate
    compression rate that was actually applied to the texture."

    Add to 8.18, "Immutable-Format Texture Images" section:

    The command

    void TexStorageAttribs2DEXT(enum target, sizei levels, enum internalformat,
                                sizei width, sizei height, const int *attrib_list);

    behaves identically to TexStorage2D, except that additional flags can
    specified in <attrib_list>.

    Similarly, the command

    void TexStorageAttribs3DEXT(enum target, sizei levels, enum internalformat,
                                sizei width, sizei height, sizei depth, const int *attrib_list);

    behaves identically to TexStorage3D, except that additional flags can be
    specified in <attrib_list>.

    For TexStorageAttribs2DEXT and TexStorageAttribs3DEXT, <attrib_list>
    specifies a list of attributes for the texture.

    All attribute names in <attrib_list> are immediately followed by the
    corresponding value. The list is terminated with GL_NONE. If an
    attribute is not specified in <attrib_list>, then the default value
    is used.

    <attrib_list> may be NULL or empty (first attribute is GL_NONE), in which
    case all attributes assume their default value as described below.

    Attributes that can be specified in <attrib_list> include
    SURFACE_COMPRESSION_EXT.

    SURFACE_COMPRESSION_EXT specifies if fixed-rate compression can be
    enabled for the texture.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT then fixed-rate
    compression is disabled.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_DEFAULT_EXT then the
    implementation may enable compression at a default, implementation-defined,
    rate.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_1BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 1 bit and less than 2 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_2BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 2 bits and less than 3 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_3BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 3 bits and less than 4 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_4BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 4 bits and less than 5 bit per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_5BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 5 bits and less than 6 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_6BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 6 bits and less than 7 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_7BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 7 bits and less than 8 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_8BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 8 bits and less than 9 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_9BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 9 bits and less than 10 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_10BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 10 bits and less than 11 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_11BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 11 bits and less than 12 bits per component.
    If its value is SURFACE_COMPRESSION_FIXED_RATE_12BPC_EXT, then the
    implementation may enable fixed-rate compression with a bitrate of at
    least 12 bits per component.

    The default value of SURFACE_COMPRESSION_EXT is
    SURFACE_COMPRESSION_FIXED_RATE_NONE_EXT.

    If <attrib_list> is neither NULL nor a value described above, the error
    INVALID_VALUE is generated.

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

Additions to Chapter 20 of the OpenGL ES 3.2 Specification (Context State Queries)

   Add to the end of section 20.3.1 Internal Format Query Parameters describing
   supported values for <pname>, their meanings, and their possible return values
   for GetInternalformativ:

   NUM_SURFACE_COMPRESSION_FIXED_RATES_EXT: The number of fixed-rate compression
   rates that would be returned by querying SURFACE_COMPRESSION_EXT is returned
   in <params>. If <internalformat> does not support any fixed-rate compression,
   zero is returned.

   SURFACE_COMPRESSION_EXT: The fixed-rate compression rates supported for
   <internalformat> and <target> are written into <params>, in order of
   ascending bitrates.

Issues

    1. Do we need to specify compression rates in this extension?

    Resolved. Yes. The GL implementation allocates these resources and need to
    know the compression ratio.

Revision History
    Version 1, 2021/11/15
      - Internal revisions
