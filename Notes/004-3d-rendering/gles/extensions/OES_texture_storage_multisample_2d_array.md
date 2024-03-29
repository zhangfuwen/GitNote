# OES_texture_storage_multisample_2d_array

Name

    OES_texture_storage_multisample_2d_array

Name Strings

    GL_OES_texture_storage_multisample_2d_array

Contact

    Nick Hoath, Imagination Technologies Ltd (nick 'dot' hoath 'at' imgtec
    'dot' com)

Contributors

    Jeff Bolz, NVIDIA Corporation (jbolz 'at' nvidia.com)
    Jon Leech (oddhack 'at' sonic.net)
    Graham Sellers (graham.sellers 'at' amd.com)
    Dominik Witczak (dominik.witczak 'at' mobica.com)
    Tobias Hector (tobias.hector 'at' imgtec.com)

Notice

    Copyright (c) 2014-2019 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete.
    Ratified by the Khronos Board of Promoters on 2014/03/14.

Version

    Last Modified Date: January 11, 2019
    Revision: 8

Number

    OpenGL ES Extension #174

Dependencies

    This extension is written against the OpenGL ES 3.1 (2015/04/29)
    specification.

Overview

    This extension provides support for a new type of immutable texture,
    two-dimensional multisample array textures. It depends on functionality
    introduced in OpenGL ES 3.1 to support two-dimensional multisample
    (non-array) textures.

New Procedures and Functions

    void TexStorage3DMultisampleOES(enum target,
                                    sizei samples,
                                    enum internalformat,
                                    sizei width,
                                    sizei height,
                                    sizei depth,
                                    boolean fixedsamplelocations);

New Tokens

    Accepted by the <target> parameter of BindTexture,
    TexStorage3DMultisampleOES, GetInternalformativ, TexParameter{if}*,
    GetTexParameter{if}v and GetTexLevelParameter{if}v. Also, the texture
    object indicated by the <texture> argument to FramebufferTextureLayer
    can be TEXTURE_2D_MULTISAMPLE_ARRAY_OES

        TEXTURE_2D_MULTISAMPLE_ARRAY_OES                0x9102

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv,
    and GetFloatv:

        TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY_OES        0x9105

    Returned by the <type> parameter of GetActiveUniform:

        SAMPLER_2D_MULTISAMPLE_ARRAY_OES                0x910B
        INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES            0x910C
        UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES   0x910D


Additions to Chapter 7 of the OpenGL ES 3.1 Specification (Programs and Shaders)

    Add to table 7.3 "OpenGL ES Shading Language type tokens" on p. 86:

    Type Name Token                               Keyword           Attrib   Xfb    Buffer
    --------------------------------------------- ----------------- ------- ------- -------
    SAMPLER_2D_MULTISAMPLE_ARRAY_OES              sampler2DMSArray  (empty) (empty) (empty)
    INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES          isampler2DMSArray (empty) (empty) (empty)
    UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES usampler2DMSArray (empty) (empty) (empty)


Additions to Chapter 8 of the OpenGL ES 3.1 Specification (Textures and
Samplers)

    Modify Section 8.1, Texture Objects

    Modify first paragraph of the section on p. 130, adding 2Dms array
    textures:

    Textures in GL are represented by ... The default texture object is
    bound to each of ... and TEXTURE_2D_MULTISAMPLE_ARRAY_OES targets during
    context initialization.


    Modify second to last paragraph of  section on p. 132:

    The texture object name space, including the initial ... and
    two-dimensional multisample array texture objects, is shared among all
    texture units. ...


    Modify Section 8.5.3, Texture Image Specification (p. 157)

    Add the new target to MAX_TEXTURE_SIZE description:

    In a similar fashion, the maximum allowable width and height of a texel
    array for a ... or two-dimensional multisample array texture must each
    be at least 2^(k-lod) ...


    Modify the introduction of section 8.8 "Multisample Textures":
    sections.)

    In addition to the texture types described in previous sections, two
    additional type of textures are supported. Multisample textures are
    similar to two-dimensional or two-dimensional array texture, except they
    contains multiple samples per texel. Multisample textures do not have
    multiple image levels, and are immutable.

    The commands

        void TexStorage2DMultisample ...
        void TexStorage3DMultisampleOES(enum target, sizei samples,
                                   enum internalformat,
                                   sizei width, sizei height, sizei depth,
                                   boolean fixedsamplelocations);

    establish ... For TexStorage3DMultisampleOES <target> must be
    TEXTURE_2D_MULTISAMPLE_ARRAY_OES. <width> and <height> are the
    dimensions in texels of the texture.


    Modify the second paragraph on p. 172:

    Upon success of TexStorage*MultisampleOES the contents of texels for
    <target> are undefined. TEXTURE_WIDTH, TEXTURE_HEIGHT, ...


    Modify the Errors section to specify existing errors as specific to
    TexStorage2DMultisample only (for <target>, and for <width> and <height>
    being too large); other existing errors are taken to apply to both
    commands. Then add new errors for TexStorage3DMultisample:

    Errors

    ...

    An INVALID_ENUM error is generated by TexStorage3DMultisample if
    <target> is not TEXTURE_2D_MULTISAMPLE_ARRAY.

    An INVALID_VALUE error is generated if <width>, <height> or <depth> is
    less than 1.

    An INVALID_VALUE error is generated by TexStorage3DMultisample if
    <width> or <height> is greater than the value of MAX_TEXTURE_SIZE.

    An INVALID_VALUE error is generated by TexStorage3DMultisample if
    <depth> is greater than the value of MAX_ARRAY_TEXTURE_LAYERS.


    Modifications to Section 8.9, "Texture Parameters", p. 173:

    Add TEXTURE_2D_MULTISAMPLE_ARRAY_OES to the texture targets accepted by
    TexParameter* in the first paragraph.


    Add to the Errors section on p. 174/175:

    Add TEXTURE_2D_MULTISAMPLE_ARRAY to the list of <target>s for which an
    INVALID_ENUM error is *not* generated.

    An INVALID_ENUM error is generated if <target> is
    TEXTURE_2D_MULTISAMPLE_ARRAY, and <pname> is any sampler state from
    table 6.13.

    An INVALID_OPERATION error is generated if <target> is
    TEXTURE_2D_MULTISAMPLE_ARRAY, and <pname> TEXTURE_BASE_LEVEL is set to a
    value other than zero.

    Modifications to Section 8.10.2, "Texture Parameter Queries"

    Modify the second paragraph of that section on p. 175 describing the
    <target> parameter of GetTexParameter*:

    <target> may be one of ... or TEXTURE_2D_MULTISAMPLE_ARRAY_OES,
    indicating the current ... or two-dimensional multisample array texture
    object, respectively.


    Modify Section 8.10.3 "Texture Level Parameter Queries" in the description
    of GetTexLevelParameter{if}v on p. 175:

    <target> may be one of ... or TEXTURE_2D_MULTISAMPLE_ARRAY_OES,
    indicating the ... or two-dimensional multisample array target.

Additions to Chapter 9 of the OpenGL ES 3.1 Specification (Framebuffers and
Framebuffer Objects)

    Modify Section 9.2.2, Attaching Images to Framebuffer Objects, p. 210:

    The command

        void FramebufferTextureLayer(enum target, enum attachment,
                                     uint texture, int level, int layer);

    operates similarly to FramebufferTexture2D, except that it
    attaches a single layer of a ...
    or two-dimensional multisample array texture level.

    ...

    If <texture> is a two-dimensional multisample array texture then <level>
    must be zero.

    Errors

    ...

    An INVALID_VALUE error is generated if <layer> is larger than the value
    of MAX_ARRAY_TEXTURE_LAYERS minus one (for two-dimensional array
    textures).

    Add two-dimensional multisample arrays to the list of texture types for
    which an INVALID_OPERATION error is *not* generated.

Additions to Chapter 19 of the OpenGL ES 3.1 Specification (Context State
Queries)

    Modifications to Section 19.3, "Internal Format Queries"

    Add to table 19.1 "Possible targets that <internalformat> can be used
    with ..."

    Target                           Usage
    -------------------------------- ----------------------------
    TEXTURE_2D_MULTISAMPLE_ARRAY_OES 2D multisample array texture

Errors

    Errors are described in the base 3.1 spec which this extension modified,
    or inline above. They are not summarized here.

New State

    (add to table 20.8, Textures (selector, state per texture unit) p. 365)

                                                                   Initial
    Get Value                                 Type     Get Command Value  Description                      Sec.
    ----------------------------------------  ----     ----------- ------ -------------------------------- ------
    TEXTURE_BINDING_2D_MULTISAMPLE_ARRAY_OES  32*xZ+   GetIntegerv 0      Texture object bound to          8.1
                                                                          TEXTURE_2D_MULTISAMPLE_ARRAY_OES

Modifications to the OpenGL ES Shading Language Specification, Version 3.10

    Including the following line in a shader can be used to control the
    language featured described in this extension:

      #extension GL_OES_texture_storage_multisample_2d_array : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

      #define GL_OES_texture_storage_multisample_2d_array 1

    Add to section 3.7 "Keywords"

    The following new sampler types are added:

      sampler2DMSArray, isampler2DMSArray, usampler2DMSArray


    Add to section 4.1 "Basic Types"

    Add the following sampler type to the "Floating Point Sampler Types (opaque)"
    table:

      sampler2DMSArray   handle for accessing a 2D multisample array
                            texture

    Add the following sampler type to the "Unsigned Integer Sampler
    Types (opaque)" table:

      usampler2DMSArray  handle for accessing an unsigned integer 2D
                            multisample array texture

    Add the following sampler type to the "Integer Sampler Types" table:

      isampler2DMSArray  handle for accessing an integer 2D multisample
                            array texture


    Add to section 8.9.2    "Texel Lookup Functions"

    Add new functions to the set of allowed texture lookup functions:

    Syntax:

      gvec4 texelFetch(gsampler2DMSArray sampler, ivec3 P, int sample)

    Description:

      Use integer texture coordinate <P> to lookup a single sample
      <sample> on the texture bound to <sampler> as described in section
      2.11.9.3 of the OpenGL ES specification "Multisample Texel Fetches".

    Syntax:

      ivec3 textureSize(gsampler2DMSArray sampler)

    Description:

      Returns the dimensions, width and height of level 0 for the
      texture bound to <sampler>, as described in section 2.11.9.4 of
      the OpenGL ES specification section "Texture Size Query".

Examples

Issues

    (1) Should mutable multisample texture support be kept?

    Resolution: No - only immutable multisample textures should be support by
       this extension

    (2) What should the minimum number of samples be?

    Resolution: The minimum is one, but there is no requirement to implement
    support for one, as this would be an unusual requirement.

    (3) Should the new sampler types have OES suffixes?

    RESOLVED: No. The non-suffixed names are reserved keywords in OpenGL ES
    Shading Language 3.10, and can be used here.

Revision History

    Rev.    Date     Author         Changes
    ----  ---------- ---------      --------------------------------------------
    8     2019/01/11 Jon Leech      Change 'int sizedinternalformat' parameter
                                    to 'enum internalformat', to follow
                                    changes to the ES 3.2 specification
                                    (KhronosGroup/OpenGL-API issue 30).
    7     2016/05/03 Tobias Hector  Fixed INVALID_OPERATION error message for
                                    TexStorage3DMultisample to match desktop and
                                    ES 3.2
                                    Rebased against release version of ES 3.1
                                    spec.
    6     2015/04/16 Jon Leech      Remove texture border width term b_t, which
                                    doesn't exist in OpenGL ES.
    5     2014/03/26 Jon Leech      Add missing GL_ prefix to the name string.
    4     2014/03/06 Jon Leech      Change limit on TexStorage3DMultisampleOES
                                    <depth> parameter to the value of
                                    MAX_ARRAY_TEXTURE_LAYERS (Bug 11135).
    3     2014/02/04 Jon Leech      Remove OES suffix from sampler keywords
                                    since they're reserved in GLSL-ES 3.10
                                    (Bug 11636).
    2     2014/01/30 Jon Leech      Remove 2D multisample non-array textures,
                                    and all common language already in the ES
                                    3.1 specification draft.
    1     2014/01/30 Jon Leech      Branch from internal XXX_texture_multisample
                                    spec.
