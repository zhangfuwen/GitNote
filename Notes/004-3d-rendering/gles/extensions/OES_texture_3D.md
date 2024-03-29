# OES_texture_3D

Name

    OES_texture_3D

Name Strings

    GL_OES_texture_3D

Contributors

    Benj Lipchak
    Robert Simpson

Contact

    Aaftab Munshi (amunshi@apple.com)

Notice

    Copyright (c) 2005-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

IP Status

    None.

Status

    Ratified by the Khronos BOP, July 22, 2005.
    
Version

    Last Modifed Date: July 24, 2007

Number

    OpenGL ES Extension #34    

Dependencies

    OpenGL ES 2.0 is required.

Overview

    This extension adds support for 3D textures.  The OpenGL ES 2.0 texture wrap
    modes and mip-mapping is supported for power of two 3D textures.  Mip-
    mapping and texture wrap modes other than CLAMP_TO_EDGE are not supported 
    for non-power of two 3D textures.
    
    The OES_texture_npot extension, if supported, will enable mip-mapping and 
    other wrap modes for non-power of two 3D textures.

Issues

    None.
 
New Tokens

    Accepted by the <target> parameter of TexImage3DOES, TexSubImage3DOES, 
    CopyTexSubImage3DOES, CompressedTexImage3DOES and 
    CompressedTexSubImage3DOES, GetTexParameteriv, and GetTexParameterfv:

        TEXTURE_3D_OES              0x806F

    Accepted by the <pname> parameter of TexParameteriv, TexParameterfv,
    GetTexParameteriv, and GetTexParameterfv:

        TEXTURE_WRAP_R_OES          0x8072

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and 
    GetFloatv:

        MAX_3D_TEXTURE_SIZE_OES     0x8073
        TEXTURE_BINDING_3D_OES      0x806A

New Procedures and Functions

    void TexImage3DOES(enum target, int level, enum internalFormat, 
                       sizei width, sizei height, sizei depth, int border, 
                       enum format, enum type, const void *pixels)

    Similar to 2D textures and cubemaps,  <internalFormat> must match <format>.
    Please refer to table 3.1 of the OpenGL ES 2.0 specification for a list of 
    valid <format> and <type> values.  No texture borders are supported.

    void TexSubImage3DOES(enum target, int level, 
                          int xoffset, int yoffset, int zoffset, 
                          sizei width, sizei height, sizei depth, 
                          enum format, enum type, const void *pixels)

    void CopyTexSubImage3DOES(enum target, int level, 
                              int xoffset, int yoffset, int zoffset, 
                              int x, int y, sizei width, sizei height)

    CopyTexSubImage3DOES is supported.  The internal format parameter can be 
    any of the base internal formats described for TexImage2D and TexImage3DOES 
    subject to the constraint that color buffer components can be dropped during
    the conversion to the base internal format, but new components cannot be 
    added.  For example, an RGB color buffer can be used to create LUMINANCE or 
    RGB textures, but not ALPHA, LUMINANCE_ALPHA, or RGBA textures.  Table 3.3 
    of the OpenGL ES 2.0 specification summarizes the allowable framebuffer and 
    base internal format combinations.

    void CompressedTexImage3DOES(enum target, int level, enum internalformat, 
                                 sizei width, sizei height, sizei depth, 
                                 int border, sizei imageSize, const void *data)

    void CompressedTexSubImage3DOES(enum target, int level, 
                                    int xoffset, int yoffset, int zoffset, 
                                    sizei width, sizei height, sizei depth, 
                                    enum format, sizei imageSize, 
                                    const void *data)

    void FramebufferTexture3DOES(enum target, enum attachment,
                                 enum textarget, uint texture,
                                 int level, int zoffset);

    FramebufferTexture3DOES is derived from FramebufferTexture3DEXT.  Please 
    refer to the EXT_framebuffer_object extension specification for a detailed 
    description of FramebufferTexture3DEXT.  The only difference is that 
    FramebufferTexture3DOES can be used to render directly into the base level 
    of a 3D texture image only.  The OES_fbo_render_mipmap extension removes
    this limitation and allows rendering to any mip-level of a 3D texture.

New Keywords

    sampler3D

Grammar changes

    The token SAMPLER3D is added to the list of tokens returned from lexical 
    analysis and the type_specifier_no_prec production.

New Built-in Functions

    texture3D()
    texture3DProj()  
    texture3DLod()
    texture3DProjLod()

New Macro Definitions

    #define GL_OES_texture_3D 1

Additions to Chapter 4 of the OpenGL ES Shading Language specification:

    Add the following to the table of basic types in section 4.1:

    Type:
        sampler3D

    Meaning:
        a handle for accessing a 3D texture

Additions to Chapter 8 of the OpenGL ES Shading Language specification:

    Add the following to the table of built-in functions in section 8.7:

    The built-in texture lookup functions texture3D, texture3DProj, 
    texture3DLod, and texture3DProjLod are optional, and must be enabled by

    #extension GL_OES_texture_3D : enable

    before being used.  

    Syntax:
        vec4 texture3D (sampler3D sampler, vec3 coord [, float bias] )
        vec4 texture3DProj (sampler3D sampler, vec4 coord [, float bias] )
        vec4 texture3DLod (sampler3D sampler, vec3 coord, float lod)
        vec4 texture3DProjLod (sampler3D sampler, vec4 coord, float lod)

    Description:
        Use the texture coordinate coord to do a texture lookup in the 3D 
        texture currently bound to sampler.  For the projective ("Proj") 
        versions, the texture coordinate is divided by coord.q.

Errors

    None.

New State

Get Value                Type    Get Command        Value    Description
---------                ----    -----------        -----    -----------
TEXTURE_BINDING_3D_OES    Z+     GetIntegerv        0        texture object 
                                                             bound to TEXTURE_3D
TEXTURE_WRAP_R_OES        1xZ2   GetTexParameteriv  REPEAT   texture coord "r"
                                                             wrap mode
MAX_3D_TEXTURE_SIZE_OES   Z+     GetIntegerv        16       maximum 3D texture 
                                                             image dimension

Revision History

7/06/2005    Aaftab Munshi    Created the extension
6/09/2006    Aaftab Munshi    Added OES suffixes
7/24/2007    Benj Lipchak     Merged in details of language changes, removed
                              OES_framebuffer_object requirement (now core),
                              reformatted to 80 columns
