# OES_depth_texture_cube_map

Name

    OES_depth_texture_cube_map

Name Strings

    GL_OES_depth_texture_cube_map

Contact

    Daniel Koch (daniel 'at' transgaming 'dot' com)

Notice

    Copyright (c) 2012-2013 The Khronos Group Inc. Copyright terms at
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

    Recommended by the OpenGL ES Working Group, September 12, 2012

Version

    Last Modifed Date: January 29, 2012
    Version 1

Number

    OpenGL ES Extension #136

Dependencies

    OpenGL ES 2.0 is required.

    OES_depth_texture is required. 

    This extension is written against the OpenGL ES 2.0 specification

    OES_packed_depth_stencil affects the definition of this extension.

Overview

    This extension extends OES_depth_texture and OES_packed_depth_stencil
    to support depth and depth-stencil cube-map textures.
   
Issues

    None
 
New Procedures and Functions

    None

New Tokens

    Accepted by the <format> parameter of TexImage2D and TexSubImage2D and
    <internalFormat> parameter of TexImage2D when <target> is one of the
    TEXTURE_CUBE_MAP_* targets:
    
        DEPTH_COMPONENT          0x1902
        DEPTH_STENCIL_OES        0x84F9
        
    Accepted by the <type> parameter of TexImage2D, TexSubImage2D when
    <target> is one of the TEXTURE_CUBE_MAP_* targets:

        UNSIGNED_SHORT           0x1403
        UNSIGNED_INT             0x1405
        DEPTH24_STENCIL8_OES     0x88F0

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Modifications to Chapter 3, section 3.8 of the OpenGL ES 2.0 specification

    Delete the following paragraph which was added by OES_depth_texture:

    "Textures with a base internal format of DEPTH_COMPONENT are supported
    by texture image specification commands only if <target> is TEXTURE_2D.
    Using this format in conjunction with any other <target> will result in
    an INVALID_OPERATION error."

    Delete the following paragraph which was added by OES_packed_depth_stencil:

    "Textures with a base internal format of DEPTH_COMPONENT or DEPTH_STENCIL_OES
    are supported by texture image specification commands only if <target> is
    TEXTURE_2D.  Using this format in conjunction with any other <target> will
    result in an INVALID_OPERATION error."

Additions to Chapter 4, of the OpenGL ES 2.0 specification

    None

Interactions with OES_packed_depth_stencil

    If OES_packed_depth_stencil is not available, any modifications based on 
    OES_packed_depth_stencil and any mention of DEPTH_STENCIL_OES
    and DEPTH24_STENCIL8_OES are omitted.

Errors

    Change the error for <targets> accepted by TexImage2D and TexSubImage2D to:

    "The error INVALID_OPERATION is generated if <target> is not TEXTURE_2D, or
    one of the TEXTURE_CUBE_MAP_* targets."

New State

    None.

Revision History
 
    01/29/2012   First Draft, split from OES_depth_texture.
