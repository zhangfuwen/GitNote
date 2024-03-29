# OES_depth_texture

Name

    OES_depth_texture

Name Strings

    GL_OES_depth_texture

Contact

    Aaftab Munshi (amunshi@apple.com)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, March 20, 2008.

Version

    Last Modifed Date: January 29, 2012

Number

    OpenGL ES Extension #44

Dependencies

    OpenGL ES 2.0 is required.
    This extension is written against the OpenGL ES 2.0 specification

Overview

    This extension defines a new texture format that stores depth values in 
    the texture.  Depth texture images are widely used for shadow casting but 
    can also  be used for other effects such as image based rendering, displacement 
    mapping etc.  

Issues

1.  Should manual generation of mipmaps for a depth texture using GenerateMipmap be supported?

    Possible Resolutions:

    a)  Allow manual generation of mipmaps.  This will ensure that GenerateMipmap works
        consistently for any texture.

    b)  Disallow.  GenerateMipmap will generate INVALID_OPERATION error for a depth texture.
        The reason for this is that doing a low-pass filter to generate depth values for 
        higher mip-levels of a depth texture does not make sense.

    Resolution: Adopt approach b).  Manual generation of mipmaps is done by averaging a 
    2 x 2 region --> 1 texel as we go from one level to the next.  This does not make much 
    sense for depth textures.  A better approach would be to take a min or max of 2 x 2 texel 
    region instead of doing an average.  Since min & max filters are not supported by GenerateMipmap, 
    the WG decided to disallow manual generation of mipmaps for a depth texture.

2.  Should GL_DEPTH_TEXTURE_MODE be used to determine whether depth textures are treated as
    LUMINANCE, INTENSITY or ALPHA textures during texture filtering and application.

    Possible Resolutions:

    a)  Adopt text from the OpenGL specification.

    b)  No longer require DEPTH_TEXTURE_MODE.  Treat depth textures always as luminance 
        textures i.e. depth value is returned as (d, d, d, 1.0) by GLSL texture* calls in the
        fragment and/or vertex shader.

    Resolution:  Adopt approach b).  We only need to support one way of treating how depth textures
    are read by the shader instead of three possible ways as supported by OpenGL.  Almost all apps on 
    desktop that use depth textures treat depth textures as LUMINANCE.  Therefore, this extension treats 
    depth textures always as luminance textures and no longer supports DEPTH_TEXTURE_MODE.

3.  How should 24-bit depth texture data be represented when specified in TexImage2D and TexSubImage2D?

    Resolution:  This is currently not supported.  Depth textures can be specified with 16-bit depth
    values i.e. <type> = UNSIGNED_SHORT or 32-bit depth values i.e. <type> = UNSIGNED_INT.

4.  Are cube-map depth textures implemented by this extension?

    Resolution: No. This is defined in OES_depth_texture_cube_map.

    A very interesting use case is rendering shadows for a point light.
    For a point light, you want to be able to render depth values into a cube-map and then 
    use this cube-map depth texture as a shadow map to compute % in shadow at each pixel.
    
    The original version of this extension supported cube-map
    depth textures, but this was contradicted by OES_packed_depth_stencil which
    only supported 2D textures. Some implementations of OES_depth_texture did
    not support cube-maps while others did so it was decided to make support for
    cube-map depth textures a separate extension.

5.  Are 3D depth textures implemented by this extension?

    Resolution:  This is not supported.  We could not come up with any use cases for
    3D depth textures.  In addition, we can always define a new extension that adds 
    this specific functionality in the future.

New Procedures and Functions

    None

New Tokens

    Accepted by the <format> parameter of TexImage2D and TexSubImage2D and
    <internalFormat> parameter of TexImage2D:
    
        DEPTH_COMPONENT          0x1902
        
    Accepted by the <type> parameter of TexImage2D, TexSubImage2D: 

        UNSIGNED_SHORT           0x1403
        UNSIGNED_INT             0x1405

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Modifications to table 2.9 (Component conversions)

        Add the following entries:

            GLType                     Conversion
            -------                    ----------
            UNSIGNED_SHORT             c / (2^16 - 1)
            UNSIGNED_INT               c / (2^32 - 1)

        NOTE:  UNSIGNED_SHORT and UNSIGNED_INT entries already exist in table 2.9 of the 
               OpenGL ES 2.0 specification and have been added here for clarity.

Additions to Chapter 3, section 3.8 of the OpenGL ES 2.0 specification

    Textures with <format> and <internalFormat> values of DEPTH_COMPONENT
    refer to a texture that contains depth component data.  <type> is used
    to determine the number of bits used to specify depth texel values.  

    A <type> value of UNSIGNED_SHORT refers to a 16-bit depth value.
  
    A <type> value of UNSIGNED_INT refers to a 32-bit depth value.
    
    As per the OpenGL ES spec, there is no guarantee that the OpenGL ES implementation 
    will use the <type> to determine how to store the depth texture internally.  
    It may choose to downsample the 32-bit depth values to 16-bit or even 24-bit.
    There is currently no way for the application to know or find out how the
    depth texture (or any texture) will be stored internally by the OpenGL ES implementation.

    Textures with a base internal format of DEPTH_COMPONENT are supported
    by texture image specification commands only if <target> is TEXTURE_2D.
    Using this format in conjunction with any other <target> will result in
    an INVALID_OPERATION error.

    CopyTexImage2D and CopyTexSubImage2D are not supported.

Additions to Chapter 4, section 4.4.2 of the OpenGL ES 2.0 specification

    Textures of <format> = DEPTH_COMPONENT are depth renderable.

Errors

    The error INVALID_OPERATION is generated if the <format> and <internalFormat>
    is DEPTH_COMPONENT and <type> is not UNSIGNED_SHORT, or UNSIGNED_INT.

    The error INVALID_OPERATION is generated if the <format> and <internalFormat>
    is not DEPTH_COMPONENT and <type> is UNSIGNED_SHORT, or UNSIGNED_INT.

    The error INVALID_OPERATION is generated if <target> is not TEXTURE_2D.

New State

    None.

Revision History
 
    1/17/2006    First Draft.
    6/14/2006    CopyTexImage2D and CopyTexSubImage2D are not supported.
    7/19/2007    Added Issues section + updates to section 3.8 +
                 add section on supporting framebuffer texture attachements
                 for depth textures.
    7/30/2007    Update issues section with adopted resolutions.
                 Updates to the Errors section.
    7/31/2007    Updates to conversion table 2.9
    9/4/2007     Added item 3 to Issues List.  
                 Resolution for issue 1 changed to b) - decided in ES WG meeting on 8/29/2007.
    9/24/2007    Removed UNSIGNED_INT_24_OES.  Added reasoning to resolutions in Issues section.
                 Removed dependencies on OES_depth24 and OES_depth32.
    10/20/2007   Added issues 4. and 5.
    10/08/2009   Changed INVALID_VALUE to INVALID_OPERATION error (bug 5209).
    01/29/2012   Move depth cube map texture support to OES_depth_texture_cube_map.
