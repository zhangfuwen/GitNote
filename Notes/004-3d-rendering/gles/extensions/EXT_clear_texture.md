# EXT_clear_texture

Name

    EXT_clear_texture

Name Strings

    GL_EXT_clear_texture

Contact

    Tobias Hector, Imagination Technologies (tobias.hector 'at' imgtec.com)

Contributors

    Contributors to the original ARB_clear_texture
    Ian Romanick
    Daniel Koch

Notice

    Copyright (c) 2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Draft

Version

    Last Modified Date: September 16, 2016
    Revision: 2

Number

    OpenGL ES Extension #269

Dependencies

    OpenGL ES 3.1 is required.

    This extension is written against the OpenGL ES 3.1 API specification.

    This extension interacts with EXT/OES_texture_buffer or OpenGL ES 3.2.

    This extension interacts with EXT/OES_texture_cube_map_array or
    OpenGL ES 3.2.

    This extension interacts with OES_texture_stencil8 or OpenGL ES 3.2.

Overview

    Texture objects are fundamental to the operation of OpenGL. They are
    used as a source for texture sampling and destination for rendering
    as well as being accessed in shaders for image load/store operations
    It is also possible to invalidate the contents of a texture. It is
    currently only possible to set texture image data to known values by
    uploading some or all of a image array from application memory or by
    attaching it to a framebuffer object and using the Clear or ClearBuffer
    commands.

    Both uploading initial texture data and clearing by attaching to a
    framebuffer have potential disadvantages when one simply wants to
    initialize texture data to a known value. Uploading initial data
    requires the application to allocate a (potentially large) chunk
    of memory and transferring that to the GL.  This can be a costly
    operation both in terms of memory bandwidth and power usage.
    Alternatively, attaching a texture level to a framebuffer to clear it
    may not be possible if the texture format isn't supported for
    rendering, or even if it is, attaching the image to a framebuffer object
    may cause the texture to be allocated in certain types of memory, which
    it may otherwise not need to be placed in.

    This extension solves these problems by providing a mechanism whereby
    the contents of a texture image array can be set to known values by
    using the ClearTexImageEXT or ClearTexSubImageEXT commands.  These commands
    can also be useful for initializing an image that will be used for
    atomic shader operations.

IP Status

    No known IP claims.

New Procedures and Functions

    void ClearTexImageEXT(uint texture, int level,
                          enum format, enum type,
                          const void * data);

    void ClearTexSubImageEXT(uint texture, int level,
                             int xoffset, int yoffset, int zoffset,
                             sizei width, sizei height, sizei depth,
                             enum format, enum type,
                             const void * data);

New Types

    None

New Tokens

    None

Additions to Chapter 7 of the OpenGL ES 3.1 API Specification (Programs and
Shaders)

    In section 7.11.2 (Shader Memory Access Synchronization) edit the
    description of the TEXTURE_UPDATE_BARRIER_BIT to add ClearTexImageEXT
    and ClearTexSubImageEXT to the list of commands that can write to
    texture images.

Additions to Chapter 8 of the OpenGL ES 3.1 API Specification (Textures and
Samplers)

    Add a new Section 8.x (Clearing Texture Image Data) after
    Section 8.17 (Immutable-Format Texture Images):

    "All or part of a texture image may be filled with a constant value
    by calling the command

      void ClearTexSubImageEXT(uint texture, int level,
                               int xoffset, int yoffset, int zoffset,
                               sizei width, sizei height, sizei depth,
                               enum format, enum type,
                               const void * data);

    with <texture> and <level> indicating which texture array image is being
    cleared.  It is an error if <texture> is zero or not the name of a texture
    object, if <texture> is a buffer texture, or if the texture image has
    a compressed internal format.

    Arguments <xoffset>, <yoffset>, and <zoffset> specify the lower left
    texel coordinates of a <width>-wide by <height>-high by <depth>-deep
    rectangular subregion of the texel array and are interpreted as they
    are in TexSubImage3D as described in section 8.6 (Alternate Texture
    Image Specification Commands). The subregion must lie within the bounds
    of the texture image, as described in that section.

    For 2D array textures, <zoffset> is interpreted as the first layer to be
    cleared and <depth> is the number of layers to clear. Cube map textures are
    treated as an array of six slices in the z-dimension, where the value
    of <zoffset> is interpreted as specifying the cube map face for the
    corresponding <layer> in table 8.25 (Layer numbers for cube map texture
    faces) and <depth> is the number of faces to clear. For cube map array
    textures, <zoffset> is the first layer-face to clear, and <depth> is the
    number of layer-faces to clear. Each layer-face is translated into an
    array layer and a cube map face as described for layer-face numbers in
    section 8.5.3 [in OpenGL ES 3.2].

    For texture types that do not have certain dimensions, this
    command treats those dimensions as having a size of 1.  For example,
    to clear a portion of a two-dimensional texture, the application would
    use <zoffset> equal to zero and <depth> equal to one.

    <format> and <type> specify the format and type of the source
    data and are interpreted as they are for TexImage3D, as described in
    section 8.4.2 (Transfer of Pixel Rectangles). Textures with a base
    internal format of DEPTH_COMPONENT, STENCIL_INDEX, or DEPTH_STENCIL
    require depth component, stencil, or depth/stencil component data
    respectively. Textures with other base internal formats require RGBA
    formats. Textures with integer internal formats (see table 8.13) require
    integer data.

    <data> is a pointer to an array of between one and four components of
    texel data that will be used as the source for the constant fill value.
    The elements of <data> are converted by the GL into the
    <internalformat> of the texture image (that was specified when the level
    was defined by any of the TexImage, TexStorage or CopyTexImage
    commands) in the manner described in section 8.4.2 (Transfer of Pixel
    Rectangles), and then used to fill the specified range of the
    destination texture level.  If <data> is NULL, then the pointer is
    ignored and the sub-range of the texture image is filled with zeros.
    If <texture> is a multisample texture, all the samples in a texel
    are cleared to the value specified by <data>.

    Errors

        An INVALID_OPERATION error is generated if <texture> is zero or not the
        name of a texture object.

        An INVALID_OPERATION error is generated if <texture> is a buffer
        texture.

        An INVALID_OPERATION error is generated if <texture> has a compressed
        internal format.

        An INVALID_OPERATION error is generated if the base internal format is
        DEPTH_COMPONENT and <format> is not DEPTH_COMPONENT.

        An INVALID_OPERATION error is generated if the base internal format is
        DEPTH_STENCIL and <format> is not DEPTH_STENCIL.

        An INVALID_OPERATION error is generated if the base internal format is
        STENCIL_INDEX and <format> is not STENCIL_INDEX.

        An INVALID_OPERATION error is generated if the base internal format is
        RGBA and the <format> is DEPTH_COMPONENT, STENCIL_INDEX, or
        DEPTH_STENCIL.

        An INVALID_OPERATION error is generated if the internal format is
        integer and <format> does not specify integer data.

        An INVALID_OPERATION error is generated if the internal format is
        not integer and <format> does specify integer data.

        An INVALID_OPERATION error is generated if the specified subregion
        does not lie within the bounds of the texture image, as described
        for TexSubImage3D in section 8.6.

    The command

      void ClearTexImageEXT(uint texture, int level,
                            enum format, enum type,
                            const void * data);

    is equivalent to calling ClearTexSubImageEXT with <xoffset>, <yoffset>,
    and <zoffset> equal to 0 and <width>, <height>, and <depth> equal
    to the dimensions of the texture image (or zero and one for dimensions
    the texture doesn't have).

    Errors

        In addition to the errors generated by ClearTexSubImageEXT:

        An INVALID_OPERATION error is generated if the image array identified by
        <level> has not previously been defined by a TexImage* or TexStorage*
        command."

Interactions with EXT/OES_texture_buffer or OpenGL ES 3.2

    If EXT_texture_buffer or equivalent functionality is not supported,
    ignore all references to buffer textures.

Interactions with EXT/OES_texture_cube_map_array or OpenGL ES 3.2

    If EXT_texture_cube_map_array or equivalent functionality is not supported,
    ignore all references to cube map array textures.

Interactions with OES_texture_stencil8 or OpenGL ES 3.2

    If OES_texture_stencil8 or equivalent functionality is not supported,
    ignore all references to stencil textures.

Errors

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if <texture> is zero or not the name of a texture object.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if <texture> is a buffer texture.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if <texture> has a compressed internal format.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the base internal format is DEPTH_COMPONENT and
    <format> is not DEPTH_COMPONENT.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the base internal format is STENCIL_INDEX and
    <format> is not STENCIL_INDEX.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the base internal format is DEPTH_STENCIL and
    <format> is not DEPTH_STENCIL.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the base internal format is RGBA and the <format>
    is DEPTH_COMPONENT, STENCIL_INDEX, or DEPTH_STENCIL.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the internal format is integer and <format> does not
    specify integer data.

    An INVALID_OPERATION error is generated by ClearTexImageEXT or
    ClearTexSubImageEXT if the internal format is not integer and <format>
    does specify integer data.

    An INVALID_OPERATION error is generated if the specified subregion
    does not lie within the bouds of the texture image, as described
    for TexSubImage3D in section 8.6.

    An INVALID_OPERATION error is generated by ClearTexImageEXT if the
    image array identified by <level> has not previously been defined.

New State

    None.

New Implementation Dependent State

    None.

Issues

    See ARB_clear_texture for relevant issues.

    1) How does this differ from ARB_clear_texture?

    RESOLVED:
      - Removed interactions with features not present in OpenGL ES:
        - 1D arrays
        - border texels
      - Removed CLEAR_TEXTURE token for GetInternalFormat* which relies
        on the ARB_internalformat_query2 extension which does not have
        equivalent functionality in OpenGL ES.
      - minor language updates to sync with OpenGL 4.5 language.

Revision History

    Revision 1, 2015/08/15, thector
      - Initial revision
    Revision 2, 2016/09/16, dgkoch
      - added interactions with texture cube map arrays and stencil textures
      - added missing suffixes on commands
      - removed CLEAR_TEXTURE, which relies on functionality not in ES.
      - minor language updates to sync with OpenGL 4.5 core language.
      - updated some section references.
