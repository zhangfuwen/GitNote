# KHR_gl_texture_2D_image

Name

    KHR_gl_texture_2D_image
    KHR_gl_texture_cubemap_image
    KHR_gl_texture_3D_image
    KHR_gl_renderbuffer_image

Name Strings

    EGL_KHR_gl_texture_2D_image
    EGL_KHR_gl_texture_cubemap_image
    EGL_KHR_gl_texture_3D_image
    EGL_KHR_gl_renderbuffer_image

Contributors

    Aaftab Munshi
    Barthold Lichtenbelt
    Gary King
    Jeff Juliano
    Jon Leech
    Jonathan Grant
    Acorn Pooley

Contacts

    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on February 11, 2008.

Version

    Version 13, October 9, 2013

Number

    EGL Extension #5

Dependencies

    All extensions require EGL 1.2 and the EGL_KHR_image extension

    These extensions are written against the wording of the EGL 1.2
    Specification.

    KHR_gl_texture_2D_image requires an OpenGL or OpenGL ES client API (any
    version of either API).

    KHR_gl_texture_cubemap_image requires an OpenGL or OpenGL ES client API
    supporting texture cube maps, either in the core API or via extensions.

    KHR_gl_texture_3D_image requires KHR_gl_texture_2D_image to be supported
    by the EGL implementation. It also requires an OpenGL or OpenGL ES
    client API supporting three-dimensional textures, either in the core API
    or via extensions.

    KHR_gl_renderbuffer_image requires KHR_gl_texture_2D_image to be
    supported by the EGL implementation. It also requires an OpenGL or
    OpenGL ES client API supporting renderbuffers, either in the core API or
    via extensions.

Overview

    The extensions specified in this document provide a mechanism for
    creating EGLImage objects from OpenGL and OpenGL ES (henceforth referred
    to collectively as 'GL') API resources, including two- and three-
    dimensional textures, cube maps and render buffers. For an overview of
    EGLImage operation, please see the EGL_KHR_image specification.

    Due to the number of available extensions for the OpenGL ES 1.1 and
    OpenGL ES 2.0 APIs, this document is organized as 4 separate extensions,
    described collectively. These extensions are separated based on the
    required underlying GL functionality (described in the dependencies
    section).

New Types

    None

New Procedures and Functions

    None

New Tokens

      Accepted in the <target> parameter of eglCreateImageKHR:

          EGL_GL_TEXTURE_2D_KHR                         0x30B1

      Accepted as an attribute in the <attr_list> parameter of
      eglCreateImageKHR:

          EGL_GL_TEXTURE_LEVEL_KHR                      0x30BC

    Added by KHR_gl_texture_cubemap_image:

      Accepted in the <target> parameter of eglCreateImageKHR:

          EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR        0x30B3
          EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR        0x30B4
          EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR        0x30B5
          EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR        0x30B6
          EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR        0x30B7
          EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR        0x30B8

    Added by KHR_gl_texture_3D_image:

      Accepted in the <target> parameter of eglCreateImageKHR:

          EGL_GL_TEXTURE_3D_KHR                         0x30B2

      Accepted as an attribute in the <attr_list> parameter of
      eglCreateImageKHR:

          EGL_GL_TEXTURE_ZOFFSET_KHR                    0x30BD

    Added by KHR_gl_renderbuffer_image:

      Accepted in the <target> parameter of eglCreateImageKHR:

          EGL_GL_RENDERBUFFER_KHR                       0x30B9


Additions to the EGL Image (EGL_KHR_image) Specification:

    Add the following to Table aaa (Legal values for eglCreateImageKHR
    <target> parameter), Section 2.5.1 (EGLImage Specification)

      +-------------------------------------+---------------------------------+
      |  <target>                           |  Notes                          |
      +-------------------------------------+---------------------------------+
      |  EGL_GL_TEXTURE_2D_KHR              |  Used for GL 2D texture images  |
      +-------------------------------------+---------------------------------+

    If KHR_gl_texture_cubemap_image is supported:

      +-----------------------------------------+-----------------------------+
      |  <target>                               |  Notes                      |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR |  Used for the +X face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR |  Used for the -X face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR |  Used for the +Y face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR |  Used for the -Y face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR |  Used for the +Z face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+
      |  EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR |  Used for the -Z face of    |
      |                                         |  GL cubemap texture images  |
      +-----------------------------------------+-----------------------------+

    If KHR_gl_texture_3D_image is supported:

      +-------------------------------------+---------------------------------+
      |  <target>                           |  Notes                          |
      +-------------------------------------+---------------------------------+
      |  EGL_GL_TEXTURE_3D_KHR              |  Used for GL 3D texture images  |
      +-------------------------------------+---------------------------------+

    If KHR_gl_renderbuffer_image is supported:

      +-------------------------------------+---------------------------------+
      |  <target>                           |  Notes                          |
      +-------------------------------------+---------------------------------+
      |  EGL_GL_RENDERBUFFER_KHR            |  Used for GL renderbuffer images|
      +-------------------------------------+---------------------------------+

    Add the following to Table bbb (Legal attributes for eglCreateImageKHR
    <attr_list> parameter), Section 2.5.1 (EGLImage Specification)

      +---------------------------+-------------------------------+----------------------------+---------+
      |                           |                               | Valid                      | Default |
      | Attribute                 |  Description                  | <target>s                  | Value   |
      +---------------------------+-------------------------------+----------------------------+---------+
      | EGL_GL_TEXTURE_LEVEL_KHR  |  Specifies the mipmap level   | EGL_GL_TEXTURE_2D_KHR,     |   0     |
      |                           |  used as the EGLImage source. | EGL_GL_TEXTURE_CUBE_MAP_*, |         |
      |                           |  Must be part of the complete | EGL_GL_TEXTURE_3D_KHR      |         |
      |                           |  texture object <buffer>      |                            |         |
      +---------------------------+-------------------------------+----------------------------+---------+

    If KHR_gl_texture_3D_image is supported:

      +----------------------------+------------------------------+----------------------------+---------+
      |                            |                              | Valid                      | Default |
      | Attribute                  |  Description                 | <target>s                  | Value   |
      +----------------------------+------------------------------+----------------------------+---------+
      | EGL_GL_TEXTURE_ZOFFSET_KHR |  Specifies the depth offset  | EGL_GL_TEXTURE_3D_KHR      |   0     |
      |                            |  of the image to use as the  |                            |         |
      |                            |  EGLImage source.  Must be   |                            |         |
      |                            |  part of the complete texture|                            |         |
      |                            |  object <buffer>             |                            |         |
      +----------------------------+------------------------------+----------------------------+---------+


    Insert the following text after paragraph 3 ("If <target> is
    NATIVE_PIXMAP_KHR...") of Section 2.5.1 (EGLImage Specification)

    "If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_3D_KHR,
    EGL_GL_RENDERBUFFER_KHR,
    EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_X_KHR,
    EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_X_KHR,
    EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Y_KHR,
    EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Y_KHR,
    EGL_GL_TEXTURE_CUBE_MAP_POSITIVE_Z_KHR, or
    EGL_GL_TEXTURE_CUBE_MAP_NEGATIVE_Z_KHR,
    <dpy> must be a valid EGLDisplay,
    and <ctx> must be a valid GL API context on that display.

    If <target> is EGL_GL_TEXTURE_2D_KHR, <buffer> must be the name of a
    nonzero, GL_TEXTURE_2D target texture object, cast into
    the type EGLClientBuffer.  <attr_list> should specify the mipmap level
    which will be used as the EGLImage source (EGL_GL_TEXTURE_LEVEL_KHR); the
    specified mipmap level must be part of <buffer>.  If not specified, the
    default value listed in Table bbb will be used, instead.  Additional
    values specified in <attr_list> are ignored.  The texture must be complete
    unless the mipmap level to be used is 0, the texture has mipmap level 0
    specified, and no other mipmap levels are specified.

    If <target> is one of the EGL_GL_TEXTURE_CUBE_MAP_* enumerants, <buffer>
    must be the name of a cube-complete, nonzero, GL_TEXTURE_CUBE_MAP (or
    equivalent in GL extensions) target texture object, cast into the type
    EGLClientBuffer.  <attr_list> should specify the mipmap level which will
    be used as the EGLImage source (EGL_GL_TEXTURE_LEVEL_KHR); the specified
    mipmap level must be part of <buffer>.  If not specified, the default
    value listed in Table bbb will be used, instead.  Additional values
    specified in <attr_list> are ignored.  The texture must be cube-complete
    unless the mipmap level to be used is 0, the texture has mipmap level 0
    specified for all faces, and no other mipmap levels are specified for any
    faces.

    If <target> is EGL_GL_TEXTURE_3D_KHR, <buffer> must be the name of a
    complete, nonzero, GL_TEXTURE_3D (or equivalent in GL extensions) target
    texture object, cast
    into the type EGLClientBuffer.  <attr_list> should specify the mipmap
    level (EGL_GL_TEXTURE_LEVEL_KHR) and z-offset (EGL_GL_TEXTURE_ZOFFSET_KHR)
    which will be used as the EGLImage source; the specified mipmap level must
    be part of <buffer>, and the specified z-offset must be smaller than the
    depth of the specified mipmap level.  If a value is not specified, the
    default value listed in Table bbb will be used, instead.  Additional
    values specified in <attr_list> are ignored.  The texture must be
    complete unless the mipmap level to be used is 0, the texture has mipmap
    level 0 specified, and no other mipmap levels are specified.

    If <target> is EGL_GL_RENDERBUFFER_KHR, <buffer> must be the name of a
    complete, nonzero, non-multisampled GL_RENDERBUFFER (or equivalent in
    extensions) target object, cast into the type EGLClientBuffer. Values
    specified in <attr_list> are ignored."

    Add the following errors to the end of the list in Section 2.5.1 (EGLImage
    Specification):

    "   * If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR, 
          EGL_GL_RENDERBUFFER_KHR or EGL_GL_TEXTURE_3D_KHR, and <dpy> is not a
          valid EGLDisplay, the error EGL_BAD_DISPLAY is generated.

        * If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR,
          EGL_GL_RENDERBUFFER_KHR or EGL_GL_TEXTURE_3D_KHR, and <ctx> is not a
          valid EGLContext, the error EGL_BAD_CONTEXT is generated.

        * If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR,
          EGL_GL_RENDERBUFFER_KHR or EGL_GL_TEXTURE_3D_KHR, and <ctx> is not a
          valid GL context, or does not match the <dpy>, the error
          EGL_BAD_MATCH is generated.

        * If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR
          or EGL_GL_TEXTURE_3D_KHR and <buffer> is not the name of a
          texture object of type <target>, the error EGL_BAD_PARAMETER
          is generated.

        * If <target> is EGL_GL_RENDERBUFFER_KHR and <buffer> is not the
          name of a renderbuffer object, or if <buffer> is the name of a
          multisampled renderbuffer object, the error EGL_BAD_PARAMETER is
          generated.

        * If EGL_GL_TEXTURE_LEVEL_KHR is nonzero, <target> is
          EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR or
          EGL_GL_TEXTURE_3D_KHR, and <buffer> is not the name of a complete
          GL texture object, the error EGL_BAD_PARAMETER is generated.

        * If EGL_GL_TEXTURE_LEVEL_KHR is 0, <target> is
          EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR or
          EGL_GL_TEXTURE_3D_KHR, <buffer> is the name of an incomplete GL
          texture object, and any mipmap levels other than mipmap level 0
          are specified, the error EGL_BAD_PARAMETER is generated.

        * If EGL_GL_TEXTURE_LEVEL_KHR is 0, <target> is
          EGL_GL_TEXTURE_2D_KHR or EGL_GL_TEXTURE_3D_KHR, <buffer> is not
          the name of a complete GL texture object, and mipmap level 0 is
          not specified, the error EGL_BAD_PARAMETER is generated.

        * If EGL_GL_TEXTURE_LEVEL_KHR is 0, <target> is
          EGL_GL_TEXTURE_CUBE_MAP_*_KHR, <buffer> is not the name of a
          complete GL texture object, and one or more faces do not have
          mipmap level 0 specified, the error EGL_BAD_PARAMETER is
          generated.

        * If <target> is EGL_GL_TEXTURE_2D_KHR,
          EGL_GL_TEXTURE_CUBE_MAP_*_KHR, EGL_GL_RENDERBUFFER_KHR or
          EGL_GL_TEXTURE_3D_KHR and <buffer> refers to the default GL
          texture object (0) for the corresponding GL target, the error
          EGL_BAD_PARAMETER is generated.

        * If <target> is EGL_GL_TEXTURE_2D_KHR, EGL_GL_TEXTURE_CUBE_MAP_*_KHR,
          or EGL_GL_TEXTURE_3D_KHR, and the value specified in <attr_list>
          for EGL_GL_TEXTURE_LEVEL_KHR is not a valid mipmap level for the
          specified GL texture object <buffer>, the error EGL_BAD_MATCH is
          generated.

        * If <target> is EGL_GL_TEXTURE_3D_KHR, and the value specified in
          <attr_list> for EGL_GL_TEXTURE_ZOFFSET_KHR exceeds the depth
          of the specified mipmap level-of-detail in <buffer>, the error
          EGL_BAD_PARAMETER is generated."


Issues

    1.  What should happen if an application attempts to create an
        EGLImage from a default OpenGL object (i.e., objects with
        a name of 0)?

        SUGGESTION:  Disallow this operation, and generate an error.

    2.  What happens when one of
            glTexImage2D
            glCopyTexImage2D
            glCompressedTexImage2D
            glTexImage3D
            glCopyTexImage3D
            glCompressedTexImage3D
        is called on a texture which has a mipmap level which is an EGLImage
        sibling?

        RESOLVED: the EGLImage sibling is orphaned.  The mipmap level and the
        EGLImage no longer have any connection.

    3.  What happens when one of
            glTexSubImage2D
            glCopyTexSubImage2D
            glCompressedTexSubImage2D
            glTexSubImage3D
            glCopyTexSubImage3D
            glCompressedTexSubImage3D
        is called on a texture which has a mipmap level which is an EGLImage
        sibling?

        RESOLVED: the EGLImage sibling is NOT orphaned.  The mipmap level
        remains an EGLImage sibling.

    4.  What happens when glGenerateMipmaps is called on a texture which has a
        mipmap level which is an EGLImage sibling?

        RESOLVED: If the texture is already complete, then the EGLImage
        sibling is not orphaned, and the mipmap level remains an EGLImage
        sibling.  However, if the texture was not complete then the
        EGLImage sibling IS orphaned. This is because the implementation
        will implicitly alter the structure of the mipmap levels.

    5.  What happens when the GL_GENERATE_MIPMAP bit causes a texture to be
        respecified.

        RESOLVED: If the texture is already complete, then the EGLImage
        sibling is not orphaned, and the mipmap level remains an EGLImage
        sibling.  However, if the texture was not complete then the
        EGLImage sibling IS orphaned. This is because the implementation
        will implicitly alter the structure of the mipmap levels.

    6.  Can an EGLImage be created from a multisampled GL image?

        RESOLVED: NO. Attempting to create an EGLImage from a multisampled
        GL renderbuffer is now an error. Attempting to create from a
        multisampled OpenGL texture image is not possible because none of
        the multisampled <target>s are supported.

    7.  Are all types of two-dimensional GL images which might
        be associated with EGLImages allowed?

        Not yet. We could add new variants of these extensions to support
        other image types such as rectangular and 2D array slice textures,
        but haven't yet seen a need to do so.

Revision History

#13 (Jon Leech, October 9, 2013) - Define interactions with and support for
    OpenGL and OpenGL ES 3.0, in addition to OpenGL ES 1/2. Add issue 7 (Bug
    10728).
#12 (Jon Leech, September 16, 2013) - Add error when specifying a
    renderbuffer <target> and passing a multisampled renderbuffer object.
    Add issue 6 describing lack of support for multisampled EGLImages (Bug
    10728).
#11 (Jon Leech, June 26, 2013) - Add error when specifying a renderbuffer
    <target> and not passing a renderbuffer object (Bug 10384).
#10 (Jon Leech, June 13, 2013) - Add a "Valid Targets" column to table bbb
    for new attributes, matching proposed changes in EGL_KHR_image_base (Bug
    10151).
#9  (Jon Leech, March 28, 2012)
    - Fix spelling of *CUBE_MAP* tokens (from CUBEMAP) to agree with
      eglext.h.
#8  (Jon Leech, February 4, 2009)
    - Change "non-default ... texture object" to "nonzero".
#7  (Bruce Merry, January 20, 2009)
    - Minor wording improvements on issues 4 and 5.
#6  (Acorn Pooley, January 13, 2009)
    - Modify completion requirement so textures with only mipmap level 0 can
      be EGLImage source siblings.  Add issues 2-5.
#5  (Jon Leech, October 8, 2008)
    - Updated status (approved as part of OpenKODE 1.0)
#4  (Jon Leech, April 7, 2007)
    - Assigned enumerant values
    - Added OpenKODE 1.0 Provisional disclaimer
#3  (December 14, 2006)
    - Changed requirement to egl 1.2 to include EGLClientBuffer type.
    - formatting to keep within 80 columns
    - added error condition descriptions for <dpy> and <ctx>
    - changed error condition for EGL_GL_TEXTURE_ZOFFSET_KHR too big to
        be EGL_BAD_PARAMETER
#2  (November 27, 2006)
    - Changed OES token to KHR
