# KHR_vg_parent_image

Name

    KHR_vg_parent_image

Name Strings

    EGL_KHR_vg_parent_image

Contributors

    Ignacio Llamas
    Gary King
    Chris Wynn

Contacts

    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on February 11, 2008.

Version

    Version 5, October 8, 2008

Number

    EGL Extension #4

Dependencies

    This extension requires EGL 1.2 and the EGL_KHR_image extension,
    and an OpenVG implementation

    This specification is written against the wording of the EGL Image
    (EGL_KHR_image) specification.

Overview

    This extension provides a mechanism for creating EGLImage objects
    from OpenVG VGImage API resources.  For an overview of EGLImage
    operation, please see the EGL_KHR_image specification.

New Types

    None

New Procedures and Functions

    None

New Tokens

          EGL_VG_PARENT_IMAGE_KHR                       0x30BA


Additions to the EGL Image (EGL_KHR_image) Specification:

    Add the following to Table aaa (Legal values for CreateImageKHR
    <target> parameter), Section 2.5.1 (EGLImage Specification)

      +--------------------------+--------------------------------------------+
      |  <target>                |  Notes                                     |
      +--------------------------+--------------------------------------------+
      |  EGL_VG_PARENT_IMAGE_KHR     |  Used for OpenVG VGImage objects           |
      +--------------------------+--------------------------------------------+

    Insert the following text after paragraph 3 ("If <target> is
    NATIVE_PIXMAP_KHR...") of Section 2.5.1 (EGLImage Specification):

    "If <target> is EGL_VG_PARENT_IMAGE_KHR, <dpy> must be a valid EGLDisplay,
    <ctx> must be a valid OpenVG API context on that display, and <buffer>
    must be a handle of a VGImage object valid in the specified context, cast
    into the type EGLClientBuffer.  Furthermore, the specified VGImage
    <buffer> must not be a child image (i.e. the value returned by
    vgGetParent(<buffer>) must be <buffer>).  If the specified VGImage
    <buffer> has any child images (i.e., vgChildImage has been previously
    called with the parent parameter set to <buffer>), all child images will
    be treated as EGLImage siblings after CreateImageKHR returns.  Any values
    specified in <attr_list> are ignored."

    Add the following errors to the end of the list in Section 2.5.1 (EGLImage
    Specification):

    "   * If <target> is EGL_VG_PARENT_IMAGE_KHR, and <dpy> is not a
          valid EGLDisplay, the error EGL_BAD_DISPLAY is generated.

        * If <target> is EGL_VG_PARENT_IMAGE_KHR and <ctx> is not a
          valid EGLContext, the error EGL_BAD_CONTEXT is generated.

        * If <target> is EGL_VG_PARENT_IMAGE_KHR and <ctx> is not a valid
          OpenVG context, the error EGL_BAD_MATCH is returned.

        * If <target> is EGL_VG_PARENT_IMAGE_KHR and <buffer> is not a handle
          to a VGImage object in the specified API context <ctx>, the error
          EGL_BAD_PARAMETER is generated.

        * If <target> is EGL_VG_PARENT_IMAGE_KHR, and the VGImage specified by
          <buffer> is a child image (i.e., vgGetParent(<buffer>) returns
          a different handle), the error EGL_BAD_ACCESS is generated."

Issues

    1.  Should this specification allow the creation of EGLImages
        from OpenVG child images?

        RESOLVED:  No.  It is believed that properly addressing the
        interaction of hardware restrictions (e.g., memory alignment),
        arbitrary image subrectangles, scissor rectangles and viewport
        rectangles may create an undue burden on implementers.  In the
        interest of providing a useful spec in a timely fashion, this
        functionality has been disallowed, with the possibility of
        providing it (if necessary) through a future layered extension.

        This restriction is shared with eglCreatePbufferFromClientBuffer;
        however, this specification allows EGL Images to be created
        from VGImages which have child images, functionality not
        previously available.

Revision History

#5  (Jon Leech, October 8, 2008)
    - Updated status (approved as part of OpenKODE 1.0)
#4  (Jon Leech, April 5, 2007)
    - Assigned enumerant values
    - Added OpenKODE 1.0 Provisional disclaimer
#3  (December 14, 2006)
    - Changed requirement to egl 1.2 to include EGLClientBuffer type.
    - added error condition descriptions for <dpy> and <ctx>
#2    (November 27, 2006)
    - Changed OES token to KHR
