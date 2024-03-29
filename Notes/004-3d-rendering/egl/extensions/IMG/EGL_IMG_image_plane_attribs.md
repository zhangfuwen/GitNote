# IMG_image_plane_attribs

Name

    IMG_image_plane_attribs

Name Strings

    EGL_IMG_image_plane_attribs

Contributors

    Ben Bowman
    Alistair Strachan

Contacts

    Tobias Hector, Imagination Technologies (tobias 'dot' hector 'at'
    imgtec 'dot' com)

Status

    Complete

Version

    Version 0.4, October 18, 2015

Number

    EGL Extension #95

Dependencies

    EGL_KHR_image_base is required.

    One of EGL_KHR_image, EGL_KHR_image_pixmap or
    EGL_ANDROID_image_native_buffer is required.

    This extension is written against the wording of the EGL 1.2
    Specification as modified by EGL_KHR_image_base,
    EGL_ANDROID_image_native_buffer and EGL_KHR_image_pixmap.
    This extension interacts with GL_OES_EGL_image and GL_EXT_texture_rg.

Overview

    This extension allows creating an EGLImage from a single plane of a
    multi-planar Android native image buffer (ANativeWindowBuffer) or
    a native pixmap (EGLNativePixmap).

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <attrib_list> parameter of eglCreateImageKHR:

        EGL_NATIVE_BUFFER_MULTIPLANE_SEPARATE_IMG        0x3105
        EGL_NATIVE_BUFFER_PLANE_OFFSET_IMG               0x3106

Additions to Chapter 2 of the EGL 1.2 Specification (EGL Operation)

    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

    Add the following to Table bbb (Legal attributes for eglCreateImageKHR
    <attr_list> parameter), Section 2.5.1 (EGLImage Specification)

      +-----------------------------+-------------------------+---------------------------+---------------+
      | Attribute                   | Description             | Valid <target>s           | Default Value |
      +-----------------------------+-------------------------+---------------------------+---------------+
      | EGL_NATIVE_BUFFER_MULTI     | Whether a multiplane    | EGL_NATIVE_BUFFER_ANDROID | EGL_FALSE     |
      | PLANE_SEPARATE_IMG          | native buffer should be | EGL_NATIVE_PIXMAP_KHR     |               |
      |                             | treated as separate     |                           |               |
      |                             | buffers                 |                           |               |
      |                             |                         |                           |               |
      | EGL_NATIVE_BUFFER_          | Which plane of a multi- | EGL_NATIVE_BUFFER_ANDROID | 0             |
      | PLANE_OFFSET_IMG            | plane native buffer is  | EGL_NATIVE_PIXMAP_KHR     |               |
      |                             | used as the EGLImage    |                           |               |
      |                             | source                  |                           |               |
      +-----------------------------+-------------------------+---------------------------+---------------+
      Table bbb. Legal attributes for eglCreateImageKHR <attrib_list> parameter

    ...

    If <target> is EGL_NATIVE_BUFFER_ANDROID or EGL_NATIVE_PIXMAP_KHR, and
    <buffer> is a handle to a valid multi-planar surface, such as a YUV420 2 or
    3 planar video surface, an EGLImage will be created from only one of the
    planes, as opposed to a single image representing all of the planes as is
    normally the case. The intention of this extension is that a call to
    glEGLImageTargetTexture2DOES or EGLImageTargetRenderbufferStorageOES with an
    EGLImage created from a single plane of a multiplanar buffer will result in
    a GL_RED or GL_RG  texture or renderbuffer, depending on the format of the
    multiplanar buffer. This allows an application to work directly in the YUV
    colorspace, rather than forcing a conversion to the linear RGB colorspace,
    potentially losing precision.

    The size of each image will represent the actual size of the data buffer
    for that plane which may mean that the size of an EGLImage created from
    plane 0 of a multi-planar buffer may not be the same as that of one
    created from plane 1, which is determined by the YUV's sampling ratio (e.g.
    a 420 will have planes 1 and 2, if present, represented by an image of half
    the width).

    Add to the list of error conditions for eglCreateImageKHR:

      "* If EGL_NATIVE_BUFFER_MULTIPLANE_SEPARATE_IMG is EGL_TRUE, and <target>
         is not EGL_NATIVE_BUFFER_ANDROID or EGL_NATIVE_PIXMAP_KHR, the error
         EGL_BAD_PARAMETER is generated.

       * If EGL_NATIVE_BUFFER_MULTIPLANE_SEPARATE_IMG is EGL_TRUE, and
         EGL_NATIVE_BUFFER_PLANE_OFFSET_IMG is greater than or equal to the
         number of planes in <buffer>, the error EGL_BAD_MATCH is generated.

       * If EGL_NATIVE_BUFFER_MULTIPLANE_SEPARATE_IMG is EGL_FALSE, and
         EGL_NATIVE_BUFFER_PLANE_OFFSET_IMG is greater than 0, the error
         EGL_BAD_PARAMETER is generated.

       * If EGL_NATIVE_BUFFER_MULTIPLANE_SEPARATE_IMG is EGL_TRUE, and the
         format of <buffer> is not supported by the implementation,
         EGL_BAD_PARAMETER is generated."

Dependencies on EGL_KHR_image_pixmap or EGL_KHR_image

    If neither of these extensions are supported, remove all references to
    native pixmaps and EGL_NATIVE_PIXMAP_KHR.

Dependencies on EGL_ANDROID_image_native_buffer

    If this extension is not supported, remove all references to
    ANativeWindowBuffer and EGL_NATIVE_BUFFER_ANDROID.

Issues

    None

Revision History

#0.4  (Tobias Hector, October, 2015)
    - Add interactions with EGL_KHR_image_pixmap/EGL_KHR_image
    - Added error language for unsupported formats
#0.3  (Jon Leech, June 13, 2013)
    - Add a "Valid Targets" column to table bbb for new attributes, matching
      proposed changes in EGL_KHR_image_base (Bug 10151). Note that this
      change implies a new error will be generated when <target> is not
      EGL_NATIVE_BUFFER_ANDROID and EGL_NATIVE_BUFFER_PLANE_OFFSET_IMG is
      specified in <attrib_list>; this falls out from the generic
      target-attribute matching error added to EGL_KHR_image_base.
#0.2  (Ben Bowman, May 30, 2012)
    - Fixed some typos
#0.1  (Ben Bowman, May 30, 2012)
    - First draft of extension .
