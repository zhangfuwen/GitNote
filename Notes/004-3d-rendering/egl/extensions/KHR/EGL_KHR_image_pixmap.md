# KHR_image_pixmap

Name

    KHR_image_pixmap

Name Strings

    EGL_KHR_image_pixmap

Contributors

    Jeff Juliano
    Gary King
    Jon Leech
    Jonathan Grant
    Barthold Lichtenbelt
    Aaftab Munshi
    Acorn Pooley
    Chris Wynn
    Ray Smith

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)
    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright (c) 2008-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete. Functionality approved (as part of KHR_image) by the
    Khronos Board of Promoters on February 11, 2008.

    Split into KHR_image_base and KHR_image_pixmap approved by the
    Khronos Technical Working Group on November 19, 2008.

Version

    Version 5, November 13, 2013

Number

    EGL Extension #9

Dependencies

    EGL 1.2 is required.

    EGL_KHR_image_base is required.

    The EGL implementation must define an EGLNativePixmapType (although it
    is not required either to export any EGLConfigs supporting rendering to
    native pixmaps, or to support eglCreatePixmapSurface).

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    This extension allows creating an EGLImage from a native pixmap
    image.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <target> parameter of eglCreateImageKHR:

        EGL_NATIVE_PIXMAP_KHR                          0x30B0

Additions to Chapter 2 of the EGL 1.4 Specification (EGL Operation)

    Rename section 2.2.2.1 "Native Surface Coordinate Systems" to "Native
    Surface and EGLImage Pixmap Coordinate Systems" and add to the end of
    the section:

   "EGLImages created with target EGL_NATIVE_PIXMAP_KHR share the same
    coordinate system as native pixmap surfaces. When that coordinate system
    is inverted, client APIs must invert their <y> coordinate when accessing
    such images as described above."
    
    Add to section 2.5.1 "EGLImage Specification" (as defined by the
    EGL_KHR_image_base specification), in the description of
    eglCreateImageKHR:

   "Values accepted for <target> are listed in Table aaa, below.

      +-------------------------+--------------------------------------------+
      |  <target>               |  Notes                                     |
      +-------------------------+--------------------------------------------+
      |  EGL_NATIVE_PIXMAP_KHR  |   Used for EGLNativePixmapType objects     |
      +-------------------------+--------------------------------------------+
       Table aaa.  Legal values for eglCreateImageKHR <target> parameter

    ...

    If <target> is EGL_NATIVE_PIXMAP_KHR, <dpy> must be a valid display, <ctx>
    must be EGL_NO_CONTEXT; <buffer> must be a handle to a valid
    NativePixmapType object, cast into the type EGLClientBuffer; and
    attributes other than EGL_IMAGE_PRESERVED_KHR are ignored."

    Add to the list of error conditions for eglCreateImageKHR:

      "* If <target> is EGL_NATIVE_PIXMAP_KHR and <buffer> is not a
         valid native pixmap handle, or if <buffer> is a native pixmap
         whose color buffer format is incompatible with the system's
         EGLImage implementation, the error EGL_BAD_PARAMETER is
         generated.

       * If <target> is EGL_NATIVE_PIXMAP_KHR, and <dpy> is not a valid
         EGLDisplay object the error EGL_BAD_DISPLAY is generated.

       * If <target> is EGL_NATIVE_PIXMAP_KHR, and <ctx> is not EGL_NO_CONTEXT,
         the error EGL_BAD_PARAMETER is generated.

       * If <target> is EGL_NATIVE_PIXMAP_KHR, and <buffer> is not a handle
         to a valid NativePixmapType object, the error EGL_BAD_PARAMETER
         is generated."

Issues

    1) Should this specification allow EGLImages to be created from native
       pixmaps which already have a pixmap surface associated with them, and
       vice versa?

       RESOLVED: Yes. There are practical usecases for this, and it is
       already the application's responsibility to handle any format
       mismatch or synchronization issues that this may allow.

Revision History

#5  (Jon Leech, November 13, 2013)
    - Add Issue #1 regarding use cases for multiple EGL consumer/producers
      of a native pixmap (Bug 7779).

#4  (Jon Leech, October 16, 2013)
    - Add language allowing native pixmap and client API image y coordinate
      convention to differ. Re-base extension against EGL 1.4 (Bug 9701).

#3  (Jon Leech, November 25, 2008)
    - Remove dependency on EGLConfig in error conditions.

#2  (Jon Leech, November 12, 2008)
    - Clarified dependency on EGLNativePixmapType such that pixmap configs
      and surfaces are not required.

#1  (Jon Leech, October 21, 2008)
    - Split native pixmap functionality from EGL_KHR_image into a layered
      extension on EGL_KHR_image_base, and note interaction with the new
      EGL_IMAGE_PRESERVED_KHR attribute.
