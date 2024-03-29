# KHR_image

Name

    KHR_image

Name Strings

    EGL_KHR_image

Contributors

    Jeff Juliano
    Gary King
    Jon Leech
    Jonathan Grant
    Barthold Lichtenbelt
    Aaftab Munshi
    Acorn Pooley
    Chris Wynn

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)
    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete.
    Approved by the Khronos Board of Promoters on February 11, 2008.

    Rewritten in terms of split functionality in KHR_image_base and
    KHR_image_pixmap, approved by the Khronos Technical Working Group
    on November 19, 2008.

Version

    Version 11, November 12, 2008

Number

    EGL Extension #3

Dependencies

    EGL 1.2 is required.

    An EGL client API, such as OpenGL ES or OpenVG, is required.

    The specifications of EGL_KHR_image_base and EGL_KHR_image_pixmap are
    required to determine the specification of this extension, although
    those extentions may not be supported.

    This extension is written against the wording of the EGL 1.2
    Specification.

Overview

    This extension defines a new EGL resource type that is suitable for
    sharing 2D arrays of image data between client APIs, the EGLImage,
    and allows creating EGLImages from EGL native pixmaps.

New Types

    As defined by EGL_KHR_image_base.

New Procedures and Functions

    As defined by EGL_KHR_image_base.

New Tokens

    As defined by EGL_KHR_image_base and EGL_KHR_image_pixmap, with the
    exception that EGL_IMAGE_PRESERVED_KHR is not defined by this
    extension.

Additions to Chapter 2 of the EGL 1.2 Specification (EGL Operation)

    EGL_KHR_image is equivalent to the combination of the functionality
    defined by EGL_KHR_image_base and EGL_KHR_image_pixmap, with the
    exception that if EGL_KHR_image is supported and EGL_KHR_image_base
    is not, the attribute EGL_IMAGE_PRESERVED_KHR is not accepted in
    <attrib_list>, However, the default value of this attribute is still
    EGL_FALSE. In this situation, image preservation is always disabled.

Issues

    None (but see the issues lists for EGL_KHR_image_base and
    EGL_KHR_image_pixmap).

Revision History

#11 (Jon Leech, November 12, 2008)
    - Clarified image preservation behavior when using this extension.
#10 (Jon Leech, October 22, 2008)
    - Update description of interactions with EGL_KHR_image_base now
      that the default value of EGL_IMAGE_PRESERVED_KHR is always FALSE.
#9  (Jon Leech, October 21, 2008)
    - Split functionality into new extensions EGL_KHR_image_base and
      EGL_KHR_image_pixmap, and defined legacy non-preserved image behavior
      when this extension is supported.
#8  (Jon Leech, October 8, 2008)
    - Updated status (approved as part of OpenKODE 1.0)
#7  (Jon Leech, November 20, 2007)
    - Corrected 'enum' to 'EGLenum' in prototypes.
#6  (Jon Leech, April 5, 2007)
    - Assigned enumerant values
    - Added OpenKODE 1.0 Provisional disclaimer
#5  (Jon Leech, February 26, 2007)
    - Add eglCreateImageKHR error if native pixmaps are not supported by
      EGL.
#4  (December 14, 2006)
    - Replaced EGL_OUT_OF_MEMORY error with EGL_BAD_ALLOC
    - add "egl" and "EGL" to names to be consistant with spec
    - formatting to keep within 80 columns
    - Changed requirement to egl 1.2 to include EGLClientBuffer type.
    - clarified some unclear error cases
    - added some new error cases related to <dpy> and <ctx>
    - add <dpy> param to eglCreateImageKHR and eglDestroyImageKHR
#3  (November 27, 2006)
    - Converted OES token to KHR token
#2  (October 20, 2006)
    - Split out API-specific image source types (VG, GL, etc.) into
      individual extensions.
    - Merged CreateImage2DOES and CreateImage3DOES functions into
      a single CreateImageOES function with an attribute-value list.
    - Removed the minimum requirements section (2.5.3), since this
      doesn't make sense without the client-API specific extensions.
      The minimum requirements should be migrated to the client-API
      specific extension specifications.
    - Added EGL_NO_IMAGE_OES default object, used as return value for
      CreateImage*OES functions in the event of error conditions.
    - Reworded issue 5, to clarify that the buffer sub-object (i.e.,
      the unique resource specified by <ctx>, <target>, <buffer>,
      and <attrib_list>) specified in CreateImage may not already be
      an EGLImage sibling (either EGLImage source or EGLImage target).
#1  Original release
