# EXT_EGL_image_array

Name

    EXT_EGL_image_array

Name Strings

    GL_EXT_EGL_image_array

Contact

    Jeff Leger, Qualcomm Technologies Inc. (jleger@qti.qualcomm.com)

Contributors

    Sam Holmes
    Jesse Hall
    Tate Hornbeck
    Daniel Koch

Status

    Complete

Version

    Last Modified Date: July 28, 2017
    Revision: 0.5

Number

    OpenGL ES Extension #278

Dependencies

    OpenGL ES 2.0 is required.

    Requires EGL 1.2 and either the EGL_KHR_image or EGL_KHR_image_base
    extensions as well as OES_EGL_image.

    This extension is written against the OpenGL ES 2.0 specification and
    the OES_EGL_image extension.

Overview

    This extension adds functionality to that provided by OES_EGL_image in
    order to support EGLImage 2D arrays. It extends the existing
    EGLImageTargetTexture2DOES entry point from OES_EGL_image. Render buffers
    are not extended to include array support.

    EGLImage 2D arrays can be created using extended versions of eglCreateImageKHR.
    For example, EGL_ANDROID_image_native_buffer can import image array native buffers
    on devices where such native buffers can be created.

New Procedures and Functions

    None.

New Tokens

    None.

Additions to Chapter 3 of the OpenGL ES 2.0 Specification

    In section 3.8.2 within the specification added by OES_EGL_Image:

        "Currently, <target> must be TEXTURE_2D or TEXTURE_2D_ARRAY."

Errors

    GL_INVALID_ENUM is generated by EGLImageTargetTexture2DOES if
     <target> is not TEXTURE_2D or TEXTURE_2D_ARRAY

    GL_INVALID_OPERATION is generated by EGLImageTargetTexture2DOES if
    <target> is not TEXTURE_2D_ARRAY and <image> has more than 1 layer.

Issues

    None.

Revision History

      Rev.  Date        Author    Changes
      ----  ----------  --------  -----------------------------------------
      0.1   06/03/2016  Sam       Initial draft
      0.2   03/09/2017  Sam       Update contact
      0.3   03/21/2017  Tate      Update errors
      0.4   03/28/2017  Jeff      Minor formatting updates.
      0.5   07/28/2017  Jeff      Fix reference to external extension.