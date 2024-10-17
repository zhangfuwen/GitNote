# HI_clientpixmap

Name

    HI_clientpixmap

Name Strings

    EGL_HI_clientpixmap

Contributors

    Guillaume Portier

Contacts

    HI support. (support_renderion 'at' hicorp.co.jp)

Status

    Shipping (Revision 3).

Version

    Last Modified Date: June 7, 2010
    Revision 3

Number

    EGL Extension #24

Dependencies

    This extension is written against the wording of the EGL 1.4
    Specification.

Overview

    The extension specified in this document provide a mechanism for
    using memory allocated by the application as a color-buffer.


New Types

    EGLClientPixmapHI : specifies the width, height, stride, format 
    and memory pointer of the pixmap to be used by the function
    eglCreatePixmapSurfaceHI to create the PixmapSurface.

    Members:
        void*		pData;
             Pointer to a memory buffer allocated by the application
             that will contain the result of the drawing operations.
             It is up to the application to ensure that the buffer
             size corresponds to iHeight * iStride * sizeof(pixel).
        EGLint		iWidth;
             Width of the buffer in pixels.
        EGLint		iHeight;
             Height of the buffer in pixels. The height of the buffer
             can be negative; in that case the result of the
             drawing operations will be vertically swapped. When
			 positive, pData will point at the bottom-left corner
             of the image; when negative, to the top-left corner.
        EGLint		iStride;
	         Stride of the buffer, in pixels. It is important to note
             that each row of the buffer must start on 32-bit
             boundaries.

New Procedures and Functions

    eglCreatePixmapSurfaceHI : creates an EGL ClientPixmap from 
    an EGLClientPixmapHI structure. eglCreatePixmapSurfaceHI usage
    is identical to eglCreatePixmapSurface. In addition the ordering
    of the color components in the color buffer can be specified by
    the surface attribute described in the EGL_HI_colorformats
    extension.

    In order to update the pointer to the data of the surface, the application 
    can call eglSurfaceAttrib with the EGL_CLIENT_PIXMAP_POINTER_HI attribute.
    See below for an example.

New Tokens

    None.

Example


    EGLClientPixmapHI pixmap;
    EGLint attrib_list[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_PIXMAP_BIT,
        // Specifying ARGB as a color format
        EGL_COLOR_FORMAT_HI, EGL_COLOR_ARGB_HI,
        EGL_NONE
    };

    // 