# NV_read_depth

Name

    NV_read_depth
    NV_read_stencil
    NV_read_depth_stencil

Name Strings

    GL_NV_read_depth
    GL_NV_read_stencil
    GL_NV_read_depth_stencil

Contact

    Greg Roth, NVIDIA Corporation (groth 'at' nvidia.com)

Contributors

    Koji Ashida, NVIDIA Corporation
    Gregory Prisament, NVIDIA Corporation
    Greg Roth, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date: November 9, 2021
    NVIDIA Revision: 5.0

Number

    OpenGL ES Extension #94

Dependencies

    Written against the OpenGL ES 2.0 Specification.

    NV_read_depth_stencil requires OES_packed_depth_stencil.

Overview

    Unextended OpenGL-ES 2.0 only supports using ReadPixels to read from the
    default color buffer of the currently-bound framebuffer.  However, it is
    useful for debugging to be able to read from depth and stencil buffers.
    This extension re-introduces these features into OpenGL-ES 2.0.

    This document describes several extensions in order to allow an
    implementation to support a subset of the total functionality.

    The NV_read_depth extension allows reading from the depth buffer using
    ReadPixels.

    The NV_read_stencil extension allows reading from the stencil buffer using
    ReadPixels.

    The NV_read_depth_stencil extension allows reading from packed
    depth-stencil buffers using ReadPixels.


New Procedures and Functions

    None.

New Tokens

    None.

Changes to Chapter 4 of the OpenGL ES 2.0 Specification
(Per-Fragment Operations and the Framebuffer)


    Section 4.3.1 (Reading Pixels), replace description of ReadPixels:

        The arguments after x and y to ReadPixels are those described in
        section 3.6.2 defining pixel rectangles. The pixel storage modes
        that apply to ReadPixels are summarized in Table 4.3.

    Section 4.3.1 (Reading Pixels) subsection "Obtaining Pixels from
    the Framebuffer", replace the first paragraph with:

        If the <format> is DEPTH_COMPONENT, then values are obtained
        from the depth buffer. If there is no depth buffer, the error
        INVALID_OPERATION occurs. If the <type> is not UNSIGNED_SHORT,
        FLOAT, or UNSIGNED_INT_24_8_OES then the error INVALID_ENUM
        occurs. If the <type> is FLOAT and the depth buffer is not a
        float buffer, an INVALID_OPERATION error occurs. If the <type>
        is UNSIGNED_SHORT or UNSIGNED_INT_24_8_OES and the depth buffer
        is a float buffer, an INVALID_OPERATION error occurs. When the
        <type> is UNSIGNED_INT_24_8_OES the values read into the 8
        stencil bits are undefined.

        If the <format> is DEPTH_STENCIL_OES, then values are taken from
        both the depth buffer and the stencil buffer. If there is no
        depth buffer or if there is no stencil buffer, then the error
        INVALID_OPERATION occurs. If the <type> is not UNSIGNED_INT_24_-
        8_OES or FLOAT_32_UNSIGNED_INT_24_8_REV_NV, then the error
        INVALID_ENUM occurs. If the <type> is FLOAT_32_UNSIGNED_INT_24_-
        8_REV_NV and the depth buffer is not a float buffer, an INVALID_-
        OPERATION error occurs. If the <type> is UNSIGNED_INT_24_8_OES
        and the depth buffer is a float buffer, an INVALID_OPERATION
        error occurs.

        If the <format> is STENCIL_INDEX, then values are taken from
        the stencil buffer. If there is no stencil buffer, then the
        error INVALID_OPERATION occurs. If the <type> is not
        UNSIGNED_BYTE, then the error INVALID_ENUM occurs.

        If <format> is a color format, then red, green, blue, and alpha
        values are obtained from the color buffer at each pixel
        location. If the framebuffer does not support alpha values then
        the value that is obtained is 1.0. Only two combinations of
        color format and type are accepted. The first is format RGBA and
        type UNSIGNED_BYTE. The second is an implementation-chosen
        format from among those defined in table 3.4, excluding formats
        LUMINANCE and LUMINANCE_ALPHA. The values of format and type for
        this format may be determined by calling GetIntegerv with the
        symbolic constants IMPLEMENTATION_COLOR_READ_FORMAT and
        IMPLEMENTATION_COLOR_READ_TYPE, respectively. The implementation-
        chosen format may vary depending on the format of the currently
        bound rendering surface. Unsupported combinations of format and
        type will generate an INVALID_OPERATION error.

Dependencies on NV_read_depth:

    If NV_read_depth is not supported, the paragraph in "Obtaining Pixels
    from the Framebuffer" describing behavior when the <format> is
    DEPTH_COMPONENT is omitted.

Dependencies on NV_read_stencil:

    If NV_read_stencil is not supported, the paragraph in "Obtaining
    Pixels from the Framebuffer" describing behavior "when the <format>
    is STENCIL_INDEX" is omitted.

Dependencies on NV_read_depth_stencil:

    If NV_read_depth_stencil is not supported, the paragraph in
    "Obtaining Pixels from the Framebuffer" describing behavior "when
    the <format> is  DEPTH_STENCIL_OES is omitted and UNSIGNED_INT_24_8_EXT is not an
    accepted <value> when <format> is DEPTH_COMPONENT.

Issues

    1. Do we need to be able to read stencil buffers individually?

      On platforms that easily support reading stencil buffers
      individually, it is useful for debugging.  However, we do not want
      to require that it is supported on platforms using packed depth-
      stencil buffers.  Therefore we use multiple extension names.

    2. Should we have FRONT/BACK, LEFT/RIGHT buffer enums for <mode>
      parameter of ReadBufferNV to be used with window system provided
      framebuffers?

      OpenGL ES 2.0 does not support stereo framebuffers, so for now we
      only support FRONT and BACK.

    3. Should this extension add reading of coverage samples?

      No.  That should be left to the EGL_NV_coverage_sample
       specification.

    4. How does this extension interact with NV_read_buffer?

      NV_read_buffer adds support for the ReadBufferNV command, which is
      used to select which color buffer to read from.  It does not
      affect the behavior of this extension, since all framebuffers have
      at most one depth and one stencil buffer.


Revision History

    Rev.    Date      Author       Changes
    ----   --------   ---------    -------------------------------------
     5     11/09/21   Jon Leech    Removed references to
                                   NV_depth_buffer_float2, which was
                                   abandoned and never published.
                                   (see KhronosGroup/OpenGL-Registry#488).
     4     06/01/11    groth       Mostly rewrote spec edits to better
                                   match the spec and more clearly
                                   describe behavior. Reformatted.
     3     03/22/09    gprisament  Split from NV_read_buffer.
                                   Broken into several extension names.
                                   Re-wrote overview section.
     2     07/03/08    kashida     Change to depend on
                                   NV_packed_depth_stencil2
     1     06/10/07    kashida     First revision.

