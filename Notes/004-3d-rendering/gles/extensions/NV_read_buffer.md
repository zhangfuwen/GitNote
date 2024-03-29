# NV_read_buffer

Name

    NV_read_buffer
    NV_read_buffer_front

Name Strings

    GL_NV_read_buffer
    GL_NV_read_buffer_front

Contact

    Greg Roth, NVIDIA Corporation (groth 'at' nvidia.com)

Contributors

    Koji Ashida, NVIDIA Corporation
    Gregory Prisament, NVIDIA Corporation
    Greg Roth, NVIDIA Corporation
    James Helferty, NVIDIA Corporation
    Antoine Chauveau, NVIDIA Corporation

Status

    Complete.

Version

    Last Modified Date: September 27, 2013
    NVIDIA Revision: 7.0

Number

    OpenGL ES Extension #93

Dependencies

    Written against the OpenGL ES 2.0 Specification.

    NV_draw_buffers affects this extension.

Overview

    Unextended OpenGL ES 2.0 only supports using ReadPixels to read from
    the default color buffer of the currently-bound framebuffer.
    However, it is useful for debugging to be able to read from
    non-default color buffers.  Particularly, when the NV_draw_buffers
    extension is supported, each framebuffer may contain multiple color
    buffers. This extension provides a mechanism to select which color
    buffer to read from.

    This document describes two extensions to allow an implementation to
    support a subset of the total functionality.

    The NV_read_buffer extension adds the command ReadBufferNV, which is
    used to select which color buffer of the currently-bound framebuffer
    to use as the source for subsequent calls to ReadPixels,
    CopyTexImage2D, and CopyTexSubImage2D. If the system-provided
    framebuffer is bound, then ReadBufferNV accepts value BACK. If a
    user-created FBO is bound, then ReadBufferNV accepts COLOR_ATTACHMENT0.
    Additionally, if the NV_draw_buffers extension is supported,
    ReadBufferNV accepts COLOR_ATTACHMENTn_NV (n is 0 to 15).

    The NV_read_buffer_front extension requires NV_read_buffer and adds
    the ability to select the system-provided FRONT color buffer as the
    source for read operations when the system-provided framebuffer is
    bound and contains both a front and back buffer.

New Procedures and Functions

    void ReadBufferNV(GLenum mode)

New Tokens

    Accepted by the <pname> parameter of GetIntegerv:

        READ_BUFFER_NV                 0x0C02

Changes to Chapter 4 of the OpenGL ES 2.0 Specification
(Per-Fragment Operations and the Framebuffer)

    Section 4.3.1 (Reading Pixels), subsection "Obtaining Pixels from
    the Framebuffer" add:

    For color formats, the read buffer from which values are obtained
    is one of the color buffers; the selection of color buffer for the
    bound framebuffer object is controlled with ReadBufferNV.

    The command

        void ReadBufferNV(enum src);

    takes a symbolic constant as argument. <src> must be FRONT, BACK,
    NONE, COLOR_ATTACHMENT0, or COLOR_ATTACHMENTi_NV, where <i> is the
    index of the color attachment point. Otherwise, an INVALID_ENUM
    error is generated. Further, the acceptable values for <src> depend
    on whether the GL is using the default framebuffer (i.e.
    FRAMEBUFFER_BINDING is zero), or a framebuffer object (i.e.
    FRAMEBUFFER_BINDING is non-zero) and whether the default framebuffer
    is single or double buffered. For more information about framebuffer
    objects, see section 4.4.

    If the object bound to FRAMEBUFFER_BINDING is not framebuffer
    complete (as defined in section 4.4.5), then ReadPixels generates
    the error INVALID_FRAMEBUFFER_OPERATION. If ReadBufferNV is supplied
    with a constant that is neither legal for the default framebuffer,
    nor legal for a framebuffer object, then the error INVALID_ENUM
    results.

    When FRAMEBUFFER_BINDING is zero, i.e. the default framebuffer,
    <src> must be FRONT, BACK or NONE. If the requested buffer is
    missing, the error INVALID_OPERATION is generated. If there is a default
    framebuffer associated with the context, the initial setting for
    READ_BUFFER_NV is BACK, otherwise it is NONE.

    When the GL is using a framebuffer object, <src> must be NONE,
    COLOR_ATTACHMENT0, or COLOR_ATTACHMENTi_NV, where <i> is the index
    of the color attachment point. Specifying COLOR_ATTACHMENT0 or
    COLOR_ATTACHMENTi_NV enables reading from the image attached to the
    framebuffer at COLOR_ATTACHMENTi_NV. For framebuffer objects, the
    initial setting for READ_BUFFER_NV is COLOR_ATTACHMENT0.

    ReadPixels generates an INVALID_OPERATION error if it attempts to
    select a color buffer while READ_BUFFER_NV is NONE or if the GL is
    using a framebuffer object (i.e., READ_FRAMEBUFFER_BINDING is non-zero)
    and the read buffer selects an attachment that has no image attached.

    Section 4.3.2 (Pixel Draw/Read State) Replace first paragraph:

    The state required for pixel operations consists of the parameters
    that are set with PixelStore. This state has been summarized in
    tables 3.1. Additional state includes an integer indicating the
    current setting of ReadBufferNV. State set with PixelStore is GL
    client state.


Dependencies on NV_read_buffer_front:

    If NV_read_buffer_front is not supported, add to the third paragraph
    describing ReadBufferNV:

    If <src> is FRONT, the error INVALID_ENUM is generated.

Dependencies on NV_draw_buffers:

    If NV_draw_buffers is not supported, change all references to
    "COLOR_ATTACHMENTi_NV, where <i> is the index of the color attachment
    point" or simply "COLOR_ATTACHMENTi_NV" to "COLOR_ATTACHMENT0". 

New State

    Add Table 6.X Framebuffer (State per framebuffer object):

        State           Type  Get Command Initial Value Description 
        --------------- ---- ------------ ------------- -----------
        READ_BUFFER_NV  Z10*  GetIntegerv see 4.2.1     Read source buffer

Issues

    1. Should we use ReadBufferNV to specify whether ReadPixels reads
      from the window system provided framebuffer or FBO?

      No. The switching is automatic with FBO binding. The read buffer
      state belongs to the rendering surface, so switching the rendering
      surface automatically switches which read buffer to use.

      This is consistent with the behavior of OpenGL 2.0 with the
      ARB_framebuffer_object extension and unextended OpenGL 3.0.

    2. Should we have FRONT/BACK, LEFT/RIGHT buffer enums for <mode>
      parameter of ReadBufferNV to be used with window system provided
      framebuffers?

      OpenGL ES 2.0 does not support stereo framebuffers, so for now we
      only support FRONT and BACK.

    3. Why separate NV_read_buffer and NV_read_buffer_front?

      SUGGESTION: Some platforms, such as those with a compositing
      window system, may be unable to read from the front buffer.
      However, we would like to allow these platforms to read from any
      of the buffers drawn to using the NV_draw_buffers extension.

    4. Should this extension allow reading from depth and stencil buffers?

      While originally part of this document, support for reading from
      depth and stencil buffers has been moved to the
      NV_read_depth_stencil extension. It is clearer to devote one
      document to the re-introduction of ReadBuffer, and a separate
      document to legalizing new format and type combinations for
      ReadPixels.

    5. Should ReadBufferNV() pass if READ_BUFFER points to a non-
       existent buffer?
       
      Early drivers followed the precedent set by Issue 55 of the
      EXT_framebuffer_object spec; ReadBufferNV() would cause an error if
      a FBO was bound and the requested buffer did not exist.

      OpenGL ES 3.0 and OpenGL 4.3 allow it to pass.

      RESOLVED: Behavior should match OpenGL ES 3.0. Application developers
      are cautioned that early Tegra drivers may exhibit the previous
      behavior.

    6. What should happen if COLOR_ATTACHMENT0, the default ReadBufferNV
       is not bound and ReadBufferNV() gets called on this attachment?

      Behavior matches the resolution of Issue 5.
      
    7. Version 6 of this specification isn't compatible with OpenGL ES 3.0.
       For contexts without a back buffer, this extension makes FRONT the
       default read buffer. ES 3.0 instead calls it BACK.
       How can this be harmonized?
       
      RESOLVED: Update the specification to match ES 3.0 behavior. This
      introduces a backwards incompatibility, but few applications are
      expected to be affected. In the EGL ecosystem where ES 2.0 is
      prevalent, only pixmaps have no backbuffer and their usage remains
      limited.
      

Revision History

    Rev.    Date      Author       Changes
    ----   --------   ---------    -------------------------------------
     7     09/27/13   achauveau    Harmonize BACK vs. FRONT selection
                                   with GLES 3.0. 
     6     07/11/13   jhelferty    Changes in behavior to match GLES 3.0
     5     06/07/11   groth        Responded to feedback. Clarified
                                   non-FBO behavior and state ownership.
                                   added a few issues.
     4     06/01/11   groth        Mostly rewrote spec edits to better
                                   match the spec and more clearly
                                   describe behavior.
     3     03/22/09   gprisament   Split depth & stencil reading into
                                   separate document
                                   (NV_read_depth_stencil).
                                   Inline dependencies on NV_draw_buffers
                                   Re-wrote overview section.
     2     07/03/08   kashida      Change to depend on
                                   NV_packed_depth_stencil2
     1     06/10/07   kashida      First revision.
