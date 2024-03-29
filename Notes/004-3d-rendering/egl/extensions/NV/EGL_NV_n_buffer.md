# NV_triple_buffer

Name

    NV_triple_buffer
    NV_quadruple_buffer

Name Strings

    EGL_NV_triple_buffer
    EGL_NV_quadruple_buffer

Contributors

    Daniel Kartch, NVIDIA Corporation
    Tom McReynolds, NVIDIA Corporation
    Santanu Thangaraj, NVIDIA Corporation

Contact

    Daniel Kartch, NVIDIA Corporation (dkartch 'at' nvidia.com)

Status

    Complete.

Version

    Version 1 - February 28, 2019

Number

    133

Extension Type

    EGL display extension

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.3
    Specification.

Overview

    NV_triple_buffer and NV_quadruple_buffer allow applications to
    request additional back buffers, in order to produce greater and
    less variable frame rates.

    This document describes two related extensions, one dependent on
    the other. Implementations may choose to support only
    NV_triple_buffer and not NV_quadruple_buffer, but not vice versa.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Added by NV_triple_buffer:

      Accepted as a value for EGL_RENDER_BUFFER in the <attrib_list>
      parameter of eglCreateWindowSurface:

          EGL_TRIPLE_BUFFER_NV                    0x3230

    Added by NV_quadruple_buffer:

      Accepted as a value for EGL_RENDER_BUFFER in the <attrib_list>
      parameter of eglCreateWindowSurface:

          EGL_QUADRUPLE_BUFFER_NV                 0x3231

Additions to the EGL 1.3 Specification:

    Insert after third sentence of second paragraph of Section 2.2.2
    (Rendering Models):

        Windows may have more than one back buffer, allowing rendering
        of a new frame to proceed while the copy requested by 
        eglSwapBuffers is still pending.

    Replace the third sentence of the EGL_RENDER_BUFFER description in
    Section 3.5.1 (Creating On-Screen Rendering Surfaces):

        If its value is EGL_BACK_BUFFER, EGL_TRIPLE_BUFFER_NV, or
        EGL_QUADRUPLE_BUFFER_NV, then client APIs should render into
        the current back buffer. The implementation should provide
        at least one, two, or three back buffers, respectively, which
        will be used in rotation each frame.

    Change first sentence of third bullet point of eglQueryContext
    description in Section 3.7.4 (Context Queries):

        If the context is bound to a window surface, then either
        EGL_SINGLE_BUFFER, EGL_BACK_BUFFER, EGL_TRIPLE_BUFFER_NV, or
        EGL_QUADRUPLE_BUFFER_NV may be returned.

    Replace first sentence of eglSwapBuffers description in
    Section 3.9.1
    (Posting to a Window):

        If surface is a back-buffered window surface, then the current
        color buffer is copied to the native window associated with
        that surface. If there is more than one back buffer, then the
        next color buffer in rotation becomes current, and rendering
        of the next frame may proceed before the copy takes place,
        provided any previous swaps from the new current buffer have
        completed.

Issues

    1. Why do we need triple-buffering?

       RESOLVED: With only a single back buffer and a non-zero swap
       interval, eglSwapBuffers must block rendering to the back-
       buffer until the copy has completed. This can leave the CPU
       and/or GPU idle, wasting valuable compute time, and possibly
       cause the next frame to be delivered later than otherwise could
       have been. Additional buffers allow rendering to continue even
       when a frame is awaiting display, maximizing our use of
       computational resources.

    2. Why quadruple-buffering? Isn't triple-buffering enough to
       produce frames as fast as the processor(s) and swap interval
       allow?

       RESOLVED: When there is only a single rendering stream
       operating on a system, triple-buffering is sufficient. However,
       if other threads are contending for resources, variable 
       latencies may be introduced. This is especially problematic
       with video, where any deviation in frame rate from the recorded
       media can produce visible artifacts. Additional buffers smooth
       out these latencies, allowing a steady frame rate.

    3. Then why not arbitrary n-buffering?

       RESOLVED: The TRIPLE/QUADRUPLE buffer specification fits nicely
       into the RENDER_BUFFER attribute already in use for
       eglCreateWindowSurface. Arbitrary buffer counts would require a
       new attribute. Additionally, case studies indicated no
       significant benefit to using more than three back buffers,
       especially when factoring in the added memory cost.

Revision History

    #2 (February 28, 2019) Santanu Thangaraj
       - Marked issues 1,2 and 3 as resolved.
       - Included extension type section.
       - Corrected line length violations.

    #1 (August 12, 2008) Daniel Kartch
       - Initial Draft
