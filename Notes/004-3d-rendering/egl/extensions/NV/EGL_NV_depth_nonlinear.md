# NV_depth_nonlinear

Name

    NV_depth_nonlinear

Name Strings

    GL_NV_depth_nonlinear
    EGL_NV_depth_nonlinear

Contact

    Gary King, NVIDIA Corporation (gking 'at' nvidia.com)

Notice

    Copyright NVIDIA Corporation, 2005 - 2007.

Status

    NVIDIA Proprietary

Version

    Last Modified: 2007/03/20
    NVIDIA Revision: 1.0

Number

    EGL Extension #18
    OpenGL ES Extension #73

Dependencies

    Written based on the wording of the OpenGL 2.0 Specification and
    EGL 1.2 Specification.

    Requires EGL 1.1.

    Requires OpenGL-ES 1.0.

    OES_framebuffer_object affects the wording of this specification.

Overview

    Due to the perspective divide, conventional integer Z-buffers have
    a hyperbolic distribution of encodings between the near plane
    and the far plane.  This can result in inaccurate depth testing,
    particularly when the number of depth buffer bits is small
    and objects are rendered near the far plane.

    Particularly when the number of depth buffer bits is limited
    (desirable and/or required in low-memory environments), artifacts
    due to this loss of precision may occur even with relatively
    modest far plane-to-near plane ratios (e.g., greater than 100:1).

    Many attempts have been made to provide alternate encodings for
    Z-buffer (or alternate formulations for the stored depth) to
    reduce the artifacts caused by perspective division, such as
    W-buffers, Z-complement buffers and floating-point 1-Z buffers.

    This extension adds a non-linear encoded Z buffer to OpenGL,
    which can improve the practically useful range of, e.g. 16-bit
    depth buffers by up to a factor of 16, greatly improving depth
    test quality in applications where the ratio between the near
    and far planes can not be as tightly controlled.

IP Status

    NVIDIA Proprietary

New Procedures and Functions

    None

New Tokens

    Accepted as a valid sized internal format by all functions accepting
    sized internal formats with a base format of DEPTH_COMPONENT

        DEPTH_COMPONENT16_NONLINEAR_NV     0x8E2C

    Accepted by the <attrib_list> parameter of eglChooseConfig and
    eglCreatePbufferSurface, and by the <attribute> parameter of
    eglGetConfigAttrib

        EGL_DEPTH_ENCODING_NV              0x30E2

    Accepted as a value in the <attrib_list> parameter of eglChooseConfig
    and eglCreatePbufferSurface, and returned in the <value> parameter
    of eglGetConfigAttrib

        EGL_DEPTH_ENCODING_NONE_NV         0
        EGL_DEPTH_ENCODING_NONLINEAR_NV    0x30E3

Changes to the OpenGL 2.0 Specification

    Add the following line to table 3.16 (p. 154)

    +--------------------------------+-----------------+------+
    |      Sized Internal Format     |  Base Internal  |  D   |
    |                                |  Format         | Bits |
    +--------------------------------+-----------------+------+
    | DEPTH_COMPONENT16_NONLINEAR_NV | DEPTH_COMPONENT |  16  |
    +--------------------------------+-----------------+------+

Changes to the EGL 1.2 Specification

    Add the following line to table 3.1 (p. 14)

    +--------------------------+------+---------------------------------------+
    |         Attribute        | Type | Notes                                 |
    +--------------------------+------+---------------------------------------+
    |   EGL_DEPTH_ENCODING_NV  | enum | Type of depth-buffer encoding employed|
    +--------------------------+------+---------------------------------------+

    Modify the description of the depth buffer in Section 3.4 (p. 15)

    "The depth buffer is used only by OpenGL ES.  It contains fragment depth 
    (Z) information generated during rasterization.  EGL_DEPTH_SIZE indicates 
    the depth of this buffer in bits, and EGL_DEPTH_ENCODING_NV indicates which
    alternate depth-buffer encoding (if any) should be used.  Legal values for
    EGL_DEPTH_ENCODING_NV are: EGL_DONT_CARE, EGL_DEPTH_ENCODING_NONE_NV and
    EGL_DEPTH_ENCODING_NONLINEAR_NV."

    Add the following line to table 3.4 (p. 20)

    +-----------------------+---------------+-----------+-------+----------+
    |       Attribute       |     Default   | Selection |  Sort |   Sort   |
    |                       |               |  Criteria | Order | Priority |
    +-----------------------+---------------+-----------+-------+----------+
    | EGL_DEPTH_ENCODING_NV | EGL_DONT_CARE |   Exact   |  None |     -    |
    +-----------------------+---------------+-------------------+----------+

Issues

    None

Revision History

#1.0 - 20.03.2007

   Renumbered enumerants.  Reformatted to 80 columns.
