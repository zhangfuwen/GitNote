# NV_fbo_color_attachments

Name

    NV_fbo_color_attachments

Name Strings

    GL_NV_fbo_color_attachments

Contributors
    Marek Zylak

Contact

    James Helferty, NVIDIA Corporation (jhelferty 'at' nvidia.com)

Status

    Complete.

Version

    Last Modified Date: Feb 2, 2015
    NVIDIA Revision: 4.0

Number

    OpenGL ES Extension #92

Dependencies

    The extension is written against the OpenGLES 2.0 specification.

    EXT_discard_framebuffer affects the definition of this extension

Overview

    This extension increases the number of available framebuffer object
    color attachment points.

IP Status

    NVIDIA Proprietary

New Procedures and Functions

    None

New Tokens

    Accepted by the <value> parameter of GetIntegerv:

        MAX_COLOR_ATTACHMENTS_NV                     0x8CDF

    Accepted by the <attachment> parameter of FramebufferRenderbuffer,
    FramebufferTexture2D and GetFramebufferAttachmentParameteriv:

        COLOR_ATTACHMENT0_NV                         0x8CE0
        COLOR_ATTACHMENT1_NV                         0x8CE1
        COLOR_ATTACHMENT2_NV                         0x8CE2
        COLOR_ATTACHMENT3_NV                         0x8CE3
        COLOR_ATTACHMENT4_NV                         0x8CE4
        COLOR_ATTACHMENT5_NV                         0x8CE5
        COLOR_ATTACHMENT6_NV                         0x8CE6
        COLOR_ATTACHMENT7_NV                         0x8CE7
        COLOR_ATTACHMENT8_NV                         0x8CE8
        COLOR_ATTACHMENT9_NV                         0x8CE9
        COLOR_ATTACHMENT10_NV                        0x8CEA
        COLOR_ATTACHMENT11_NV                        0x8CEB
        COLOR_ATTACHMENT12_NV                        0x8CEC
        COLOR_ATTACHMENT13_NV                        0x8CED
        COLOR_ATTACHMENT14_NV                        0x8CEE
        COLOR_ATTACHMENT15_NV                        0x8CEF

    The COLOR_ATTACHMENT0_NV constant is equal to the COLOR_ATTACHMENT0
    constant.

    Each COLOR_ATTACHMENTi_NV adheres to COLOR_ATTACHMENTi_NV =
    COLOR_ATTACHMENT0_NV + i.

Changes to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    Replace the second paragraph of Section 4.4.1 (Binding and Managing
    Framebuffer Objects) with the following:

    The namespace for framebuffer objects is the unsigned integers, with
    zero reserved by OpenGL ES to refer to the default framebuffer. A
    framebuffer object is created by binding an unused name to the
    target FRAMEBUFFER. The binding is effected by calling

        void BindFramebuffer(enum target, uint framebuffer);

    with target set to FRAMEBUFFER and framebuffer set to the unused
    name.  The resulting framebuffer object is a new state vector. There
    is a number of color attachment points, plus one each for the depth
    and stencil attachment points. The number of color attachment points
    is equal to the state of MAX_COLOR_ATTACHMENTS_NV.

New Implementation Dependent State

    Add to Table 6.18 (Implementation Dependent Values)

    Get value                Type Get Cmnd    Minimum Value Description             Sec.
    ------------------------ ---- ----------- ------------- -----------             -----
    MAX_COLOR_ATTACHMENTS_NV Z+   GetIntegerv 1             Number of framebuffer   4.4.1
                                                            color attachment points

Interactions with EXT_discard_framebuffer

    If EXT_discard_framebuffer is present, in Section 4.5 modify the
    arguments accepted by DiscardFramebufferEXT as follows: If a
    framebuffer object is bound to <target>,  <attachments> may also
    contain tokens COLOR_ATTACHMENTm_NV where <m> is greater than or
    equal to 0 and less than MAX_COLOR_ATTACHMENTS_NV.

Interactions with other extensions

    Rendering to color attachments COLOR_ATTACHMENT[1..15]_NV can be
    activated through the GL_NV_draw_buffers extension.

Issues

    None.

Revision History

    Rev.    Date      Author       Changes
    ----   --------   ---------    -------------------------------------
     4     02/02/15   jhelferty    Add interaction with DiscardFramebuffer.
     3     05/10/10   mzylak       Fixed references to core specification.
     2     05/06/10   mzylak       Small fixes and clarifications.
     1     05/05/10   mzylak       First revision.
