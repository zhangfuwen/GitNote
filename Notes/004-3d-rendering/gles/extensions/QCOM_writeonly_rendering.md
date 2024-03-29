# QCOM_writeonly_rendering

Name

    QCOM_writeonly_rendering

Name Strings

    GL_QCOM_writeonly_rendering

Contributors

    Benj Lipchak (Original Author)
    Maurice Ribble

Contact

    Maurice Ribble (mribble 'at' qualcomm.com)

Notice

    Copyright Qualcomm 2009.

IP Status

    Qualcomm Proprietary.

Status

    Complete.

Version

    Last Modified Date: July 7, 2009
    Revision: #2

Number

    OpenGL ES Extension #61

Dependencies

    OpenGL ES 1.0 is required.

    This extension is written against the OpenGL ES 2.0 specification.

Overview

This extension defines a specialized "write-only" rendering mode that
may offer a performance boost for simple 2D rendering.

Some applications render large frame buffers with simple geometry, very
little overdraw, and no need for the following per-fragment stages:
depth/stencil testing, Multisampling, blending, and Logic Operations.
Applications rendering a windowed desktop or other 2D GUI might fit
into this profile.

Applications that match this profile can enable ``write-only'' rendering
mode.  Performance may be improved in this mode through single-pass
rendering directly to system memory as compared with multipass tile-based
rendering into on-chip memory.  However, since the write-path to system memory
is generally lower bandwidth, any gains are most likely for 2D applications
rendering to large frame buffers with little overdraw.

On some HW, the GPU is not able to simultaneously read and write to system
memory framebuffers, so enabling this mode also implicitly disables any per-
fragment operations that may read from the frame buffer.  In addition, this
mode implicitly disables any reads and writes from the depth buffer.

To enable write-only rendering, an OpenGL application will call
glEnable(GL_WRITEONLY_RENDERING_QCOM).  When write-only rendering is enabled,
the following per-fragment stages are disabled regardless of the associated
GL enables:  multisample, depth_test, stencil_test, blending, and color_logic_Op.
In addition, write-only rendering will implicitly disable all depth writes
regardless of the value set via glDepthMask().    The alpha_test and scissor_test
stages function as normal.

To disable fast-rendering, call glDisable(GL_ WRITEONLY_RENDERING_QCOM).  Any
stages previously disabled by write-only rendering will return to their current
GL state.

IP Status

    There is no intellectual property associated with this extension.

Issues

(1)  How can alpha-blending be implemented with write-only rendering?

RESOLVED:  For ES2.0 applications that want to use write-only rendering combined
with alpha-blending, this must be done in the GLSL fragment shader, by binding the
RT as a texture.  Application will handle all coherency issues using glFinish().

(2)  Can write-only logicOps be supported?

RESOLVED:  It is possible we could allow some LogicOps (those that don't need to
read from the destination).  For simplicity, the extension doesn't allow any LogicOps
with write-only rendering.

(3)  Can Multisample be supported?

RESOLVED:  Write-only rendering with multisampling enabled may not work on
all HW.  For now, MSAA is not allowed with WRITEONLY_RENDERING.

(4)  Can the depth or stencil buffer be cleared while write-only rendering is
enabled?

RESOLVED:  No, while write-only rendering is enabled, all reads and writes to the
depth/stencil buffer implicitly disabled.  Clears of the depth buffer are ignored.
For example, calling glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT |
GL_STENCIL_BUFFER_BIT) would only clear the color buffer.


New Procedures and Functions


New Tokens

    Accepted by the <cap> parameter of Enable, Disable, and by the

        WRITEONLY_RENDERING_QCOM                      0x8823


    None.


Additions to Chapter 4 of the OpenGL 1.4 Specification (Per-Fragment
Operations and the Frame Buffer)

    TBD

Additions to Chapter 5 of the OpenGL 1.4 Specification (Special
Errors

    None.

New State


Get Value                                 Type            Command      Value
---------                                 ----            -------     -------
WRITEONLY_RENDERING_QCOM                  bool          IsEnabled


Revision History
#03    09/30/2009    Maurice Ribble     Fixed some AMD stuff I missed.
#02    07/07/2009    Maurice Ribble     Update due to the AMD->Qualcomm move.
#01    ??            Benj Lipchak       Initial version.
