# MESA_framebuffer_swap_xy

Name

    MESA_framebuffer_swap_xy

Name Strings

    GL_MESA_framebuffer_swap_xy

Contact

    Cici Ruan <ruanc@chromium.org>

Contributors

    Cici Ruan, Google
    Kristian Kristensen‎, Google
    Fritz Koenig, Google
    Rob Clark, Google
    Chad Versace, Google

Status

    Proposal

Version

    Last Modified Date: April 8, 2020
    Revision: 1

Number

    OpenGL Extension 549
    OpenGL ES Extension 328

Dependencies

    Requires OpenGL ES 3.1 or OpenGL 4.3 for FramebufferParameteri.

Overview

    This extension defines a new framebuffer parameter,
    GL_FRAMEBUFFER_SWAP_XY_MESA, that changes the behavior of the reads and
    writes to the framebuffer attachment points. When
    GL_FRAMEBUFFER_SWAP_XY_MESA is GL_TRUE, render commands and pixel transfer
    operations access the backing store of each attachment point with an
    xy-swapped coordinate system. This xy-inversion is relative to the
    coordinate system set when GL_FRAMEBUFFER_SWAP_XY_MESA is GL_FALSE.

    Access through TexSubImage2D and similar calls will notice the effect of
    the swap when they are not attached to framebuffer objects because
    GL_FRAMEBUFFER_SWAP_XY_MESA is associated with the framebuffer object and
    not the attachment points.

    The application should notice the display width and height are also swapped
    when GL_FRAMEBUFFER_SWAP_XY_MESA is GL_TRUE.

    This extension is mainly for pre-rotation and recommended to use it with
    MESA_framebuffer_flip_x and MESA_framebuffer_flip_y to have rotated
    result.

IP Status

    None

Issues

    None

New Procedures and Functions

    None

New Types

    None

New Tokens

    Accepted by the <pname> argument of FramebufferParameteri and
    GetFramebufferParameteriv:

        GL_FRAMEBUFFER_SWAP_XY_MESA                      0x8BBD

    Interactions with OpenGL ES 3.1 and any other versions and
    extensions that provide the entry points FramebufferParameteri
    and GetFramebufferParameteriv

    Token GL_FRAMEBUFFER_SWAP_XY_MESA is accepted as the <pname> argument of
    FramebufferParameteri and GetFramebufferParameteriv.

Errors

    None

Revision History

    Version 1, 2020/04/08
      - Initial draft (Cici Ruan)
