# EXT_multisampled_render_to_texture2

Name

    EXT_multisampled_render_to_texture2

Name Strings

    GL_EXT_multisampled_render_to_texture2

Contributors

    Jeff Leger
    Maurice Ribble
    Krzysztof Kosinski
    Craig Donner
    Tobias Hector
    Jan-Harald Fredriksen
    Nigel Williams

Contact

    Jeff Leger (jleger 'at' qti.qualcomm.com)

Status

    Complete

Version

    Last Modified Date: February 27, 2017
    Revision: 3

Number

    OpenGL ES Extension #275

Dependencies

    This requires support of EXT_multisample_render_to_texture or an equivalent
    extension.

    This interacts with OVR_multiview_multisampled_render_to_texture.

Overview

    The <attachment> parameters for FramebufferTexture2DMultisampleEXT is no
    longer required to be COLOR_ATTACHMENT0.  The attachment parameter now
    matches what is allowed in FramebufferTexture2D.  This means values like
    GL_COLOR_ATTACHMENTi, GL_DEPTH_ATTACHMENT, GL_STENCIL_ATTACHMENT, or
    GL_DEPTH_STENCIL_ATTACHMENT may be used.
    After the application has rendered into the mutisampled buffer, the application
    should be careful to not trigger an implicit flush by performing a client side
    read of the buffer (readpixels, copyteximage, blitframebuffer, etc) before any
    subsequent rendering which uses the contents of the buffer. This may cause the
    attachment to be downsampled before the following draw, which would potentially
    cause corruption.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    None

Additions to Section 4.4.3 of the OpenGL ES 2.0 Specification
(Renderbuffer Objects)

    Remove the following wording describing FramebufferTexture2DMultisampleEXT:

        "and have the same restrictions.  attachment must be COLOR_ATTACHMENT0."

    In the description of FramebufferTexture2DMultisampleEXT, after the sentence
    "After such a resolve, the contents of the multisample buffer become undefined.",
    add the following sentence:

        "If texture is a depth or stencil texture, the contents of the multisample
        buffer is discarded rather than resolved - equivalent to the application
        calling InvalidateFramebuffer for this attachment."

Errors

    Remove this error:
    The error INVALID_ENUM is generated if FramebufferTexture2DMultisampleEXT
    is called with an <attachment> that is not COLOR_ATTACHMENT0.

Issues

    1. How should downsampling depth/stencil surfaces be handled?

    Proposed: Ideally, when using this extension, depth/stencil attachments should
    always be discarded/invalidated by the application *before* the rendering is
    submitted.  If the application fails to do so, the implementation is not required
    to preserve the contents of those attachments.  A depth/stencil resolve is
    equivalent to InvalidateFramebuffer for those attachments.

    Revision History

    Revision 1, 2016/9/15
      - First draft of extension
    Revision 2, 2016/12/2
      - Added interaction with OVR_multiview_multisampled_render_to_texture
      - Added issue for downsampling depth
    Revision 3, 2017/02/27
      - Final version. A depth/stencil resolve is equivalent
        to InvalidateFramebuffer
