# EXT_discard_framebuffer

Name

    EXT_discard_framebuffer

Name Strings

    GL_EXT_discard_framebuffer

Contributors

    Benji Bowman, Imagination Technologies
    John Rosasco, Apple
    Richard Schreyer, Apple
    Stuart Smith, Imagination Technologies
    Michael Swift, Apple
    
Contacts

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

Version

    Last Modified Date: September 15, 2009
    Revision: #7

Number

    OpenGL ES Extension #64

Dependencies

    OpenGL ES 1.0 is required.
    
    Written based on the wording of the OpenGL ES 2.0 specification.

    Requires OES_framebuffer_object or OpenGL ES 2.0.

Overview

    This extension provides a new command, DiscardFramebufferEXT, which 
    causes the contents of the named framebuffer attachable images to become 
    undefined.  The contents of the specified buffers are undefined until a 
    subsequent operation modifies the content, and only the modified region 
    is guaranteed to hold valid content.  Effective usage of this command 
    may provide an implementation with new optimization opportunities.

    Some OpenGL ES implementations cache framebuffer images in a small pool 
    of fast memory.  Before rendering, these implementations must load the
    existing contents of one or more of the logical buffers (color, depth, 
    stencil, etc.) into this memory.  After rendering, some or all of these 
    buffers are likewise stored back to external memory so their contents can
    be used again in the future.  In many applications, some or all of the 
    logical buffers  are cleared at the start of rendering.  If so, the 
    effort to load or store those buffers is wasted.

    Even without this extension, if a frame of rendering begins with a full-
    screen Clear, an OpenGL ES implementation may optimize away the loading
    of framebuffer contents prior to rendering the frame.  With this extension, 
    an application can use DiscardFramebufferEXT to signal that framebuffer 
    contents will no longer be needed.  In this case an OpenGL ES 
    implementation may also optimize away the storing back of framebuffer 
    contents after rendering the frame.

Issues

    1) Should DiscardFramebufferEXT's argument be a list of COLOR_ATTACHMENTx 
    enums, or should it use the same bitfield from Clear and BlitFramebuffer?
        
        RESOLVED: We'll use a sized list of framebuffer attachments.  This
        will give us some future-proofing for when MRTs and multisampled
        FBOs are supported.
        
    2) What happens if the app discards only one of the depth and stencil
    attachments, but those are backed by the same packed_depth_stencil buffer?
    
        a) Generate an error
        b) Both images become undefined
        c) Neither image becomes undefined
        d) Only one of the images becomes undefined
        
        RESOLVED: (b) which sort of falls out of Issue 4.
        
    3) How should DiscardFramebufferEXT interact with the default framebuffer?

        a) Generate an error
        b) Ignore the hint silently
        c) The contents of the specified attachments become undefined

        RESOLVED: (c), with appropriate wording to map FBO attachments to
        the corresponding default framebuffer's logical buffers

    4) What happens when you discard an attachment that doesn't exist?  This is 
    the case where a framebuffer is complete but doesn't have, for example, a
    stencil attachment, yet the app tries to discard the stencil attachment.

        a) Generate an error
        b) Ignore the hint silently

        RESOLVED: (b) for two reasons.  First, this is just a hint anyway, and
        if we required error detection, then suddenly an implementation can't
        trivially ignore it.  Second, this is consistent with Clear, which 
        ignores specified buffers that aren't present.
    
New Procedures and Functions

    void DiscardFramebufferEXT(enum target, 
                               sizei numAttachments, 
                               const enum *attachments);

New Tokens

    Accepted in the <attachments> parameter of DiscardFramebufferEXT when the
    default framebuffer is bound to <target>:

        COLOR_EXT                                0x1800
        DEPTH_EXT                                0x1801
        STENCIL_EXT                              0x1802

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment 
Operations and the Framebuffer)
    
    Introduce new section 4.5:

    "4.5 Discarding Framebuffer Contents

    The GL provides a means for discarding portions of every pixel in a 
    particular buffer, effectively leaving its contents undefined.  The 
    command

        void DiscardFramebufferEXT(enum target, 
                                   sizei numAttachments, 
                                   const enum *attachments);

    effectively signals to the GL that it need not preserve all contents of
    a bound framebuffer object.  <numAttachments> indicates how many 
    attachments are supplied in the <attachments> list.  If an attachment is 
    specified that does not exist in the framebuffer bound to <target>, it is 
    ignored.  <target> must be FRAMEBUFFER.  
    
    If a framebuffer object is bound to <target>, then <attachments> may 
    contain COLOR_ATTACHMENT0, DEPTH_ATTACHMENT, and/or STENCIL_ATTACHMENT.  If
    the framebuffer object is not complete, DiscardFramebufferEXT may be 
    ignored.

    If the default framebuffer is bound to <target>, then <attachment> may 
    contain COLOR, identifying the color buffer; DEPTH, identifying the depth 
    buffer; or STENCIL, identifying the stencil buffer."

Errors

    The error INVALID_ENUM is generated if DiscardFramebufferEXT is called
    with a <target> that is not FRAMEBUFFER.
    
    The error INVALID_ENUM is generated if DiscardFramebufferEXT is called with
    a token other than COLOR_ATTACHMENT0, DEPTH_ATTACHMENT, or 
    STENCIL_ATTACHMENT in its <attachments> list when a framebuffer object is
    bound to <target>.

    The error INVALID_ENUM is generated if DiscardFramebufferEXT is called with
    a token other than COLOR_EXT, DEPTH_EXT, or STENCIL_EXT in its 
    <attachments> list when the default framebuffer is bound to <target>.

    The error INVALID_VALUE is generated if DiscardFramebufferEXT is called 
    with <numAttachments> less than zero.

Revision History

    09/15/2009  Benj Lipchak
        Make attachments argument const enum*.
        
    09/07/2009  Benj Lipchak
        Minor clarification to overview text.
        
    08/18/2009  Benj Lipchak
        Replace null-terminated list with sized list, loosen error checking,
        and use separate attachment tokens for default framebuffers.

    07/15/2009  Benj Lipchak
        Minor changes to overview, change GLenum to enum, whitespace fixes.

    07/14/2009  Benj Lipchak
        Rename entrypoint to DiscardFramebufferEXT to follow verb/object naming
        style, and rename entire extension to match.  Replace bitfield with
        null-terminated attachment list.  Add actual spec diffs.  Update
        overview, issues list, and errors.

    04/30/2009  Richard Schreyer
        General revision, removed the combined resolve-and-discard feature.
        
    04/30/2008  Michael Swift
        First draft of extension.

TODO:
- provide examples of intended usage
